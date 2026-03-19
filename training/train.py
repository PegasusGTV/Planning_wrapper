"""
Training script for the unified 2D DiT — single GPU.

Usage
-----
  python train.py --config config.yaml
  python train.py --config config.yaml --resume runs/v1/checkpoints/step_0010000.pt
  python train.py --config config.yaml model.depth=8 train.lr=2e-4
"""

import argparse
from itertools import cycle
import math
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from model      import DiT, ModelConfig
from diffusion  import FlowMatching, DDIM, NormStats, Stats
from dataset    import NpzDataset, compute_norm_stats
from scheduling import generate_scheduling_matrix


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.model   = model
        self.decay   = decay
        self.shadow  = {n: p.data.clone()
                        for n, p in model.named_parameters() if p.requires_grad}
        self._backup: dict = {}

    @torch.no_grad()
    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply_shadow(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self._backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self._backup:
                p.data.copy_(self._backup[n])
        self._backup.clear()

    def state_dict(self):          return {k: v.clone() for k, v in self.shadow.items()}
    def load_state_dict(self, sd): self.shadow = {k: v.clone() for k, v in sd.items()}
    
class SNRWeighting:
    def __init__(self, diffusion, snr_clip: float = 5.0):
        if isinstance(diffusion, DDIM):
            ac  = diffusion.scheduler.alphas_cumprod
            snr = ac / (1.0 - ac)
        else:  # FlowMatching — approximate SNR from t
            # t=0 → clean, t=1 → noise; SNR ≈ (1-t)^2 / t^2
            t   = torch.linspace(1e-3, 1.0 - 1e-3, diffusion.num_train_timesteps)
            snr = ((1.0 - t) / t) ** 2

        self.snr         = snr
        self.clipped_snr = snr.clone().clamp_(max=snr_clip)

    def weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        # timesteps: [B, T] → weights: [B, T]
        snr         = self.snr        .to(timesteps.device)[timesteps.long()]
        clipped_snr = self.clipped_snr.to(timesteps.device)[timesteps.long()]
        return clipped_snr / snr.clamp(min=1e-8)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

# def normalize(seq: torch.Tensor, norm: NormStats) -> Tuple[torch.Tensor, torch.Tensor]:
#     """Split and normalise a packed [B, T, SD+AD] tensor."""
#     SD = norm.state.mean.shape[0]
#     return (
#         norm.state .normalize(seq[..., :SD]),
#         norm.action.normalize(seq[..., SD:]),
#     )

# def denormalize(
#     states: torch.Tensor, actions: torch.Tensor, norm: NormStats
# ) -> torch.Tensor:
#     """Denormalise and repack to [B, T, SD+AD]."""
#     return torch.cat([
#         norm.state .denormalize(states),
#         norm.action.denormalize(actions),
#     ], dim=-1)

def normalize(seq: torch.Tensor, norm: Stats, SD: int) -> Tuple[torch.Tensor, torch.Tensor]:
    bundle = norm.normalize(seq)      
    return bundle[..., :SD], bundle[..., SD:]

def denormalize(states: torch.Tensor, actions: torch.Tensor, norm: Stats) -> torch.Tensor:
    bundle = torch.cat([states, actions], dim=-1)
    return norm.denormalize(bundle)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(
    model:     torch.nn.Module,
    seq:       torch.Tensor,
    diffusion: Union[FlowMatching, DDIM],
    norm:      NormStats,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    grad_clip: Optional[float],
    snr:       SNRWeighting,                  # ← add this
) -> dict:
    seq  = seq.to(device)
    B, T = seq.shape[:2]

    s_0, a_0  = normalize(seq, norm, model.state_dim)
    # s_noise   = torch.randn_like(s_0)
    # a_noise   = torch.randn_like(a_0)
    s_noise = torch.randn_like(s_0).clamp(-20.0, 20.0)
    a_noise = torch.randn_like(a_0).clamp(-20.0, 20.0)
    timesteps = diffusion.sample_timesteps(B, device, num_frames=T)

    # Condition first frame at low noise ~50% of the time
    # if torch.rand(1).item() < 0.5:
    #     max_cond_t = diffusion.num_train_timesteps // 4
    #     timesteps[:, 0] = torch.randint(0, max_cond_t, (B,), device=device)

    s_t = diffusion.add_noise(s_0, s_noise, timesteps)
    a_t = diffusion.add_noise(a_0, a_noise, timesteps)

    s_pred, a_pred = model(s_t, a_t, timesteps)

    s_target = diffusion.get_target(s_0, s_noise, timesteps)
    a_target = diffusion.get_target(a_0, a_noise, timesteps)

    # Per-token SNR weights, shape [B, T] → [B, T, 1] for broadcasting
    w = snr.weights(timesteps).unsqueeze(-1)

    s_loss = (F.mse_loss(s_pred, s_target, reduction="none") * w).mean()
    a_loss = (F.mse_loss(a_pred, a_target, reduction="none") * w).mean()
    loss   = s_loss + a_loss

    optimizer.zero_grad()
    loss.backward()

    grad_norm = None
    if grad_clip is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()

    optimizer.step()

    metrics = {
        "loss": loss.item(), "s_loss": s_loss.item(), "a_loss": a_loss.item(),
        "t_mean": timesteps.float().mean().item(),
    }
    if grad_norm is not None:
        metrics["grad_norm"] = grad_norm
    return metrics


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model:       torch.nn.Module,
    val_loader:  DataLoader,
    diffusion:   Union[FlowMatching, DDIM],
    norm:        NormStats,
    ema:         EMA,
    device:      torch.device,
    num_batches: int = 10,
) -> dict:
    ema.apply_shadow()
    model.eval()
    total, count = 0.0, 0
    try:
        for seq in val_loader:
            if count >= num_batches:
                break
            seq  = seq.to(device)
            B, T = seq.shape[:2]

            s_0, a_0 = normalize(seq, norm, model.state_dim)
            s_noise, a_noise = torch.randn_like(s_0), torch.randn_like(a_0)
            ts = diffusion.sample_timesteps(B, device, num_frames=T)

            s_t = diffusion.add_noise(s_0, s_noise, ts)
            a_t = diffusion.add_noise(a_0, a_noise, ts)

            s_pred, a_pred = model(s_t, a_t, ts)

            loss = (
                F.mse_loss(s_pred, diffusion.get_target(s_0, s_noise, ts))
                + F.mse_loss(a_pred, diffusion.get_target(a_0, a_noise, ts))
            )
            total += loss.item()
            count += 1
    finally:
        ema.restore()
        model.train()
    return {"loss": total / max(count, 1)}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model:                torch.nn.Module,
    seq:                  torch.Tensor,        # [B, T, SD+AD]  raw
    diffusion:            Union[FlowMatching, DDIM],
    norm:                 NormStats,
    n_steps:              int,
    device:               torch.device,
    cond_frames:          List[int] = (),
    df_schedule:          str       = "full_sequence",
    df_uncertainty_scale: float     = 1.0,
) -> dict:
    B, T, _ = seq.shape
    SD       = model.state_dim

    s_0, a_0 = normalize(seq.to(device), norm, SD)   # [B, T, SD], [B, T, AD]
    cond_frames = None
    # Initialise from noise
    # s_t = torch.randn_like(s_0)
    # a_t = torch.randn_like(a_0)
    s_t = torch.randn_like(s_0).clamp(-20.0, 20.0)
    a_t = torch.randn_like(a_0).clamp(-20.0, 20.0)

    if cond_frames:
        cond_idx = list(cond_frames)
        s_t[:, cond_idx] = s_0[:, cond_idx]
        a_t[:, cond_idx] = a_0[:, cond_idx]

    # Scheduling matrix (over non-conditioned frames only)
    gen_mask    = torch.ones(T, dtype=torch.bool, device=device)
    if cond_frames:
        gen_mask[list(cond_frames)] = False
    gen_indices = gen_mask.nonzero(as_tuple=True)[0]
    n_gen       = len(gen_indices)

    K_full  = generate_scheduling_matrix(n_gen, n_steps, df_schedule, df_uncertainty_scale)
    K_float = K_full.astype(np.float64) / n_steps * diffusion.num_train_timesteps
    K_ts    = torch.from_numpy(
        np.clip(K_float.astype(np.int64), 0, diffusion.num_train_timesteps - 1)
    ).to(device)   # [M, n_gen]

    is_fm = isinstance(diffusion, FlowMatching)
    if not is_fm:
        ac = diffusion.scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)
        
    for m in range(len(K_ts) - 1):
        curr_k = K_ts[m]
        next_k = K_ts[m + 1]

        t_batch = torch.zeros(B, T, device=device, dtype=torch.long)
        t_batch[:, gen_indices] = curr_k.unsqueeze(0).expand(B, -1)

        s_pred, a_pred = model(s_t, a_t, t_batch)   # [B, T, SD], [B, T, AD]

        if is_fm:
            dt    = (curr_k - next_k).float() / diffusion.num_train_timesteps
            s_new = s_t[:, gen_indices] - dt[None, :, None] * s_pred[:, gen_indices]
            a_new = a_t[:, gen_indices] - dt[None, :, None] * a_pred[:, gen_indices]
        else:
            a_t_coef  = ac[curr_k][None, :, None]
            a_p_coef  = ac[next_k][None, :, None]
            s_eps, a_eps = s_pred[:, gen_indices], a_pred[:, gen_indices]
            s_x0  = (s_t[:, gen_indices] - (1 - a_t_coef).sqrt() * s_eps) / a_t_coef.sqrt().clamp(min=1e-8)
            a_x0  = (a_t[:, gen_indices] - (1 - a_t_coef).sqrt() * a_eps) / a_t_coef.sqrt().clamp(min=1e-8)
            s_new = a_p_coef.sqrt() * s_x0 + (1 - a_p_coef).sqrt() * s_eps
            a_new = a_p_coef.sqrt() * a_x0 + (1 - a_p_coef).sqrt() * a_eps

        changed = (curr_k != next_k)[None, :, None].expand(B, n_gen, -1)
        s_t[:, gen_indices] = torch.where(changed.expand_as(s_new), s_new, s_t[:, gen_indices])
        a_t[:, gen_indices] = torch.where(changed.expand_as(a_new), a_new, a_t[:, gen_indices])

        if cond_frames:
            s_t[:, cond_idx] = s_0[:, cond_idx]
            a_t[:, cond_idx] = a_0[:, cond_idx]
            
    s_mse_norm = F.mse_loss(s_t, s_0).item()
    a_mse_norm = F.mse_loss(a_t, a_0).item()

    pred_raw = denormalize(s_t, a_t, norm)
    gt_raw   = seq.to(device)
    
    # In run_inference, add before denormalization:
    

    return {
        "mse":        F.mse_loss(pred_raw, gt_raw).item(),
        "state_mse":  F.mse_loss(pred_raw[..., :SD], gt_raw[..., :SD]).item(),
        "action_mse": F.mse_loss(pred_raw[..., SD:], gt_raw[..., SD:]).item(),
        "state_mse_norm":  s_mse_norm,    # ← add these
        "action_mse_norm": a_mse_norm,
        "pred": pred_raw,
        "gt":   gt_raw,
    }

    # return {
    #     "mse":        F.mse_loss(pred_raw,           gt_raw          ).item(),
    #     "state_mse":  F.mse_loss(pred_raw[..., :SD], gt_raw[..., :SD]).item(),
    #     "action_mse": F.mse_loss(pred_raw[..., SD:], gt_raw[..., SD:]).item(),
    #     "pred":       pred_raw,
    #     "gt":         gt_raw,
    # }


# ---------------------------------------------------------------------------
# 3D point-cloud visualisation helpers (mirrors scripts/vis_lowdim.py)
# ---------------------------------------------------------------------------

_VIS_BLOCK_COLOR = np.array([0.20, 0.47, 0.96])   # blue
_VIS_HAND_COLOR  = np.array([0.91, 0.47, 0.10])   # orange
_VIS_GHOST_COLOR = np.array([0.20, 0.95, 0.60])   # green

_VIS_BLOCK_HALF = 0.025
_VIS_T_BOXES = [
    (np.array([0.0, -0.0375/2, 0.0]),                np.array([0.05, 0.0125, 0.02])),
    (np.array([0.0, 4*0.0125 - 0.0375/2, 0.0]),      np.array([0.0125, 0.075*0.75, 0.02])),
]
_VIS_STICK_RADIUS = 0.008
_VIS_STICK_LENGTH = 0.10
_VIS_STICK_OFFSET_Z = -0.05


def _vis_rot6d_to_rotmat(r6: np.ndarray) -> np.ndarray:
    r1 = r6[:3].astype(np.float64)
    r2 = r6[3:].astype(np.float64)
    a1 = r1 / np.linalg.norm(r1)
    a2 = r2 - np.dot(r2, a1) * a1
    a2 = a2 / np.linalg.norm(a2)
    a3 = np.cross(a1, a2)
    return np.stack([a1, a2, a3], axis=-1)


def _vis_apply_pose_9d(pts: np.ndarray, pose_9d: np.ndarray) -> np.ndarray:
    pos = pose_9d[:3].astype(np.float64)
    R   = _vis_rot6d_to_rotmat(pose_9d[3:9])
    return (pts.astype(np.float64) @ R.T + pos).astype(np.float32)


def _vis_sample_box(offset, half, n, rng):
    n_init = n * 20
    hx, hy, hz = half
    areas  = np.array([4*hy*hz, 4*hx*hz, 4*hx*hy], dtype=np.float64)
    counts = np.round(areas / areas.sum() * n_init).astype(int)
    counts[-1] += n_init - counts.sum()
    pts = []
    if counts[0] > 0:
        s = rng.choice([-1.0, 1.0], counts[0])
        pts.append(np.stack([s*hx, rng.uniform(-hy,hy,counts[0]), rng.uniform(-hz,hz,counts[0])], 1))
    if counts[1] > 0:
        s = rng.choice([-1.0, 1.0], counts[1])
        pts.append(np.stack([rng.uniform(-hx,hx,counts[1]), s*hy, rng.uniform(-hz,hz,counts[1])], 1))
    if counts[2] > 0:
        s = rng.choice([-1.0, 1.0], counts[2])
        pts.append(np.stack([rng.uniform(-hx,hx,counts[2]), rng.uniform(-hy,hy,counts[2]), s*hz], 1))
    initial = np.concatenate(pts).astype(np.float32) + np.asarray(offset, dtype=np.float32)
    idx = rng.choice(len(initial), size=min(n, len(initial)), replace=False)
    return initial[idx]


def _vis_sample_cylinder(offset_z, radius, length, n, rng):
    n_init = n * 20
    th  = rng.uniform(0, 2*np.pi, n_init)
    z   = rng.uniform(-length/2, length/2, n_init) + offset_z
    pts = np.stack([radius*np.cos(th), radius*np.sin(th), z], 1).astype(np.float32)
    idx = rng.choice(len(pts), size=min(n, len(pts)), replace=False)
    return pts[idx]


def _vis_build_templates(n_block: int, n_hand: int, rng, use_T: bool) -> dict:
    if use_T:
        areas     = [2*(4*hy*hz+4*hx*hz+4*hx*hy) for _, (hx,hy,hz) in _VIS_T_BOXES]
        total     = sum(areas)
        t_parts   = [_vis_sample_box(off, half, max(1, int(n_block*a/total)), rng)
                     for (off, half), a in zip(_VIS_T_BOXES, areas)]
        block_tpl = np.concatenate(t_parts)
    else:
        block_tpl = _vis_sample_box(
            np.zeros(3), np.full(3, _VIS_BLOCK_HALF), n_block, rng
        )
    hand_tpl = _vis_sample_cylinder(
        _VIS_STICK_OFFSET_Z, _VIS_STICK_RADIUS, _VIS_STICK_LENGTH, n_hand, rng
    )
    return {"block": block_tpl, "hand": hand_tpl}


def _vis_render_single_frame(
    states_gt:   np.ndarray,   # [18]
    actions_gt:  np.ndarray,   # [9]
    states_pr:   np.ndarray,   # [18]
    actions_pr:  np.ndarray,   # [9]
    templates:   dict,
    lims:        list,          # [(xlo,xhi),(ylo,yhi),(zlo,zhi)]
    t:           int,
    T:           int,
    elev:        float,
    azim:        float,
    ghost_alpha: float,
    cond_frames: List[int],
    dpi:         int = 100,
) -> np.ndarray:
    """Render one GIF frame: GT (left) and Pred (right) side-by-side 3D scatter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 5), dpi=dpi)

    for col, (states, actions, label) in enumerate([
        (states_gt, actions_gt, f"GT   t={t}/{T-1}"),
        (states_pr, actions_pr, f"Pred t={t}/{T-1}"),
    ]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")

        tcp_9d   = states[0:9]
        block_9d = states[9:18]
        act_9d   = actions[0:9]

        hand_pts  = _vis_apply_pose_9d(templates["hand"],  tcp_9d)
        block_pts = _vis_apply_pose_9d(templates["block"], block_9d)
        ghost_pts = _vis_apply_pose_9d(templates["hand"],  act_9d)

        ax.scatter(block_pts[:,0], block_pts[:,1], block_pts[:,2],
                   s=1.2, c=[_VIS_BLOCK_COLOR.tolist()], depthshade=True,
                   alpha=0.6, label="block")
        ax.scatter(hand_pts[:,0],  hand_pts[:,1],  hand_pts[:,2],
                   s=2.5, c=[_VIS_HAND_COLOR.tolist()],  depthshade=True,
                   alpha=0.9, label="TCP (now)")
        ax.scatter(ghost_pts[:,0], ghost_pts[:,1], ghost_pts[:,2],
                   s=2.5, c=[_VIS_GHOST_COLOR.tolist()], depthshade=True,
                   alpha=ghost_alpha, label="action target")

        if t in cond_frames:
            ax.scatter(*hand_pts.mean(0), marker="D", s=60,
                       c=[[1.0, 1.0, 0.2]], zorder=6, alpha=0.9)

        (xlo, xhi), (ylo, yhi), (zlo, zhi) = lims
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi); ax.set_zlim(zlo, zhi)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("X", fontsize=6); ax.set_ylabel("Y", fontsize=6)
        ax.set_zlabel("Z", fontsize=6); ax.tick_params(labelsize=5)
        ax.set_title(label, fontsize=8)
        if col == 0:
            ax.legend(fontsize=6, loc="upper right", markerscale=3)

    fig.suptitle(
        "Blue=block  Orange=TCP  Green=action target  ◆=cond frame",
        fontsize=7,
    )
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    w, h_px = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = buf.reshape(h_px, w, 4)[:, :, 1:]   # ARGB -> RGB
    plt.close(fig)
    return img.copy()


# ---------------------------------------------------------------------------
# Prediction visualisation — one GIF per sample, GT vs Pred side-by-side
# ---------------------------------------------------------------------------

def visualize_predictions(
    model:       torch.nn.Module,
    seq:         torch.Tensor,             # [B, T, SD+AD] raw
    diffusion:   Union[FlowMatching, DDIM],
    norm:        NormStats,
    n_steps:     int,
    device:      torch.device,
    save_dir:    Path,
    step:        int,
    cond_frames: List[int] = (),
    n_samples:   int = 4,
    use_T_block: bool = False,
    every:       int = 1,
    n_block_pts: int = 400,
    n_hand_pts:  int = 250,
    elev:        float = 30.0,
    azim:        float = -60.0,
    ghost_alpha: float = 0.35,
    gif_fps:     float = 10.0,
) -> None:
    """
    Run inference on n_samples trajectories and save one GIF per sample to
        <save_dir>/step_XXXXXXX/sample_N.gif

    Each GIF animates over the T timesteps (subsampled by `every`).
    Every frame shows GT (left) and Predicted (right) as 3D point clouds,
    matching the exact style of scripts/vis_lowdim.py.
    """
    import imageio

    n_samples = min(n_samples, seq.shape[0])
    sub_seq   = seq[:n_samples].to(device)

    model.eval()
    with torch.no_grad():
        result = run_inference(
            model, sub_seq, diffusion, norm,
            n_steps=n_steps, device=device,
            cond_frames=cond_frames,
        )
    model.train()

    SD      = model.state_dim   # 18
    pred_np = result["pred"].cpu().numpy()   # [n_samples, T, 27]
    gt_np   = result["gt"].cpu().numpy()

    # Fixed-seed templates so point cloud shape doesn't jitter across calls
    rng       = np.random.default_rng(0)
    templates = _vis_build_templates(n_block_pts, n_hand_pts, rng, use_T_block)

    step_dir = save_dir / f"step_{step:07d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    T      = gt_np.shape[1]
    frames = list(range(0, T, every))

    for i in range(n_samples):
        gt_states   = gt_np  [i, :, :SD]   # [T, 18]
        gt_actions  = gt_np  [i, :, SD:]   # [T, 9]
        pr_states   = pred_np[i, :, :SD]
        pr_actions  = pred_np[i, :, SD:]

        # Axis limits computed from GT so GT and Pred share the same scale
        # all_pos = np.concatenate([gt_states[:, 0:3], gt_states[:, 9:12]], axis=0)

        # With this:
        all_pos = np.concatenate([
            gt_states[:, 0:3], gt_states[:, 9:12],
            pr_states[:, 0:3], pr_states[:, 9:12],
        ], axis=0)
        all_pos = all_pos[np.isfinite(all_pos).all(axis=1)]  # filter NaN/inf from bad preds
        pad = 0.06
        ctr = all_pos.mean(0)
        r   = max((all_pos.max(0) - all_pos.min(0)).max() / 2 + pad, 0.08)
        lims = [(float(ctr[j] - r), float(ctr[j] + r)) for j in range(3)]

        gif_frames = []
        for t in frames:
            img = _vis_render_single_frame(
                states_gt   = gt_states[t],
                actions_gt  = gt_actions[t],
                states_pr   = pr_states[t],
                actions_pr  = pr_actions[t],
                templates   = templates,
                lims        = lims,
                t           = t,
                T           = T,
                elev        = elev,
                azim        = azim,
                ghost_alpha = ghost_alpha,
                cond_frames = cond_frames,
            )
            gif_frames.append(img)

        out_path = step_dir / f"sample_{i}.gif"
        imageio.mimsave(str(out_path), gif_frames,
                        duration=1.0 / gif_fps, loop=0)
        print(f"  ↳ saved GIF -> {out_path}")

# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path: Path, model, ema: EMA, optimizer, lr_sched, step: int, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step":      step,
        "model":     model.state_dict(),
        "ema":       ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": lr_sched.state_dict(),
        "cfg":       OmegaConf.to_container(cfg),
    }, path)
    print(f"  ✓ saved {path}")


def load_checkpoint(path: str, model, optimizer, lr_sched, ema: EMA) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    lr_sched.load_state_dict(ckpt["scheduler"])
    if "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    return ckpt["step"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg):
    torch.manual_seed(cfg.seed)

    # 1. Check for NVIDIA GPU
    if torch.cuda.is_available():
        device_name = "cuda"
    # 2. Check for Apple Silicon GPU (Metal Performance Shaders)
    elif torch.backends.mps.is_available():
        device_name = "mps"
    # 3. Final Fallback
    else:
        device_name = "cpu"

    device = torch.device(cfg.get("device", device_name))
    print(f"Device: {device}")

    run_dir  = Path(cfg.log.dir) / cfg.run_name
    ckpt_dir = run_dir / "checkpoints"
    vis_dir  = run_dir / "visualisations"   # ← new: where PNGs are saved
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    print(f"Run: {run_dir}")

    # Data
    train_ds = NpzDataset(cfg.data.train_path, cfg.data.seq_len)
    val_ds   = NpzDataset(cfg.data.val_path,   cfg.data.seq_len)

    if cfg.train.overfit_batches > 0:
        print(f"Overfit mode: caching {cfg.train.overfit_batches} batch(es)…")
        tmp_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True)
        cached     = [next(iter(tmp_loader)) for _ in range(cfg.train.overfit_batches)]
        train_iter = cycle(cached)
        val_loader = cached
    else:
        train_loader = DataLoader(
            train_ds, batch_size=cfg.data.batch_size, shuffle=True,
            num_workers=cfg.data.num_workers, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.data.batch_size, shuffle=False,
            num_workers=cfg.data.num_workers, pin_memory=True,
        )
        train_iter = cycle(train_loader)

    # Normalisation stats
    # print("Computing normalisation stats…")
    # norm = compute_norm_stats(train_ds)
    # print(f"  state  mean={norm.state.mean.mean():.3f}  std={norm.state.std.mean():.3f}")
    # print(f"  action mean={norm.action.mean.mean():.3f}  std={norm.action.std.mean():.3f}")
    # Change this:
    # print(f"  state  mean={norm.state.mean.mean():.3f}  std={norm.state.std.mean():.3f}")
    # print(f"  action mean={norm.action.mean.mean():.3f}  std={norm.action.std.mean():.3f}")

    # To this:
    norm = compute_norm_stats(train_ds)
    SD = train_ds.state_dim  # or however you access it
    print(f"  state  mean={norm.mean[:SD].mean():.3f}  std={norm.std[:SD].mean():.3f}")
    print(f"  action mean={norm.mean[SD:].mean():.3f}  std={norm.std[SD:].mean():.3f}")

    # Model
    model_cfg = ModelConfig(**OmegaConf.to_container(cfg.model))
    model     = DiT(model_cfg).to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    # Diffusion backend
    if cfg.denoise_method == "flow_matching":
        diffusion   = FlowMatching(
            num_train_timesteps=cfg.flow_matching.num_train_timesteps,
            shift=cfg.flow_matching.shift,
        )
        n_inf_steps = cfg.flow_matching.inference_steps
    elif cfg.denoise_method == "ddim":
        diffusion   = DDIM(
            num_train_timesteps=cfg.ddim.num_train_timesteps,
            beta_schedule=cfg.ddim.beta_schedule,
            clip_sample=cfg.ddim.clip_sample,
            clip_sample_range=cfg.ddim.clip_sample_range,
        )
        n_inf_steps = cfg.ddim.inference_steps
    else:
        raise ValueError(f"Unknown denoise_method: {cfg.denoise_method!r}")
    
    snr_weighting = SNRWeighting(diffusion, snr_clip=cfg.train.get("snr_clip", 5.0))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    # With this:
    warmup_steps = cfg.train.get("warmup_steps", 10000)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        # cosine decay after warmup
        progress = (step - warmup_steps) / max(cfg.train.steps - warmup_steps, 1)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ema = EMA(model, decay=cfg.train.ema_decay)

    start_step = 0
    if cfg.resume:
        start_step = load_checkpoint(cfg.resume, model, optimizer, lr_sched, ema)
        print(f"Resumed from step {start_step}")

    cond_frames = list(cfg.log.get("inference_cond_frames", []))

    # Stash a fixed visualisation batch sampled with wide stride so the
    # n_samples sequences come from very different parts of the dataset —
    # consecutive windows from an unshuffled loader would be nearly identical
    # because NpzDataset slides a 1-step window, making samples 0,1,2,3
    # differ by only one timestep.
    _vis_batch: Optional[torch.Tensor] = None
    try:
        n_vis = cfg.log.get("vis_n_samples", 4)
        ds_len = len(val_ds)
        # Pick n_vis indices spread uniformly across the dataset
        vis_indices = [int(i * ds_len / n_vis) for i in range(n_vis)]
        _vis_batch = torch.stack([val_ds[idx] for idx in vis_indices])
    except Exception as e:
        print(f"  [warn] could not build vis batch: {e}")

    # -------------------------------------------------------------------------
    model.train()
    step, log_loss = start_step, 0.0
    print(f"Training for {cfg.train.steps} steps…")

    while step < cfg.train.steps:
        seq     = next(train_iter)
        metrics = train_step(
            model, seq, diffusion, norm, optimizer, device,
            cfg.train.grad_clip, snr_weighting,        # ← add snr_weighting
        )
        lr_sched.step()
        ema.update()
        log_loss += metrics["loss"]

        # Logging
        if step % cfg.log.every == 0 and step > 0:
            avg      = log_loss / cfg.log.every
            log_loss = 0.0
            lr       = lr_sched.get_last_lr()[0]
            # In the logging block, extend the msg:
            msg = (f"step {step:06d} | loss {avg:.4f}"
                f" | s_loss {metrics['s_loss']:.4f} | a_loss {metrics['a_loss']:.4f}"
                f" | t_mean {metrics['t_mean']:.0f} | lr {lr:.2e}")
            if "grad_norm" in metrics:
                msg += f" | gnorm {metrics['grad_norm']:.2f}"
            print(msg)

        # Validation + visualisation (same frequency)
        if step > 0 and step % cfg.log.val_every == 0:
            val_m = validate(model, val_loader, diffusion, norm, ema, device, cfg.log.val_batches)
            print(f"step {step:06d} | val_loss {val_m['loss']:.4f}")

            # ── Prediction visualisation ─────────────────────────────────────
            if _vis_batch is not None:
                ema.apply_shadow()
                try:
                    visualize_predictions(
                        model       = model,
                        seq         = _vis_batch,
                        diffusion   = diffusion,
                        norm        = norm,
                        n_steps     = n_inf_steps,
                        device      = device,
                        save_dir    = vis_dir,
                        step        = step,
                        cond_frames = cond_frames,
                        n_samples   = cfg.log.get("vis_n_samples",   4),
                        use_T_block = cfg.log.get("vis_use_T_block",  False),
                        every       = cfg.log.get("vis_every",        5),
                        ghost_alpha = cfg.log.get("vis_ghost_alpha",  0.35),
                        elev        = cfg.log.get("vis_elev",         30.0),
                        azim        = cfg.log.get("vis_azim",        -60.0),
                    )
                except Exception as e:
                    print(f"  [warn] visualisation failed at step {step}: {e}")
                finally:
                    ema.restore()
                    model.train()

        # Inference / sampling (kept at its own cadence)
        if step > 0 and step % cfg.log.sample_every == 0:
            ema.apply_shadow()
            model.eval()
            try:
                inf = run_inference(
                    model, seq.to(device), diffusion, norm,
                    n_steps=n_inf_steps,
                    device=device,
                    cond_frames=cond_frames,
                )
                # print(f"step {step:06d} | state_mse {inf['state_mse']:.4f}"
                #       f" | action_mse {inf['action_mse']:.4f}")
                print(f"step {step:06d} | state_mse {inf['state_mse']:.4f} (norm: {inf['state_mse_norm']:.4f})"
                        f" | action_mse {inf['action_mse']:.4f} (norm: {inf['action_mse_norm']:.4f})")
            finally:
                ema.restore()
                model.train()

        # Checkpoint
        if step > 0 and step % cfg.log.save_every == 0:
            save_checkpoint(
                ckpt_dir / f"step_{step:07d}.pt",
                model, ema, optimizer, lr_sched, step, cfg,
            )

        step += 1

    save_checkpoint(
        ckpt_dir / "final.pt", model, ema, optimizer, lr_sched, step, cfg,
    )
    print("Training complete!")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--resume",  default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    if args.resume:
        cfg.resume = args.resume

    print(OmegaConf.to_yaml(cfg))
    main(cfg)