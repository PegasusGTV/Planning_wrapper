"""
Guided inference script for the 2D DiT model (standard guidance only).

Usage:
    python inference_guided_dit.py --checkpoint runs/v1/checkpoints/final.pt \\
                                   --target_tcp_xy 0.5 0.3 \\
                                   --guidance_scale 500.0
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from model     import DiT, ModelConfig
from diffusion import DDIM, Stats
from dataset   import NpzDataset, compute_norm_stats

from train import (
    _vis_build_templates,
    _vis_apply_pose_9d,
    _VIS_BLOCK_COLOR,
    _VIS_HAND_COLOR,
)

_VIS_TARGET_COLOR = np.array([0.20, 0.95, 0.60])   # green


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class GuidedInferenceConfig:
    # --- Guidance ---
    guidance_scale: float  = 1000.0
    num_inner_steps: int   = 1
    max_grad_norm: float   = 3.0

    # --- Diffusion ---
    num_inference_steps: int = 50

    # --- Target: desired raw (un-normalised) TCP XY at the final frame ---
    target_tcp_xy: Optional[List[float]] = None   # [x, y]

    # --- Output ---
    num_samples: int   = 4
    output_name: str   = "guided_dit"
    no_visualize: bool = False
    gif_fps: float     = 10.0
    vis_every: int     = 1
    block_shape: str   = "cube"
    vis_elev: float    = 90.0
    vis_azim: float    = -90.0


# ============================================================================
# Guidance
# # ============================================================================

# def clip_gradient(grad: torch.Tensor, max_norm: float) -> torch.Tensor:
#     norm = grad.norm()
#     return grad * (max_norm / (norm + 1e-8)) if norm > max_norm else grad


def x0_from_eps(
    x_t: torch.Tensor,
    eps: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """x_0 = (x_t - sqrt(1 - a_t) * eps) / sqrt(a_t)"""
    a_t = alphas_cumprod[t.long()]
    return (x_t - (1.0 - a_t).sqrt() * eps) / a_t.sqrt().clamp(min=1e-8)


# def guidance_score(
#     x_0_est: torch.Tensor,        # [B, T, D]  normalised
#     target_tcp_xy: torch.Tensor,  # [B, 2]     raw
#     norm: Stats,
# ) -> torch.Tensor:
#     """
#     Negative mean L2 distance from the final-frame TCP prediction to the target.

#     Critically, we denormalise x_0 before computing distances so the gradient
#     magnitude is in meaningful units — exactly like get_score() in the original
#     inference_guided.py which calls batch.prediction_stats.denormalize() first.
#     """
#     x_0_raw  = norm.denormalize(x_0_est)    # [B, T, D]
#     pred_tcp = x_0_raw[:, -1, 0:2]          # [B, 2]  TCP XY at final frame
#     return -torch.mean(torch.norm(pred_tcp - target_tcp_xy, dim=-1))


# def guided_step(
#     model: nn.Module,
#     diffusion: DDIM,
#     norm: Stats,
#     x_t: torch.Tensor,                      # [B, T, D]
#     t: torch.Tensor,                         # scalar timestep
#     config: GuidedInferenceConfig,
#     target_tcp_xy: Optional[torch.Tensor],  # [B, 2] raw, or None
# ) -> torch.Tensor:
#     B, T_seq, D = x_t.shape
#     t_batch = t.expand(B).unsqueeze(1).expand(B, T_seq)   # [B, T]
#     ac      = diffusion.scheduler.alphas_cumprod.to(x_t.device)

#     x_guided = x_t.detach().clone()

#     if config.guidance_scale != 0.0 and target_tcp_xy is not None:
#         for _ in range(config.num_inner_steps):
#             x_guided.requires_grad_(True)

#             eps_pred = model(x_guided, t_batch)
#             x_0_est  = x0_from_eps(x_guided, eps_pred, ac, t)

#             score = guidance_score(x_0_est, target_tcp_xy, norm)
#             grad  = torch.autograd.grad(score, x_guided)[0]
#             grad  = clip_gradient(grad, config.max_grad_norm)

#             x_guided = x_guided.detach() + config.guidance_scale * grad
#             # no requires_grad_ needed here — the loop re-enables it at the top

#     with torch.no_grad():
#         eps_pred = model(x_guided.detach(), t_batch)
#         x_prev   = diffusion.scheduler.step(
#             model_output=eps_pred,
#             timestep=t,
#             sample=x_guided.detach(),
#         ).prev_sample

#     return x_prev

def clip_gradient(grad: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Clips gradients per-sequence in the batch, not globally."""
    B = grad.shape[0]
    # Compute norm per sequence: [B, 1, 1]
    norms = grad.view(B, -1).norm(dim=1).view(B, 1, 1)
    scale = max_norm / (norms + 1e-8)
    scale = torch.clamp(scale, max=1.0)
    return grad * scale

def guidance_score(
    x_0_est: torch.Tensor,        # [B, T, D]  normalised
    target_tcp_xy: torch.Tensor,  # [B, 2]     raw
    norm: Stats,
) -> torch.Tensor:
    x_0_raw  = norm.denormalize(x_0_est)    # [B, T, D]
    pred_tcp = x_0_raw[:, -1, 0:2]          # [B, 2]
    
    # guide on every frame, not just the final one — this gives a stronger signal and empirically works better
    # full_tcp = x_0_raw[:, :, 0:2]              # [B, T, 2]
    # Negative sum of L2 distances from all predicted TCPs to the target
    # return - torch.sum(torch.norm(full_tcp - target_tcp_xy.unsqueeze(1), dim=-1))
    # Use SUM so the gradient scale doesn't artificially shrink with larger batches
    return -torch.sum(torch.norm(pred_tcp - target_tcp_xy, dim=-1))

def guided_step(
    model: nn.Module,
    diffusion: DDIM,
    norm: Stats,
    x_t: torch.Tensor,
    t: torch.Tensor,
    config: GuidedInferenceConfig,
    target_tcp_xy: Optional[torch.Tensor],
) -> torch.Tensor:
    B, T_seq, D = x_t.shape
    t_batch = t.expand(B).unsqueeze(1).expand(B, T_seq)
    ac = diffusion.scheduler.alphas_cumprod.to(x_t.device)

    # 1. Standard forward pass (no gradients through the DiT)
    with torch.no_grad():
        eps_pred = model(x_t, t_batch)

    if config.guidance_scale != 0.0 and target_tcp_xy is not None:
        # We only need gradients with respect to the input x_t
        x_in = x_t.detach().requires_grad_(True)
        
        with torch.enable_grad():
            # Critically: detach eps_pred so we don't backprop through the model
            x_0_est = x0_from_eps(x_in, eps_pred.detach(), ac, t)
            
            score = guidance_score(x_0_est, target_tcp_xy, norm)
            grad  = torch.autograd.grad(score, x_in)[0]
            grad  = clip_gradient(grad, config.max_grad_norm)

        # 2. Modify the noise prediction, NOT x_t
        # score is negative distance (we want to maximize it), so grad points to target.
        # Following the classifier guidance formulation, we subtract from eps.
        a_t = ac[t.long()]
        eps_pred = eps_pred - (1.0 - a_t).sqrt() * config.guidance_scale * grad
        
        # modify the x_in instead of eps_pred — this is more stable and gives better results in practice
        # x_t = x_in + config.guidance_scale * grad
        
        # x_t = x_in + config.guidance_scale * grad
        # with torch.no_grad():
        #     eps_pred = model(x_t, t_batch)  # recompute for the new x_t

    # 3. Standard DDIM step using the modified noise
    with torch.no_grad():
        x_prev = diffusion.scheduler.step(
            model_output=eps_pred,
            timestep=t,
            sample=x_t,
        ).prev_sample

    return x_prev


# ============================================================================
# Full denoising loop
# ============================================================================

# def run_guided_inference(
#     model: nn.Module,
#     diffusion: DDIM,
#     norm: Stats,
#     x_0_raw: torch.Tensor,       # [B, T, D]
#     config: GuidedInferenceConfig,
#     device: torch.device,
# ) -> torch.Tensor:
#     """Returns denormalised predictions [B, T, D]."""
#     B, T_seq, D = x_0_raw.shape

#     # Keep target in raw space — no normalisation needed
#     target_tcp_xy = None
#     if config.target_tcp_xy is not None:
#         target_tcp_xy = torch.tensor(
#             config.target_tcp_xy, dtype=torch.float32, device=device
#         ).unsqueeze(0).expand(B, -1)   # [B, 2]

#     diffusion.scheduler.set_timesteps(config.num_inference_steps, device=device)
#     x_t = torch.randn((B, T_seq, D), device=device)

#     for t in diffusion.scheduler.timesteps:
#         x_t = guided_step(
#             model=model, diffusion=diffusion, norm=norm,
#             x_t=x_t, t=t, config=config, target_tcp_xy=target_tcp_xy,
#         )

#     return norm.denormalize(x_t)

def run_guided_inference(
    model: nn.Module,
    diffusion: DDIM,
    norm: Stats,
    x_0_raw: torch.Tensor,       # [B*num_samples, T, D]
    config: GuidedInferenceConfig,
    device: torch.device,
) -> torch.Tensor:
    """Returns denormalised predictions [B*num_samples, T, D]."""
    B, T_seq, D = x_0_raw.shape

    target_tcp_xy = None
    if config.target_tcp_xy is not None:
        target_tcp_xy = torch.tensor(
            config.target_tcp_xy, dtype=torch.float32, device=device
        ).unsqueeze(0).expand(B, -1)   # [B, 2]

    diffusion.scheduler.set_timesteps(config.num_inference_steps, device=device)
    x_t = torch.randn((B, T_seq, D), device=device)

    for t in diffusion.scheduler.timesteps:
        x_t = guided_step(
            model=model, diffusion=diffusion, norm=norm,
            x_t=x_t, t=t, config=config, target_tcp_xy=target_tcp_xy,
        )

    return norm.denormalize(x_t)


# ============================================================================
# Visualisation — mirrors train.py exactly; adds green cloud at target TCP
# ============================================================================

# def _render_frame(
#     gt_tcp_9d:     np.ndarray,
#     gt_block_9d:   np.ndarray,
#     pr_tcp_9d:     np.ndarray,
#     pr_block_9d:   np.ndarray,
#     templates:     dict,
#     lims:          list,
#     t:             int,
#     T:             int,
#     elev:          float,
#     azim:          float,
#     target_tcp_9d: Optional[np.ndarray] = None,
#     dpi:           int = 100,
# ) -> np.ndarray:
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt

#     fig = plt.figure(figsize=(10, 5), dpi=dpi)

#     for col, (tcp_9d, block_9d, label) in enumerate([
#         (gt_tcp_9d,  gt_block_9d,  f"GT   t={t}/{T-1}"),
#         (pr_tcp_9d,  pr_block_9d,  f"Pred t={t}/{T-1}"),
#     ]):
#         ax = fig.add_subplot(1, 2, col + 1, projection="3d")

#         hand_pts  = _vis_apply_pose_9d(templates["hand"],  tcp_9d)
#         block_pts = _vis_apply_pose_9d(templates["block"], block_9d)

#         ax.scatter(*block_pts.T, s=1.2, c=[_VIS_BLOCK_COLOR.tolist()],
#                    alpha=0.6, depthshade=True, label="block")
#         ax.scatter(*hand_pts.T,  s=2.5, c=[_VIS_HAND_COLOR.tolist()],
#                    alpha=0.9, depthshade=True, label="TCP")

#         if target_tcp_9d is not None:
#             target_pts = _vis_apply_pose_9d(templates["hand"], target_tcp_9d)
#             ax.scatter(*target_pts.T, s=2.5, c=[_VIS_TARGET_COLOR.tolist()],
#                        alpha=0.5, depthshade=True, label="target")

#         (xlo, xhi), (ylo, yhi), (zlo, zhi) = lims
#         ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi); ax.set_zlim(zlo, zhi)
#         ax.set_box_aspect([1, 1, 1])
#         ax.view_init(elev=elev, azim=azim)
#         ax.set_xlabel("X", fontsize=6); ax.set_ylabel("Y", fontsize=6)
#         ax.set_zlabel("Z", fontsize=6); ax.tick_params(labelsize=5)
#         ax.set_title(label, fontsize=8)
#         if col == 0:
#             ax.legend(fontsize=6, loc="upper right", markerscale=3)

#     suptitle = "Blue=block  Orange=TCP"
#     if target_tcp_9d is not None:
#         suptitle += "  Green=target"
#     fig.suptitle(suptitle, fontsize=7)
#     fig.tight_layout(pad=0.5)
#     fig.canvas.draw()
#     w, h_px = fig.canvas.get_width_height()
#     buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
#     img = buf.reshape(h_px, w, 4)[:, :, 1:]
#     plt.close(fig)
#     return img.copy()


# def visualize_predictions_guided(
#     gt_raw:   torch.Tensor,
#     pred_raw: torch.Tensor,
#     dataset,
#     save_dir: Path,
#     prefix:   str,
#     config:   GuidedInferenceConfig,
# ) -> None:
#     import imageio

#     save_dir.mkdir(parents=True, exist_ok=True)

#     gt   = gt_raw.cpu().numpy()
#     pred = pred_raw.cpu().numpy()
#     B, T, _ = gt.shape

#     tcp_z   = dataset.mean_tcp_z
#     block_z = dataset.mean_block_z
#     identity_rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

#     def make_pose_9d(xy: np.ndarray, z: float) -> np.ndarray:
#         return np.array([xy[0], xy[1], z, *identity_rot6d], dtype=np.float32)

#     target_tcp_9d = None
#     if config.target_tcp_xy is not None:
#         target_tcp_9d = make_pose_9d(np.array(config.target_tcp_xy), tcp_z)

#     rng       = np.random.default_rng(0)
#     templates = _vis_build_templates(400, 250, rng, config.block_shape)
#     frame_indices = list(range(0, T, config.vis_every))

#     for b in range(B):
#         gt_tcp_9d   = np.stack([make_pose_9d(gt[b,   t, 0:2], tcp_z)   for t in range(T)])
#         gt_block_9d = np.stack([make_pose_9d(gt[b,   t, 2:4], block_z) for t in range(T)])
#         pr_tcp_9d   = np.stack([make_pose_9d(pred[b, t, 0:2], tcp_z)   for t in range(T)])
#         pr_block_9d = np.stack([make_pose_9d(pred[b, t, 2:4], block_z) for t in range(T)])

#         all_pts_list = (
#             [_vis_apply_pose_9d(templates["hand"],  p) for p in gt_tcp_9d] +
#             [_vis_apply_pose_9d(templates["block"], p) for p in gt_block_9d] +
#             [_vis_apply_pose_9d(templates["hand"],  p) for p in pr_tcp_9d] +
#             [_vis_apply_pose_9d(templates["block"], p) for p in pr_block_9d]
#         )
#         if target_tcp_9d is not None:
#             all_pts_list.append(_vis_apply_pose_9d(templates["hand"], target_tcp_9d))

#         all_pts = np.concatenate(all_pts_list)
#         pad = 0.05
#         ctr = all_pts.mean(0)
#         r   = max((all_pts.max(0) - all_pts.min(0)).max() / 2 + pad, 0.1)
#         lims = [(float(ctr[j] - r), float(ctr[j] + r)) for j in range(3)]

#         gif_frames = []
#         for t_idx in frame_indices:
#             img = _render_frame(
#                 gt_tcp_9d    = gt_tcp_9d[t_idx],
#                 gt_block_9d  = gt_block_9d[t_idx],
#                 pr_tcp_9d    = pr_tcp_9d[t_idx],
#                 pr_block_9d  = pr_block_9d[t_idx],
#                 templates    = templates,
#                 lims         = lims,
#                 t            = t_idx,
#                 T            = T,
#                 elev         = config.vis_elev,
#                 azim         = config.vis_azim,
#                 target_tcp_9d= target_tcp_9d,
#             )
#             gif_frames.append(img)

#         out_path = save_dir / f"{prefix}_b{b:03d}.gif"
#         imageio.mimsave(str(out_path), gif_frames, duration=1.0 / config.gif_fps, loop=0)
#         print(f"  ↳ saved {out_path}")

def _render_frame(
    gt_tcp_9d:      np.ndarray,
    gt_block_9d:    np.ndarray,
    pr_tcp_9d_list: list[np.ndarray],   # one entry per sample
    pr_block_9d_list: list[np.ndarray],
    templates:      dict,
    lims:           list,
    t:              int,
    T:              int,
    elev:           float,
    azim:           float,
    target_tcp_9d:  Optional[np.ndarray] = None,
    dpi:            int = 100,
) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap

    fig = plt.figure(figsize=(10, 5), dpi=dpi)
    cmap = get_cmap("tab10")
    n_samples = len(pr_tcp_9d_list)

    for col, (tcp_9d, block_9d, label) in enumerate([
        (gt_tcp_9d, gt_block_9d, f"GT   t={t}/{T-1}"),
        (None,      None,        f"Pred t={t}/{T-1}  (n={n_samples})"),
    ]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")

        if col == 0:
            hand_pts  = _vis_apply_pose_9d(templates["hand"],  tcp_9d)
            block_pts = _vis_apply_pose_9d(templates["block"], block_9d)
            ax.scatter(*block_pts.T, s=1.2, c=[_VIS_BLOCK_COLOR.tolist()],
                       alpha=0.6, depthshade=True, label="block")
            ax.scatter(*hand_pts.T,  s=2.5, c=[_VIS_HAND_COLOR.tolist()],
                       alpha=0.9, depthshade=True, label="TCP")
        else:
            for s, (pr_tcp, pr_block) in enumerate(zip(pr_tcp_9d_list, pr_block_9d_list)):
                color = cmap(s % 10)
                hand_pts  = _vis_apply_pose_9d(templates["hand"],  pr_tcp)
                block_pts = _vis_apply_pose_9d(templates["block"], pr_block)
                ax.scatter(*block_pts.T, s=1.2, c=[color], alpha=0.4, depthshade=True)
                ax.scatter(*hand_pts.T,  s=2.5, c=[color], alpha=0.7, depthshade=True,
                           label=f"s{s}")

        if target_tcp_9d is not None:
            target_pts = _vis_apply_pose_9d(templates["hand"], target_tcp_9d)
            ax.scatter(*target_pts.T, s=2.5, c=[_VIS_TARGET_COLOR.tolist()],
                       alpha=0.5, depthshade=True, label="target")

        (xlo, xhi), (ylo, yhi), (zlo, zhi) = lims
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi); ax.set_zlim(zlo, zhi)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("X", fontsize=6); ax.set_ylabel("Y", fontsize=6)
        ax.set_zlabel("Z", fontsize=6); ax.tick_params(labelsize=5)
        ax.set_title(label, fontsize=8)
        ax.legend(fontsize=6, loc="upper right", markerscale=3)

    suptitle = "Blue=block  Orange=TCP  colours=samples"
    if target_tcp_9d is not None:
        suptitle += "  Green=target"
    fig.suptitle(suptitle, fontsize=7)
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    w, h_px = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = buf.reshape(h_px, w, 4)[:, :, 1:]
    plt.close(fig)
    return img.copy()


def visualize_predictions_guided(
    gt_raw:    torch.Tensor,
    pred_raws: list[torch.Tensor],   # all samples for this item
    dataset,
    save_dir:  Path,
    prefix:    str,
    config:    GuidedInferenceConfig,
) -> None:
    import imageio

    save_dir.mkdir(parents=True, exist_ok=True)

    gt    = gt_raw.cpu().numpy()
    preds = [p.cpu().numpy() for p in pred_raws]
    B, T, _ = gt.shape

    tcp_z   = dataset.mean_tcp_z
    block_z = dataset.mean_block_z
    identity_rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    def make_pose_9d(xy: np.ndarray, z: float) -> np.ndarray:
        return np.array([xy[0], xy[1], z, *identity_rot6d], dtype=np.float32)

    target_tcp_9d = None
    if config.target_tcp_xy is not None:
        target_tcp_9d = make_pose_9d(np.array(config.target_tcp_xy), tcp_z)

    rng       = np.random.default_rng(0)
    templates = _vis_build_templates(400, 250, rng, config.block_shape)
    frame_indices = list(range(0, T, config.vis_every))

    for b in range(B):
        gt_tcp_9d   = np.stack([make_pose_9d(gt[b, t, 0:2], tcp_z)   for t in range(T)])
        gt_block_9d = np.stack([make_pose_9d(gt[b, t, 2:4], block_z) for t in range(T)])

        pr_tcp_9d_list   = [np.stack([make_pose_9d(p[b, t, 0:2], tcp_z)   for t in range(T)]) for p in preds]
        pr_block_9d_list = [np.stack([make_pose_9d(p[b, t, 2:4], block_z) for t in range(T)]) for p in preds]

        all_pts_list = (
            [_vis_apply_pose_9d(templates["hand"],  p) for p in gt_tcp_9d] +
            [_vis_apply_pose_9d(templates["block"], p) for p in gt_block_9d] +
            [pt for tcp_list in pr_tcp_9d_list   for pt in [_vis_apply_pose_9d(templates["hand"],  p) for p in tcp_list]] +
            [pt for blk_list in pr_block_9d_list for pt in [_vis_apply_pose_9d(templates["block"], p) for p in blk_list]]
        )
        if target_tcp_9d is not None:
            all_pts_list.append(_vis_apply_pose_9d(templates["hand"], target_tcp_9d))

        all_pts = np.concatenate(all_pts_list)
        pad = 0.05
        ctr = all_pts.mean(0)
        r   = max((all_pts.max(0) - all_pts.min(0)).max() / 2 + pad, 0.1)
        lims = [(float(ctr[j] - r), float(ctr[j] + r)) for j in range(3)]

        gif_frames = []
        for t_idx in frame_indices:
            img = _render_frame(
                gt_tcp_9d        = gt_tcp_9d[t_idx],
                gt_block_9d      = gt_block_9d[t_idx],
                pr_tcp_9d_list   = [arr[t_idx] for arr in pr_tcp_9d_list],
                pr_block_9d_list = [arr[t_idx] for arr in pr_block_9d_list],
                templates        = templates,
                lims             = lims,
                t                = t_idx,
                T                = T,
                elev             = config.vis_elev,
                azim             = config.vis_azim,
                target_tcp_9d    = target_tcp_9d,
            )
            gif_frames.append(img)

        out_path = save_dir / f"{prefix}_b{b:03d}.gif"
        imageio.mimsave(str(out_path), gif_frames, duration=1.0 / config.gif_fps, loop=0)
        print(f"  ↳ saved {out_path}")

# ============================================================================
# Model loading
# ============================================================================

def load_model_and_diffusion(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg  = OmegaConf.create(ckpt["cfg"])

    model_cfg = ModelConfig(**OmegaConf.to_container(cfg.model))
    model     = DiT(model_cfg).to(device).eval()

    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"], strict=False)
        print("Loaded EMA weights")
    else:
        model.load_state_dict(ckpt["model"], strict=False)
        print("Loaded regular weights")

    diffusion = DDIM(
        num_train_timesteps=cfg.ddim.num_train_timesteps,
        beta_schedule=cfg.ddim.beta_schedule,
        clip_sample=cfg.ddim.clip_sample,
        clip_sample_range=cfg.ddim.clip_sample_range,
    )
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model, diffusion, cfg


# ============================================================================
# Main inference loop
# ============================================================================

def run_inference(args, config: GuidedInferenceConfig):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, diffusion, train_cfg = load_model_and_diffusion(args.checkpoint, device)

    train_ds = NpzDataset(
        train_cfg.data.train_path,
        train_cfg.data.seq_len,
        train_cfg.data.stride,
    )
    norm = compute_norm_stats(train_ds)

    data_path = args.data_path or train_cfg.data.val_path
    val_ds    = NpzDataset(data_path, train_cfg.data.seq_len,
                           getattr(train_cfg.data, "stride", 1))

    output_dir = Path("./inference_results") / config.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    print(f"\n{'='*60}")
    print("GUIDED INFERENCE — DiT model")
    print(f"  guidance_scale  : {config.guidance_scale}")
    print(f"  num_inner_steps : {config.num_inner_steps}")
    print(f"  inference_steps : {config.num_inference_steps}")
    print(f"  num_samples     : {config.num_samples}")
    print(f"  target_tcp_xy   : {config.target_tcp_xy}")
    print(f"{'='*60}\n")

    all_mse   = []
    max_items = args.max_items or len(val_ds)

    # for item_idx in tqdm(range(min(max_items, len(val_ds))), desc="Processing"):
    #     seq = val_ds[item_idx]
    #     if seq.ndim == 2:
    #         seq = seq.unsqueeze(0)
    #     seq = seq.to(device)

    #     all_preds = []
    #     for _ in range(config.num_samples):
    #         pred_raw = run_guided_inference(
    #             model=model, diffusion=diffusion, norm=norm,
    #             x_0_raw=seq, config=config, device=device,
    #         )
    #         all_preds.append(pred_raw)

    #     mse_vals = [F.mse_loss(p, seq).item() for p in all_preds]
    #     avg_mse  = float(np.mean(mse_vals))
    #     all_mse.append(avg_mse)

    #     if args.verbose:
    #         # Also print how close the final TCP is to the target
    #         if config.target_tcp_xy is not None:
    #             target = torch.tensor(config.target_tcp_xy, device=device)
    #             dists  = [torch.norm(p[0, -1, 0:2] - target).item() for p in all_preds]
    #             print(f"  item {item_idx:04d} | avg MSE = {avg_mse:.4f} "
    #                   f"| tcp dist to target = {np.mean(dists):.4f} "
    #                   f"(best {min(dists):.4f})")
    #         else:
    #             print(f"  item {item_idx:04d} | avg MSE = {avg_mse:.4f}")

    #     if not config.no_visualize:
    #         viz_dir = output_dir / "visualizations"
    #         for s, pred in enumerate(all_preds):
    #             visualize_predictions_guided(
    #                 gt_raw   = seq,
    #                 pred_raw = pred,
    #                 dataset  = val_ds,
    #                 save_dir = viz_dir,
    #                 prefix   = f"item{item_idx:04d}_s{s}",
    #                 config   = config,
    #             )
    
    for item_idx in tqdm(range(min(max_items, len(val_ds))), desc="Processing"):
        seq = val_ds[item_idx]
        if seq.ndim == 2:
            seq = seq.unsqueeze(0)               # [1, T, D]
        seq = seq.to(device)

        # tile the single item into a batch of num_samples
        seq_batched = seq.expand(config.num_samples, -1, -1)  # [num_samples, T, D]

        pred_raw = run_guided_inference(
            model=model, diffusion=diffusion, norm=norm,
            x_0_raw=seq_batched, config=config, device=device,
        )                                        # [num_samples, T, D]

        # split back into a list so the rest of the code is unchanged
        all_preds = [pred_raw[s].unsqueeze(0) for s in range(config.num_samples)]

        mse_vals = [F.mse_loss(p, seq).item() for p in all_preds]
        avg_mse  = float(np.mean(mse_vals))
        all_mse.append(avg_mse)

        if args.verbose:
            if config.target_tcp_xy is not None:
                target = torch.tensor(config.target_tcp_xy, device=device)
                dists  = [torch.norm(p[0, -1, 0:2] - target).item() for p in all_preds]
                print(f"  item {item_idx:04d} | avg MSE = {avg_mse:.4f} "
                    f"| tcp dist to target = {np.mean(dists):.4f} "
                    f"(best {min(dists):.4f})")
            else:
                print(f"  item {item_idx:04d} | avg MSE = {avg_mse:.4f}")

        if not config.no_visualize:
            viz_dir = output_dir / "visualizations"
            visualize_predictions_guided(
                gt_raw    = seq,
                pred_raws = all_preds,
                dataset   = val_ds,
                save_dir  = viz_dir,
                prefix    = f"item{item_idx:04d}",
                config    = config,
            )

    avg_mse_total = float(np.nanmean(all_mse))
    print(f"\n{'='*60}")
    print(f"RESULTS  ({len(all_mse)} items)")
    print(f"  Avg MSE (raw): {avg_mse_total:.6f}")
    print(f"{'='*60}\n")

    metrics_out = {"num_items": len(all_mse), "avg_mse": avg_mse_total, "config": asdict(config)}
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Saved metrics → {output_dir / 'metrics.json'}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Guided inference for the DiT model")

    parser.add_argument("--checkpoint",          type=str,   required=True)
    parser.add_argument("--data_path",           type=str,   default=None)
    parser.add_argument("--max_items",           type=int,   default=None)

    parser.add_argument("--guidance_scale",      type=float, default=None)
    parser.add_argument("--num_inner_steps",     type=int,   default=None)
    parser.add_argument("--max_grad_norm",       type=float, default=200)
    parser.add_argument("--num_inference_steps", type=int,   default=None)
    parser.add_argument("--num_samples",         type=int,   default=None)

    parser.add_argument("--target_tcp_xy", type=float, nargs=2, default=None,
                        metavar=("X", "Y"),
                        help="Raw-space TCP XY target at the final frame.")

    parser.add_argument("--output_name",   type=str,   default=None)
    parser.add_argument("--no_visualize",  action="store_true")
    parser.add_argument("--gif_fps",       type=float, default=None)
    parser.add_argument("--vis_every",     type=int,   default=None)
    parser.add_argument("--block_shape",   type=str,   default="circle",
                        choices=["cube", "T", "circle"])
    parser.add_argument("--vis_elev",      type=float, default=None)
    parser.add_argument("--vis_azim",      type=float, default=None)
    parser.add_argument("--device",        type=str,   default="cuda")
    parser.add_argument("--verbose",       action="store_true")

    args = parser.parse_args()

    config = GuidedInferenceConfig()
    overrides = {
        "guidance_scale":       args.guidance_scale,
        "num_inner_steps":      args.num_inner_steps,
        "max_grad_norm":        args.max_grad_norm,
        "num_inference_steps":  args.num_inference_steps,
        "num_samples":          args.num_samples,
        "target_tcp_xy":        args.target_tcp_xy,
        "output_name":          args.output_name,
        "no_visualize":         True if args.no_visualize else None,
        "gif_fps":              args.gif_fps,
        "vis_every":            args.vis_every,
        "block_shape":          args.block_shape,
        "vis_elev":             args.vis_elev,
        "vis_azim":             args.vis_azim,
    }
    for field, value in overrides.items():
        if value is not None:
            setattr(config, field, value)

    run_inference(args, config)


if __name__ == "__main__":
    main()