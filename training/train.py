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
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from model      import DiT, ModelConfig
from diffusion  import FlowMatching, DDIM, NormStats
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


# ---------------------------------------------------------------------------
# Normalisation helpers  — replace the two pack/unpack helpers with these
# ---------------------------------------------------------------------------

def normalize(seq: torch.Tensor, norm: NormStats) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split and normalise a packed [B, T, SD+AD] tensor."""
    SD = norm.state.mean.shape[0]
    return (
        norm.state .normalize(seq[..., :SD]),
        norm.action.normalize(seq[..., SD:]),
    )

def denormalize(
    states: torch.Tensor, actions: torch.Tensor, norm: NormStats
) -> torch.Tensor:
    """Denormalise and repack to [B, T, SD+AD]."""
    return torch.cat([
        norm.state .denormalize(states),
        norm.action.denormalize(actions),
    ], dim=-1)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(
    model:     torch.nn.Module,
    seq:       torch.Tensor,               # [B, T, SD+AD]  raw
    diffusion: Union[FlowMatching, DDIM],
    norm:      NormStats,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    grad_clip: Optional[float],
) -> dict:
    seq  = seq.to(device)
    B, T = seq.shape[:2]

    s_0, a_0 = normalize(seq, norm)                              # [B, T, SD], [B, T, AD]

    s_noise = torch.randn_like(s_0)
    a_noise = torch.randn_like(a_0)

    timesteps = diffusion.sample_timesteps(B, device, num_frames=T)   # [B, T]

    s_t = diffusion.add_noise(s_0, s_noise, timesteps)
    a_t = diffusion.add_noise(a_0, a_noise, timesteps)

    s_pred, a_pred = model(s_t, a_t, timesteps)

    s_target = diffusion.get_target(s_0, s_noise, timesteps)
    a_target = diffusion.get_target(a_0, a_noise, timesteps)

    loss = F.mse_loss(s_pred, s_target) + F.mse_loss(a_pred, a_target)

    optimizer.zero_grad()
    loss.backward()

    grad_norm = None
    if grad_clip is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()

    optimizer.step()

    metrics = {"loss": loss.item(), "t_mean": timesteps.float().mean().item()}
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

            s_0, a_0 = normalize(seq, norm)
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
    df_schedule:          str       = "pyramid",
    df_uncertainty_scale: float     = 1.0,
) -> dict:
    B, T, _ = seq.shape
    SD       = norm.state.mean.shape[0]

    s_0, a_0 = normalize(seq.to(device), norm)   # [B, T, SD], [B, T, AD]

    # Initialise from noise
    s_t = torch.randn_like(s_0)
    a_t = torch.randn_like(a_0)

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

    pred_raw = denormalize(s_t, a_t, norm)
    gt_raw   = seq.to(device)

    return {
        "mse":        F.mse_loss(pred_raw,           gt_raw          ).item(),
        "state_mse":  F.mse_loss(pred_raw[..., :SD], gt_raw[..., :SD]).item(),
        "action_mse": F.mse_loss(pred_raw[..., SD:], gt_raw[..., SD:]).item(),
        "pred":       pred_raw,
        "gt":         gt_raw,
    }
# ---------------------------------------------------------------------------
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
        val_loader = cached   # iterate directly in validate
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
    print("Computing normalisation stats…")
    norm = compute_norm_stats(train_ds)
    print(f"  state  mean={norm.state.mean.mean():.3f}  std={norm.state.std.mean():.3f}")
    print(f"  action mean={norm.action.mean.mean():.3f}  std={norm.action.std.mean():.3f}")

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

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.steps, eta_min=cfg.train.lr * 0.01
    )
    ema = EMA(model, decay=cfg.train.ema_decay)

    start_step = 0
    if cfg.resume:
        start_step = load_checkpoint(cfg.resume, model, optimizer, lr_sched, ema)
        print(f"Resumed from step {start_step}")

    cond_frames = list(cfg.log.get("inference_cond_frames", []))

    # -------------------------------------------------------------------------
    model.train()
    step, log_loss = start_step, 0.0
    print(f"Training for {cfg.train.steps} steps…")

    while step < cfg.train.steps:
        seq     = next(train_iter)
        metrics = train_step(model, seq, diffusion, norm, optimizer, device, cfg.train.grad_clip)
        lr_sched.step()
        ema.update()
        log_loss += metrics["loss"]

        # Logging
        if step % cfg.log.every == 0 and step > 0:
            avg      = log_loss / cfg.log.every
            log_loss = 0.0
            lr       = lr_sched.get_last_lr()[0]
            msg      = (f"step {step:06d} | loss {avg:.4f}"
                        f" | t_mean {metrics['t_mean']:.0f} | lr {lr:.2e}")
            if "grad_norm" in metrics:
                msg += f" | gnorm {metrics['grad_norm']:.2f}"
            print(msg)

        # Validation
        if step > 0 and step % cfg.log.val_every == 0:
            val_m = validate(model, val_loader, diffusion, norm, ema, device, cfg.log.val_batches)
            print(f"step {step:06d} | val_loss {val_m['loss']:.4f}")

        # Inference / sampling
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
                print(f"step {step:06d} | state_mse {inf['state_mse']:.4f}"
                      f" | action_mse {inf['action_mse']:.4f}")
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