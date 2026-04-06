# """
# Training script for the unified 2D DiT — single GPU.

# Usage
# -----
#   python train.py --config config.yaml
#   python train.py --config config.yaml --resume runs/v1/checkpoints/step_0010000.pt
#   python train.py --config config.yaml model.depth=8 train.lr=2e-4
# """

# import argparse
# from itertools import cycle
# import math
# from pathlib import Path
# from typing import List, Optional, Tuple
# import numpy as np
# import torch
# import torch.nn.functional as F
# from omegaconf import OmegaConf
# from torch.utils.data import DataLoader

# from model      import DiT, ModelConfig
# from diffusion  import DDIM
# from dataset    import NpzDataset, DeltaStats, compute_norm_stats
# from scheduling import generate_scheduling_matrix


# # ---------------------------------------------------------------------------
# # EMA
# # ---------------------------------------------------------------------------

# class EMA:
#     def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
#         self.model   = model
#         self.decay   = decay
#         self.shadow  = {n: p.data.clone()
#                         for n, p in model.named_parameters() if p.requires_grad}
#         self._backup: dict = {}

#     @torch.no_grad()
#     def update(self):
#         for n, p in self.model.named_parameters():
#             if p.requires_grad and n in self.shadow:
#                 self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

#     def apply_shadow(self):
#         for n, p in self.model.named_parameters():
#             if p.requires_grad and n in self.shadow:
#                 self._backup[n] = p.data.clone()
#                 p.data.copy_(self.shadow[n])

#     def restore(self):
#         for n, p in self.model.named_parameters():
#             if p.requires_grad and n in self._backup:
#                 p.data.copy_(self._backup[n])
#         self._backup.clear()

#     def state_dict(self):          return {k: v.clone() for k, v in self.shadow.items()}
#     def load_state_dict(self, sd): self.shadow = {k: v.clone() for k, v in sd.items()}


# # ---------------------------------------------------------------------------
# # Delta → absolute position
# # ---------------------------------------------------------------------------

# def delta_to_abs(delta: np.ndarray, init_xy: np.ndarray) -> np.ndarray:
#     """[T, 2] per-frame deltas + [2] initial position → [T, 2] absolute."""
#     return init_xy[None, :] + np.cumsum(delta, axis=0)


# # ---------------------------------------------------------------------------
# # Training step
# # ---------------------------------------------------------------------------

# def train_step(
#     model:                torch.nn.Module,
#     seq:                  torch.Tensor,        # [B, T, 6]  raw
#     diffusion:            DDIM,
#     norm:                 DeltaStats,
#     optimizer:            torch.optim.Optimizer,
#     device:               torch.device,
#     grad_clip:            Optional[float],
#     delta_loss_weight:    float = 1.0,
#     relative_loss_weight: float = 0.1,
# ) -> dict:
#     seq   = seq.to(device)
#     x_0   = norm.normalize(seq)           # [B, T, 6]  in [-1, 1]
#     B, T  = x_0.shape[:2]

#     noise     = torch.randn_like(x_0)
#     timesteps = diffusion.sample_timesteps(B, device, num_frames=T)
#     x_t       = diffusion.add_noise(x_0, noise, timesteps)

#     x_pred = model(x_t, timesteps)        # [B, T, 6]

#     delta_loss    = F.mse_loss(x_pred[..., :4], noise[..., :4])
#     relative_loss = F.mse_loss(x_pred[..., 4:], noise[..., 4:])
#     loss = delta_loss_weight * delta_loss + relative_loss_weight * relative_loss

#     optimizer.zero_grad()
#     loss.backward()

#     grad_norm = None
#     if grad_clip is not None:
#         grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()

#     optimizer.step()

#     return {
#         "loss":          loss.item(),
#         "delta_loss":    delta_loss.item(),
#         "relative_loss": relative_loss.item(),
#         "t_mean":        timesteps.float().mean().item(),
#         **({"grad_norm": grad_norm} if grad_norm is not None else {}),
#     }


# # ---------------------------------------------------------------------------
# # Validation  (noise-prediction MSE in normalised space)
# # ---------------------------------------------------------------------------

# @torch.no_grad()
# def validate(
#     model:       torch.nn.Module,
#     val_loader:  DataLoader,
#     diffusion:   DDIM,
#     norm:        DeltaStats,
#     ema:         EMA,
#     device:      torch.device,
#     num_batches: int = 10,
# ) -> dict:
#     ema.apply_shadow()
#     model.eval()
#     total_delta = total_rel = count = 0
#     try:
#         for seq in val_loader:
#             if count >= num_batches:
#                 break
#             x_0   = norm.normalize(seq.to(device))
#             noise = torch.randn_like(x_0)
#             ts    = diffusion.sample_timesteps(x_0.shape[0], device, num_frames=x_0.shape[1])
#             x_t   = diffusion.add_noise(x_0, noise, ts)
#             pred  = model(x_t, ts)
#             total_delta += F.mse_loss(pred[..., :4], noise[..., :4]).item()
#             total_rel   += F.mse_loss(pred[..., 4:], noise[..., 4:]).item()
#             count += 1
#     finally:
#         ema.restore()
#         model.train()
#     n = max(count, 1)
#     return {"delta_loss": total_delta / n, "relative_loss": total_rel / n}


# # ---------------------------------------------------------------------------
# # Inference MSE  (full DDIM sampling, MSE in normalised space)
# #
# # Why normalised space?
# # ---------------------
# # The scheduling-matrix DDIM x0 estimate is:
# #   x0 = (x_t - sqrt(1-alpha)*eps) / sqrt(alpha)
# # When alpha≈0 (high noise timestep), sqrt(alpha)→0 and the estimate explodes
# # even with the 1e-8 clamp, making raw-space MSE meaningless until the model
# # is near-perfectly trained.  Reporting MSE in the normalised space in which
# # the model operates gives a stable, interpretable signal from step 1.
# # ---------------------------------------------------------------------------

# @torch.no_grad()
# def run_inference(
#     model:                torch.nn.Module,
#     seq:                  torch.Tensor,        # [B, T, 6]  raw
#     diffusion:            DDIM,
#     norm:                 DeltaStats,
#     n_steps:              int,
#     device:               torch.device,
#     cond_frames:          List[int] = (),
#     df_schedule:          str       = "pyramid",
#     df_uncertainty_scale: float     = 1.0,
# ) -> dict:
#     """
#     Runs full DDIM denoising and returns MSE metrics in *normalised* space.
#     This avoids the numerical blow-up that occurs when computing x0 estimates
#     at high-noise timesteps in raw (denormalised) coordinates.
#     """
#     B, T, _ = seq.shape

#     x_0_norm = norm.normalize(seq.to(device))   # [B, T, 6]  ground truth in [-1, 1]
#     x_t      = torch.randn_like(x_0_norm)

#     if cond_frames:
#         x_t[:, list(cond_frames)] = x_0_norm[:, list(cond_frames)]

#     gen_mask    = torch.ones(T, dtype=torch.bool, device=device)
#     if cond_frames:
#         gen_mask[list(cond_frames)] = False
#     gen_indices = gen_mask.nonzero(as_tuple=True)[0]
#     n_gen       = len(gen_indices)

#     K_full  = generate_scheduling_matrix(n_gen, n_steps, df_schedule, df_uncertainty_scale)
#     K_float = K_full.astype(np.float64) / n_steps * diffusion.num_train_timesteps
#     K_ts    = torch.from_numpy(
#         np.clip(K_float.astype(np.int64), 0, diffusion.num_train_timesteps - 1)
#     ).to(device)   # [M, n_gen]

#     ac = diffusion.scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)

#     for m in range(len(K_ts) - 1):
#         curr_k = K_ts[m]       # [n_gen]
#         next_k = K_ts[m + 1]   # [n_gen]

#         t_batch = torch.zeros(B, T, device=device, dtype=torch.long)
#         t_batch[:, gen_indices] = curr_k.unsqueeze(0).expand(B, -1)

#         x_pred = model(x_t, t_batch)   # [B, T, 6]

#         # --- DDIM update only for generated frames ---
#         # Shape: [1, n_gen, 1] to broadcast over [B, n_gen, 6]
#         a_t = ac[curr_k][None, :, None]   # alpha_t  [1, n_gen, 1]
#         a_p = ac[next_k][None, :, None]   # alpha_{t-1}

#         eps  = x_pred[:, gen_indices]     # predicted noise  [B, n_gen, 6]
#         xt_g = x_t[:, gen_indices]        # current noisy    [B, n_gen, 6]

#         # Predict x0 — safe because we only update frames where alpha > 0
#         # (K_ts never has curr_k == 0 except at the final step which we skip)
#         x0   = (xt_g - (1 - a_t).sqrt() * eps) / a_t.sqrt().clamp(min=1e-6)
#         x0   = x0.clamp(-5.0, 5.0)   # guard rail: keeps x0 in a sane range
#         x_new = a_p.sqrt() * x0 + (1 - a_p).sqrt() * eps

#         changed = (curr_k != next_k)[None, :, None].expand_as(x_new)
#         x_t[:, gen_indices] = torch.where(changed, x_new, xt_g)

#         if cond_frames:
#             x_t[:, list(cond_frames)] = x_0_norm[:, list(cond_frames)]

#     # --- MSE in normalised space ---
#     # x_t should now be ≈ x_0_norm; comparing in this space is numerically stable.
#     pred_norm = x_t   # [B, T, 6]  denoised, in [-1, 1] ideally
#     gt_norm   = x_0_norm

#     return {
#         "mse":          F.mse_loss(pred_norm,           gt_norm          ).item(),
#         "agent_mse":    F.mse_loss(pred_norm[..., :2],  gt_norm[..., :2] ).item(),
#         "block_mse":    F.mse_loss(pred_norm[..., 2:4], gt_norm[..., 2:4]).item(),
#         "relative_mse": F.mse_loss(pred_norm[..., 4:],  gt_norm[..., 4:] ).item(),
#         # Also expose denormalised pred for downstream use (e.g. custom eval)
#         "pred": norm.denormalize(pred_norm),
#         "gt":   seq.to(device),
#     }


# # ---------------------------------------------------------------------------
# # Point-cloud visualisation helpers
# # ---------------------------------------------------------------------------

# _VIS_BLOCK_COLOR = np.array([0.20, 0.47, 0.96])
# _VIS_HAND_COLOR  = np.array([0.91, 0.47, 0.10])
# _VIS_CUBE_HALF     = 0.025
# _VIS_CIRCLE_RADIUS = 0.025
# _VIS_T_BOXES = [
#     (np.array([0.0, -0.0375/2,             0.0]), np.array([0.05,       0.0125,      0.02])),
#     (np.array([0.0,  4*0.0125 - 0.0375/2, 0.0]), np.array([0.0125, 0.075*0.75,      0.02])),
# ]
# _VIS_STICK_RADIUS = 0.008
# _VIS_STICK_LENGTH = 0.10


# def _vis_sample_box(offset, half, n, rng):
#     n_init = n * 20
#     hx, hy, hz = half
#     areas  = np.array([4*hy*hz, 4*hx*hz, 4*hx*hy], dtype=np.float64)
#     counts = np.round(areas / areas.sum() * n_init).astype(int)
#     counts[-1] += n_init - counts.sum()
#     pts = []
#     if counts[0] > 0:
#         s = rng.choice([-1.0, 1.0], counts[0])
#         pts.append(np.stack([s*hx, rng.uniform(-hy, hy, counts[0]), rng.uniform(-hz, hz, counts[0])], 1))
#     if counts[1] > 0:
#         s = rng.choice([-1.0, 1.0], counts[1])
#         pts.append(np.stack([rng.uniform(-hx, hx, counts[1]), s*hy, rng.uniform(-hz, hz, counts[1])], 1))
#     if counts[2] > 0:
#         s = rng.choice([-1.0, 1.0], counts[2])
#         pts.append(np.stack([rng.uniform(-hx, hx, counts[2]), rng.uniform(-hy, hy, counts[2]), s*hz], 1))
#     pts = np.concatenate(pts).astype(np.float32) + offset.astype(np.float32)
#     return pts[rng.choice(len(pts), size=min(n, len(pts)), replace=False)]


# def _vis_sample_cylinder_z(radius, length, n, rng):
#     n_init = n * 20
#     th  = rng.uniform(0, 2 * np.pi, n_init)
#     z   = rng.uniform(-length / 2, length / 2, n_init)
#     pts = np.stack([radius * np.cos(th), radius * np.sin(th), z], axis=1).astype(np.float32)
#     return pts[rng.choice(len(pts), size=min(n, len(pts)), replace=False)]


# def _vis_sample_circle_block(radius, n, rng):
#     n_init = n * 20
#     th  = rng.uniform(0, 2 * np.pi, n_init)
#     z   = rng.uniform(-radius, radius, n_init)
#     pts = np.stack([radius * np.cos(th), radius * np.sin(th), z], axis=1).astype(np.float32)
#     return pts[rng.choice(len(pts), size=min(n, len(pts)), replace=False)]


# def _vis_build_templates(n_block, n_hand, rng, block_shape="cube"):
#     if block_shape == "T":
#         areas = [2*(4*hy*hz + 4*hx*hz + 4*hx*hy) for _, (hx, hy, hz) in _VIS_T_BOXES]
#         total = sum(areas)
#         parts = [_vis_sample_box(off, half, max(1, int(n_block * a / total)), rng)
#                  for (off, half), a in zip(_VIS_T_BOXES, areas)]
#         block_tpl = np.concatenate(parts)
#     elif block_shape == "circle":
#         block_tpl = _vis_sample_circle_block(_VIS_CIRCLE_RADIUS, n_block, rng)
#     else:
#         block_tpl = _vis_sample_box(np.zeros(3), np.full(3, _VIS_CUBE_HALF), n_block, rng)
#     return {"block": block_tpl,
#             "hand":  _vis_sample_cylinder_z(_VIS_STICK_RADIUS, _VIS_STICK_LENGTH, n_hand, rng)}


# def _vis_rot6d_to_rotmat(r6):
#     a1 = r6[:3] / np.linalg.norm(r6[:3])
#     a2 = r6[3:] - np.dot(r6[3:], a1) * a1
#     a2 = a2 / np.linalg.norm(a2)
#     return np.stack([a1, a2, np.cross(a1, a2)], axis=-1)


# def _vis_apply_pose_9d(pts, pose_9d):
#     pos = pose_9d[:3].astype(np.float64)
#     R   = _vis_rot6d_to_rotmat(pose_9d[3:9].astype(np.float64))
#     return (pts.astype(np.float64) @ R.T + pos).astype(np.float32)


# def _vis_render_frame(gt_tcp_9d, gt_block_9d, pr_tcp_9d, pr_block_9d,
#                       templates, lims, t, T, elev, azim, dpi=100):
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt

#     fig = plt.figure(figsize=(10, 5), dpi=dpi)
#     for col, (tcp_9d, block_9d, label) in enumerate([
#         (gt_tcp_9d,  gt_block_9d,  f"GT   t={t}/{T-1}"),
#         (pr_tcp_9d,  pr_block_9d,  f"Pred t={t}/{T-1}"),
#     ]):
#         ax = fig.add_subplot(1, 2, col + 1, projection="3d")
#         ax.scatter(*_vis_apply_pose_9d(templates["block"], block_9d).T,
#                    s=1.2, c=[_VIS_BLOCK_COLOR.tolist()], alpha=0.6, depthshade=True, label="block")
#         ax.scatter(*_vis_apply_pose_9d(templates["hand"],  tcp_9d).T,
#                    s=2.5, c=[_VIS_HAND_COLOR.tolist()],  alpha=0.9, depthshade=True, label="TCP")
#         (xlo, xhi), (ylo, yhi), (zlo, zhi) = lims
#         ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi); ax.set_zlim(zlo, zhi)
#         ax.set_box_aspect([1, 1, 1])
#         ax.view_init(elev=elev, azim=azim)
#         ax.set_xlabel("X", fontsize=6); ax.set_ylabel("Y", fontsize=6)
#         ax.set_zlabel("Z", fontsize=6); ax.tick_params(labelsize=5)
#         ax.set_title(label, fontsize=8)
#         if col == 0:
#             ax.legend(fontsize=6, loc="upper right", markerscale=3)
#     fig.suptitle("Blue = block   Orange = TCP", fontsize=7)
#     fig.tight_layout(pad=0.5)
#     fig.canvas.draw()
#     w, h_px = fig.canvas.get_width_height()
#     buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
#     img = buf.reshape(h_px, w, 4)[:, :, 1:]
#     plt.close(fig)
#     return img.copy()


# def visualize_predictions(
#     model,
#     dataset:          NpzDataset,
#     seq:              torch.Tensor,    # [B, T, 6]  raw
#     initial_abs_xy:   np.ndarray,      # [B, 4]
#     diffusion:        DDIM,
#     norm:             DeltaStats,
#     n_steps:          int,
#     device:           torch.device,
#     save_dir:         Path,
#     step:             int,
#     n_samples:        int   = 4,
#     block_shape:      str   = "cube",
#     every:            int   = 1,
#     elev:             float = 90.0,
#     azim:             float = -90.0,
#     gif_fps:          float = 10.0,
# ) -> None:
#     import imageio

#     n_samples = min(n_samples, seq.shape[0])
#     sub_seq   = seq[:n_samples].to(device)
#     init_xy   = initial_abs_xy[:n_samples]

#     model.eval()
#     with torch.no_grad():
#         x_t = torch.randn_like(norm.normalize(sub_seq))
#         diffusion.scheduler.set_timesteps(n_steps, device=device)
#         for t_step in diffusion.scheduler.timesteps:
#             t_batch = t_step.expand(n_samples).unsqueeze(1).expand(n_samples, x_t.shape[1])
#             x_t     = diffusion.scheduler.step(model(x_t, t_batch), t_step, x_t).prev_sample
#     model.train()

#     pred = norm.denormalize(x_t).cpu().numpy()   # [B, T, 6]
#     gt   = sub_seq.cpu().numpy()

#     tcp_z          = dataset.mean_tcp_z
#     block_z        = dataset.mean_block_z
#     identity_rot6d = np.array([1., 0., 0., 0., 1., 0.], dtype=np.float64)

#     def _pose_9d(xy, z):
#         return np.array([xy[0], xy[1], z, *identity_rot6d], dtype=np.float64)

#     rng           = np.random.default_rng(0)
#     templates     = _vis_build_templates(400, 250, rng, block_shape)
#     T_len         = gt.shape[1]
#     frame_indices = list(range(0, T_len, every))

#     step_dir = save_dir / f"step_{step:07d}"
#     step_dir.mkdir(parents=True, exist_ok=True)

#     for i in range(n_samples):
#         init_tcp_xy   = init_xy[i, :2]
#         init_block_xy = init_xy[i, 2:]

#         gt_tcp_abs   = delta_to_abs(gt[i,   :, :2],  init_tcp_xy)
#         gt_block_abs = delta_to_abs(gt[i,   :, 2:4], init_block_xy)
#         pr_tcp_abs   = delta_to_abs(pred[i, :, :2],  init_tcp_xy)
#         pr_block_abs = delta_to_abs(pred[i, :, 2:4], init_block_xy)

#         gt_tcp_9d   = np.stack([_pose_9d(gt_tcp_abs[t],   tcp_z)   for t in range(T_len)])
#         gt_block_9d = np.stack([_pose_9d(gt_block_abs[t], block_z) for t in range(T_len)])
#         pr_tcp_9d   = np.stack([_pose_9d(pr_tcp_abs[t],   tcp_z)   for t in range(T_len)])
#         pr_block_9d = np.stack([_pose_9d(pr_block_abs[t], block_z) for t in range(T_len)])

#         all_pts = np.concatenate([
#             *[_vis_apply_pose_9d(templates["hand"],  p) for p in gt_tcp_9d],
#             *[_vis_apply_pose_9d(templates["block"], p) for p in gt_block_9d],
#             *[_vis_apply_pose_9d(templates["hand"],  p) for p in pr_tcp_9d],
#             *[_vis_apply_pose_9d(templates["block"], p) for p in pr_block_9d],
#         ])
#         pad  = 0.05
#         ctr  = all_pts.mean(0)
#         r    = max((all_pts.max(0) - all_pts.min(0)).max() / 2 + pad, 0.1)
#         lims = [(float(ctr[j] - r), float(ctr[j] + r)) for j in range(3)]

#         gif_frames = [
#             _vis_render_frame(
#                 gt_tcp_9d=gt_tcp_9d[t_idx], gt_block_9d=gt_block_9d[t_idx],
#                 pr_tcp_9d=pr_tcp_9d[t_idx], pr_block_9d=pr_block_9d[t_idx],
#                 templates=templates, lims=lims, t=t_idx, T=T_len, elev=elev, azim=azim,
#             )
#             for t_idx in frame_indices
#         ]

#         out_path = step_dir / f"sample_{i}.gif"
#         imageio.mimsave(str(out_path), gif_frames, duration=1.0 / gif_fps, loop=0)
#         print(f"  -> saved {out_path}")


# # ---------------------------------------------------------------------------
# # Checkpoint helpers
# # ---------------------------------------------------------------------------

# def save_checkpoint(path, model, ema, optimizer, lr_sched, step, cfg):
#     path.parent.mkdir(parents=True, exist_ok=True)
#     torch.save({
#         "step": step, "model": model.state_dict(), "ema": ema.state_dict(),
#         "optimizer": optimizer.state_dict(), "scheduler": lr_sched.state_dict(),
#         "cfg": OmegaConf.to_container(cfg),
#     }, path)
#     print(f"  saved {path}")


# def load_checkpoint(path, model, optimizer, lr_sched, ema) -> int:
#     ckpt = torch.load(path, map_location="cpu")
#     model.load_state_dict(ckpt["model"])
#     optimizer.load_state_dict(ckpt["optimizer"])
#     lr_sched.load_state_dict(ckpt["scheduler"])
#     if "ema" in ckpt:
#         ema.load_state_dict(ckpt["ema"])
#     return ckpt["step"]


# # ---------------------------------------------------------------------------
# # Main
# # ---------------------------------------------------------------------------

# def main(cfg):
#     torch.manual_seed(cfg.seed)

#     device_name = ("cuda" if torch.cuda.is_available()
#                    else "mps" if torch.backends.mps.is_available() else "cpu")
#     device = torch.device(cfg.get("device", device_name))
#     print(f"Device: {device}")

#     run_dir  = Path(cfg.log.dir) / cfg.run_name
#     ckpt_dir = run_dir / "checkpoints"
#     vis_dir  = run_dir / "visualisations"
#     run_dir.mkdir(parents=True, exist_ok=True)
#     ckpt_dir.mkdir(exist_ok=True)
#     OmegaConf.save(cfg, run_dir / "config.yaml")
#     print(f"Run: {run_dir}")

#     train_ds = NpzDataset(cfg.data.train_path, cfg.data.seq_len, cfg.data.stride)
#     val_ds   = NpzDataset(cfg.data.val_path,   cfg.data.seq_len, cfg.data.stride)

#     if cfg.train.overfit_batches > 0:
#         print(f"Overfit mode: caching {cfg.train.overfit_batches} batch(es)...")
#         tmp_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True)
#         cached     = [next(iter(tmp_loader)) for _ in range(cfg.train.overfit_batches)]
#         train_iter = cycle(cached)
#         val_loader = cached
#     else:
#         train_loader = DataLoader(
#             train_ds, batch_size=cfg.data.batch_size, shuffle=True,
#             num_workers=cfg.data.num_workers, pin_memory=True, drop_last=True,
#         )
#         val_loader = DataLoader(
#             val_ds, batch_size=cfg.data.batch_size, shuffle=False,
#             num_workers=cfg.data.num_workers, pin_memory=True,
#         )
#         train_iter = cycle(train_loader)

#     norm = compute_norm_stats(train_ds)

#     model_cfg = ModelConfig(**OmegaConf.to_container(cfg.model))
#     model     = DiT(model_cfg).to(device)
#     print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

#     diffusion = DDIM(
#         num_train_timesteps=cfg.ddim.num_train_timesteps,
#         beta_schedule=cfg.ddim.beta_schedule,
#         clip_sample=cfg.ddim.clip_sample,
#         clip_sample_range=cfg.ddim.clip_sample_range,
#     )
#     n_inf_steps = cfg.ddim.inference_steps

#     optimizer = torch.optim.AdamW(
#         model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
#     )

#     warmup_steps = cfg.train.get("warmup_steps", 10000)
#     def lr_lambda(step):
#         if step < warmup_steps:
#             return step / max(warmup_steps, 1)
#         progress = (step - warmup_steps) / max(cfg.train.steps - warmup_steps, 1)
#         return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

#     lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
#     ema      = EMA(model, decay=cfg.train.ema_decay)

#     delta_w    = cfg.train.get("delta_loss_weight",    1.0)
#     relative_w = cfg.train.get("relative_loss_weight", 1.0)

#     start_step = 0
#     if cfg.resume:
#         start_step = load_checkpoint(cfg.resume, model, optimizer, lr_sched, ema)
#         print(f"Resumed from step {start_step}")

#     _vis_batch:      Optional[torch.Tensor] = None
#     _vis_initial_xy: Optional[np.ndarray]   = None
#     try:
#         n_vis       = cfg.log.get("vis_n_samples", 4)
#         vis_indices = [int(i * len(val_ds) / n_vis) for i in range(n_vis)]
#         _vis_batch      = torch.stack([val_ds[i]                for i in vis_indices])
#         _vis_initial_xy = np.stack(  [val_ds.get_abs_initial(i) for i in vis_indices])
#     except Exception as e:
#         print(f"  [warn] could not build vis batch: {e}")

#     model.train()
#     step = start_step
#     log_loss = log_delta = log_rel = 0.0
#     print(f"Training for {cfg.train.steps} steps...")

#     while step < cfg.train.steps:
#         seq     = next(train_iter)
#         metrics = train_step(
#             model, seq, diffusion, norm, optimizer, device, cfg.train.grad_clip,
#             delta_loss_weight=delta_w, relative_loss_weight=relative_w,
#         )
#         lr_sched.step()
#         ema.update()
#         log_loss  += metrics["loss"]
#         log_delta += metrics["delta_loss"]
#         log_rel   += metrics["relative_loss"]

#         if step % cfg.log.every == 0 and step > 0:
#             n   = cfg.log.every
#             lr  = lr_sched.get_last_lr()[0]
#             msg = (f"step {step:06d}"
#                    f" | loss {log_loss/n:.4f}"
#                    f" | delta {log_delta/n:.4f}"
#                    f" | rel {log_rel/n:.4f}"
#                    f" | t_mean {metrics['t_mean']:.0f}"
#                    f" | lr {lr:.2e}")
#             if "grad_norm" in metrics:
#                 msg += f" | gnorm {metrics['grad_norm']:.2f}"
#             print(msg)
#             log_loss = log_delta = log_rel = 0.0

#         if step > 0 and step % cfg.log.val_every == 0:
#             val_m = validate(model, val_loader, diffusion, norm, ema, device, cfg.log.val_batches)
#             # Both metrics are noise-prediction MSE in normalised space.
#             # A well-trained model should drive these toward 0; random baseline ≈ 1.0.
#             print(f"step {step:06d}"
#                   f" | val_delta {val_m['delta_loss']:.4f}"
#                   f" | val_rel {val_m['relative_loss']:.4f}")

#             if _vis_batch is not None and _vis_initial_xy is not None:
#                 ema.apply_shadow()
#                 try:
#                     visualize_predictions(
#                         model=model, dataset=val_ds, seq=_vis_batch,
#                         initial_abs_xy=_vis_initial_xy, diffusion=diffusion,
#                         norm=norm, n_steps=n_inf_steps, device=device,
#                         save_dir=vis_dir, step=step,
#                         n_samples   = cfg.log.get("vis_n_samples",   4),
#                         block_shape = cfg.log.get("vis_block_shape", "cube"),
#                         every       = cfg.log.get("vis_every",       1),
#                         elev        = cfg.log.get("vis_elev",        30.0),
#                         azim        = cfg.log.get("vis_azim",       -60.0),
#                         gif_fps     = cfg.log.get("vis_gif_fps",     10.0),
#                     )
#                 except Exception as e:
#                     import traceback; traceback.print_exc()
#                     print(f"  [warn] visualisation failed: {e}")
#                 finally:
#                     ema.restore()
#                     model.train()

#         # Inference MSE — all values in normalised space, should decrease from ~1.0
#         if step > 0 and step % cfg.log.sample_every == 0:
#             ema.apply_shadow()
#             model.eval()
#             try:
#                 inf = run_inference(
#                     model=model, seq=next(train_iter).to(device),
#                     diffusion=diffusion, norm=norm, n_steps=n_inf_steps, device=device,
#                 )
#                 print(f"step {step:06d} [norm-space]"
#                       f" | mse {inf['mse']:.4f}"
#                       f" | agent {inf['agent_mse']:.4f}"
#                       f" | block {inf['block_mse']:.4f}"
#                       f" | rel {inf['relative_mse']:.4f}")
#             finally:
#                 ema.restore()
#                 model.train()

#         if step > 0 and step % cfg.log.save_every == 0:
#             save_checkpoint(ckpt_dir / f"step_{step:07d}.pt",
#                             model, ema, optimizer, lr_sched, step, cfg)

#         step += 1

#     save_checkpoint(ckpt_dir / "final.pt", model, ema, optimizer, lr_sched, step, cfg)
#     print("Training complete!")


# # ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config",  default="config.yaml")
#     parser.add_argument("--resume",  default=None)
#     parser.add_argument("overrides", nargs="*")
#     args = parser.parse_args()

#     cfg = OmegaConf.load(args.config)
#     if args.overrides:
#         cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
#     if args.resume:
#         cfg.resume = args.resume

#     print(OmegaConf.to_yaml(cfg))
#     main(cfg)

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
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from model      import DiT, ModelConfig
from diffusion  import DDIM
from dataset    import NpzDataset, DeltaStats, compute_norm_stats
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
# Delta → absolute position
# ---------------------------------------------------------------------------

def delta_to_abs(delta: np.ndarray, init_xy: np.ndarray) -> np.ndarray:
    """[T, 2] per-frame deltas + [2] initial position → [T, 2] absolute."""
    return init_xy[None, :] + np.cumsum(delta, axis=0)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(
    model:                torch.nn.Module,
    seq:                  torch.Tensor,        # [B, T, 6]  raw
    diffusion:            DDIM,
    norm:                 DeltaStats,
    optimizer:            torch.optim.Optimizer,
    device:               torch.device,
    grad_clip:            Optional[float],
    delta_loss_weight:    float = 1.0,
    relative_loss_weight: float = 0.1,
    consistency_loss_weight: float = 0.1,
) -> dict:
    seq   = seq.to(device)
    x_0   = norm.normalize(seq)           # [B, T, 6]  in [-1, 1]
    B, T  = x_0.shape[:2]

    noise     = torch.randn_like(x_0)
    timesteps = diffusion.sample_timesteps(B, device, num_frames=T)
    x_t       = diffusion.add_noise(x_0, noise, timesteps)

    # Model now predicts x_0 directly (M0-prediction), not noise
    x_pred = model(x_t, timesteps)        # [B, T, 6]  predicted clean signal

    # Primary losses: Huber (smooth L1) against the clean signal, matching MAGNet eq. 15
    delta_loss    = F.huber_loss(x_pred[..., :4], x_0[..., :4])
    relative_loss = F.huber_loss(x_pred[..., 4:], x_0[..., 4:])

    # Consistency loss (L_c): the change in predicted relative position across time
    # must equal the difference in predicted block vs agent deltas — eq. 16.
    # relative[t] - relative[t-1]  =  block_delta[t] - agent_delta[t]
    pred_rel_delta  = x_pred[..., 1:, 4:] - x_pred[..., :-1, 4:]    # [B, T-1, 2]
    pred_delta_diff = x_pred[..., 1:, 2:4] - x_pred[..., 1:, 0:2]   # [B, T-1, 2]
    consistency_loss = F.huber_loss(pred_rel_delta, pred_delta_diff)

    loss = (delta_loss_weight    * delta_loss
            + relative_loss_weight * relative_loss
            + consistency_loss_weight * consistency_loss)

    optimizer.zero_grad()
    loss.backward()

    grad_norm = None
    if grad_clip is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()

    optimizer.step()

    return {
        "loss":             loss.item(),
        "delta_loss":       delta_loss.item(),
        "relative_loss":    relative_loss.item(),
        "consistency_loss": consistency_loss.item(),
        "t_mean":           timesteps.float().mean().item(),
        **({"grad_norm": grad_norm} if grad_norm is not None else {}),
    }


# ---------------------------------------------------------------------------
# Validation  (M0-prediction Huber loss in normalised space)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model:       torch.nn.Module,
    val_loader:  DataLoader,
    diffusion:   DDIM,
    norm:        DeltaStats,
    ema:         EMA,
    device:      torch.device,
    num_batches: int = 10,
) -> dict:
    ema.apply_shadow()
    model.eval()
    total_delta = total_rel = total_consistency = count = 0
    try:
        for seq in val_loader:
            if count >= num_batches:
                break
            x_0   = norm.normalize(seq.to(device))
            noise = torch.randn_like(x_0)
            ts    = diffusion.sample_timesteps(x_0.shape[0], device, num_frames=x_0.shape[1])
            x_t   = diffusion.add_noise(x_0, noise, ts)
            pred  = model(x_t, ts)   # predicts x_0

            total_delta += F.huber_loss(pred[..., :4], x_0[..., :4]).item()
            total_rel   += F.huber_loss(pred[..., 4:], x_0[..., 4:]).item()

            pred_rel_delta  = pred[..., 1:, 4:] - pred[..., :-1, 4:]
            pred_delta_diff = pred[..., 1:, 2:4] - pred[..., 1:, 0:2]
            total_consistency += F.huber_loss(pred_rel_delta, pred_delta_diff).item()

            count += 1
    finally:
        ema.restore()
        model.train()
    n = max(count, 1)
    return {
        "delta_loss":       total_delta / n,
        "relative_loss":    total_rel / n,
        "consistency_loss": total_consistency / n,
    }


# ---------------------------------------------------------------------------
# Inference MSE  (full DDIM sampling, MSE in normalised space)
#
# With M0-prediction the model directly outputs x_0_hat at each denoising step.
# We derive eps from x_0_hat for the standard DDIM update formula, rather than
# deriving x_0 from a predicted eps as in the original epsilon-prediction version.
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model:                torch.nn.Module,
    seq:                  torch.Tensor,        # [B, T, 6]  raw
    diffusion:            DDIM,
    norm:                 DeltaStats,
    n_steps:              int,
    device:               torch.device,
    cond_frames:          List[int] = (),
    df_schedule:          str       = "pyramid",
    df_uncertainty_scale: float     = 1.0,
) -> dict:
    """
    Runs full DDIM denoising with M0-prediction and returns MSE metrics in
    *normalised* space.
    """
    B, T, _ = seq.shape

    x_0_norm = norm.normalize(seq.to(device))   # [B, T, 6]  ground truth in [-1, 1]
    x_t      = torch.randn_like(x_0_norm)

    if cond_frames:
        x_t[:, list(cond_frames)] = x_0_norm[:, list(cond_frames)]

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

    ac = diffusion.scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)

    for m in range(len(K_ts) - 1):
        curr_k = K_ts[m]       # [n_gen]
        next_k = K_ts[m + 1]   # [n_gen]

        t_batch = torch.zeros(B, T, device=device, dtype=torch.long)
        t_batch[:, gen_indices] = curr_k.unsqueeze(0).expand(B, -1)

        # Model predicts x_0 directly (M0-prediction)
        x0_pred = model(x_t, t_batch)   # [B, T, 6]

        # --- DDIM update only for generated frames ---
        a_t = ac[curr_k][None, :, None]   # alpha_t    [1, n_gen, 1]
        a_p = ac[next_k][None, :, None]   # alpha_{t-1}

        x0_g = x0_pred[:, gen_indices].clamp(-5.0, 5.0)   # predicted x_0  [B, n_gen, 6]
        xt_g = x_t[:, gen_indices]                         # current noisy  [B, n_gen, 6]

        # Derive eps from the predicted x_0 for the DDIM update:
        #   x_t = sqrt(a_t)*x_0 + sqrt(1-a_t)*eps
        #   => eps = (x_t - sqrt(a_t)*x_0) / sqrt(1-a_t)
        eps  = (xt_g - a_t.sqrt() * x0_g) / (1.0 - a_t).sqrt().clamp(min=1e-6)

        x_new = a_p.sqrt() * x0_g + (1 - a_p).sqrt() * eps

        changed = (curr_k != next_k)[None, :, None].expand_as(x_new)
        x_t[:, gen_indices] = torch.where(changed, x_new, xt_g)

        if cond_frames:
            x_t[:, list(cond_frames)] = x_0_norm[:, list(cond_frames)]

    # --- MSE in normalised space ---
    pred_norm = x_t
    gt_norm   = x_0_norm

    return {
        "mse":          F.mse_loss(pred_norm,           gt_norm          ).item(),
        "agent_mse":    F.mse_loss(pred_norm[..., :2],  gt_norm[..., :2] ).item(),
        "block_mse":    F.mse_loss(pred_norm[..., 2:4], gt_norm[..., 2:4]).item(),
        "relative_mse": F.mse_loss(pred_norm[..., 4:],  gt_norm[..., 4:] ).item(),
        "pred": norm.denormalize(pred_norm),
        "gt":   seq.to(device),
    }


# ---------------------------------------------------------------------------
# Point-cloud visualisation helpers
# ---------------------------------------------------------------------------

_VIS_BLOCK_COLOR = np.array([0.20, 0.47, 0.96])
_VIS_HAND_COLOR  = np.array([0.91, 0.47, 0.10])
_VIS_CUBE_HALF     = 0.025
_VIS_CIRCLE_RADIUS = 0.025
_VIS_T_BOXES = [
    (np.array([0.0, -0.0375/2,             0.0]), np.array([0.05,       0.0125,      0.02])),
    (np.array([0.0,  4*0.0125 - 0.0375/2, 0.0]), np.array([0.0125, 0.075*0.75,      0.02])),
]
_VIS_STICK_RADIUS = 0.008
_VIS_STICK_LENGTH = 0.10


def _vis_sample_box(offset, half, n, rng):
    n_init = n * 20
    hx, hy, hz = half
    areas  = np.array([4*hy*hz, 4*hx*hz, 4*hx*hy], dtype=np.float64)
    counts = np.round(areas / areas.sum() * n_init).astype(int)
    counts[-1] += n_init - counts.sum()
    pts = []
    if counts[0] > 0:
        s = rng.choice([-1.0, 1.0], counts[0])
        pts.append(np.stack([s*hx, rng.uniform(-hy, hy, counts[0]), rng.uniform(-hz, hz, counts[0])], 1))
    if counts[1] > 0:
        s = rng.choice([-1.0, 1.0], counts[1])
        pts.append(np.stack([rng.uniform(-hx, hx, counts[1]), s*hy, rng.uniform(-hz, hz, counts[1])], 1))
    if counts[2] > 0:
        s = rng.choice([-1.0, 1.0], counts[2])
        pts.append(np.stack([rng.uniform(-hx, hx, counts[2]), rng.uniform(-hy, hy, counts[2]), s*hz], 1))
    pts = np.concatenate(pts).astype(np.float32) + offset.astype(np.float32)
    return pts[rng.choice(len(pts), size=min(n, len(pts)), replace=False)]


def _vis_sample_cylinder_z(radius, length, n, rng):
    n_init = n * 20
    th  = rng.uniform(0, 2 * np.pi, n_init)
    z   = rng.uniform(-length / 2, length / 2, n_init)
    pts = np.stack([radius * np.cos(th), radius * np.sin(th), z], axis=1).astype(np.float32)
    return pts[rng.choice(len(pts), size=min(n, len(pts)), replace=False)]


def _vis_sample_circle_block(radius, n, rng):
    n_init = n * 20
    th  = rng.uniform(0, 2 * np.pi, n_init)
    z   = rng.uniform(-radius, radius, n_init)
    pts = np.stack([radius * np.cos(th), radius * np.sin(th), z], axis=1).astype(np.float32)
    return pts[rng.choice(len(pts), size=min(n, len(pts)), replace=False)]


def _vis_build_templates(n_block, n_hand, rng, block_shape="cube"):
    if block_shape == "T":
        areas = [2*(4*hy*hz + 4*hx*hz + 4*hx*hy) for _, (hx, hy, hz) in _VIS_T_BOXES]
        total = sum(areas)
        parts = [_vis_sample_box(off, half, max(1, int(n_block * a / total)), rng)
                 for (off, half), a in zip(_VIS_T_BOXES, areas)]
        block_tpl = np.concatenate(parts)
    elif block_shape == "circle":
        block_tpl = _vis_sample_circle_block(_VIS_CIRCLE_RADIUS, n_block, rng)
    else:
        block_tpl = _vis_sample_box(np.zeros(3), np.full(3, _VIS_CUBE_HALF), n_block, rng)
    return {"block": block_tpl,
            "hand":  _vis_sample_cylinder_z(_VIS_STICK_RADIUS, _VIS_STICK_LENGTH, n_hand, rng)}


def _vis_rot6d_to_rotmat(r6):
    a1 = r6[:3] / np.linalg.norm(r6[:3])
    a2 = r6[3:] - np.dot(r6[3:], a1) * a1
    a2 = a2 / np.linalg.norm(a2)
    return np.stack([a1, a2, np.cross(a1, a2)], axis=-1)


def _vis_apply_pose_9d(pts, pose_9d):
    pos = pose_9d[:3].astype(np.float64)
    R   = _vis_rot6d_to_rotmat(pose_9d[3:9].astype(np.float64))
    return (pts.astype(np.float64) @ R.T + pos).astype(np.float32)


def _vis_render_frame(gt_tcp_9d, gt_block_9d, pr_tcp_9d, pr_block_9d,
                      templates, lims, t, T, elev, azim, dpi=100):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 5), dpi=dpi)
    for col, (tcp_9d, block_9d, label) in enumerate([
        (gt_tcp_9d,  gt_block_9d,  f"GT   t={t}/{T-1}"),
        (pr_tcp_9d,  pr_block_9d,  f"Pred t={t}/{T-1}"),
    ]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        ax.scatter(*_vis_apply_pose_9d(templates["block"], block_9d).T,
                   s=1.2, c=[_VIS_BLOCK_COLOR.tolist()], alpha=0.6, depthshade=True, label="block")
        ax.scatter(*_vis_apply_pose_9d(templates["hand"],  tcp_9d).T,
                   s=2.5, c=[_VIS_HAND_COLOR.tolist()],  alpha=0.9, depthshade=True, label="TCP")
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = lims
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi); ax.set_zlim(zlo, zhi)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("X", fontsize=6); ax.set_ylabel("Y", fontsize=6)
        ax.set_zlabel("Z", fontsize=6); ax.tick_params(labelsize=5)
        ax.set_title(label, fontsize=8)
        if col == 0:
            ax.legend(fontsize=6, loc="upper right", markerscale=3)
    fig.suptitle("Blue = block   Orange = TCP", fontsize=7)
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    w, h_px = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = buf.reshape(h_px, w, 4)[:, :, 1:]
    plt.close(fig)
    return img.copy()


def visualize_predictions(
    model,
    dataset:          NpzDataset,
    seq:              torch.Tensor,    # [B, T, 6]  raw
    initial_abs_xy:   np.ndarray,      # [B, 4]
    diffusion:        DDIM,
    norm:             DeltaStats,
    n_steps:          int,
    device:           torch.device,
    save_dir:         Path,
    step:             int,
    n_samples:        int   = 4,
    block_shape:      str   = "cube",
    every:            int   = 1,
    elev:             float = 90.0,
    azim:             float = -90.0,
    gif_fps:          float = 10.0,
) -> None:
    import imageio

    n_samples = min(n_samples, seq.shape[0])
    sub_seq   = seq[:n_samples].to(device)
    init_xy   = initial_abs_xy[:n_samples]

    model.eval()
    with torch.no_grad():
        x_t = torch.randn_like(norm.normalize(sub_seq))
        diffusion.scheduler.set_timesteps(n_steps, device=device)
        for t_step in diffusion.scheduler.timesteps:
            # Use [B, T] timesteps matching training convention
            t_batch = t_step.view(1, 1).expand(n_samples, x_t.shape[1])
            x_t     = diffusion.scheduler.step(model(x_t, t_batch), t_step, x_t).prev_sample
    model.train()

    pred = norm.denormalize(x_t).cpu().numpy()   # [B, T, 6]
    gt   = sub_seq.cpu().numpy()

    tcp_z          = dataset.mean_tcp_z
    block_z        = dataset.mean_block_z
    identity_rot6d = np.array([1., 0., 0., 0., 1., 0.], dtype=np.float64)

    def _pose_9d(xy, z):
        return np.array([xy[0], xy[1], z, *identity_rot6d], dtype=np.float64)

    rng           = np.random.default_rng(0)
    templates     = _vis_build_templates(400, 250, rng, block_shape)
    T_len         = gt.shape[1]
    frame_indices = list(range(0, T_len, every))

    step_dir = save_dir / f"step_{step:07d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_samples):
        init_tcp_xy   = init_xy[i, :2]
        init_block_xy = init_xy[i, 2:]

        gt_tcp_abs   = delta_to_abs(gt[i,   :, :2],  init_tcp_xy)
        gt_block_abs = delta_to_abs(gt[i,   :, 2:4], init_block_xy)
        pr_tcp_abs   = delta_to_abs(pred[i, :, :2],  init_tcp_xy)
        pr_block_abs = delta_to_abs(pred[i, :, 2:4], init_block_xy)

        gt_tcp_9d   = np.stack([_pose_9d(gt_tcp_abs[t],   tcp_z)   for t in range(T_len)])
        gt_block_9d = np.stack([_pose_9d(gt_block_abs[t], block_z) for t in range(T_len)])
        pr_tcp_9d   = np.stack([_pose_9d(pr_tcp_abs[t],   tcp_z)   for t in range(T_len)])
        pr_block_9d = np.stack([_pose_9d(pr_block_abs[t], block_z) for t in range(T_len)])

        all_pts = np.concatenate([
            *[_vis_apply_pose_9d(templates["hand"],  p) for p in gt_tcp_9d],
            *[_vis_apply_pose_9d(templates["block"], p) for p in gt_block_9d],
            *[_vis_apply_pose_9d(templates["hand"],  p) for p in pr_tcp_9d],
            *[_vis_apply_pose_9d(templates["block"], p) for p in pr_block_9d],
        ])
        pad  = 0.05
        ctr  = all_pts.mean(0)
        r    = max((all_pts.max(0) - all_pts.min(0)).max() / 2 + pad, 0.1)
        lims = [(float(ctr[j] - r), float(ctr[j] + r)) for j in range(3)]

        gif_frames = [
            _vis_render_frame(
                gt_tcp_9d=gt_tcp_9d[t_idx], gt_block_9d=gt_block_9d[t_idx],
                pr_tcp_9d=pr_tcp_9d[t_idx], pr_block_9d=pr_block_9d[t_idx],
                templates=templates, lims=lims, t=t_idx, T=T_len, elev=elev, azim=azim,
            )
            for t_idx in frame_indices
        ]

        out_path = step_dir / f"sample_{i}.gif"
        imageio.mimsave(str(out_path), gif_frames, duration=1.0 / gif_fps, loop=0)
        print(f"  -> saved {out_path}")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, ema, optimizer, lr_sched, step, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step, "model": model.state_dict(), "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": lr_sched.state_dict(),
        "cfg": OmegaConf.to_container(cfg),
    }, path)
    print(f"  saved {path}")


def load_checkpoint(path, model, optimizer, lr_sched, ema) -> int:
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

    device_name = ("cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(cfg.get("device", device_name))
    print(f"Device: {device}")

    run_dir  = Path(cfg.log.dir) / cfg.run_name
    ckpt_dir = run_dir / "checkpoints"
    vis_dir  = run_dir / "visualisations"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    print(f"Run: {run_dir}")

    train_ds = NpzDataset(cfg.data.train_path, cfg.data.seq_len, cfg.data.stride)
    val_ds   = NpzDataset(cfg.data.val_path,   cfg.data.seq_len, cfg.data.stride)

    if cfg.train.overfit_batches > 0:
        print(f"Overfit mode: caching {cfg.train.overfit_batches} batch(es)...")
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

    norm = compute_norm_stats(train_ds)

    model_cfg = ModelConfig(**OmegaConf.to_container(cfg.model))
    model     = DiT(model_cfg).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    diffusion = DDIM(
        num_train_timesteps=cfg.ddim.num_train_timesteps,
        beta_schedule=cfg.ddim.beta_schedule,
        clip_sample=cfg.ddim.clip_sample,
        clip_sample_range=cfg.ddim.clip_sample_range,
    )
    n_inf_steps = cfg.ddim.inference_steps

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    warmup_steps = cfg.train.get("warmup_steps", 10000)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(cfg.train.steps - warmup_steps, 1)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ema      = EMA(model, decay=cfg.train.ema_decay)

    delta_w       = cfg.train.get("delta_loss_weight",       1.0)
    relative_w    = cfg.train.get("relative_loss_weight",    0.1)
    consistency_w = cfg.train.get("consistency_loss_weight", 0.1)

    start_step = 0
    if cfg.resume:
        start_step = load_checkpoint(cfg.resume, model, optimizer, lr_sched, ema)
        print(f"Resumed from step {start_step}")

    _vis_batch:      Optional[torch.Tensor] = None
    _vis_initial_xy: Optional[np.ndarray]   = None
    try:
        n_vis       = cfg.log.get("vis_n_samples", 4)
        vis_indices = [int(i * len(val_ds) / n_vis) for i in range(n_vis)]
        _vis_batch      = torch.stack([val_ds[i]                for i in vis_indices])
        _vis_initial_xy = np.stack(  [val_ds.get_abs_initial(i) for i in vis_indices])
    except Exception as e:
        print(f"  [warn] could not build vis batch: {e}")

    model.train()
    step = start_step
    log_loss = log_delta = log_rel = log_consistency = 0.0
    print(f"Training for {cfg.train.steps} steps...")

    while step < cfg.train.steps:
        seq     = next(train_iter)
        metrics = train_step(
            model, seq, diffusion, norm, optimizer, device, cfg.train.grad_clip,
            delta_loss_weight=delta_w,
            relative_loss_weight=relative_w,
            consistency_loss_weight=consistency_w,
        )
        lr_sched.step()
        ema.update()
        log_loss        += metrics["loss"]
        log_delta       += metrics["delta_loss"]
        log_rel         += metrics["relative_loss"]
        log_consistency += metrics["consistency_loss"]

        if step % cfg.log.every == 0 and step > 0:
            n   = cfg.log.every
            lr  = lr_sched.get_last_lr()[0]
            msg = (f"step {step:06d}"
                   f" | loss {log_loss/n:.4f}"
                   f" | delta {log_delta/n:.4f}"
                   f" | rel {log_rel/n:.4f}"
                   f" | cons {log_consistency/n:.4f}"
                   f" | t_mean {metrics['t_mean']:.0f}"
                   f" | lr {lr:.2e}")
            if "grad_norm" in metrics:
                msg += f" | gnorm {metrics['grad_norm']:.2f}"
            print(msg)
            log_loss = log_delta = log_rel = log_consistency = 0.0

        if step > 0 and step % cfg.log.val_every == 0:
            val_m = validate(model, val_loader, diffusion, norm, ema, device, cfg.log.val_batches)
            # All metrics are Huber loss in normalised space against the clean signal.
            # A well-trained model should drive these toward 0.
            print(f"step {step:06d}"
                  f" | val_delta {val_m['delta_loss']:.4f}"
                  f" | val_rel {val_m['relative_loss']:.4f}"
                  f" | val_cons {val_m['consistency_loss']:.4f}")

            if _vis_batch is not None and _vis_initial_xy is not None:
                ema.apply_shadow()
                try:
                    visualize_predictions(
                        model=model, dataset=val_ds, seq=_vis_batch,
                        initial_abs_xy=_vis_initial_xy, diffusion=diffusion,
                        norm=norm, n_steps=n_inf_steps, device=device,
                        save_dir=vis_dir, step=step,
                        n_samples   = cfg.log.get("vis_n_samples",   4),
                        block_shape = cfg.log.get("vis_block_shape", "cube"),
                        every       = cfg.log.get("vis_every",       1),
                        elev        = cfg.log.get("vis_elev",        30.0),
                        azim        = cfg.log.get("vis_azim",       -60.0),
                        gif_fps     = cfg.log.get("vis_gif_fps",     10.0),
                    )
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"  [warn] visualisation failed: {e}")
                finally:
                    ema.restore()
                    model.train()

        # Inference MSE — all values in normalised space, should decrease from ~1.0
        if step > 0 and step % cfg.log.sample_every == 0:
            ema.apply_shadow()
            model.eval()
            try:
                inf = run_inference(
                    model=model, seq=next(train_iter).to(device),
                    diffusion=diffusion, norm=norm, n_steps=n_inf_steps, device=device,
                )
                print(f"step {step:06d} [norm-space]"
                      f" | mse {inf['mse']:.4f}"
                      f" | agent {inf['agent_mse']:.4f}"
                      f" | block {inf['block_mse']:.4f}"
                      f" | rel {inf['relative_mse']:.4f}")
            finally:
                ema.restore()
                model.train()

        if step > 0 and step % cfg.log.save_every == 0:
            save_checkpoint(ckpt_dir / f"step_{step:07d}.pt",
                            model, ema, optimizer, lr_sched, step, cfg)

        step += 1

    save_checkpoint(ckpt_dir / "final.pt", model, ema, optimizer, lr_sched, step, cfg)
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