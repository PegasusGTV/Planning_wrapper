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
# from typing import List, Optional, Tuple, Union
# import numpy as np
# import torch
# import torch.nn.functional as F
# from omegaconf import OmegaConf
# from torch.utils.data import DataLoader

# from model      import DiT, ModelConfig
# from diffusion  import DDIM, Stats
# from dataset    import NpzDataset, compute_norm_stats
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
    
# class SNRWeighting:
#     def __init__(self, diffusion, snr_clip: float = 5.0):
#         if isinstance(diffusion, DDIM):
#             ac  = diffusion.scheduler.alphas_cumprod
#             snr = ac / (1.0 - ac)
#         else:  # FlowMatching — approximate SNR from t
#             # t=0 → clean, t=1 → noise; SNR ≈ (1-t)^2 / t^2
#             t   = torch.linspace(1e-3, 1.0 - 1e-3, diffusion.num_train_timesteps)
#             snr = ((1.0 - t) / t) ** 2

#         self.snr         = snr
#         self.clipped_snr = snr.clone().clamp_(max=snr_clip)

#     def weights(self, timesteps: torch.Tensor) -> torch.Tensor:
#         # timesteps: [B, T] → weights: [B, T]
#         snr         = self.snr        .to(timesteps.device)[timesteps.long()]
#         clipped_snr = self.clipped_snr.to(timesteps.device)[timesteps.long()]
#         return clipped_snr / snr.clamp(min=1e-8)


# # ---------------------------------------------------------------------------
# # Normalisation helpers
# # ---------------------------------------------------------------------------

# def normalize(seq: torch.Tensor, norm: Stats, SD: int) -> Tuple[torch.Tensor, torch.Tensor]:
#     bundle = norm.normalize(seq)      
#     return bundle[..., :SD], bundle[..., SD:]

# def denormalize(states: torch.Tensor, actions: torch.Tensor, norm: Stats) -> torch.Tensor:
#     bundle = torch.cat([states, actions], dim=-1)
#     return norm.denormalize(bundle)



# # ---------------------------------------------------------------------------
# # Training step
# # ---------------------------------------------------------------------------

# def train_step(
#     model:     torch.nn.Module,
#     seq:       torch.Tensor,               # [B, T, SD+AD]  raw
#     diffusion: DDIM,
#     norm:      Stats,
#     optimizer: torch.optim.Optimizer,
#     device:    torch.device,
#     grad_clip: Optional[float],
# ) -> dict:
#     seq  = seq.to(device)
#     B, T = seq.shape[:2]

#     s_0, a_0 = normalize(seq, norm, model.state_dim)                          # [B, T, SD], [B, T, AD]

#     s_noise = torch.randn_like(s_0)
#     a_noise = torch.randn_like(a_0)

#     timesteps = diffusion.sample_timesteps(B, device, num_frames=T)   # [B, T]

#     s_t = diffusion.add_noise(s_0, s_noise, timesteps)
#     a_t = diffusion.add_noise(a_0, a_noise, timesteps)

#     x_t = torch.cat([s_t, a_t], dim=-1)   # [B, T, SD+AD]
    
#     x_pred = model(x_t, timesteps)

#     # x_target = diffusion.get_target(x_pred, timesteps)
#     # loss = F.mse_loss(x_pred, x_target)
    
#     noise = torch.cat([s_noise, a_noise], dim=-1)
#     loss = F.mse_loss(x_pred, noise)

#     optimizer.zero_grad()
#     loss.backward()

#     grad_norm = None
#     if grad_clip is not None:
#         grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()

#     optimizer.step()

#     metrics = {"loss": loss.item(), "t_mean": timesteps.float().mean().item()}
#     if grad_norm is not None:
#         metrics["grad_norm"] = grad_norm
#     return metrics


# # ---------------------------------------------------------------------------
# # Validation
# # ---------------------------------------------------------------------------

# @torch.no_grad()
# def validate(
#     model:       torch.nn.Module,
#     val_loader:  DataLoader,
#     diffusion:   DDIM,
#     norm:        Stats,
#     ema:         EMA,
#     device:      torch.device,
#     num_batches: int = 10,
# ) -> dict:
#     ema.apply_shadow()
#     model.eval()
#     total, count = 0.0, 0
#     try:
#         for seq in val_loader:
#             if count >= num_batches:
#                 break
#             seq  = seq.to(device)
#             B, T = seq.shape[:2]

#             s_0, a_0 = normalize(seq, norm, model.state_dim)
#             s_noise, a_noise = torch.randn_like(s_0), torch.randn_like(a_0)
#             ts = diffusion.sample_timesteps(B, device, num_frames=T)

#             s_t = diffusion.add_noise(s_0, s_noise, ts)
#             a_t = diffusion.add_noise(a_0, a_noise, ts)

#             x_t = torch.cat([s_t, a_t], dim=-1)
#             x_pred = model(x_t, ts)
            
#             noise = torch.cat([s_noise, a_noise], dim=-1)

#             loss = (
#                 F.mse_loss(x_pred, noise)
#             )
#             total += loss.item()
#             count += 1
#     finally:
#         ema.restore()
#         model.train()
#     return {"loss": total / max(count, 1)}


# # ---------------------------------------------------------------------------
# # Inference
# # ---------------------------------------------------------------------------

# @torch.no_grad()
# def run_inference(
#     model:                torch.nn.Module,
#     seq:                  torch.Tensor,        # [B, T, SD+AD]  raw
#     diffusion:            DDIM,
#     norm:                 Stats,
#     n_steps:              int,
#     device:               torch.device,
#     cond_frames:          List[int] = (),
#     df_schedule:          str       = "pyramid",
#     df_uncertainty_scale: float     = 1.0,
# ) -> dict:
#     B, T, D = seq.shape
#     SD       = model.state_dim
#     AD      = D - SD

#     s_0, a_0 = normalize(seq.to(device), norm, model.state_dim)   # [B, T, SD], [B, T, AD]

#     # Initialise from noise
#     s_t = torch.randn_like(s_0)
#     a_t = torch.randn_like(a_0)

#     if cond_frames:
#         cond_idx = list(cond_frames)
#         s_t[:, cond_idx] = s_0[:, cond_idx]
#         a_t[:, cond_idx] = a_0[:, cond_idx]

#     # Scheduling matrix (over non-conditioned frames only)
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
#         x_t = torch.cat([s_t, a_t], dim=-1)   # [B, T, SD+AD]
#         curr_k = K_ts[m]
#         next_k = K_ts[m + 1]

#         t_batch = torch.zeros(B, T, device=device, dtype=torch.long)
#         t_batch[:, gen_indices] = curr_k.unsqueeze(0).expand(B, -1)

#         x_pred = model(x_t, t_batch)   # [B, T, SD+AD]

#         a_t_coef  = ac[curr_k][None, :, None]
#         a_p_coef  = ac[next_k][None, :, None]
#         s_eps, a_eps = x_pred[:, gen_indices, :SD], x_pred[:, gen_indices, SD:SD+AD]
#         s_x0  = (s_t[:, gen_indices] - (1 - a_t_coef).sqrt() * s_eps) / a_t_coef.sqrt().clamp(min=1e-8)
#         a_x0  = (a_t[:, gen_indices] - (1 - a_t_coef).sqrt() * a_eps) / a_t_coef.sqrt().clamp(min=1e-8)
#         s_new = a_p_coef.sqrt() * s_x0 + (1 - a_p_coef).sqrt() * s_eps
#         a_new = a_p_coef.sqrt() * a_x0 + (1 - a_p_coef).sqrt() * a_eps

#         changed = (curr_k != next_k)[None, :, None].expand(B, n_gen, -1)
#         s_t[:, gen_indices] = torch.where(changed.expand_as(s_new), s_new, s_t[:, gen_indices])
#         a_t[:, gen_indices] = torch.where(changed.expand_as(a_new), a_new, a_t[:, gen_indices])

#         if cond_frames:
#             s_t[:, cond_idx] = s_0[:, cond_idx]
#             a_t[:, cond_idx] = a_0[:, cond_idx]

#     pred_raw = denormalize(s_t, a_t, norm)
#     gt_raw   = seq.to(device)

#     return {
#         "mse":        F.mse_loss(pred_raw,           gt_raw          ).item(),
#         "state_mse":  F.mse_loss(pred_raw[..., :SD], gt_raw[..., :SD]).item(),
#         "action_mse": F.mse_loss(pred_raw[..., SD:], gt_raw[..., SD:]).item(),
#         "pred":       pred_raw,
#         "gt":         gt_raw,
#     }

    

# # def train_step(model, seq, diffusion, norm: Stats, optimizer, device, grad_clip):
# #     seq = seq.to(device)          # [B, T, 27]
# #     B, T, D = seq.shape

# #     x_0   = norm.normalize(seq)   # [B, T, 27] → [-1, 1] per pose dim
# #     noise = torch.randn_like(x_0)
# #     ts    = diffusion.sample_timesteps(B, device)  # [B]

# #     x_t      = diffusion.add_noise(x_0, noise, ts)
# #     eps_pred = model(x_t, ts)     # [B, T, 27]

# #     loss = F.mse_loss(eps_pred, noise)

# #     optimizer.zero_grad()
# #     loss.backward()
# #     if grad_clip:
# #         torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
# #     optimizer.step()

# #     return {"loss": loss.item(), "t_mean": ts.float().mean().item()}


# # @torch.no_grad()
# # def validate(model, val_loader, diffusion, norm: Stats, ema, device, num_batches=10):
# #     ema.apply_shadow()
# #     model.eval()
# #     total, count = 0.0, 0
# #     try:
# #         for seq in val_loader:
# #             if count >= num_batches:
# #                 break
# #             seq  = seq.to(device)
# #             x_0  = norm.normalize(seq)
# #             noise = torch.randn_like(x_0)
# #             ts    = diffusion.sample_timesteps(seq.shape[0], device)
# #             x_t   = diffusion.add_noise(x_0, noise, ts)
# #             eps_pred = model(x_t, ts)
# #             total += F.mse_loss(eps_pred, noise).item()
# #             count += 1
# #     finally:
# #         ema.restore()
# #         model.train()
# #     return {"loss": total / max(count, 1)}


# # @torch.no_grad()
# # def run_inference(model, seq, diffusion, norm: Stats, n_steps, device):
# #     B, T, D = seq.shape
# #     x_0_norm = norm.normalize(seq.to(device))

# #     x_t = torch.randn((B, T, D), device=device)

# #     diffusion.scheduler.set_timesteps(n_steps, device=device)
# #     for t in diffusion.scheduler.timesteps:
# #         eps_pred = model(x_t, t.expand(B))
# #         x_t = diffusion.scheduler.step(
# #             model_output=eps_pred,
# #             timestep=t,
# #             sample=x_t,
# #         ).prev_sample

# #     mse_norm = F.mse_loss(x_t, x_0_norm).item()
# #     pred_raw = norm.denormalize(x_t)
# #     gt_raw   = seq.to(device)

# #     return {
# #         "mse":            F.mse_loss(pred_raw, gt_raw).item(),
# #         "state_mse":      F.mse_loss(pred_raw[..., :model.cfg.state_dim], gt_raw[..., :model.cfg.state_dim]).item(),
# #         "action_mse":     F.mse_loss(pred_raw[..., model.cfg.state_dim:], gt_raw[..., model.cfg.state_dim:]).item(),
# #         "mse_norm":       mse_norm,
# #         "pred":           pred_raw,
# #         "gt":             gt_raw,
# #     }

# # ---------------------------------------------------------------------------
# # 3D point-cloud visualisation helpers (mirrors scripts/vis_lowdim.py)
# # ---------------------------------------------------------------------------

# _VIS_BLOCK_COLOR = np.array([0.20, 0.47, 0.96])   # blue
# _VIS_HAND_COLOR  = np.array([0.91, 0.47, 0.10])   # orange
# _VIS_GHOST_COLOR = np.array([0.20, 0.95, 0.60])   # green

# _VIS_BLOCK_HALF = 0.025
# _VIS_T_BOXES = [
#     (np.array([0.0, -0.0375/2, 0.0]),                np.array([0.05, 0.0125, 0.02])),
#     (np.array([0.0, 4*0.0125 - 0.0375/2, 0.0]),      np.array([0.0125, 0.075*0.75, 0.02])),
# ]
# _VIS_STICK_RADIUS = 0.008
# _VIS_STICK_LENGTH = 0.10
# _VIS_STICK_OFFSET_Z = -0.05


# def _vis_rot6d_to_rotmat(r6: np.ndarray) -> np.ndarray:
#     r1 = r6[:3].astype(np.float64)
#     r2 = r6[3:].astype(np.float64)
#     a1 = r1 / np.linalg.norm(r1)
#     a2 = r2 - np.dot(r2, a1) * a1
#     a2 = a2 / np.linalg.norm(a2)
#     a3 = np.cross(a1, a2)
#     return np.stack([a1, a2, a3], axis=-1)


# def _vis_apply_pose_9d(pts: np.ndarray, pose_9d: np.ndarray) -> np.ndarray:
#     pos = pose_9d[:3].astype(np.float64)
#     R   = _vis_rot6d_to_rotmat(pose_9d[3:9])
#     return (pts.astype(np.float64) @ R.T + pos).astype(np.float32)


# def _vis_sample_box(offset, half, n, rng):
#     n_init = n * 20
#     hx, hy, hz = half
#     areas  = np.array([4*hy*hz, 4*hx*hz, 4*hx*hy], dtype=np.float64)
#     counts = np.round(areas / areas.sum() * n_init).astype(int)
#     counts[-1] += n_init - counts.sum()
#     pts = []
#     if counts[0] > 0:
#         s = rng.choice([-1.0, 1.0], counts[0])
#         pts.append(np.stack([s*hx, rng.uniform(-hy,hy,counts[0]), rng.uniform(-hz,hz,counts[0])], 1))
#     if counts[1] > 0:
#         s = rng.choice([-1.0, 1.0], counts[1])
#         pts.append(np.stack([rng.uniform(-hx,hx,counts[1]), s*hy, rng.uniform(-hz,hz,counts[1])], 1))
#     if counts[2] > 0:
#         s = rng.choice([-1.0, 1.0], counts[2])
#         pts.append(np.stack([rng.uniform(-hx,hx,counts[2]), rng.uniform(-hy,hy,counts[2]), s*hz], 1))
#     initial = np.concatenate(pts).astype(np.float32) + np.asarray(offset, dtype=np.float32)
#     idx = rng.choice(len(initial), size=min(n, len(initial)), replace=False)
#     return initial[idx]


# def _vis_sample_cylinder(offset_z, radius, length, n, rng):
#     n_init = n * 20
#     th  = rng.uniform(0, 2*np.pi, n_init)
#     z   = rng.uniform(-length/2, length/2, n_init) + offset_z
#     pts = np.stack([radius*np.cos(th), radius*np.sin(th), z], 1).astype(np.float32)
#     idx = rng.choice(len(pts), size=min(n, len(pts)), replace=False)
#     return pts[idx]


# def _vis_build_templates(n_block: int, n_hand: int, rng, use_T: bool) -> dict:
#     if use_T:
#         areas     = [2*(4*hy*hz+4*hx*hz+4*hx*hy) for _, (hx,hy,hz) in _VIS_T_BOXES]
#         total     = sum(areas)
#         t_parts   = [_vis_sample_box(off, half, max(1, int(n_block*a/total)), rng)
#                      for (off, half), a in zip(_VIS_T_BOXES, areas)]
#         block_tpl = np.concatenate(t_parts)
#     else:
#         block_tpl = _vis_sample_box(
#             np.zeros(3), np.full(3, _VIS_BLOCK_HALF), n_block, rng
#         )
#     hand_tpl = _vis_sample_cylinder(
#         _VIS_STICK_OFFSET_Z, _VIS_STICK_RADIUS, _VIS_STICK_LENGTH, n_hand, rng
#     )
#     return {"block": block_tpl, "hand": hand_tpl}


# def _vis_render_single_frame(
#     states_gt:   np.ndarray,   # [18]
#     actions_gt:  np.ndarray,   # [9]
#     states_pr:   np.ndarray,   # [18]
#     actions_pr:  np.ndarray,   # [9]
#     templates:   dict,
#     lims:        list,          # [(xlo,xhi),(ylo,yhi),(zlo,zhi)]
#     t:           int,
#     T:           int,
#     elev:        float,
#     azim:        float,
#     ghost_alpha: float,
#     cond_frames: List[int],
#     dpi:         int = 100,
# ) -> np.ndarray:
#     """Render one GIF frame: GT (left) and Pred (right) side-by-side 3D scatter."""
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt

#     fig = plt.figure(figsize=(10, 5), dpi=dpi)

#     for col, (states, actions, label) in enumerate([
#         (states_gt, actions_gt, f"GT   t={t}/{T-1}"),
#         (states_pr, actions_pr, f"Pred t={t}/{T-1}"),
#     ]):
#         ax = fig.add_subplot(1, 2, col + 1, projection="3d")

#         tcp_9d   = states[0:9]
#         block_9d = states[9:18]
#         act_9d   = actions[0:9]

#         hand_pts  = _vis_apply_pose_9d(templates["hand"],  tcp_9d)
#         block_pts = _vis_apply_pose_9d(templates["block"], block_9d)
#         ghost_pts = _vis_apply_pose_9d(templates["hand"],  act_9d)

#         ax.scatter(block_pts[:,0], block_pts[:,1], block_pts[:,2],
#                    s=1.2, c=[_VIS_BLOCK_COLOR.tolist()], depthshade=True,
#                    alpha=0.6, label="block")
#         ax.scatter(hand_pts[:,0],  hand_pts[:,1],  hand_pts[:,2],
#                    s=2.5, c=[_VIS_HAND_COLOR.tolist()],  depthshade=True,
#                    alpha=0.9, label="TCP (now)")
#         ax.scatter(ghost_pts[:,0], ghost_pts[:,1], ghost_pts[:,2],
#                    s=2.5, c=[_VIS_GHOST_COLOR.tolist()], depthshade=True,
#                    alpha=ghost_alpha, label="action target")

#         if t in cond_frames:
#             ax.scatter(*hand_pts.mean(0), marker="D", s=60,
#                        c=[[1.0, 1.0, 0.2]], zorder=6, alpha=0.9)

#         (xlo, xhi), (ylo, yhi), (zlo, zhi) = lims
#         ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi); ax.set_zlim(zlo, zhi)
#         ax.set_box_aspect([1, 1, 1])
#         ax.view_init(elev=elev, azim=azim)
#         ax.set_xlabel("X", fontsize=6); ax.set_ylabel("Y", fontsize=6)
#         ax.set_zlabel("Z", fontsize=6); ax.tick_params(labelsize=5)
#         ax.set_title(label, fontsize=8)
#         if col == 0:
#             ax.legend(fontsize=6, loc="upper right", markerscale=3)

#     fig.suptitle(
#         "Blue=block  Orange=TCP  Green=action target  ◆=cond frame",
#         fontsize=7,
#     )
#     fig.tight_layout(pad=0.5)
#     fig.canvas.draw()
#     w, h_px = fig.canvas.get_width_height()
#     buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
#     img = buf.reshape(h_px, w, 4)[:, :, 1:]   # ARGB -> RGB
#     plt.close(fig)
#     return img.copy()


# # ---------------------------------------------------------------------------
# # Prediction visualisation — one GIF per sample, GT vs Pred side-by-side
# # ---------------------------------------------------------------------------
# def visualize_predictions(
#     model, dataset, seq, diffusion, norm,
#     n_steps, device, save_dir, step,
#     n_samples=4, use_T_block=False, every=1,
#     elev=30.0, azim=-60.0, gif_fps=10.0,
# ) -> None:
#     import imageio

#     n_samples = min(n_samples, seq.shape[0])
#     sub_seq   = seq[:n_samples].to(device)

#     # Run inference
#     model.eval()
#     with torch.no_grad():
#         x_t = torch.randn_like(sub_seq)
#         diffusion.scheduler.set_timesteps(n_steps, device=device)
#         for t in diffusion.scheduler.timesteps:
#             t_batch = t.expand(n_samples).unsqueeze(1).expand(n_samples, x_t.shape[1])  # [B, T]
#             eps = model(x_t, t_batch)
#             x_t = diffusion.scheduler.step(eps, t, x_t).prev_sample
#     model.train()

#     pred_xy = norm.denormalize(x_t).cpu().numpy()         # [B, T, 4]
#     gt_xy   = norm.denormalize(sub_seq).cpu().numpy()     # [B, T, 4]

#     # Fixed z heights — just use something visually reasonable
#     tcp_z   = dataset.mean_tcp_z
#     block_z = dataset.mean_block_z

#     # Fixed identity-ish rotations so block/hand always look correct
#     # rot6d identity = first two columns of I = [1,0,0, 0,1,0]
#     identity_rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

#     rng       = np.random.default_rng(0)
#     templates = _vis_build_templates(400, 250, rng, use_T_block)

#     step_dir = save_dir / f"step_{step:07d}"
#     step_dir.mkdir(parents=True, exist_ok=True)

#     T      = gt_xy.shape[1]
#     frames = list(range(0, T, every))

#     for i in range(n_samples):

#         # Build all poses for this sample up front
#         def build_poses(xy):
#             # xy: [T, 4]
#             tcp_poses   = np.concatenate([xy[:, 0:2], np.full((T, 1), tcp_z),   np.tile(identity_rot6d, (T,1))], axis=1)  # [T, 9]
#             block_poses = np.concatenate([xy[:, 2:4], np.full((T, 1), block_z), np.tile(identity_rot6d, (T,1))], axis=1)  # [T, 9]
#             return tcp_poses, block_poses

#         gt_tcp_poses,   gt_block_poses   = build_poses(gt_xy[i])
#         pred_tcp_poses, pred_block_poses = build_poses(pred_xy[i])

#         # Compute scene limits ONCE across all frames and both gt/pred
#         # so axes never shift between frames
#         all_hand_pts  = [_vis_apply_pose_9d(templates["hand"],  p) for p in np.concatenate([gt_tcp_poses,   pred_tcp_poses])]
#         all_block_pts = [_vis_apply_pose_9d(templates["block"], p) for p in np.concatenate([gt_block_poses, pred_block_poses])]
#         all_pts = np.concatenate(all_hand_pts + all_block_pts, axis=0)

#         pad = 0.05
#         ctr = all_pts.mean(0)
#         r   = max((all_pts.max(0) - all_pts.min(0)).max() / 2 + pad, 0.1)
#         lims = [(float(ctr[j] - r), float(ctr[j] + r)) for j in range(3)]

#         gif_frames = []
#         for t_idx in frames:
#             img = _render_xy_frame(
#                 gt_tcp_9d    = gt_tcp_poses[t_idx],
#                 gt_block_9d  = gt_block_poses[t_idx],
#                 pr_tcp_9d    = pred_tcp_poses[t_idx],
#                 pr_block_9d  = pred_block_poses[t_idx],
#                 templates    = templates,
#                 lims         = lims,     # fixed bounds
#                 t            = t_idx,
#                 T            = T,
#                 elev         = elev,
#                 azim         = azim,
#             )
#             gif_frames.append(img)

#         out_path = step_dir / f"sample_{i}.gif"
#         imageio.mimsave(str(out_path), gif_frames, duration=1.0/gif_fps, loop=0)
#         print(f"  ↳ saved {out_path}")


# def _render_xy_frame(
#     gt_tcp_9d, gt_block_9d,
#     pr_tcp_9d, pr_block_9d,
#     templates, lims, t, T, elev, azim, dpi=100,
# ) -> np.ndarray:
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt

#     fig = plt.figure(figsize=(10, 5), dpi=dpi)

#     for col, (tcp_9d, block_9d, label) in enumerate([
#         (gt_tcp_9d,  gt_block_9d,  f"GT   t={t}/{T-1}"),
#         (pr_tcp_9d,  pr_block_9d,  f"Pred t={t}/{T-1}"),
#     ]):
#         ax = fig.add_subplot(1, 2, col+1, projection="3d")

#         hand_pts  = _vis_apply_pose_9d(templates["hand"],  tcp_9d)
#         block_pts = _vis_apply_pose_9d(templates["block"], block_9d)

#         ax.scatter(block_pts[:,0], block_pts[:,1], block_pts[:,2],
#                    s=1.2, c=[_VIS_BLOCK_COLOR.tolist()], alpha=0.6, depthshade=True)
#         ax.scatter(hand_pts[:,0],  hand_pts[:,1],  hand_pts[:,2],
#                    s=2.5, c=[_VIS_HAND_COLOR.tolist()],  alpha=0.9, depthshade=True)

#         # Fixed axes — never shifts between frames
#         (xlo, xhi), (ylo, yhi), (zlo, zhi) = lims
#         ax.set_xlim(xlo, xhi)
#         ax.set_ylim(ylo, yhi)
#         ax.set_zlim(zlo, zhi)
#         ax.set_box_aspect([1, 1, 1])
#         ax.view_init(elev=elev, azim=azim)
#         ax.set_title(label, fontsize=9)
#         ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
#         ax.tick_params(labelsize=6)

#     fig.suptitle("Blue=block  Orange=TCP", fontsize=8)
#     fig.tight_layout(pad=0.5)
#     fig.canvas.draw()
#     w, h = fig.canvas.get_width_height()
#     buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
#     img = buf.reshape(h, w, 4)[:, :, 1:]
#     plt.close(fig)
#     return img.copy()

# # Checkpoint helpers
# # ---------------------------------------------------------------------------

# def save_checkpoint(path: Path, model, ema: EMA, optimizer, lr_sched, step: int, cfg):
#     path.parent.mkdir(parents=True, exist_ok=True)
#     torch.save({
#         "step":      step,
#         "model":     model.state_dict(),
#         "ema":       ema.state_dict(),
#         "optimizer": optimizer.state_dict(),
#         "scheduler": lr_sched.state_dict(),
#         "cfg":       OmegaConf.to_container(cfg),
#     }, path)
#     print(f"  ✓ saved {path}")


# def load_checkpoint(path: str, model, optimizer, lr_sched, ema: EMA) -> int:
#     ckpt = torch.load(path, map_location="cpu")
#     model.load_state_dict(ckpt["model"])
#     optimizer.load_state_dict(ckpt["optimizer"])
#     lr_sched.load_state_dict(ckpt["scheduler"])
#     if "ema" in ckpt:
#         ema.load_state_dict(ckpt["ema"])
#     return ckpt["step"]

# def main(cfg):
#     torch.manual_seed(cfg.seed)

#     if torch.cuda.is_available():
#         device_name = "cuda"
#     elif torch.backends.mps.is_available():
#         device_name = "mps"
#     else:
#         device_name = "cpu"
#     device = torch.device(cfg.get("device", device_name))
#     print(f"Device: {device}")

#     run_dir  = Path(cfg.log.dir) / cfg.run_name
#     ckpt_dir = run_dir / "checkpoints"
#     vis_dir  = run_dir / "visualisations"
#     run_dir.mkdir(parents=True, exist_ok=True)
#     ckpt_dir.mkdir(exist_ok=True)
#     OmegaConf.save(cfg, run_dir / "config.yaml")
#     print(f"Run: {run_dir}")

#     # Data
#     train_ds = NpzDataset(cfg.data.train_path, cfg.data.seq_len)
#     val_ds   = NpzDataset(cfg.data.val_path,   cfg.data.seq_len)

#     if cfg.train.overfit_batches > 0:
#         print(f"Overfit mode: caching {cfg.train.overfit_batches} batch(es)…")
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

#     # Normalization
#     norm = compute_norm_stats(train_ds)

#     # Model
#     model_cfg = ModelConfig(**OmegaConf.to_container(cfg.model))
#     model     = DiT(model_cfg).to(device)
#     n_params  = sum(p.numel() for p in model.parameters())
#     print(f"Model: {n_params:,} parameters")

#     # Diffusion
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

#     start_step = 0
#     if cfg.resume:
#         start_step = load_checkpoint(cfg.resume, model, optimizer, lr_sched, ema)
#         print(f"Resumed from step {start_step}")

#     # Fixed vis batch — spread uniformly across val dataset
#     _vis_batch: Optional[torch.Tensor] = None
#     try:
#         n_vis      = cfg.log.get("vis_n_samples", 4)
#         vis_indices = [int(i * len(val_ds) / n_vis) for i in range(n_vis)]
#         _vis_batch  = torch.stack([val_ds[i] for i in vis_indices])  # [n_vis, T, 4]
#     except Exception as e:
#         print(f"  [warn] could not build vis batch: {e}")

#     # Training loop
#     model.train()
#     step, log_loss = start_step, 0.0
#     print(f"Training for {cfg.train.steps} steps…")

#     while step < cfg.train.steps:
#         seq     = next(train_iter)
#         metrics = train_step(model, seq, diffusion, norm, optimizer, device, cfg.train.grad_clip)
#         lr_sched.step()
#         ema.update()
#         log_loss += metrics["loss"]

#         # Logging
#         if step % cfg.log.every == 0 and step > 0:
#             avg      = log_loss / cfg.log.every
#             log_loss = 0.0
#             lr       = lr_sched.get_last_lr()[0]
#             msg      = (f"step {step:06d} | loss {avg:.4f}"
#                         f" | t_mean {metrics['t_mean']:.0f} | lr {lr:.2e}")
#             if "grad_norm" in metrics:
#                 msg += f" | gnorm {metrics['grad_norm']:.2f}"
#             print(msg)

#         # Validation
#         if step > 0 and step % cfg.log.val_every == 0:
#             val_m = validate(model, val_loader, diffusion, norm, ema, device, cfg.log.val_batches)
#             print(f"step {step:06d} | val_loss {val_m['loss']:.4f}")

#             # Visualisation
#             if _vis_batch is not None:
#                 ema.apply_shadow()
#                 try:
#                     visualize_predictions(
#                         model       = model,
#                         dataset     = val_ds,
#                         seq         = _vis_batch,
#                         diffusion   = diffusion,
#                         norm        = norm,
#                         n_steps     = n_inf_steps,
#                         device      = device,
#                         save_dir    = vis_dir,
#                         step        = step,
#                         n_samples   = cfg.log.get("vis_n_samples",  4),
#                         use_T_block = cfg.log.get("vis_use_T_block", False),
#                         every       = cfg.log.get("vis_every",       1),
#                         elev        = cfg.log.get("vis_elev",        30.0),
#                         azim        = cfg.log.get("vis_azim",       -60.0),
#                         gif_fps     = cfg.log.get("vis_gif_fps",     10.0),
#                     )
#                 except Exception as e:
#                     import traceback
#                     print(f"  [warn] visualisation failed: {e}")
#                     traceback.print_exc()
#                 finally:
#                     ema.restore()
#                     model.train()

#         # Inference MSE
#         if step > 0 and step % cfg.log.sample_every == 0:
#             ema.apply_shadow()
#             model.eval()
#             try:
#                 inf = run_inference(
#                     model     = model,
#                     seq       = next(train_iter).to(device),
#                     diffusion = diffusion,
#                     norm      = norm,
#                     n_steps   = n_inf_steps,
#                     device    = device,
#                 )
#                 print(f"step {step:06d} | mse {inf['mse']:.4f}")
#             finally:
#                 ema.restore()
#                 model.train()

#         # Checkpoint
#         if step > 0 and step % cfg.log.save_every == 0:
#             save_checkpoint(
#                 ckpt_dir / f"step_{step:07d}.pt",
#                 model, ema, optimizer, lr_sched, step, cfg,
#             )

#         step += 1

#     save_checkpoint(
#         ckpt_dir / "final.pt", model, ema, optimizer, lr_sched, step, cfg,
#     )
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
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from model      import DiT, ModelConfig
from diffusion  import DDIM, Stats
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
    seq:       torch.Tensor,               # [B, T, SD+AD]  raw
    diffusion: DDIM,
    norm:      Stats,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    grad_clip: Optional[float],
) -> dict:
    seq  = seq.to(device)
    B, T = seq.shape[:2]

    s_0, a_0 = normalize(seq, norm, model.state_dim)                          # [B, T, SD], [B, T, AD]

    s_noise = torch.randn_like(s_0)
    a_noise = torch.randn_like(a_0)

    timesteps = diffusion.sample_timesteps(B, device, num_frames=T)   # [B, T]

    s_t = diffusion.add_noise(s_0, s_noise, timesteps)
    a_t = diffusion.add_noise(a_0, a_noise, timesteps)

    x_t = torch.cat([s_t, a_t], dim=-1)   # [B, T, SD+AD]
    
    x_pred = model(x_t, timesteps)

    noise = torch.cat([s_noise, a_noise], dim=-1)
    loss = F.mse_loss(x_pred, noise)

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
    diffusion:   DDIM,
    norm:        Stats,
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

            x_t = torch.cat([s_t, a_t], dim=-1)
            x_pred = model(x_t, ts)
            
            noise = torch.cat([s_noise, a_noise], dim=-1)

            loss = (
                F.mse_loss(x_pred, noise)
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
    diffusion:            DDIM,
    norm:                 Stats,
    n_steps:              int,
    device:               torch.device,
    cond_frames:          List[int] = (),
    df_schedule:          str       = "pyramid",
    df_uncertainty_scale: float     = 1.0,
) -> dict:
    B, T, D = seq.shape
    SD       = model.state_dim
    AD      = D - SD

    s_0, a_0 = normalize(seq.to(device), norm, model.state_dim)   # [B, T, SD], [B, T, AD]

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

    ac = diffusion.scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)

    for m in range(len(K_ts) - 1):
        x_t = torch.cat([s_t, a_t], dim=-1)   # [B, T, SD+AD]
        curr_k = K_ts[m]
        next_k = K_ts[m + 1]

        t_batch = torch.zeros(B, T, device=device, dtype=torch.long)
        t_batch[:, gen_indices] = curr_k.unsqueeze(0).expand(B, -1)

        x_pred = model(x_t, t_batch)   # [B, T, SD+AD]

        a_t_coef  = ac[curr_k][None, :, None]
        a_p_coef  = ac[next_k][None, :, None]
        s_eps, a_eps = x_pred[:, gen_indices, :SD], x_pred[:, gen_indices, SD:SD+AD]
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


# # ---------------------------------------------------------------------------
# # Point-cloud visualisation helpers (XY-only model, fixed Z + identity rot)
# # ---------------------------------------------------------------------------

# _VIS_BLOCK_COLOR = np.array([0.20, 0.47, 0.96])   # blue
# _VIS_HAND_COLOR  = np.array([0.91, 0.47, 0.10])   # orange

# # Geometry — must match push_env.py constants
# _VIS_CUBE_HALF     = 0.025
# _VIS_CIRCLE_RADIUS = 0.025
# _VIS_T_BOXES = [                                        # (local_offset, half_extents)
#     (np.array([0.0, -0.0375/2,             0.0]), np.array([0.05,       0.0125,      0.02])),
#     (np.array([0.0,  4*0.0125 - 0.0375/2, 0.0]), np.array([0.0125, 0.075*0.75,      0.02])),
# ]
# _VIS_STICK_RADIUS   = 0.008
# _VIS_STICK_LENGTH   = 0.10
# _VIS_STICK_OFFSET_Z = 0.0   # stick centre is 10 cm above the TCP contact point


# def _vis_sample_box(offset: np.ndarray, half: np.ndarray, n: int, rng) -> np.ndarray:
#     """Uniform surface samples on a box centred at `offset` with half-extents `half`."""
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
#     idx = rng.choice(len(pts), size=min(n, len(pts)), replace=False)
#     return pts[idx]


# def _vis_sample_cylinder_z(offset_z: float, radius: float, length: float,
#                             n: int, rng) -> np.ndarray:
#     """Surface samples on a cylinder whose axis runs along Z, centred at z=offset_z."""
#     n_init = n * 20
#     th  = rng.uniform(0, 2 * np.pi, n_init)
#     z   = rng.uniform(-length / 2, length / 2, n_init) + offset_z
#     pts = np.stack([radius * np.cos(th), radius * np.sin(th), z], axis=1).astype(np.float32)
#     idx = rng.choice(len(pts), size=min(n, len(pts)), replace=False)
#     return pts[idx]


# def _vis_sample_circle_block(radius: float, n: int, rng) -> np.ndarray:
#     """
#     Surface samples for the circle block.

#     push_env.py spawns it as a cylinder rotated by euler2quat(0, pi/2, 0),
#     which swings the default Z-axis cylinder so its axis aligns with world X.
#     We reproduce that here: axis along X, radial spread in YZ.
#     """
#     n_init = n * 20
#     th  = rng.uniform(0, 2 * np.pi, n_init)
#     z   = rng.uniform(-radius, radius, n_init)
#     pts = np.stack([radius * np.cos(th), radius * np.sin(th), z], axis=1).astype(np.float32)
#     idx = rng.choice(len(pts), size=min(n, len(pts)), replace=False)
#     return pts[idx]

# def _vis_build_templates(n_block: int, n_hand: int, rng,
#                           block_shape: str = "cube") -> dict:
#     """
#     Build fixed local-frame point-cloud templates (called once, reused every frame).

#     Args:
#         n_block:     target point count for the block cloud
#         n_hand:      target point count for the hand / TCP cloud
#         rng:         numpy RNG
#         block_shape: "cube" | "T" | "circle"
#     """
#     if block_shape == "T":
#         areas     = [2 * (4*hy*hz + 4*hx*hz + 4*hx*hy)
#                      for _, (hx, hy, hz) in _VIS_T_BOXES]
#         total     = sum(areas)
#         parts     = [_vis_sample_box(off, half, max(1, int(n_block * a / total)), rng)
#                      for (off, half), a in zip(_VIS_T_BOXES, areas)]
#         block_tpl = np.concatenate(parts)
#     elif block_shape == "circle":
#         block_tpl = _vis_sample_circle_block(_VIS_CIRCLE_RADIUS, n_block, rng)
#     else:   # "cube"
#         block_tpl = _vis_sample_box(
#             np.zeros(3), np.full(3, _VIS_CUBE_HALF), n_block, rng
#         )

#     # TCP stick: vertical cylinder extending upward from the contact point
#     hand_tpl = _vis_sample_cylinder_z(
#         _VIS_STICK_OFFSET_Z, _VIS_STICK_RADIUS, _VIS_STICK_LENGTH, n_hand, rng
#     )
#     return {"block": block_tpl, "hand": hand_tpl}


# def _vis_place(template: np.ndarray, xy: np.ndarray, z: float) -> np.ndarray:
#     """
#     Translate a local-frame template to world position (xy[0], xy[1], z).

#     The model predicts XY only; block/hand orientation is not modelled, so
#     we simply shift the pre-built template rather than applying a full pose.
#     """
#     offset = np.array([xy[0], xy[1], z], dtype=np.float32)
#     return template + offset


# def _vis_render_frame(
#     gt_tcp_xy:   np.ndarray,   # (2,)  ground-truth TCP XY
#     gt_block_xy: np.ndarray,   # (2,)  ground-truth block XY
#     pr_tcp_xy:   np.ndarray,   # (2,)  predicted TCP XY
#     pr_block_xy: np.ndarray,   # (2,)  predicted block XY
#     templates:   dict,
#     tcp_z:       float,
#     block_z:     float,
#     lims:        list,          # [(xlo,xhi), (ylo,yhi), (zlo,zhi)] — fixed across frames
#     t:           int,
#     T:           int,
#     elev:        float,
#     azim:        float,
#     dpi:         int = 100,
# ) -> np.ndarray:
#     """Render one GIF frame: GT (left) and Pred (right) side-by-side 3D scatter."""
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt

#     fig = plt.figure(figsize=(10, 5), dpi=dpi)

#     for col, (tcp_xy, block_xy, label) in enumerate([
#         (gt_tcp_xy,  gt_block_xy,  f"GT   t={t}/{T-1}"),
#         (pr_tcp_xy,  pr_block_xy,  f"Pred t={t}/{T-1}"),
#     ]):
#         ax = fig.add_subplot(1, 2, col + 1, projection="3d")

#         hand_pts  = _vis_place(templates["hand"],  tcp_xy,   tcp_z)
#         block_pts = _vis_place(templates["block"], block_xy, block_z)

#         ax.scatter(*block_pts.T, s=1.2, c=[_VIS_BLOCK_COLOR.tolist()],
#                    alpha=0.6, depthshade=True, label="block")
#         ax.scatter(*hand_pts.T,  s=2.5, c=[_VIS_HAND_COLOR.tolist()],
#                    alpha=0.9, depthshade=True, label="TCP")

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
#     img = buf.reshape(h_px, w, 4)[:, :, 1:]   # ARGB → RGB
#     plt.close(fig)
#     return img.copy()


# def visualize_predictions(
#     model,
#     dataset,
#     seq:         torch.Tensor,      # [B, T, 4]  raw XY data
#     diffusion,
#     norm,
#     n_steps:     int,
#     device:      torch.device,
#     save_dir:    Path,
#     step:        int,
#     n_samples:   int   = 4,
#     block_shape: str   = "cube",    # "cube" | "T" | "circle"
#     every:       int   = 1,
#     elev:        float = 30.0,
#     azim:        float = -60.0,
#     gif_fps:     float = 10.0,
# ) -> None:
#     import imageio

#     n_samples = min(n_samples, seq.shape[0])
#     sub_seq   = seq[:n_samples].to(device)

#     # ── Run inference (simple unconditional DDIM loop) ────────────────────
#     model.eval()
#     with torch.no_grad():
#         x_t = torch.randn_like(sub_seq)
#         diffusion.scheduler.set_timesteps(n_steps, device=device)
#         for t in diffusion.scheduler.timesteps:
#             t_batch = t.expand(n_samples).unsqueeze(1).expand(n_samples, x_t.shape[1])
#             eps     = model(x_t, t_batch)
#             x_t     = diffusion.scheduler.step(eps, t, x_t).prev_sample
#     model.train()

#     pred_xy = norm.denormalize(x_t).cpu().numpy()       # [B, T, 4]
#     gt_xy   = norm.denormalize(sub_seq).cpu().numpy()   # [B, T, 4]

#     # Fixed Z heights from dataset statistics
#     tcp_z   = dataset.mean_tcp_z
#     block_z = dataset.mean_block_z

#     rng       = np.random.default_rng(0)
#     templates = _vis_build_templates(400, 250, rng, block_shape)
#     T_len     = gt_xy.shape[1]
#     frame_indices = list(range(0, T_len, every))

#     step_dir = save_dir / f"step_{step:07d}"
#     step_dir.mkdir(parents=True, exist_ok=True)

#     for i in range(n_samples):
#         # ── Compute scene limits once across all frames and both GT/pred ──
#         # so the camera never shifts between GIF frames.
#         all_hand_pts  = [_vis_place(templates["hand"],  xy, tcp_z)
#                          for xy in np.concatenate([gt_xy[i, :, 0:2], pred_xy[i, :, 0:2]])]
#         all_block_pts = [_vis_place(templates["block"], xy, block_z)
#                          for xy in np.concatenate([gt_xy[i, :, 2:4], pred_xy[i, :, 2:4]])]
#         all_pts = np.concatenate(all_hand_pts + all_block_pts, axis=0)
#         pad = 0.05
#         ctr = all_pts.mean(0)
#         r   = max((all_pts.max(0) - all_pts.min(0)).max() / 2 + pad, 0.1)
#         lims = [(float(ctr[j] - r), float(ctr[j] + r)) for j in range(3)]

#         gif_frames = []
#         for t_idx in frame_indices:
#             img = _vis_render_frame(
#                 gt_tcp_xy   = gt_xy[i,   t_idx, 0:2],
#                 gt_block_xy = gt_xy[i,   t_idx, 2:4],
#                 pr_tcp_xy   = pred_xy[i, t_idx, 0:2],
#                 pr_block_xy = pred_xy[i, t_idx, 2:4],
#                 templates   = templates,
#                 tcp_z       = tcp_z,
#                 block_z     = block_z,
#                 lims        = lims,
#                 t           = t_idx,
#                 T           = T_len,
#                 elev        = elev,
#                 azim        = azim,
#             )
#             gif_frames.append(img)

#         out_path = step_dir / f"sample_{i}.gif"
#         imageio.mimsave(str(out_path), gif_frames, duration=1.0 / gif_fps, loop=0)
#         print(f"  ↳ saved {out_path}")



 
# ---------------------------------------------------------------------------
# Point-cloud visualisation helpers (XY-only model, fixed Z + identity rot)
# ---------------------------------------------------------------------------
 
_VIS_BLOCK_COLOR = np.array([0.20, 0.47, 0.96])   # blue
_VIS_HAND_COLOR  = np.array([0.91, 0.47, 0.10])   # orange
 
# Geometry — must match push_env.py constants
_VIS_CUBE_HALF     = 0.025
_VIS_CIRCLE_RADIUS = 0.025
_VIS_T_BOXES = [                                        # (local_offset, half_extents)
    (np.array([0.0, -0.0375/2,             0.0]), np.array([0.05,       0.0125,      0.02])),
    (np.array([0.0,  4*0.0125 - 0.0375/2, 0.0]), np.array([0.0125, 0.075*0.75,      0.02])),
]
_VIS_STICK_RADIUS = 0.008
_VIS_STICK_LENGTH = 0.10
 
 
def _vis_sample_box(offset: np.ndarray, half: np.ndarray, n: int, rng) -> np.ndarray:
    """Uniform surface samples on a box centred at `offset` with half-extents `half`."""
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
    idx = rng.choice(len(pts), size=min(n, len(pts)), replace=False)
    return pts[idx]
 
 
# def _vis_sample_cylinder_z(radius: float, length: float, n: int, rng) -> np.ndarray:
#     """
#     Surface samples for the TCP stick, centred at the origin.
 
#     The stick is a cylinder rotated 90° about Y (euler2quat(0, pi/2, 0)),
#     so its axis runs along world X — same convention as the circle block.
#     Radial spread is in YZ; X is the axial direction.
#     """
#     n_init = n * 20
#     th  = rng.uniform(0, 2 * np.pi, n_init)
#     x   = rng.uniform(-length / 2, length / 2, n_init)   # axis along X
#     pts = np.stack([x, radius * np.cos(th), radius * np.sin(th)], axis=1).astype(np.float32)
#     idx = rng.choice(len(pts), size=min(n, len(pts)), replace=False)
#     return pts[idx]
 
 
# def _vis_sample_circle_block(radius: float, n: int, rng) -> np.ndarray:
#     """
#     Surface samples for the circle block.
 
#     push_env.py spawns it as a cylinder rotated by euler2quat(0, pi/2, 0),
#     which swings the default Z-axis cylinder so its axis aligns with world X.
#     We reproduce that here: axis along X, radial spread in YZ.
#     """
#     n_init = n * 20
#     th  = rng.uniform(0, 2 * np.pi, n_init)
#     x   = rng.uniform(-radius, radius, n_init)          # axis runs along X
#     pts = np.stack([x, radius * np.cos(th), radius * np.sin(th)], axis=1).astype(np.float32)
#     idx = rng.choice(len(pts), size=min(n, len(pts)), replace=False)
#     return pts[idx]

def _vis_sample_cylinder_z(radius: float, length: float, n: int, rng) -> np.ndarray:
    """
    Surface samples for the TCP stick, centred at the origin.
 
    Axis runs along world Z (upright). Radial spread is in the XY plane.
    """
    n_init = n * 20
    th  = rng.uniform(0, 2 * np.pi, n_init)
    z   = rng.uniform(-length / 2, length / 2, n_init)   # axis along Z
    
    # Stack order: [x, y, z] -> [r*cos, r*sin, linear]
    pts = np.stack([radius * np.cos(th), radius * np.sin(th), z], axis=1).astype(np.float32)
    
    idx = rng.choice(len(pts), size=min(n, len(pts)), replace=False)
    return pts[idx]
 
 
def _vis_sample_circle_block(radius: float, n: int, rng) -> np.ndarray:
    """
    Surface samples for the circle block.
 
    Axis runs along world Z. Radial spread is in the XY plane.
    """
    n_init = n * 20
    th  = rng.uniform(0, 2 * np.pi, n_init)
    z   = rng.uniform(-radius, radius, n_init)          # axis runs along Z
    
    # Stack order: [x, y, z] -> [r*cos, r*sin, linear]
    pts = np.stack([radius * np.cos(th), radius * np.sin(th), z], axis=1).astype(np.float32)
    
    idx = rng.choice(len(pts), size=min(n, len(pts)), replace=False)
    return pts[idx]
 
 
def _vis_build_templates(n_block: int, n_hand: int, rng,
                          block_shape: str = "cube") -> dict:
    """
    Build fixed local-frame point-cloud templates (called once, reused every frame).
 
    Args:
        n_block:     target point count for the block cloud
        n_hand:      target point count for the hand / TCP cloud
        rng:         numpy RNG
        block_shape: "cube" | "T" | "circle"
    """
    if block_shape == "T":
        areas     = [2 * (4*hy*hz + 4*hx*hz + 4*hx*hy)
                     for _, (hx, hy, hz) in _VIS_T_BOXES]
        total     = sum(areas)
        parts     = [_vis_sample_box(off, half, max(1, int(n_block * a / total)), rng)
                     for (off, half), a in zip(_VIS_T_BOXES, areas)]
        block_tpl = np.concatenate(parts)
    elif block_shape == "circle":
        block_tpl = _vis_sample_circle_block(_VIS_CIRCLE_RADIUS, n_block, rng)
    else:   # "cube"
        block_tpl = _vis_sample_box(
            np.zeros(3), np.full(3, _VIS_CUBE_HALF), n_block, rng
        )
 
    # TCP stick: vertical cylinder centred at the origin; _vis_place adds the world position
    hand_tpl = _vis_sample_cylinder_z(
        _VIS_STICK_RADIUS, _VIS_STICK_LENGTH, n_hand, rng
    )
    return {"block": block_tpl, "hand": hand_tpl}
 
 
def _vis_rot6d_to_rotmat(r6: np.ndarray) -> np.ndarray:
    """(6,) 6D rotation → (3,3) rotation matrix via Gram-Schmidt."""
    r1 = r6[:3].astype(np.float64)
    r2 = r6[3:].astype(np.float64)
    a1 = r1 / np.linalg.norm(r1)
    a2 = r2 - np.dot(r2, a1) * a1
    a2 = a2 / np.linalg.norm(a2)
    a3 = np.cross(a1, a2)
    return np.stack([a1, a2, a3], axis=-1)
 
 
def _vis_apply_pose_9d(pts: np.ndarray, pose_9d: np.ndarray) -> np.ndarray:
    """Transform local-frame point cloud by a 9D pose [pos(3), rot6d(6)]."""
    pos = pose_9d[:3].astype(np.float64)
    R   = _vis_rot6d_to_rotmat(pose_9d[3:9])
    return (pts.astype(np.float64) @ R.T + pos).astype(np.float32)
 
 
def _vis_render_frame(
    gt_tcp_9d:   np.ndarray,   # (9,)  GT TCP pose
    gt_block_9d: np.ndarray,   # (9,)  GT block pose
    pr_tcp_9d:   np.ndarray,   # (9,)  pred TCP pose  (XY replaced, rot/Z from GT)
    pr_block_9d: np.ndarray,   # (9,)  pred block pose
    templates:   dict,
    lims:        list,          # [(xlo,xhi),(ylo,yhi),(zlo,zhi)] — fixed across frames
    t:           int,
    T:           int,
    elev:        float,
    azim:        float,
    dpi:         int = 100,
) -> np.ndarray:
    """Render one GIF frame: GT (left) and Pred (right) side-by-side 3D scatter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    fig = plt.figure(figsize=(10, 5), dpi=dpi)
 
    for col, (tcp_9d, block_9d, label) in enumerate([
        (gt_tcp_9d,  gt_block_9d,  f"GT   t={t}/{T-1}"),
        (pr_tcp_9d,  pr_block_9d,  f"Pred t={t}/{T-1}"),
    ]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
 
        hand_pts  = _vis_apply_pose_9d(templates["hand"],  tcp_9d)
        block_pts = _vis_apply_pose_9d(templates["block"], block_9d)
 
        ax.scatter(*block_pts.T, s=1.2, c=[_VIS_BLOCK_COLOR.tolist()],
                   alpha=0.6, depthshade=True, label="block")
        ax.scatter(*hand_pts.T,  s=2.5, c=[_VIS_HAND_COLOR.tolist()],
                   alpha=0.9, depthshade=True, label="TCP")
 
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
    img = buf.reshape(h_px, w, 4)[:, :, 1:]   # ARGB → RGB
    plt.close(fig)
    return img.copy()
 
 
def visualize_predictions(
    model,
    dataset,
    seq:         torch.Tensor,      # [B, T, SD+AD]  raw — states(18) + actions(9)
    diffusion,
    norm,
    n_steps:     int,
    device:      torch.device,
    save_dir:    Path,
    step:        int,
    n_samples:   int   = 4,
    block_shape: str   = "cube",    # "cube" | "T" | "circle"
    every:       int   = 1,
    elev:        float = 90.0,
    azim:        float = -90.0,
    gif_fps:     float = 10.0,
) -> None:
    import imageio
 
    n_samples = min(n_samples, seq.shape[0])
    sub_seq   = seq[:n_samples].to(device)
 
    # ── Run inference (simple unconditional DDIM loop) ────────────────────
    model.eval()
    with torch.no_grad():
        x_t = torch.randn_like(sub_seq)
        diffusion.scheduler.set_timesteps(n_steps, device=device)
        for t in diffusion.scheduler.timesteps:
            t_batch = t.expand(n_samples).unsqueeze(1).expand(n_samples, x_t.shape[1])
            eps     = model(x_t, t_batch)
            x_t     = diffusion.scheduler.step(eps, t, x_t).prev_sample
    model.train()
 
    # GT is already raw; only the model output needs denormalizing
    pred_raw = norm.denormalize(x_t).cpu().numpy()    # [B, T, 4]  tcp_x, tcp_y, block_x, block_y
    gt_raw   = sub_seq.cpu().numpy()                  # [B, T, 4]
 
    tcp_z   = dataset.mean_tcp_z
    block_z = dataset.mean_block_z
 
    # Identity 6D rotation (first two columns of I): [1,0,0, 0,1,0]
    identity_rot6d = np.array([1.0, 0.0, 0.0,  0.0, 1.0, 0.0], dtype=np.float32)
 
    def _make_pose_9d(xy: np.ndarray, z: float) -> np.ndarray:
        """(2,) XY + fixed Z + identity rotation → (9,) pose."""
        return np.array([xy[0], xy[1], z,
                         *identity_rot6d], dtype=np.float32)
 
    rng       = np.random.default_rng(0)
    templates = _vis_build_templates(400, 250, rng, block_shape)
    T_len     = gt_raw.shape[1]
    frame_indices = list(range(0, T_len, every))
 
    step_dir = save_dir / f"step_{step:07d}"
    step_dir.mkdir(parents=True, exist_ok=True)
 
    for i in range(n_samples):
        gt_tcp_9d   = np.stack([_make_pose_9d(gt_raw[i, t, 0:2],   tcp_z)   for t in range(T_len)])  # (T, 9)
        gt_block_9d = np.stack([_make_pose_9d(gt_raw[i, t, 2:4],   block_z) for t in range(T_len)])  # (T, 9)
        pr_tcp_9d   = np.stack([_make_pose_9d(pred_raw[i, t, 0:2], tcp_z)   for t in range(T_len)])  # (T, 9)
        pr_block_9d = np.stack([_make_pose_9d(pred_raw[i, t, 2:4], block_z) for t in range(T_len)])  # (T, 9)
 
        # ── Fixed scene limits across all frames and both GT/pred ─────────
        all_pts = np.concatenate([
            np.concatenate([_vis_apply_pose_9d(templates["hand"],  p) for p in gt_tcp_9d]),
            np.concatenate([_vis_apply_pose_9d(templates["block"], p) for p in gt_block_9d]),
            np.concatenate([_vis_apply_pose_9d(templates["hand"],  p) for p in pr_tcp_9d]),
            np.concatenate([_vis_apply_pose_9d(templates["block"], p) for p in pr_block_9d]),
        ])
        pad = 0.05
        ctr = all_pts.mean(0)
        r   = max((all_pts.max(0) - all_pts.min(0)).max() / 2 + pad, 0.1)
        lims = [(float(ctr[j] - r), float(ctr[j] + r)) for j in range(3)]
 
        gif_frames = []
        for t_idx in frame_indices:
            img = _vis_render_frame(
                gt_tcp_9d   = gt_tcp_9d[t_idx],
                gt_block_9d = gt_block_9d[t_idx],
                pr_tcp_9d   = pr_tcp_9d[t_idx],
                pr_block_9d = pr_block_9d[t_idx],
                templates   = templates,
                lims        = lims,
                t           = t_idx,
                T           = T_len,
                elev        = elev,
                azim        = azim,
            )
            gif_frames.append(img)
 
        out_path = step_dir / f"sample_{i}.gif"
        imageio.mimsave(str(out_path), gif_frames, duration=1.0 / gif_fps, loop=0)
        print(f"  ↳ saved {out_path}")
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

def main(cfg):
    torch.manual_seed(cfg.seed)

    if torch.cuda.is_available():
        device_name = "cuda"
    elif torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"
    device = torch.device(cfg.get("device", device_name))
    print(f"Device: {device}")

    run_dir  = Path(cfg.log.dir) / cfg.run_name
    ckpt_dir = run_dir / "checkpoints"
    vis_dir  = run_dir / "visualisations"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    print(f"Run: {run_dir}")

    # Data
    train_ds = NpzDataset(cfg.data.train_path, cfg.data.seq_len, cfg.data.stride)
    val_ds   = NpzDataset(cfg.data.val_path,   cfg.data.seq_len, cfg.data.stride)

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

    # Normalization
    norm = compute_norm_stats(train_ds)

    # Model
    model_cfg = ModelConfig(**OmegaConf.to_container(cfg.model))
    model     = DiT(model_cfg).to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    # Diffusion
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

    start_step = 0
    if cfg.resume:
        start_step = load_checkpoint(cfg.resume, model, optimizer, lr_sched, ema)
        print(f"Resumed from step {start_step}")

    # Fixed vis batch — spread uniformly across val dataset
    _vis_batch: Optional[torch.Tensor] = None
    try:
        n_vis       = cfg.log.get("vis_n_samples", 4)
        vis_indices = [int(i * len(val_ds) / n_vis) for i in range(n_vis)]
        _vis_batch  = torch.stack([val_ds[i] for i in vis_indices])
    except Exception as e:
        print(f"  [warn] could not build vis batch: {e}")

    # Training loop
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

            # Visualisation
            if _vis_batch is not None:
                ema.apply_shadow()
                try:
                    visualize_predictions(
                        model       = model,
                        dataset     = val_ds,
                        seq         = _vis_batch,
                        diffusion   = diffusion,
                        norm        = norm,
                        n_steps     = n_inf_steps,
                        device      = device,
                        save_dir    = vis_dir,
                        step        = step,
                        n_samples   = cfg.log.get("vis_n_samples",   4),
                        block_shape = cfg.log.get("vis_block_shape", "cube"),
                        every       = cfg.log.get("vis_every",       1),
                        elev        = cfg.log.get("vis_elev",        30.0),
                        azim        = cfg.log.get("vis_azim",       -60.0),
                        gif_fps     = cfg.log.get("vis_gif_fps",     10.0),
                    )
                except Exception as e:
                    import traceback
                    print(f"  [warn] visualisation failed: {e}")
                    traceback.print_exc()
                finally:
                    ema.restore()
                    model.train()

        # Inference MSE
        if step > 0 and step % cfg.log.sample_every == 0:
            ema.apply_shadow()
            model.eval()
            try:
                inf = run_inference(
                    model     = model,
                    seq       = next(train_iter).to(device),
                    diffusion = diffusion,
                    norm      = norm,
                    n_steps   = n_inf_steps,
                    device    = device,
                )
                print(f"step {step:06d} | mse {inf['mse']:.4f}")
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