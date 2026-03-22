# #!/usr/bin/env python3
# """
# scripts/make_lowdim_dataset.py

# Convert a RecordEpisode HDF5 file into a low-dimensional NPZ dataset.

# Rotation representation
# -----------------------
# All orientations use the 6D rotation representation from:
#   "On the Continuity of Rotation Representations in Neural Networks"
#   Zhou et al., CVPR 2019.

# Given rotation matrix R = [r1 | r2 | r3], the 6D rep is the first two
# columns: [r1, r2] (shape 6).  This is continuous everywhere, unlike
# quaternions (double cover, discontinuous) or Euler angles (gimbal lock).

# Recovery:
#   a1 = normalize(r1)
#   a2 = normalize(r2 - (r2 · a1) * a1)
#   a3 = a1 × a2

# State  (18-dim): tcp  pos(3) + tcp  6D-rot(6)  ||  block pos(3) + block 6D-rot(6)
# Action  (9-dim): next-step tcp pos(3) + tcp 6D-rot(6)  — absolute pose, not a delta

# Source fields (both scalar-first quaternion: qw, qx, qy, qz):
#   obs/extra/tcp_pose   (T+1, 7)
#   obs/extra/block_pose (T+1, 7)

# Alignment:
#   state[t]  = tcp_pose[t]  ||  block_pose[t]
#   action[t] = tcp_pose[t+1]          <- absolute next-step TCP pose

# The last state (t = T-1) has no t+1 pose, so we repeat tcp_pose[T-1] as the
# final action (i.e. "stay where you are"), keeping array lengths equal.

# Usage
# -----
#     python scripts/make_lowdim_dataset.py \
#         --h5  demos/PushBoundary/scripted/scripted.h5 \
#         --out datasets/push_lowdim.npz

#     python scripts/make_lowdim_dataset.py --h5 ... --out ... --trim_len 4000
#     python scripts/make_lowdim_dataset.py --h5 ... --out ... --no_outlier_filter
# """

# from __future__ import annotations

# import argparse
# import sys
# from pathlib import Path

# import h5py
# import numpy as np

# _HERE = Path(__file__).resolve().parent
# sys.path.insert(0, str(_HERE.parent))


# # ─────────────────────────────────────────────────────────────────────────────
# # Rotation helpers
# # ─────────────────────────────────────────────────────────────────────────────

# def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
#     """
#     (N, 4) quaternions (qw, qx, qy, qz) → (N, 3, 3) rotation matrices.
#     Handles batch dimension.
#     """
#     q  = q / np.linalg.norm(q, axis=-1, keepdims=True)   # normalise
#     w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

#     R = np.stack([
#         1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y),
#           2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x),
#           2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y),
#     ], axis=-1).reshape(*q.shape[:-1], 3, 3)
#     return R


# def rotmat_to_6d(R: np.ndarray) -> np.ndarray:
#     """
#     (N, 3, 3) rotation matrices → (N, 6) 6D representation.
#     Takes the first two columns: [R[:, :, 0], R[:, :, 1]] flattened.
#     """
#     # R[:, :, 0] is first column, R[:, :, 1] is second column
#     return np.concatenate([R[..., 0], R[..., 1]], axis=-1)   # (N, 6)


# def rot6d_to_rotmat(r6: np.ndarray) -> np.ndarray:
#     """
#     (N, 6) 6D representation → (N, 3, 3) rotation matrices (Gram-Schmidt).
#     """
#     r1 = r6[..., :3]
#     r2 = r6[..., 3:]

#     a1 = r1 / np.linalg.norm(r1, axis=-1, keepdims=True)
#     a2 = r2 - (r2 * a1).sum(axis=-1, keepdims=True) * a1
#     a2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True)
#     a3 = np.cross(a1, a2)

#     return np.stack([a1, a2, a3], axis=-1)   # (N, 3, 3)  columns = axes


# def pose7_to_9d(pose: np.ndarray) -> np.ndarray:
#     """
#     (N, 7) [pos(3), qw, qx, qy, qz] → (N, 9) [pos(3), 6D-rot(6)].
#     """
#     pos  = pose[:, :3]                                  # (N, 3)
#     quat = pose[:, 3:]                                  # (N, 4) wxyz
#     R    = _quat_wxyz_to_rotmat(quat)                   # (N, 3, 3)
#     rot6 = rotmat_to_6d(R)                              # (N, 6)
#     return np.concatenate([pos, rot6], axis=-1)         # (N, 9)


# # ─────────────────────────────────────────────────────────────────────────────
# # H5 helpers
# # ─────────────────────────────────────────────────────────────────────────────

# def _traj_keys(f: h5py.File) -> list[str]:
#     keys = [k for k in f.keys() if k.startswith("traj_")]
#     keys.sort(key=lambda k: int(k.split("_")[1]))
#     return keys


# def _traj_len(traj: h5py.Group) -> int:
#     return traj["actions"].shape[0]   # number of steps (= T)


# def _iqr_min_len(lengths: list[int]) -> int:
#     arr  = np.array(lengths, dtype=float)
#     q1, q3 = np.percentile(arr, [25, 75])
#     lower   = q1 - 1.5 * (q3 - q1)
#     non_out = arr[arr >= lower]
#     return int((non_out if len(non_out) else arr).min())


# def _extract(traj: h5py.Group, trim_len: int) -> tuple[np.ndarray, np.ndarray] | None:
#     """
#     Returns:
#       states  (trim_len, 18):  tcp_9d || block_9d  at step t
#       actions (trim_len,  9):  tcp_9d              at step t+1
#     or None if the trajectory is shorter than trim_len.
#     """
#     T = traj["actions"].shape[0]
#     if T < trim_len:
#         return None

#     # obs arrays have T+1 entries (one per state including the final one)
#     tcp_pose_raw   = np.array(traj["obs"]["extra"]["tcp_pose"],   dtype=np.float32)   # (T+1, 7)
#     block_pose_raw = np.array(traj["obs"]["extra"]["block_pose"], dtype=np.float32)   # (T+1, 7)

#     # Convert every pose to 9D
#     tcp_9d   = pose7_to_9d(tcp_pose_raw)    # (T+1, 9)
#     block_9d = pose7_to_9d(block_pose_raw)  # (T+1, 9)

#     # States: use steps 0..trim_len-1
#     states = np.concatenate([tcp_9d[:trim_len], block_9d[:trim_len]], axis=-1)   # (trim_len, 18)

#     # Actions: absolute TCP pose at the NEXT step (t+1)
#     # For the last step (t = trim_len-1) we repeat the final pose ("stay put")
#     next_tcp = np.concatenate([
#         tcp_9d[1:trim_len],        # steps 1..trim_len-1  → (trim_len-1, 9)
#         tcp_9d[trim_len:trim_len+1],  # step trim_len (or repeat last if at end)
#     ], axis=0)                     # (trim_len, 9)

#     # Guard: if trim_len == T+1 the above slice could be empty; fall back to repeat
#     if next_tcp.shape[0] < trim_len:
#         repeat = np.tile(tcp_9d[-1:], (trim_len - next_tcp.shape[0], 1))
#         next_tcp = np.concatenate([next_tcp, repeat], axis=0)

#     actions = next_tcp.astype(np.float32)   # (trim_len, 9)

#     return states.astype(np.float32), actions


# # ─────────────────────────────────────────────────────────────────────────────
# # Main
# # ─────────────────────────────────────────────────────────────────────────────

# def build_dataset(
#     h5_path: str,
#     out_path: str,
#     trim_len: int | None,
#     no_outlier_filter: bool,
#     verbose: bool,
# ) -> None:
#     with h5py.File(h5_path, "r") as f:
#         keys = _traj_keys(f)
#         if not keys:
#             raise ValueError(f"No traj_* keys found in {h5_path}")

#         print(f"Found {len(keys)} trajectories in {h5_path}")

#         lengths  = [_traj_len(f[k]) for k in keys]
#         arr_len  = np.array(lengths)
#         print(f"\nTrajectory length stats:")
#         print(f"  min={arr_len.min()}  max={arr_len.max()}  "
#               f"mean={arr_len.mean():.1f}  median={np.median(arr_len):.1f}")

#         if trim_len is not None:
#             T = trim_len
#             print(f"\nUsing user-specified trim_len={T}")
#         elif no_outlier_filter:
#             T = int(arr_len.min())
#             print(f"\nNo outlier filter → trimming to global minimum: T={T}")
#         else:
#             T = _iqr_min_len(lengths)
#             q1, q3 = np.percentile(arr_len, [25, 75])
#             lower   = q1 - 1.5 * (q3 - q1)
#             n_out   = int((arr_len < lower).sum())
#             print(f"\nIQR outlier filter:  Q1={q1:.0f}  Q3={q3:.0f}  lower_fence={lower:.0f}")
#             if n_out:
#                 print(f"  → {n_out} short outlier(s) excluded from trim target")
#             print(f"  → trimming all trajectories to T={T}")

#         all_states:  list[np.ndarray] = []
#         all_actions: list[np.ndarray] = []
#         skipped = 0

#         for k in keys:
#             result = _extract(f[k], T)
#             if result is None:
#                 skipped += 1
#                 if verbose:
#                     print(f"  [skip] {k}  len={_traj_len(f[k])} < {T}")
#                 continue
#             all_states.append(result[0])
#             all_actions.append(result[1])

#         if not all_states:
#             raise RuntimeError(f"All trajectories shorter than trim_len={T}.")

#         print(f"\nKept {len(all_states)} / {len(keys)} trajectories  "
#               f"(skipped {skipped})")

#     states_np  = np.stack(all_states,  axis=0)   # (N, T, 18)
#     actions_np = np.stack(all_actions, axis=0)   # (N, T,  9)

#     print(f"\nDataset shape:")
#     print(f"  states:  {states_np.shape}  dtype={states_np.dtype}")
#     print(f"  actions: {actions_np.shape} dtype={actions_np.dtype}")
#     print(f"\nState layout  (18-dim):")
#     print(f"  [0:3]   tcp position")
#     print(f"  [3:9]   tcp 6D rotation  (first two cols of rotation matrix)")
#     print(f"  [9:12]  block position")
#     print(f"  [12:18] block 6D rotation")
#     print(f"\nAction layout  (9-dim):")
#     print(f"  [0:3]   next-step tcp position   (absolute, world frame)")
#     print(f"  [3:9]   next-step tcp 6D rotation (absolute, world frame)")
#     print(f"\nSample (traj 0, step 0):")
#     print(f"  tcp  pos    = {states_np[0, 0, 0:3]}")
#     print(f"  tcp  rot6d  = {states_np[0, 0, 3:9]}")
#     print(f"  block pos   = {states_np[0, 0, 9:12]}")
#     print(f"  block rot6d = {states_np[0, 0, 12:18]}")
#     print(f"  action      = {actions_np[0, 0]}")

#     out_path = Path(out_path)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     np.savez(out_path, states=states_np, actions=actions_np)
#     print(f"\nSaved → {out_path}")


# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--h5",               required=True)
#     p.add_argument("--out",              required=True)
#     p.add_argument("--trim_len",         type=int, default=None)
#     p.add_argument("--no_outlier_filter",action="store_true")
#     p.add_argument("--verbose",          action="store_true")
#     return p.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     build_dataset(args.h5, args.out, args.trim_len, args.no_outlier_filter, args.verbose)

#!/usr/bin/env python3
# """
# scripts/make_lowdim_dataset.py

# Convert a RecordEpisode HDF5 file into a low-dimensional NPZ dataset.

# Rotation representation
# -----------------------
# All orientations use the 6D rotation representation from:
#   "On the Continuity of Rotation Representations in Neural Networks"
#   Zhou et al., CVPR 2019.

# Given rotation matrix R = [r1 | r2 | r3], the 6D rep is the first two
# columns: [r1, r2] (shape 6).  This is continuous everywhere, unlike
# quaternions (double cover, discontinuous) or Euler angles (gimbal lock).

# Recovery:
#   a1 = normalize(r1)
#   a2 = normalize(r2 - (r2 · a1) * a1)
#   a3 = a1 × a2

# State  (18-dim): tcp  pos(3) + tcp  6D-rot(6)  ||  block pos(3) + block 6D-rot(6)
# Action  (9-dim): next-step tcp pos(3) + tcp 6D-rot(6)  — absolute pose, not a delta

# Source fields (both scalar-first quaternion: qw, qx, qy, qz):
#   obs/extra/tcp_pose   (T+1, 7)
#   obs/extra/block_pose (T+1, 7)

# Alignment:
#   state[t]  = tcp_pose[t]  ||  block_pose[t]
#   action[t] = tcp_pose[t+1]          <- absolute next-step TCP pose

# The last state (t = T-1) has no t+1 pose, so we repeat tcp_pose[T-1] as the
# final action (i.e. "stay where you are"), keeping array lengths equal.

# Usage
# -----
#     python scripts/make_lowdim_dataset.py \
#         --h5  demos/PushBoundary/scripted/scripted.h5 \
#         --out datasets/push_lowdim.npz

#     python scripts/make_lowdim_dataset.py --h5 ... --out ... --trim_len 4000
#     python scripts/make_lowdim_dataset.py --h5 ... --out ... --no_outlier_filter
#     python scripts/make_lowdim_dataset.py --h5 ... --out ... --skip_first 10
#     python scripts/make_lowdim_dataset.py --h5 ... --out ... --knocked_over_thresh 30
# """

# from __future__ import annotations

# import argparse
# import sys
# from pathlib import Path

# import h5py
# import numpy as np

# _HERE = Path(__file__).resolve().parent
# sys.path.insert(0, str(_HERE.parent))


# # ─────────────────────────────────────────────────────────────────────────────
# # Rotation helpers
# # ─────────────────────────────────────────────────────────────────────────────

# def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
#     """
#     (N, 4) quaternions (qw, qx, qy, qz) → (N, 3, 3) rotation matrices.
#     Handles batch dimension.
#     """
#     q  = q / np.linalg.norm(q, axis=-1, keepdims=True)   # normalise
#     w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

#     R = np.stack([
#         1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y),
#           2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x),
#           2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y),
#     ], axis=-1).reshape(*q.shape[:-1], 3, 3)
#     return R


# def rotmat_to_6d(R: np.ndarray) -> np.ndarray:
#     """
#     (N, 3, 3) rotation matrices → (N, 6) 6D representation.
#     Takes the first two columns: [R[:, :, 0], R[:, :, 1]] flattened.
#     """
#     # R[:, :, 0] is first column, R[:, :, 1] is second column
#     return np.concatenate([R[..., 0], R[..., 1]], axis=-1)   # (N, 6)


# def rot6d_to_rotmat(r6: np.ndarray) -> np.ndarray:
#     """
#     (N, 6) 6D representation → (N, 3, 3) rotation matrices (Gram-Schmidt).
#     """
#     r1 = r6[..., :3]
#     r2 = r6[..., 3:]

#     a1 = r1 / np.linalg.norm(r1, axis=-1, keepdims=True)
#     a2 = r2 - (r2 * a1).sum(axis=-1, keepdims=True) * a1
#     a2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True)
#     a3 = np.cross(a1, a2)

#     return np.stack([a1, a2, a3], axis=-1)   # (N, 3, 3)  columns = axes


# def pose7_to_9d(pose: np.ndarray) -> np.ndarray:
#     """
#     (N, 7) [pos(3), qw, qx, qy, qz] → (N, 9) [pos(3), 6D-rot(6)].
#     """
#     pos  = pose[:, :3]                                  # (N, 3)
#     quat = pose[:, 3:]                                  # (N, 4) wxyz
#     R    = _quat_wxyz_to_rotmat(quat)                   # (N, 3, 3)
#     rot6 = rotmat_to_6d(R)                              # (N, 6)
#     return np.concatenate([pos, rot6], axis=-1)         # (N, 9)


# # ─────────────────────────────────────────────────────────────────────────────
# # Knockover detection
# # ─────────────────────────────────────────────────────────────────────────────

# def _geodesic_angles_from_ref(R_seq: np.ndarray, R_ref: np.ndarray) -> np.ndarray:
#     """
#     Geodesic angle (radians) between each rotation in R_seq and a reference R_ref.

#     Args:
#         R_seq: (N, 3, 3) sequence of rotation matrices
#         R_ref: (3, 3)    reference rotation matrix (e.g. the initial block pose)

#     Returns:
#         angles: (N,) in radians, one per timestep

#     The geodesic angle is the rotation-angle of R_ref^T @ R_seq[t], which equals
#     arccos((trace - 1) / 2).  This is the standard axis-angle magnitude and is
#     invariant to the choice of reference frame, and catches tipping about any axis.
#     """
#     R_rel = R_ref.T @ R_seq                             # (N, 3, 3)
#     trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
#     cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
#     return np.arccos(cos_angle)                         # (N,) radians


# def _block_knocked_over(
#     block_pose_raw: np.ndarray,
#     skip_first: int,
#     threshold_deg: float,
# ) -> tuple[bool, float]:
#     """
#     Return (knocked_over, max_deviation_deg).

#     knocked_over is True if the block's orientation ever deviates more than
#     `threshold_deg` degrees from its orientation at frame `skip_first`
#     (the first kept frame after the warm-up skip).

#     We use the full geodesic (axis-angle magnitude) so the check is sensitive
#     to tipping about any axis, not just rotation around a single world axis.

#     Args:
#         block_pose_raw: (T+1, 7)  raw poses for every obs step
#         skip_first:     int       number of leading warm-up frames to ignore
#         threshold_deg:  float     maximum allowed orientation deviation in degrees
#     """
#     R_all = _quat_wxyz_to_rotmat(block_pose_raw[:, 3:])   # (T+1, 3, 3)
#     R_ref = R_all[skip_first]                              # reference = first kept frame
#     angles_rad = _geodesic_angles_from_ref(R_all[skip_first:], R_ref)
#     # print ref and and final angles for debugging, in degress
#     print("  block_ref_deg =", np.rad2deg(_geodesic_angles_from_ref(R_ref[None], R_ref)))
#     print("  block_final_deg =", np.rad2deg(_geodesic_angles_from_ref(R_all[-1:], R_ref)))
#     max_deg = float(np.rad2deg(angles_rad.max()))
#     return max_deg > threshold_deg, max_deg


# # ─────────────────────────────────────────────────────────────────────────────
# # H5 helpers
# # ─────────────────────────────────────────────────────────────────────────────

# def _traj_keys(f: h5py.File) -> list[str]:
#     keys = [k for k in f.keys() if k.startswith("traj_")]
#     keys.sort(key=lambda k: int(k.split("_")[1]))
#     return keys


# def _traj_len(traj: h5py.Group) -> int:
#     return traj["actions"].shape[0]   # number of steps (= T)


# def _iqr_min_len(lengths: list[int]) -> int:
#     arr  = np.array(lengths, dtype=float)
#     q1, q3 = np.percentile(arr, [25, 75])
#     lower   = q1 - 1.5 * (q3 - q1)
#     non_out = arr[arr >= lower]
#     return int((non_out if len(non_out) else arr).min())


# def _extract(
#     traj: h5py.Group,
#     trim_len: int,
#     skip_first: int,
#     knocked_over_thresh: float,
#     verbose: bool,
#     key: str,
# ) -> tuple[np.ndarray, np.ndarray] | None:
#     """
#     Returns:
#       states  (trim_len - skip_first, 18):  tcp_9d || block_9d  at step t
#       actions (trim_len - skip_first,  9):  tcp_9d              at step t+1
#     or None if the trajectory is too short or the block is knocked over.

#     Args:
#         traj:                h5py.Group for this trajectory
#         trim_len:            total steps to use before skipping (must be <= T)
#         skip_first:          drop this many leading frames from the final arrays
#         knocked_over_thresh: max allowed geodesic deviation (degrees) of block
#                              orientation from its pose at frame `skip_first`
#         verbose:             print knockover details
#         key:                 trajectory name, used only for verbose output
#     """
#     T = traj["actions"].shape[0]
#     if T < trim_len:
#         return None

#     # obs arrays have T+1 entries (one per state including the final one)
#     tcp_pose_raw   = np.array(traj["obs"]["extra"]["tcp_pose"],   dtype=np.float32)   # (T+1, 7)
#     block_pose_raw = np.array(traj["obs"]["extra"]["block_pose"], dtype=np.float32)   # (T+1, 7)

#     # ── Knockover check ────────────────────────────────────────────────────
#     # Uses the full raw block_pose_raw so skip_first is interpreted as an
#     # absolute frame index into the original (unskipped) trajectory.
#     if knocked_over_thresh < 180.0:   # 180° disables the check
#         knocked, max_dev = _block_knocked_over(block_pose_raw, skip_first, knocked_over_thresh)
#         if knocked:
#             if verbose:
#                 print(f"  [knock] {key}  max_block_rot={max_dev:.1f}° > {knocked_over_thresh}°")
#             return None
#         elif verbose:
#             print(f"  [ok]    {key}  max_block_rot={max_dev:.1f}°")

#     # ── Convert every pose to 9D ───────────────────────────────────────────
#     tcp_9d   = pose7_to_9d(tcp_pose_raw)    # (T+1, 9)
#     block_9d = pose7_to_9d(block_pose_raw)  # (T+1, 9)

#     # ── States: steps skip_first..trim_len-1 ──────────────────────────────
#     start = skip_first
#     end   = trim_len
#     states = np.concatenate(
#         [tcp_9d[start:end], block_9d[start:end]], axis=-1
#     )   # (trim_len - skip_first, 18)

#     # ── Actions: absolute TCP pose at the NEXT step (t+1) ─────────────────
#     # For the last kept step we repeat the final pose ("stay put").
#     next_tcp = np.concatenate([
#         tcp_9d[start + 1 : end],            # steps start+1..end-1  → (end-start-1, 9)
#         tcp_9d[end : end + 1],              # step `end` (or repeat last if at boundary)
#     ], axis=0)                              # (end - start, 9)

#     # Guard: if trim_len == T+1 the slice above could be one frame short
#     kept = end - start
#     if next_tcp.shape[0] < kept:
#         repeat = np.tile(tcp_9d[-1:], (kept - next_tcp.shape[0], 1))
#         next_tcp = np.concatenate([next_tcp, repeat], axis=0)

#     actions = next_tcp.astype(np.float32)   # (kept, 9)

#     return states.astype(np.float32), actions


# # ─────────────────────────────────────────────────────────────────────────────
# # Main
# # ─────────────────────────────────────────────────────────────────────────────

# def build_dataset(
#     h5_path: str,
#     out_path: str,
#     trim_len: int | None,
#     no_outlier_filter: bool,
#     skip_first: int,
#     knocked_over_thresh: float,
#     verbose: bool,
# ) -> None:
#     with h5py.File(h5_path, "r") as f:
#         keys = _traj_keys(f)
#         if not keys:
#             raise ValueError(f"No traj_* keys found in {h5_path}")

#         print(f"Found {len(keys)} trajectories in {h5_path}")

#         lengths  = [_traj_len(f[k]) for k in keys]
#         arr_len  = np.array(lengths)
#         print(f"\nTrajectory length stats:")
#         print(f"  min={arr_len.min()}  max={arr_len.max()}  "
#               f"mean={arr_len.mean():.1f}  median={np.median(arr_len):.1f}")

#         if skip_first > 0:
#             print(f"\nSkipping first {skip_first} frames of each trajectory")

#         if trim_len is not None:
#             T = trim_len
#             print(f"\nUsing user-specified trim_len={T}")
#         elif no_outlier_filter:
#             T = int(arr_len.min())
#             print(f"\nNo outlier filter → trimming to global minimum: T={T}")
#         else:
#             T = _iqr_min_len(lengths)
#             q1, q3 = np.percentile(arr_len, [25, 75])
#             lower   = q1 - 1.5 * (q3 - q1)
#             n_out   = int((arr_len < lower).sum())
#             print(f"\nIQR outlier filter:  Q1={q1:.0f}  Q3={q3:.0f}  lower_fence={lower:.0f}")
#             if n_out:
#                 print(f"  → {n_out} short outlier(s) excluded from trim target")
#             print(f"  → trimming all trajectories to T={T}")

#         if skip_first >= T:
#             raise ValueError(
#                 f"skip_first={skip_first} >= trim_len={T}: no frames would remain."
#             )

#         if knocked_over_thresh < 180.0:
#             print(f"\nKnockover filter: dropping trajectories where block rotates "
#                   f"> {knocked_over_thresh}° from its pose at frame {skip_first}")

#         all_states:  list[np.ndarray] = []
#         all_actions: list[np.ndarray] = []
#         skipped_short    = 0
#         skipped_knocked  = 0

#         for k in keys:
#             result = _extract(
#                 f[k], T, skip_first, knocked_over_thresh, verbose, k
#             )
#             if result is None:
#                 traj_len = _traj_len(f[k])
#                 if traj_len < T:
#                     skipped_short += 1
#                 else:
#                     skipped_knocked += 1
#                 continue
#             all_states.append(result[0])
#             all_actions.append(result[1])

#         if not all_states:
#             raise RuntimeError(
#                 f"No trajectories survived filtering "
#                 f"(trim_len={T}, skip_first={skip_first}, "
#                 f"knocked_over_thresh={knocked_over_thresh})."
#             )

#         print(f"\nKept {len(all_states)} / {len(keys)} trajectories")
#         if skipped_short:
#             print(f"  skipped {skipped_short} (too short < {T})")
#         if skipped_knocked:
#             print(f"  skipped {skipped_knocked} (block knocked over > {knocked_over_thresh}°)")

#     states_np  = np.stack(all_states,  axis=0)   # (N, T-skip, 18)
#     actions_np = np.stack(all_actions, axis=0)   # (N, T-skip,  9)

#     print(f"\nDataset shape:")
#     print(f"  states:  {states_np.shape}  dtype={states_np.dtype}")
#     print(f"  actions: {actions_np.shape} dtype={actions_np.dtype}")
#     print(f"\nState layout  (18-dim):")
#     print(f"  [0:3]   tcp position")
#     print(f"  [3:9]   tcp 6D rotation  (first two cols of rotation matrix)")
#     print(f"  [9:12]  block position")
#     print(f"  [12:18] block 6D rotation")
#     print(f"\nAction layout  (9-dim):")
#     print(f"  [0:3]   next-step tcp position   (absolute, world frame)")
#     print(f"  [3:9]   next-step tcp 6D rotation (absolute, world frame)")
#     print(f"\nSample (traj 0, step 0):")
#     print(f"  tcp  pos    = {states_np[0, 0, 0:3]}")
#     print(f"  tcp  rot6d  = {states_np[0, 0, 3:9]}")
#     print(f"  block pos   = {states_np[0, 0, 9:12]}")
#     print(f"  block rot6d = {states_np[0, 0, 12:18]}")
#     print(f"  action      = {actions_np[0, 0]}")

#     out_path = Path(out_path)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     np.savez(out_path, states=states_np, actions=actions_np)
#     print(f"\nSaved → {out_path}")


# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--h5",               required=True)
#     p.add_argument("--out",              required=True)
#     p.add_argument("--trim_len",         type=int,   default=None)
#     p.add_argument("--no_outlier_filter",action="store_true")
#     p.add_argument(
#         "--skip_first",
#         type=int,
#         default=0,
#         help="Drop this many leading frames from every trajectory (default: 0). "
#              "Useful to skip the robot settling / initialisation period.",
#     )
#     p.add_argument(
#         "--knocked_over_thresh",
#         type=float,
#         default=30.0,
#         help="Geodesic angle threshold in degrees (default: 30). Trajectories where "
#              "the block's orientation deviates more than this from its pose at frame "
#              "skip_first are dropped entirely. Set to 180 to disable.",
#     )
#     p.add_argument("--verbose",          action="store_true")
#     return p.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     build_dataset(
#         args.h5,
#         args.out,
#         args.trim_len,
#         args.no_outlier_filter,
#         args.skip_first,
#         args.knocked_over_thresh,
#         args.verbose,
#     )

# #!/usr/bin/env python3
# """
# scripts/make_lowdim_dataset.py

# Convert a RecordEpisode HDF5 file into a low-dimensional NPZ dataset.

# Rotation representation
# -----------------------
# All orientations use the 6D rotation representation from:
#   "On the Continuity of Rotation Representations in Neural Networks"
#   Zhou et al., CVPR 2019.

# Given rotation matrix R = [r1 | r2 | r3], the 6D rep is the first two
# columns: [r1, r2] (shape 6).  This is continuous everywhere, unlike
# quaternions (double cover, discontinuous) or Euler angles (gimbal lock).

# Recovery:
#   a1 = normalize(r1)
#   a2 = normalize(r2 - (r2 · a1) * a1)
#   a3 = a1 × a2

# State  (18-dim): tcp  pos(3) + tcp  6D-rot(6)  ||  block pos(3) + block 6D-rot(6)
# Action  (9-dim): next-step tcp pos(3) + tcp 6D-rot(6)  — absolute pose, not a delta

# Source fields (both scalar-first quaternion: qw, qx, qy, qz):
#   obs/extra/tcp_pose   (T+1, 7)
#   obs/extra/block_pose (T+1, 7)

# Alignment:
#   state[t]  = tcp_pose[t]  ||  block_pose[t]
#   action[t] = tcp_pose[t+1]          <- absolute next-step TCP pose

# The last state (t = T-1) has no t+1 pose, so we repeat tcp_pose[T-1] as the
# final action (i.e. "stay where you are"), keeping array lengths equal.

# Usage
# -----
#     python scripts/make_lowdim_dataset.py \
#         --h5  demos/PushBoundary/scripted/scripted.h5 \
#         --out datasets/push_lowdim.npz

#     python scripts/make_lowdim_dataset.py --h5 ... --out ... --trim_len 4000
#     python scripts/make_lowdim_dataset.py --h5 ... --out ... --no_outlier_filter
#     python scripts/make_lowdim_dataset.py --h5 ... --out ... --skip_first 10
#     python scripts/make_lowdim_dataset.py --h5 ... --out ... --knocked_over_thresh 30
# """

# from __future__ import annotations

# import argparse
# import sys
# from pathlib import Path

# import h5py
# import numpy as np

# _HERE = Path(__file__).resolve().parent
# sys.path.insert(0, str(_HERE.parent))


# # ─────────────────────────────────────────────────────────────────────────────
# # Rotation helpers
# # ─────────────────────────────────────────────────────────────────────────────

# def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
#     """
#     (N, 4) quaternions (qw, qx, qy, qz) → (N, 3, 3) rotation matrices.
#     Handles batch dimension.
#     """
#     q  = q / np.linalg.norm(q, axis=-1, keepdims=True)   # normalise
#     w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

#     R = np.stack([
#         1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y),
#           2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x),
#           2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y),
#     ], axis=-1).reshape(*q.shape[:-1], 3, 3)
#     return R


# def rotmat_to_6d(R: np.ndarray) -> np.ndarray:
#     """
#     (N, 3, 3) rotation matrices → (N, 6) 6D representation.
#     Takes the first two columns: [R[:, :, 0], R[:, :, 1]] flattened.
#     """
#     # R[:, :, 0] is first column, R[:, :, 1] is second column
#     return np.concatenate([R[..., 0], R[..., 1]], axis=-1)   # (N, 6)


# def rot6d_to_rotmat(r6: np.ndarray) -> np.ndarray:
#     """
#     (N, 6) 6D representation → (N, 3, 3) rotation matrices (Gram-Schmidt).
#     """
#     r1 = r6[..., :3]
#     r2 = r6[..., 3:]

#     a1 = r1 / np.linalg.norm(r1, axis=-1, keepdims=True)
#     a2 = r2 - (r2 * a1).sum(axis=-1, keepdims=True) * a1
#     a2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True)
#     a3 = np.cross(a1, a2)

#     return np.stack([a1, a2, a3], axis=-1)   # (N, 3, 3)  columns = axes


# def pose7_to_9d(pose: np.ndarray) -> np.ndarray:
#     """
#     (N, 7) [pos(3), qw, qx, qy, qz] → (N, 9) [pos(3), 6D-rot(6)].
#     """
#     pos  = pose[:, :3]                                  # (N, 3)
#     quat = pose[:, 3:]                                  # (N, 4) wxyz
#     R    = _quat_wxyz_to_rotmat(quat)                   # (N, 3, 3)
#     rot6 = rotmat_to_6d(R)                              # (N, 6)
#     return np.concatenate([pos, rot6], axis=-1)         # (N, 9)


# # ─────────────────────────────────────────────────────────────────────────────
# # Knockover detection
# # ─────────────────────────────────────────────────────────────────────────────

# def _y_tipping_angles_from_ref(R_seq: np.ndarray, R_ref: np.ndarray) -> np.ndarray:
#     """
#     Y-axis tipping angle (radians) of each rotation in R_seq relative to R_ref.

#     Args:
#         R_seq: (N, 3, 3) sequence of rotation matrices
#         R_ref: (3, 3)    reference rotation matrix (e.g. the initial block pose)

#     Returns:
#         angles: (N,) in radians, always >= 0

#     Method:
#         We extract the block's local Z column (the "up" direction in world
#         frame) from each rotation matrix and project it onto the world XZ
#         plane by zeroing the Y component.  The angle between the projected
#         reference up-vector and each frame's projected up-vector equals the
#         amount the block has tipped around the world Y axis.

#         Projecting onto XZ means X-axis tipping and yaw (spin around the
#         block's own vertical axis) do not contribute to the measurement.
#     """
#     # Local Z column = block "up" direction expressed in world frame
#     up_seq = R_seq[:, :, 2]                                 # (N, 3)
#     up_ref = R_ref[:, 2]                                    # (3,)

#     # Project onto world XZ plane to isolate rotation around world Y
#     _xz = np.array([1.0, 0.0, 1.0], dtype=np.float64)
#     up_seq_xz = up_seq * _xz                                # (N, 3)
#     up_ref_xz = up_ref * _xz                                # (3,)

#     # Normalise — clamp away from zero so arccos is stable even for large tilts
#     up_seq_xz /= np.linalg.norm(up_seq_xz, axis=-1, keepdims=True).clip(min=1e-8)
#     up_ref_xz /= max(float(np.linalg.norm(up_ref_xz)), 1e-8)

#     cos_angle = np.clip((up_seq_xz * up_ref_xz).sum(axis=-1), -1.0, 1.0)
#     return np.arccos(cos_angle)                             # (N,) radians


# def _block_knocked_over(
#     block_pose_raw: np.ndarray,
#     skip_first: int,
#     threshold_deg: float,
# ) -> tuple[bool, float]:
#     """
#     Return (knocked_over, max_deviation_deg).

#     knocked_over is True if the block's orientation ever deviates more than
#     `threshold_deg` degrees from its orientation at frame `skip_first`
#     (the first kept frame after the warm-up skip).

#     We use the full geodesic (axis-angle magnitude) so the check is sensitive
#     to tipping about any axis, not just rotation around a single world axis.

#     Args:
#         block_pose_raw: (T+1, 7)  raw poses for every obs step
#         skip_first:     int       number of leading warm-up frames to ignore
#         threshold_deg:  float     maximum allowed orientation deviation in degrees
#     """
#     R_all = _quat_wxyz_to_rotmat(block_pose_raw[:, 3:])   # (T+1, 3, 3)
#     R_ref = R_all[skip_first]                              # reference = first kept frame
#     angles_rad = _y_tipping_angles_from_ref(R_all[skip_first:], R_ref)
#     max_deg = float(np.rad2deg(angles_rad.max()))
#     return max_deg > threshold_deg, max_deg


# # ─────────────────────────────────────────────────────────────────────────────
# # H5 helpers
# # ─────────────────────────────────────────────────────────────────────────────

# def _traj_keys(f: h5py.File) -> list[str]:
#     keys = [k for k in f.keys() if k.startswith("traj_")]
#     keys.sort(key=lambda k: int(k.split("_")[1]))
#     return keys


# def _traj_len(traj: h5py.Group) -> int:
#     return traj["actions"].shape[0]   # number of steps (= T)


# def _iqr_min_len(lengths: list[int]) -> int:
#     arr  = np.array(lengths, dtype=float)
#     q1, q3 = np.percentile(arr, [25, 75])
#     lower   = q1 - 1.5 * (q3 - q1)
#     non_out = arr[arr >= lower]
#     return int((non_out if len(non_out) else arr).min())


# def _extract(
#     traj: h5py.Group,
#     trim_len: int,
#     skip_first: int,
#     knocked_over_thresh: float,
#     verbose: bool,
#     key: str,
# ) -> tuple[np.ndarray, np.ndarray] | None:
#     """
#     Returns:
#       states  (trim_len - skip_first, 18):  tcp_9d || block_9d  at step t
#       actions (trim_len - skip_first,  9):  tcp_9d              at step t+1
#     or None if the trajectory is too short or the block is knocked over.

#     Args:
#         traj:                h5py.Group for this trajectory
#         trim_len:            total steps to use before skipping (must be <= T)
#         skip_first:          drop this many leading frames from the final arrays
#         knocked_over_thresh: max allowed geodesic deviation (degrees) of block
#                              orientation from its pose at frame `skip_first`
#         verbose:             print knockover details
#         key:                 trajectory name, used only for verbose output
#     """
#     T = traj["actions"].shape[0]
#     if T < trim_len:
#         return None

#     # obs arrays have T+1 entries (one per state including the final one)
#     tcp_pose_raw   = np.array(traj["obs"]["extra"]["tcp_pose"],   dtype=np.float32)   # (T+1, 7)
#     block_pose_raw = np.array(traj["obs"]["extra"]["block_pose"], dtype=np.float32)   # (T+1, 7)

#     # ── Knockover check ────────────────────────────────────────────────────
#     # Uses the full raw block_pose_raw so skip_first is interpreted as an
#     # absolute frame index into the original (unskipped) trajectory.
#     if knocked_over_thresh < 180.0:   # 180° disables the check
#         knocked, max_dev = _block_knocked_over(block_pose_raw, skip_first, knocked_over_thresh)
#         if knocked:
#             if verbose:
#                 print(f"  [knock] {key}  max_block_rot={max_dev:.1f}° > {knocked_over_thresh}°")
#             return None
#         elif verbose:
#             print(f"  [ok]    {key}  max_block_rot={max_dev:.1f}°")

#     # ── Convert every pose to 9D ───────────────────────────────────────────
#     tcp_9d   = pose7_to_9d(tcp_pose_raw)    # (T+1, 9)
#     block_9d = pose7_to_9d(block_pose_raw)  # (T+1, 9)

#     # ── States: steps skip_first..trim_len-1 ──────────────────────────────
#     start = skip_first
#     end   = trim_len
#     states = np.concatenate(
#         [tcp_9d[start:end], block_9d[start:end]], axis=-1
#     )   # (trim_len - skip_first, 18)

#     # ── Actions: absolute TCP pose at the NEXT step (t+1) ─────────────────
#     # For the last kept step we repeat the final pose ("stay put").
#     next_tcp = np.concatenate([
#         tcp_9d[start + 1 : end],            # steps start+1..end-1  → (end-start-1, 9)
#         tcp_9d[end : end + 1],              # step `end` (or repeat last if at boundary)
#     ], axis=0)                              # (end - start, 9)

#     # Guard: if trim_len == T+1 the slice above could be one frame short
#     kept = end - start
#     if next_tcp.shape[0] < kept:
#         repeat = np.tile(tcp_9d[-1:], (kept - next_tcp.shape[0], 1))
#         next_tcp = np.concatenate([next_tcp, repeat], axis=0)

#     actions = next_tcp.astype(np.float32)   # (kept, 9)

#     return states.astype(np.float32), actions


# # ─────────────────────────────────────────────────────────────────────────────
# # Main
# # ─────────────────────────────────────────────────────────────────────────────

# def build_dataset(
#     h5_path: str,
#     out_path: str,
#     trim_len: int | None,
#     no_outlier_filter: bool,
#     skip_first: int,
#     knocked_over_thresh: float,
#     verbose: bool,
# ) -> None:
#     with h5py.File(h5_path, "r") as f:
#         keys = _traj_keys(f)
#         if not keys:
#             raise ValueError(f"No traj_* keys found in {h5_path}")

#         print(f"Found {len(keys)} trajectories in {h5_path}")

#         lengths  = [_traj_len(f[k]) for k in keys]
#         arr_len  = np.array(lengths)
#         print(f"\nTrajectory length stats:")
#         print(f"  min={arr_len.min()}  max={arr_len.max()}  "
#               f"mean={arr_len.mean():.1f}  median={np.median(arr_len):.1f}")

#         if skip_first > 0:
#             print(f"\nSkipping first {skip_first} frames of each trajectory")

#         if trim_len is not None:
#             T = trim_len
#             print(f"\nUsing user-specified trim_len={T}")
#         elif no_outlier_filter:
#             T = int(arr_len.min())
#             print(f"\nNo outlier filter → trimming to global minimum: T={T}")
#         else:
#             T = _iqr_min_len(lengths)
#             q1, q3 = np.percentile(arr_len, [25, 75])
#             lower   = q1 - 1.5 * (q3 - q1)
#             n_out   = int((arr_len < lower).sum())
#             print(f"\nIQR outlier filter:  Q1={q1:.0f}  Q3={q3:.0f}  lower_fence={lower:.0f}")
#             if n_out:
#                 print(f"  → {n_out} short outlier(s) excluded from trim target")
#             print(f"  → trimming all trajectories to T={T}")

#         if skip_first >= T:
#             raise ValueError(
#                 f"skip_first={skip_first} >= trim_len={T}: no frames would remain."
#             )

#         if knocked_over_thresh < 180.0:
#             print(f"\nKnockover filter: dropping trajectories where block rotates "
#                   f"> {knocked_over_thresh}° from its pose at frame {skip_first}")

#         all_states:  list[np.ndarray] = []
#         all_actions: list[np.ndarray] = []
#         skipped_short    = 0
#         skipped_knocked  = 0

#         for k in keys:
#             result = _extract(
#                 f[k], T, skip_first, knocked_over_thresh, verbose, k
#             )
#             if result is None:
#                 traj_len = _traj_len(f[k])
#                 if traj_len < T:
#                     skipped_short += 1
#                 else:
#                     skipped_knocked += 1
#                 continue
#             all_states.append(result[0])
#             all_actions.append(result[1])

#         if not all_states:
#             raise RuntimeError(
#                 f"No trajectories survived filtering "
#                 f"(trim_len={T}, skip_first={skip_first}, "
#                 f"knocked_over_thresh={knocked_over_thresh})."
#             )

#         print(f"\nKept {len(all_states)} / {len(keys)} trajectories")
#         if skipped_short:
#             print(f"  skipped {skipped_short} (too short < {T})")
#         if skipped_knocked:
#             print(f"  skipped {skipped_knocked} (block knocked over > {knocked_over_thresh}°)")

#     states_np  = np.stack(all_states,  axis=0)   # (N, T-skip, 18)
#     actions_np = np.stack(all_actions, axis=0)   # (N, T-skip,  9)

#     print(f"\nDataset shape:")
#     print(f"  states:  {states_np.shape}  dtype={states_np.dtype}")
#     print(f"  actions: {actions_np.shape} dtype={actions_np.dtype}")
#     print(f"\nState layout  (18-dim):")
#     print(f"  [0:3]   tcp position")
#     print(f"  [3:9]   tcp 6D rotation  (first two cols of rotation matrix)")
#     print(f"  [9:12]  block position")
#     print(f"  [12:18] block 6D rotation")
#     print(f"\nAction layout  (9-dim):")
#     print(f"  [0:3]   next-step tcp position   (absolute, world frame)")
#     print(f"  [3:9]   next-step tcp 6D rotation (absolute, world frame)")
#     print(f"\nSample (traj 0, step 0):")
#     print(f"  tcp  pos    = {states_np[0, 0, 0:3]}")
#     print(f"  tcp  rot6d  = {states_np[0, 0, 3:9]}")
#     print(f"  block pos   = {states_np[0, 0, 9:12]}")
#     print(f"  block rot6d = {states_np[0, 0, 12:18]}")
#     print(f"  action      = {actions_np[0, 0]}")

#     out_path = Path(out_path)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     np.savez(out_path, states=states_np, actions=actions_np)
#     print(f"\nSaved → {out_path}")


# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--h5",               required=True)
#     p.add_argument("--out",              required=True)
#     p.add_argument("--trim_len",         type=int,   default=None)
#     p.add_argument("--no_outlier_filter",action="store_true")
#     p.add_argument(
#         "--skip_first",
#         type=int,
#         default=0,
#         help="Drop this many leading frames from every trajectory (default: 0). "
#              "Useful to skip the robot settling / initialisation period.",
#     )
#     p.add_argument(
#         "--knocked_over_thresh",
#         type=float,
#         default=30.0,
#         help="Geodesic angle threshold in degrees (default: 30). Trajectories where "
#              "the block's orientation deviates more than this from its pose at frame "
#              "skip_first are dropped entirely. Set to 180 to disable.",
#     )
#     p.add_argument("--verbose",          action="store_true")
#     return p.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     build_dataset(
#         args.h5,
#         args.out,
#         args.trim_len,
#         args.no_outlier_filter,
#         args.skip_first,
#         args.knocked_over_thresh,
#         args.verbose,
#     )

#!/usr/bin/env python3
"""
scripts/make_lowdim_dataset.py

Convert a RecordEpisode HDF5 file into a low-dimensional NPZ dataset.
Pads sequences to the maximum length and saves their actual valid lengths.

Rotation representation
-----------------------
All orientations use the 6D rotation representation from:
  "On the Continuity of Rotation Representations in Neural Networks"
  Zhou et al., CVPR 2019.

Given rotation matrix R = [r1 | r2 | r3], the 6D rep is the first two
columns: [r1, r2] (shape 6).  This is continuous everywhere, unlike
quaternions (double cover, discontinuous) or Euler angles (gimbal lock).

Recovery:
  a1 = normalize(r1)
  a2 = normalize(r2 - (r2 · a1) * a1)
  a3 = a1 × a2

State  (18-dim): tcp  pos(3) + tcp  6D-rot(6)  ||  block pos(3) + block 6D-rot(6)
Action  (9-dim): next-step tcp pos(3) + tcp 6D-rot(6)  — absolute pose, not a delta

Source fields (both scalar-first quaternion: qw, qx, qy, qz):
  obs/extra/tcp_pose   (T+1, 7)
  obs/extra/block_pose (T+1, 7)

Alignment:
  state[t]  = tcp_pose[t]  ||  block_pose[t]
  action[t] = tcp_pose[t+1]          <- absolute next-step TCP pose

The last state (t = T-1) has no t+1 pose, so we repeat tcp_pose[T-1] as the
final action (i.e. "stay where you are"), keeping array lengths equal.

Usage
-----
    python scripts/make_lowdim_dataset.py \
        --h5  demos/PushBoundary/scripted/scripted.h5 \
        --out datasets/push_lowdim.npz

    python scripts/make_lowdim_dataset.py --h5 ... --out ... --min_len 4000
    python scripts/make_lowdim_dataset.py --h5 ... --out ... --no_outlier_filter
    python scripts/make_lowdim_dataset.py --h5 ... --out ... --skip_first 10
    python scripts/make_lowdim_dataset.py --h5 ... --out ... --knocked_over_thresh 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Rotation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """
    (N, 4) quaternions (qw, qx, qy, qz) → (N, 3, 3) rotation matrices.
    Handles batch dimension.
    """
    q  = q / np.linalg.norm(q, axis=-1, keepdims=True)   # normalise
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = np.stack([
        1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y),
          2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x),
          2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y),
    ], axis=-1).reshape(*q.shape[:-1], 3, 3)
    return R


def rotmat_to_6d(R: np.ndarray) -> np.ndarray:
    """
    (N, 3, 3) rotation matrices → (N, 6) 6D representation.
    Takes the first two columns: [R[:, :, 0], R[:, :, 1]] flattened.
    """
    return np.concatenate([R[..., 0], R[..., 1]], axis=-1)   # (N, 6)


def rot6d_to_rotmat(r6: np.ndarray) -> np.ndarray:
    """
    (N, 6) 6D representation → (N, 3, 3) rotation matrices (Gram-Schmidt).
    """
    r1 = r6[..., :3]
    r2 = r6[..., 3:]

    a1 = r1 / np.linalg.norm(r1, axis=-1, keepdims=True)
    a2 = r2 - (r2 * a1).sum(axis=-1, keepdims=True) * a1
    a2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True)
    a3 = np.cross(a1, a2)

    return np.stack([a1, a2, a3], axis=-1)   # (N, 3, 3)  columns = axes


def pose7_to_9d(pose: np.ndarray) -> np.ndarray:
    """
    (N, 7) [pos(3), qw, qx, qy, qz] → (N, 9) [pos(3), 6D-rot(6)].
    """
    pos  = pose[:, :3]                                  # (N, 3)
    quat = pose[:, 3:]                                  # (N, 4) wxyz
    R    = _quat_wxyz_to_rotmat(quat)                   # (N, 3, 3)
    rot6 = rotmat_to_6d(R)                              # (N, 6)
    return np.concatenate([pos, rot6], axis=-1)         # (N, 9)


# ─────────────────────────────────────────────────────────────────────────────
# Knockover detection
# ─────────────────────────────────────────────────────────────────────────────

def _y_tipping_angles_from_ref(R_seq: np.ndarray, R_ref: np.ndarray) -> np.ndarray:
    """
    Y-axis tipping angle (radians) of each rotation in R_seq relative to R_ref.
    """
    up_seq = R_seq[:, :, 2]                                 # (N, 3)
    up_ref = R_ref[:, 2]                                    # (3,)

    _xz = np.array([1.0, 0.0, 1.0], dtype=np.float64)
    up_seq_xz = up_seq * _xz                                # (N, 3)
    up_ref_xz = up_ref * _xz                                # (3,)

    up_seq_xz /= np.linalg.norm(up_seq_xz, axis=-1, keepdims=True).clip(min=1e-8)
    up_ref_xz /= max(float(np.linalg.norm(up_ref_xz)), 1e-8)

    cos_angle = np.clip((up_seq_xz * up_ref_xz).sum(axis=-1), -1.0, 1.0)
    return np.arccos(cos_angle)                             # (N,) radians


def _block_knocked_over(
    block_pose_raw: np.ndarray,
    skip_first: int,
    threshold_deg: float,
) -> tuple[bool, float]:
    """
    Return (knocked_over, max_deviation_deg).
    """
    R_all = _quat_wxyz_to_rotmat(block_pose_raw[:, 3:])   # (T+1, 3, 3)
    R_ref = R_all[skip_first]                             # reference = first kept frame
    angles_rad = _y_tipping_angles_from_ref(R_all[skip_first:], R_ref)
    max_deg = float(np.rad2deg(angles_rad.max()))
    return max_deg > threshold_deg, max_deg


# ─────────────────────────────────────────────────────────────────────────────
# H5 helpers
# ─────────────────────────────────────────────────────────────────────────────

def _traj_keys(f: h5py.File) -> list[str]:
    keys = [k for k in f.keys() if k.startswith("traj_")]
    keys.sort(key=lambda k: int(k.split("_")[1]))
    return keys


def _traj_len(traj: h5py.Group) -> int:
    return traj["actions"].shape[0]   # number of steps (= T)


def _iqr_min_len(lengths: list[int]) -> int:
    arr  = np.array(lengths, dtype=float)
    q1, q3 = np.percentile(arr, [25, 75])
    lower   = q1 - 1.5 * (q3 - q1)
    non_out = arr[arr >= lower]
    return int((non_out if len(non_out) else arr).min())


def _extract_trajectory(
    traj: h5py.Group,
    skip_first: int,
    knocked_over_thresh: float,
    verbose: bool,
    key: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Extracts the FULL valid length of a trajectory minus skipped frames.
    Returns:
      states  (T - skip_first, 18)
      actions (T - skip_first,  9)
    or None if the block is knocked over or trajectory is too short after skip.
    """
    T = traj["actions"].shape[0]

    if T <= skip_first:
        return None

    tcp_pose_raw   = np.array(traj["obs"]["extra"]["tcp_pose"],   dtype=np.float32)
    block_pose_raw = np.array(traj["obs"]["extra"]["block_pose"], dtype=np.float32)

    # ── Knockover check ────────────────────────────────────────────────────
    if knocked_over_thresh < 180.0:
        knocked, max_dev = _block_knocked_over(block_pose_raw, skip_first, knocked_over_thresh)
        if knocked:
            if verbose:
                print(f"  [knock] {key}  max_block_rot={max_dev:.1f}° > {knocked_over_thresh}°")
            return None
        elif verbose:
            print(f"  [ok]    {key}  max_block_rot={max_dev:.1f}°")

    # ── Convert every pose to 9D ───────────────────────────────────────────
    tcp_9d   = pose7_to_9d(tcp_pose_raw)    # (T+1, 9)
    block_9d = pose7_to_9d(block_pose_raw)  # (T+1, 9)

    # ── States: steps skip_first..T-1 ─────────────────────────────────────
    start = skip_first
    end   = T
    states = np.concatenate(
        [tcp_9d[start:end], block_9d[start:end]], axis=-1
    )   # (T - skip_first, 18)

    # ── Actions: absolute TCP pose at the NEXT step (t+1) ─────────────────
    next_tcp = np.concatenate([
        tcp_9d[start + 1 : end],            # steps start+1..end-1
        tcp_9d[end : end + 1],              # step `end` (or repeat last if at boundary)
    ], axis=0)                              

    # Guard 
    kept = end - start
    if next_tcp.shape[0] < kept:
        repeat = np.tile(tcp_9d[-1:], (kept - next_tcp.shape[0], 1))
        next_tcp = np.concatenate([next_tcp, repeat], axis=0)

    actions = next_tcp.astype(np.float32)   # (kept, 9)
    return states.astype(np.float32), actions


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    h5_path: str,
    out_path: str,
    min_len_arg: int | None,
    no_outlier_filter: bool,
    skip_first: int,
    knocked_over_thresh: float,
    verbose: bool,
) -> None:
    with h5py.File(h5_path, "r") as f:
        keys = _traj_keys(f)
        if not keys:
            raise ValueError(f"No traj_* keys found in {h5_path}")

        print(f"Found {len(keys)} trajectories in {h5_path}")

        lengths  = [_traj_len(f[k]) for k in keys]
        arr_len  = np.array(lengths)
        print(f"\nTrajectory length stats:")
        print(f"  min={arr_len.min()}  max={arr_len.max()}  "
              f"mean={arr_len.mean():.1f}  median={np.median(arr_len):.1f}")

        if skip_first > 0:
            print(f"\nSkipping first {skip_first} frames of each trajectory")

        # Determine the cutoff for dropping trajectories
        if min_len_arg is not None:
            min_len = min_len_arg
            print(f"\nUsing user-specified min_len={min_len} to drop short trajectories")
        elif no_outlier_filter:
            min_len = int(arr_len.min())
            print(f"\nNo outlier filter → drop threshold is global minimum: {min_len}")
        else:
            min_len = _iqr_min_len(lengths)
            q1, q3 = np.percentile(arr_len, [25, 75])
            lower   = q1 - 1.5 * (q3 - q1)
            n_out   = int((arr_len < lower).sum())
            print(f"\nIQR outlier filter:  Q1={q1:.0f}  Q3={q3:.0f}  lower_fence={lower:.0f}")
            if n_out:
                print(f"  → {n_out} short outlier(s) will be dropped")
            print(f"  → minimum required length: {min_len}")

        if skip_first >= min_len:
            raise ValueError(
                f"skip_first={skip_first} >= min_len={min_len}: no frames would remain."
            )

        if knocked_over_thresh < 180.0:
            print(f"\nKnockover filter: dropping trajectories where block rotates "
                  f"> {knocked_over_thresh}° from its pose at frame {skip_first}")

        all_states:  list[np.ndarray] = []
        all_actions: list[np.ndarray] = []
        valid_lengths: list[int] = []
        
        skipped_short    = 0
        skipped_knocked  = 0

        # Extract all valid sequences at their native lengths
        for k in keys:
            traj_len = _traj_len(f[k])
            
            # 1. Drop if it doesn't meet the minimum length
            if traj_len < min_len:
                skipped_short += 1
                continue
                
            # 2. Extract unpadded data and check for knockovers
            result = _extract_trajectory(
                f[k], skip_first, knocked_over_thresh, verbose, k
            )
            
            if result is None:
                skipped_knocked += 1
                continue
                
            s, a = result
            all_states.append(s)
            all_actions.append(a)
            valid_lengths.append(s.shape[0])

        if not all_states:
            raise RuntimeError(
                f"No trajectories survived filtering "
                f"(min_len={min_len}, skip_first={skip_first}, "
                f"knocked_over_thresh={knocked_over_thresh})."
            )

        print(f"\nKept {len(all_states)} / {len(keys)} trajectories")
        if skipped_short:
            print(f"  dropped {skipped_short} (too short < {min_len})")
        if skipped_knocked:
            print(f"  dropped {skipped_knocked} (block knocked over > {knocked_over_thresh}°)")

    # ── Zero-padding to uniform shape ──────────────────────────────────────
    max_len = max(valid_lengths)
    N = len(all_states)
    
    print(f"\nPadding sequences to max valid length: {max_len}")
    
    # Initialize padded arrays with zeros
    states_np  = np.zeros((N, max_len, 18), dtype=np.float32)
    actions_np = np.zeros((N, max_len,  9), dtype=np.float32)
    valid_lengths_np = np.array(valid_lengths, dtype=np.int32)
    
    # Fill in the actual data
    for i, (s, a) in enumerate(zip(all_states, all_actions)):
        L = s.shape[0]
        states_np[i, :L, :] = s
        actions_np[i, :L, :] = a

    print(f"\nDataset shape:")
    print(f"  states:        {states_np.shape} dtype={states_np.dtype}")
    print(f"  actions:       {actions_np.shape} dtype={actions_np.dtype}")
    print(f"  valid_lengths: {valid_lengths_np.shape}     dtype={valid_lengths_np.dtype}")
    print(f"\nState layout  (18-dim):")
    print(f"  [0:3]   tcp position")
    print(f"  [3:9]   tcp 6D rotation  (first two cols of rotation matrix)")
    print(f"  [9:12]  block position")
    print(f"  [12:18] block 6D rotation")
    print(f"\nAction layout  (9-dim):")
    print(f"  [0:3]   next-step tcp position   (absolute, world frame)")
    print(f"  [3:9]   next-step tcp 6D rotation (absolute, world frame)")
    print(f"\nSample (traj 0, step 0):")
    print(f"  tcp  pos    = {states_np[0, 0, 0:3]}")
    print(f"  tcp  rot6d  = {states_np[0, 0, 3:9]}")
    print(f"  block pos   = {states_np[0, 0, 9:12]}")
    print(f"  block rot6d = {states_np[0, 0, 12:18]}")
    print(f"  action      = {actions_np[0, 0]}")
    print(f"  valid_len   = {valid_lengths_np[0]}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save states, actions, and valid lengths together
    np.savez(out_path, states=states_np, actions=actions_np, valid_lengths=valid_lengths_np)
    print(f"\nSaved → {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5",               required=True)
    p.add_argument("--out",              required=True)
    p.add_argument(
        "--min_len",         
        type=int,   
        default=None,
        help="Drop trajectories shorter than this length. If not provided, determined by IQR."
    )
    p.add_argument("--no_outlier_filter",action="store_true")
    p.add_argument(
        "--skip_first",
        type=int,
        default=0,
        help="Drop this many leading frames from every trajectory (default: 0). "
             "Useful to skip the robot settling / initialisation period.",
    )
    p.add_argument(
        "--knocked_over_thresh",
        type=float,
        default=30.0,
        help="Geodesic angle threshold in degrees (default: 30). Trajectories where "
             "the block's orientation deviates more than this from its pose at frame "
             "skip_first are dropped entirely. Set to 180 to disable.",
    )
    p.add_argument("--verbose",          action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        args.h5,
        args.out,
        args.min_len,
        args.no_outlier_filter,
        args.skip_first,
        args.knocked_over_thresh,
        args.verbose,
    )