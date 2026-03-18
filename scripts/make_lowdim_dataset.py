#!/usr/bin/env python3
"""
scripts/make_lowdim_dataset.py

Convert a RecordEpisode HDF5 file into a low-dimensional NPZ dataset.

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

    python scripts/make_lowdim_dataset.py --h5 ... --out ... --trim_len 4000
    python scripts/make_lowdim_dataset.py --h5 ... --out ... --no_outlier_filter
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
    # R[:, :, 0] is first column, R[:, :, 1] is second column
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


def _extract(traj: h5py.Group, trim_len: int) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Returns:
      states  (trim_len, 18):  tcp_9d || block_9d  at step t
      actions (trim_len,  9):  tcp_9d              at step t+1
    or None if the trajectory is shorter than trim_len.
    """
    T = traj["actions"].shape[0]
    if T < trim_len:
        return None

    # obs arrays have T+1 entries (one per state including the final one)
    tcp_pose_raw   = np.array(traj["obs"]["extra"]["tcp_pose"],   dtype=np.float32)   # (T+1, 7)
    block_pose_raw = np.array(traj["obs"]["extra"]["block_pose"], dtype=np.float32)   # (T+1, 7)

    # Convert every pose to 9D
    tcp_9d   = pose7_to_9d(tcp_pose_raw)    # (T+1, 9)
    block_9d = pose7_to_9d(block_pose_raw)  # (T+1, 9)

    # States: use steps 0..trim_len-1
    states = np.concatenate([tcp_9d[:trim_len], block_9d[:trim_len]], axis=-1)   # (trim_len, 18)

    # Actions: absolute TCP pose at the NEXT step (t+1)
    # For the last step (t = trim_len-1) we repeat the final pose ("stay put")
    next_tcp = np.concatenate([
        tcp_9d[1:trim_len],        # steps 1..trim_len-1  → (trim_len-1, 9)
        tcp_9d[trim_len:trim_len+1],  # step trim_len (or repeat last if at end)
    ], axis=0)                     # (trim_len, 9)

    # Guard: if trim_len == T+1 the above slice could be empty; fall back to repeat
    if next_tcp.shape[0] < trim_len:
        repeat = np.tile(tcp_9d[-1:], (trim_len - next_tcp.shape[0], 1))
        next_tcp = np.concatenate([next_tcp, repeat], axis=0)

    actions = next_tcp.astype(np.float32)   # (trim_len, 9)

    return states.astype(np.float32), actions


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    h5_path: str,
    out_path: str,
    trim_len: int | None,
    no_outlier_filter: bool,
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

        if trim_len is not None:
            T = trim_len
            print(f"\nUsing user-specified trim_len={T}")
        elif no_outlier_filter:
            T = int(arr_len.min())
            print(f"\nNo outlier filter → trimming to global minimum: T={T}")
        else:
            T = _iqr_min_len(lengths)
            q1, q3 = np.percentile(arr_len, [25, 75])
            lower   = q1 - 1.5 * (q3 - q1)
            n_out   = int((arr_len < lower).sum())
            print(f"\nIQR outlier filter:  Q1={q1:.0f}  Q3={q3:.0f}  lower_fence={lower:.0f}")
            if n_out:
                print(f"  → {n_out} short outlier(s) excluded from trim target")
            print(f"  → trimming all trajectories to T={T}")

        all_states:  list[np.ndarray] = []
        all_actions: list[np.ndarray] = []
        skipped = 0

        for k in keys:
            result = _extract(f[k], T)
            if result is None:
                skipped += 1
                if verbose:
                    print(f"  [skip] {k}  len={_traj_len(f[k])} < {T}")
                continue
            all_states.append(result[0])
            all_actions.append(result[1])

        if not all_states:
            raise RuntimeError(f"All trajectories shorter than trim_len={T}.")

        print(f"\nKept {len(all_states)} / {len(keys)} trajectories  "
              f"(skipped {skipped})")

    states_np  = np.stack(all_states,  axis=0)   # (N, T, 18)
    actions_np = np.stack(all_actions, axis=0)   # (N, T,  9)

    print(f"\nDataset shape:")
    print(f"  states:  {states_np.shape}  dtype={states_np.dtype}")
    print(f"  actions: {actions_np.shape} dtype={actions_np.dtype}")
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

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, states=states_np, actions=actions_np)
    print(f"\nSaved → {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5",               required=True)
    p.add_argument("--out",              required=True)
    p.add_argument("--trim_len",         type=int, default=None)
    p.add_argument("--no_outlier_filter",action="store_true")
    p.add_argument("--verbose",          action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(args.h5, args.out, args.trim_len, args.no_outlier_filter, args.verbose)