"""
utils/playback_utils.py
=======================
Extract low-dimensional states from predicted particle-cloud trajectories
and save them to a self-contained NPZ file that playback_predictions.py can load.

NPZ layout
----------
  n_hand             : ()      int   – hand-particle count (gripper first in cloud)
  n_block            : ()      int   – block-particle count
  context_len        : ()      int   – number of context frames
  tcp_positions      : (T, 3)        – centroid of hand particles, centered frame
  tcp_quaternions    : (T, 4)        – [w,x,y,z], identity if no template
  block_positions    : (T, 3)        – centroid of block particles, centered frame
  block_quaternions  : (T, 4)        – [w,x,y,z]
  target_block_pos   : (3,)          – last finite target in centered frame
  initial_block_pos  : (3,)          – block centroid at frame 0 (centering anchor)
  hand_template      : (n_hand,3) or empty  – stored if provided
  block_template     : (n_block,3) or empty – stored if provided
  predicted_positions: (T, N, 3)     – full particle cloud (for visualisation / debug)
  gt_positions       : (T, N, 3)     – ground-truth cloud (may be zeros in future frames)
  all_targets        : (T, N, 3)     – target array (NaN where no target)

Coordinate frame note
---------------------
All positions are in the *centered* frame used during data preprocessing
(h5_to_wds.py subtracts the bounding-box centre of the block trajectory).
`playback_predictions.py` re-anchors the initial block centroid to
BOUNDARY_CENTER in the simulator world frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch

    def _to_np(x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32)
except ImportError:
    def _to_np(x) -> np.ndarray:
        return np.asarray(x, dtype=np.float32)


from utils.rigid_fit import estimate_tcp_pose, estimate_block_pose


# ─────────────────────────────────────────────────────────────────────────────
# Core extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_trajectories(
    predicted_positions: np.ndarray,   # (T, N_pts, 3)  centered frame
    n_hand: int,
    n_block: int,
    hand_template: Optional[np.ndarray] = None,   # (n_hand,  3) local frame
    block_template: Optional[np.ndarray] = None,  # (n_block, 3) local frame
) -> dict[str, np.ndarray]:
    """
    Decompose a predicted particle trajectory into TCP and block pose trajectories.

    Parameters
    ----------
    predicted_positions : (T, N_pts, 3)
        Predicted particle cloud in the centered coordinate frame.
        Particle ordering: gripper first [0:n_hand], block after [n_hand:n_hand+n_block].
    n_hand, n_block : int
        Particle counts as used during data creation.
    hand_template, block_template : optional (N, 3) arrays
        Template point clouds in the object's local frame.  When provided,
        Kabsch fitting is used for accurate rotation recovery; otherwise
        centroid + PCA / identity is used.

    Returns
    -------
    dict with keys:
        tcp_positions    (T, 3)
        tcp_quaternions  (T, 4)  [w,x,y,z]
        block_positions  (T, 3)
        block_quaternions(T, 4)
    """
    hand_pts  = predicted_positions[:, :n_hand,  :]   # (T, n_hand,  3)
    block_pts = predicted_positions[:, n_hand:n_hand+n_block, :]  # (T, n_block, 3)

    tcp_pos, tcp_quat     = estimate_tcp_pose(hand_pts,   hand_template)
    block_pos, block_quat = estimate_block_pose(block_pts, block_template)

    return dict(
        tcp_positions=tcp_pos,
        tcp_quaternions=tcp_quat,
        block_positions=block_pos,
        block_quaternions=block_quat,
    )


def _find_target_block_pos(
    all_targets: np.ndarray,    # (T, N_pts, 3)  NaN where no target
    n_hand: int,
    n_block: int,
) -> np.ndarray:                # (3,)
    """
    Extract the target block centroid as the last finite target in the cloud.
    Falls back to the last block particle centroid if no explicit target exists.
    """
    T = all_targets.shape[0]
    block_tgt = all_targets[:, n_hand:n_hand+n_block, :]  # (T, n_block, 3)

    for t in range(T - 1, -1, -1):
        frame = block_tgt[t]
        if np.isfinite(frame).all():
            return frame.mean(0).astype(np.float32)

    # Fallback: last finite centroid
    for t in range(T - 1, -1, -1):
        frame = block_tgt[t]
        mask  = np.isfinite(frame).all(axis=-1)
        if mask.any():
            return frame[mask].mean(0).astype(np.float32)

    # Nothing finite — return zeros
    return np.zeros(3, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Save / load
# ─────────────────────────────────────────────────────────────────────────────

def save_playback_npz(
    results: dict,
    n_hand: int,
    n_block: int,
    context_len: int,
    output_path: str | Path,
    sample_idx: int = 0,
    hand_template: Optional[np.ndarray] = None,
    block_template: Optional[np.ndarray] = None,
) -> Path:
    """
    Extract low-dim states from inference results and save to *output_path*.npz.

    Parameters
    ----------
    results : dict
        Output of ``autoregressive_rollout`` — must contain
        ``predicted_positions``, ``gt_positions``, ``all_targets``
        as tensors of shape (B, T, N, 3).
    n_hand, n_block : int
        Particle counts (gripper first, block second).
    context_len : int
        Number of context (history) frames.
    output_path : str or Path
        Destination path (will have .npz appended if missing).
    sample_idx : int
        Which batch element to save.
    hand_template, block_template : optional (N,3) numpy arrays
        If provided, Kabsch rigid-body fitting is used for orientation.
    """
    output_path = Path(output_path)
    if output_path.suffix != ".npz":
        output_path = output_path.with_suffix(".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pred_all = _to_np(results["predicted_positions"])   # (B, T, N, 3)
    gt_all   = _to_np(results["gt_positions"])
    tgt_all  = _to_np(results["all_targets"])

    pred = pred_all[sample_idx]   # (T, N, 3)
    gt   = gt_all[sample_idx]
    tgt  = tgt_all[sample_idx]

    # Pose trajectories
    traj = extract_trajectories(pred, n_hand, n_block, hand_template, block_template)

    # Target block position
    target_block_pos = _find_target_block_pos(tgt, n_hand, n_block)

    # Centering anchor = block centroid at the first context frame
    initial_block_pos = pred[0, n_hand:n_hand+n_block, :].mean(0).astype(np.float32)

    np.savez(
        output_path,
        # Metadata
        n_hand=n_hand,
        n_block=n_block,
        context_len=context_len,
        # TCP trajectory (centered frame)
        tcp_positions=traj["tcp_positions"],
        tcp_quaternions=traj["tcp_quaternions"],
        # Block trajectory (centered frame)
        block_positions=traj["block_positions"],
        block_quaternions=traj["block_quaternions"],
        # Target and anchor
        target_block_pos=target_block_pos,
        initial_block_pos=initial_block_pos,
        # Templates (empty arrays if None, so npz is self-contained)
        hand_template=hand_template if hand_template is not None
                      else np.empty((0, 3), dtype=np.float32),
        block_template=block_template if block_template is not None
                       else np.empty((0, 3), dtype=np.float32),
        # Full clouds for debugging / visualisation
        predicted_positions=pred.astype(np.float32),
        gt_positions=gt.astype(np.float32),
        all_targets=tgt.astype(np.float32),
    )

    print(f"  Saved playback data → {output_path}")
    return output_path


def save_playback_batch(
    results: dict,
    n_hand: int,
    n_block: int,
    context_len: int,
    output_dir: str | Path,
    file_stem: str = "playback",
    hand_template: Optional[np.ndarray] = None,
    block_template: Optional[np.ndarray] = None,
) -> list[Path]:
    """
    Save a playback NPZ for every sample in the batch.

    Returns a list of saved file paths.
    """
    pred_all = _to_np(results["predicted_positions"])
    B = pred_all.shape[0]
    paths = []
    for b in range(B):
        p = save_playback_npz(
            results=results,
            n_hand=n_hand,
            n_block=n_block,
            context_len=context_len,
            output_path=Path(output_dir) / f"{file_stem}_{b:04d}",
            sample_idx=b,
            hand_template=hand_template,
            block_template=block_template,
        )
        paths.append(p)
    return paths


def load_playback_npz(path: str | Path) -> dict:
    """
    Load a playback NPZ and return a plain dict of numpy arrays.

    The dict has the same keys as written by ``save_playback_npz``, plus
    convenience scalars:
        n_hand_int, n_block_int, context_len_int
    """
    data = np.load(path, allow_pickle=False)
    out  = {k: data[k] for k in data.files}
    out["n_hand_int"]      = int(data["n_hand"])
    out["n_block_int"]     = int(data["n_block"])
    out["context_len_int"] = int(data["context_len"])
    return out