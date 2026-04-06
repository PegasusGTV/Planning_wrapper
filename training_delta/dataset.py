"""
Trajectory dataset — per-frame delta + relative-position representation.

Each sample returns a [seq_len, 6] tensor:
  [0:2]  agent_delta  — tcp  XY displacement from previous frame (0 at t=0)
  [2:4]  block_delta  — block XY displacement from previous frame (0 at t=0)
  [4:6]  relative     — block_xy - tcp_xy  (absolute separation at each frame)

The relative channel is an input feature for both tokens; it is also predicted
by the model so that the two token streams stay geometrically consistent.

Expected NPZ format
-------------------
  states:        float32  [N_traj, max_len, state_dim]
  valid_lengths: int32    [N_traj]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Normalisation stats  (6 independent channels)
# ---------------------------------------------------------------------------

@dataclass
class DeltaStats:
    means: torch.Tensor   # [6]
    stds:  torch.Tensor   # [6]

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        means = self.means.to(x.device)
        stds  = self.stds.to(x.device)
        return (x - means) / (stds + 1e-8)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        means = self.means.to(x.device)
        stds  = self.stds.to(x.device)
        return x * (stds + 1e-8) + means


def compute_norm_stats(dataset: NpzDataset) -> DeltaStats:
    all_agent, all_block, all_relative = [], [], []

    for n in range(len(dataset.valid_lengths_strided)):
        T_v = int(dataset.valid_lengths_strided[n])
        if T_v < 2:
            continue
        xy       = dataset.xy[n, :T_v]
        tcp_xy   = xy[:, :2]
        block_xy = xy[:, 2:]
        all_agent.append(tcp_xy[1:]   - tcp_xy[:-1])
        all_block.append(block_xy[1:] - block_xy[:-1])
        all_relative.append(block_xy  - tcp_xy)

    ad = np.concatenate(all_agent,    axis=0)
    bd = np.concatenate(all_block,    axis=0)
    rd = np.concatenate(all_relative, axis=0)

    means = np.concatenate([ad.mean(0), bd.mean(0), rd.mean(0)]).astype(np.float32)
    stds  = np.concatenate([ad.std(0),  bd.std(0),  rd.std(0) ]).astype(np.float32)

    print("Delta normalisation stats (per channel):")
    labels = ["agent_dx", "agent_dy", "block_dx", "block_dy", "rel_x", "rel_y"]
    for lbl, m, s in zip(labels, means, stds):
        print(f"  {lbl}: mean={m:.4f}  std={s:.4f}")

    return DeltaStats(means=torch.from_numpy(means), stds=torch.from_numpy(stds))

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NpzDataset(Dataset):
    """
    Extracts absolute XY from the 18D state vector:
      tcp_xy   <- states[..., 0:2]
      block_xy <- states[..., 9:11]

    Each window produces a [seq_len, 6] tensor:
      agent_delta[0]  = (0, 0)
      agent_delta[t]  = tcp_xy[t]   - tcp_xy[t-1]
      block_delta[0]  = (0, 0)
      block_delta[t]  = block_xy[t] - block_xy[t-1]
      relative[t]     = block_xy[t] - tcp_xy[t]
    """

    def __init__(self, path: str, seq_len: int, stride: int):
        data          = np.load(path)
        states        = data["states"].astype(np.float32)
        valid_lengths = data["valid_lengths"].astype(np.int32)

        states = states[:, ::stride, :]
        self.valid_lengths_strided = valid_lengths // stride

        N, max_T, _ = states.shape

        tcp_xy   = states[..., 0:2]
        block_xy = states[..., 9:11]

        self.xy = np.concatenate([tcp_xy, block_xy], axis=-1)   # [N, max_T, 4]

        valid_tcp_z, valid_block_z = [], []
        for n in range(N):
            T_v = self.valid_lengths_strided[n]
            valid_tcp_z.append(states[n, :T_v, 2])
            valid_block_z.append(states[n, :T_v, 11])
        self.mean_tcp_z   = float(np.concatenate(valid_tcp_z).mean())
        self.mean_block_z = float(np.concatenate(valid_block_z).mean())

        self.seq_len   = seq_len
        self.state_dim = 6   # agent_delta(2) + block_delta(2) + relative(2)

        self.indices: List[Tuple[int, int]] = []
        for n in range(N):
            T_v = self.valid_lengths_strided[n]
            for t in range(T_v - seq_len + 1):
                self.indices.append((n, t))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Returns [seq_len, 6]: agent_delta | block_delta | relative."""
        n, t = self.indices[idx]
        window = self.xy[n, t : t + self.seq_len]   # [seq_len, 4]

        tcp_xy   = window[:, :2]
        block_xy = window[:, 2:]

        agent_delta     = np.zeros_like(tcp_xy)
        agent_delta[1:] = tcp_xy[1:]   - tcp_xy[:-1]

        block_delta     = np.zeros_like(block_xy)
        block_delta[1:] = block_xy[1:] - block_xy[:-1]

        relative = block_xy - tcp_xy   # [seq_len, 2]

        return torch.from_numpy(
            np.concatenate([agent_delta, block_delta, relative], axis=-1)  # [seq_len, 6]
        )

    def get_abs_initial(self, idx: int) -> np.ndarray:
        """[tcp_x, tcp_y, block_x, block_y] at the first frame of window `idx`."""
        n, t = self.indices[idx]
        return self.xy[n, t].copy()   # [4]
