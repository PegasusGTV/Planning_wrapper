"""
Trajectory dataset.

Expected NPZ format
-------------------
  states:  float32  [N_traj, T_total, state_dim]
  actions: float32  [N_traj, T_total, action_dim]

NpzDataset slices each trajectory into overlapping windows of length seq_len,
returning a single packed tensor  seq  [T, state_dim + action_dim].

There is no context/future split here — that distinction belongs entirely to
the training / inference logic in train.py, not in the data layer.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from diffusion import NormStats, Stats


class TrajectoryDataset(Dataset):
    """
    Abstract base.  __getitem__ must return:
      seq: Tensor [T, state_dim + action_dim]
    """
    pass


class NpzDataset(TrajectoryDataset):
    def __init__(self, path: str, seq_len: int):
        data            = np.load(path)
        self.states     = data["states"].astype(np.float32)    # [N, T_total, SD]
        self.actions    = data["actions"].astype(np.float32)   # [N, T_total, AD]
        self.seq_len    = seq_len
        self.state_dim  = self.states.shape[-1]
        self.action_dim = self.actions.shape[-1]

        N, T_total, _ = self.states.shape
        self.indices = [
            (n, t)
            for n in range(N)
            for t in range(T_total - seq_len + 1)
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        n, t = self.indices[idx]
        s = torch.from_numpy(self.states [n, t : t + self.seq_len])  # [T, SD]
        a = torch.from_numpy(self.actions[n, t : t + self.seq_len])  # [T, AD]
        return torch.cat([s, a], dim=-1)                              # [T, SD+AD]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def compute_norm_stats(dataset: TrajectoryDataset, max_samples: int = 50_000) -> NormStats:
    """
    Compute per-dimension mean/std over up to max_samples dataset items.
    """
    SD = dataset.state_dim
    all_seqs = []
    for i in range(min(len(dataset), max_samples)):
        all_seqs.append(dataset[i])                    # [T, SD+AD]

    data = torch.stack(all_seqs).reshape(-1, all_seqs[0].shape[-1])  # [N*T, SD+AD]
    s    = data[:, :SD]
    a    = data[:, SD:]

    return NormStats(
        state  = Stats(s.mean(0), s.std(0).clamp(min=1e-6)),
        action = Stats(a.mean(0), a.std(0).clamp(min=1e-6)),
    )