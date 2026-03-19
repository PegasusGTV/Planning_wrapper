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

from diffusion import Stats
    
class NpzDataset(Dataset):
    """
    Extracts 4D state: [tcp_x, tcp_y, block_x, block_y]
    from the full 18D state (tcp_9d = pos(3)+rot6d(6), block_9d = pos(3)+rot6d(6))
    """
    def __init__(self, path: str, seq_len: int):
        data = np.load(path)
        states  = data["states"].astype(np.float32)   # [N, T, 18]
        N, T, _ = states.shape

        # tcp pos is states[..., 0:3], block pos is states[..., 9:12]
        tcp_xy   = states[..., 0:2]   # [N, T, 2]
        block_xy = states[..., 9:11]  # [N, T, 2]
        self.xy = np.concatenate([tcp_xy, block_xy], axis=-1)  # [N, T, 4]

        # Keep full states for visualization (z + rotations)
        self.states_full = states   # [N, T, 18]

        self.seq_len   = seq_len
        self.state_dim = 4
        self.indices   = [(n, t) for n in range(N) for t in range(T - seq_len + 1)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        n, t = self.indices[idx]
        return torch.from_numpy(self.xy[n, t:t + self.seq_len])  # [T, 4]

def compute_norm_stats(dataset: NpzDataset) -> Stats:
    all_xy = torch.from_numpy(dataset.xy.reshape(-1, 4))  # [N*T, 4]

    # Pool tcp and block together for shared x and y stats
    all_x = torch.cat([all_xy[:, 0], all_xy[:, 2]])  # tcp_x + block_x
    all_y = torch.cat([all_xy[:, 1], all_xy[:, 3]])  # tcp_y + block_y

    x_min, x_max = all_x.min().item(), all_x.max().item()
    y_min, y_max = all_y.min().item(), all_y.max().item()

    print("Shared normalization ranges:")
    print(f"  x: [{x_min:.3f}, {x_max:.3f}]")
    print(f"  y: [{y_min:.3f}, {y_max:.3f}]")

    return Stats(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)