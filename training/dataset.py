# """
# Trajectory dataset.

# Expected NPZ format
# -------------------
#   states:  float32  [N_traj, T_total, state_dim]
#   actions: float32  [N_traj, T_total, action_dim]

# NpzDataset slices each trajectory into overlapping windows of length seq_len,
# returning a single packed tensor  seq  [T, state_dim + action_dim].

# There is no context/future split here — that distinction belongs entirely to
# the training / inference logic in train.py, not in the data layer.
# """

# import numpy as np
# import torch
# from torch.utils.data import Dataset

# from diffusion import Stats
    
# class NpzDataset(Dataset):
#     """
#     Extracts 4D state: [tcp_x, tcp_y, block_x, block_y]
#     from the full 18D state (tcp_9d = pos(3)+rot6d(6), block_9d = pos(3)+rot6d(6))
#     """
#     def __init__(self, path: str, seq_len: int, stride: int):
#         data = np.load(path)
#         states  = data["states"].astype(np.float32)   # [N, T, 18]
#         states  = states[:, ::stride, :]
#         N, T, _ = states.shape

#         tcp_xy   = states[..., 0:2]   # [N, T, 2]
#         block_xy = states[..., 9:11]  # [N, T, 2]
#         self.xy = np.concatenate([tcp_xy, block_xy], axis=-1)  # [N, T, 4]

#         # Mean z values for visualization
#         self.mean_tcp_z   = float(states[..., 2].mean())   # tcp_z   is states[..., 2]
#         self.mean_block_z = float(states[..., 11].mean())  # block_z is states[..., 11]

#         self.states_full = states
#         self.seq_len   = seq_len
#         self.state_dim = 4
#         self.indices   = [(n, t) for n in range(N) for t in range(T - seq_len + 1)]

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         n, t = self.indices[idx]
#         return torch.from_numpy(self.xy[n, t:t + self.seq_len])  # [T, 4]

# def compute_norm_stats(dataset: NpzDataset) -> Stats:
#     all_xy = torch.from_numpy(dataset.xy.reshape(-1, 4))  # [N*T, 4]

#     # Pool tcp and block together for shared x and y stats
#     all_x = torch.cat([all_xy[:, 0], all_xy[:, 2]])  # tcp_x + block_x
#     all_y = torch.cat([all_xy[:, 1], all_xy[:, 3]])  # tcp_y + block_y

#     x_min, x_max = all_x.min().item(), all_x.max().item()
#     y_min, y_max = all_y.min().item(), all_y.max().item()

#     print("Shared normalization ranges:")
#     print(f"  x: [{x_min:.3f}, {x_max:.3f}]")
#     print(f"  y: [{y_min:.3f}, {y_max:.3f}]")

#     return Stats(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

"""
Trajectory dataset.

Expected NPZ format
-------------------
  states:        float32  [N_traj, max_len, state_dim]
  actions:       float32  [N_traj, max_len, action_dim]
  valid_lengths: int32    [N_traj]

NpzDataset slices each trajectory into overlapping windows of length seq_len,
returning a single packed tensor seq [seq_len, state_dim]. It safely ignores
any zero-padded steps at the end of trajectories.
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
    def __init__(self, path: str, seq_len: int, stride: int):
        data = np.load(path)
        states = data["states"].astype(np.float32)   # [N, max_len, 18]
        valid_lengths = data["valid_lengths"].astype(np.int32) # [N]
        
        # Apply stride to the arrays and the valid lengths
        states = states[:, ::stride, :]
        self.valid_lengths_strided = valid_lengths // stride
        
        N, max_T, _ = states.shape

        tcp_xy   = states[..., 0:2]   # [N, max_T, 2]
        block_xy = states[..., 9:11]  # [N, max_T, 2]
        self.xy = np.concatenate([tcp_xy, block_xy], axis=-1)  # [N, max_T, 4]

        # Calculate mean z values for visualization, ignoring padded zeros
        valid_tcp_z = []
        valid_block_z = []
        for n in range(N):
            T_valid = self.valid_lengths_strided[n]
            valid_tcp_z.append(states[n, :T_valid, 2])
            valid_block_z.append(states[n, :T_valid, 11])
            
        self.mean_tcp_z   = float(np.concatenate(valid_tcp_z).mean())
        self.mean_block_z = float(np.concatenate(valid_block_z).mean())

        self.states_full = states
        self.seq_len   = seq_len
        self.state_dim = 4
        
        # Build valid indices ensuring windows don't cross into the padded zeros
        self.indices = []
        for n in range(N):
            T_valid = self.valid_lengths_strided[n]
            # Only add windows that fit entirely within the valid length of this specific trajectory
            for t in range(T_valid - self.seq_len + 1):
                self.indices.append((n, t))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        n, t = self.indices[idx]
        # Because of how we built self.indices, this slice is guaranteed to be 100% valid data
        return torch.from_numpy(self.xy[n, t:t + self.seq_len])  # [seq_len, 4]


def compute_norm_stats(dataset: NpzDataset) -> Stats:
    # We must rebuild the flattened xy array using ONLY the valid steps
    valid_xy_list = []
    for n in range(len(dataset.valid_lengths_strided)):
        T_valid = dataset.valid_lengths_strided[n]
        valid_xy_list.append(dataset.xy[n, :T_valid, :])
        
    all_xy_np = np.concatenate(valid_xy_list, axis=0) # [Total_valid_steps, 4]
    all_xy = torch.from_numpy(all_xy_np)

    # Pool tcp and block together for shared x and y stats
    all_x = torch.cat([all_xy[:, 0], all_xy[:, 2]])  # tcp_x + block_x
    all_y = torch.cat([all_xy[:, 1], all_xy[:, 3]])  # tcp_y + block_y

    x_min, x_max = all_x.min().item(), all_x.max().item()
    y_min, y_max = all_y.min().item(), all_y.max().item()

    print("Shared normalization ranges (ignoring padding):")
    print(f"  x: [{x_min:.3f}, {x_max:.3f}]")
    print(f"  y: [{y_min:.3f}, {y_max:.3f}]")

    return Stats(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)