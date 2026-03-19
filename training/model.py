import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    state_dim: int   = 4    # tcp_x, tcp_y, block_x, block_y
    seq_len:   int   = 10
    dim:       int   = 128
    depth:     int   = 3
    heads:     int   = 4
    dropout:   float = 0.1
    mlp_ratio: float = 4.0

class SinusoidalEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B]
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float()[:, None] * freqs[None]
        emb  = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)  # [B, dim]


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        hidden     = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x    = x + h
        x    = x + self.mlp(self.norm2(x))
        return x
    
    
class DiT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg       = cfg
        self.state_dim = cfg.state_dim
        d              = cfg.dim

        self.input_proj = nn.Linear(cfg.state_dim, d)
        self.pos_embed  = nn.Parameter(torch.zeros(cfg.seq_len, d))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.time_embed = SinusoidalEmbed(d)
        self.time_proj  = nn.Linear(d, d)

        self.blocks = nn.ModuleList([
            Block(d, cfg.heads, cfg.dropout, cfg.mlp_ratio)
            for _ in range(cfg.depth)
        ])

        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.state_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x:         torch.Tensor,  # [B, T, 4]
        timesteps: torch.Tensor,  # [B]
    ) -> torch.Tensor:            # [B, T, 4]
        B, T, _ = x.shape

        tokens = self.input_proj(x) + self.pos_embed[:T]      # [B, T, d]
        t_emb  = self.time_proj(self.time_embed(timesteps))   # [B, d]
        tokens = tokens + t_emb.unsqueeze(1)

        for block in self.blocks:
            tokens = block(tokens)

        return self.head(self.norm(tokens))  # [B, T, 4]