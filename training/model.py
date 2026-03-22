import math
from dataclasses import dataclass
from typing import Optional, Tuple

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
    causal:    bool  = False  # ← new

# class SinusoidalEmbed(nn.Module):
#     def __init__(self, dim: int):
#         super().__init__()
#         self.dim = dim
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
#         )

#     def forward(self, t: torch.Tensor) -> torch.Tensor:
#         # t: [B]
#         half  = self.dim // 2
#         freqs = torch.exp(
#             -math.log(10000) * torch.arange(half, device=t.device).float() / half
#         )
#         args = t.float()[:, None] * freqs[None]
#         emb  = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
#         return self.mlp(emb)  # [B, dim]


# class Block(nn.Module):
#     def __init__(self, dim: int, heads: int, dropout: float, mlp_ratio: float):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
#         self.attn  = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
#         hidden     = int(dim * mlp_ratio)
#         self.mlp   = nn.Sequential(
#             nn.Linear(dim, hidden), nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden, dim),
#             nn.Dropout(dropout),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
#         x    = x + h
#         x    = x + self.mlp(self.norm2(x))
#         return x


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding  (unchanged)
# ---------------------------------------------------------------------------

class SinusoidalEmbed(nn.Module):
    def __init__(self, dim: int, max_period: int = 10_000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        shape  = t.shape
        t_flat = t.reshape(-1).float()
        half   = self.dim // 2
        freqs  = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device).float() / half
        )
        args = t_flat[:, None] * freqs[None]
        emb  = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb).reshape(*shape, self.dim)


# ---------------------------------------------------------------------------
# AdaLN-Zero  (unchanged)
# ---------------------------------------------------------------------------

class AdaLNZero(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm       = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 3 * dim, bias=True),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = self.modulation(c).chunk(3, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x, gate


# ---------------------------------------------------------------------------
# Transformer block  (unchanged from previous rewrite)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0, mlp_ratio: float = 4.0):
        super().__init__()
        assert dim % heads == 0
        self.heads    = heads
        self.head_dim = dim // heads
        self.drop     = dropout

        self.adaLN1 = AdaLNZero(dim)
        self.qkv    = nn.Linear(dim, dim * 3, bias=True)
        self.proj   = nn.Linear(dim, dim, bias=True)

        self.adaLN2 = AdaLNZero(dim)
        hidden      = int(dim * mlp_ratio)
        self.mlp    = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(
        self,
        x:         torch.Tensor,                   # [B, 2T, dim]
        c:         torch.Tensor,                   # [B, 2T, dim]
        attn_mask: Optional[torch.Tensor] = None,  # [2T, 2T] additive float
    ) -> torch.Tensor:
        B, N, d = x.shape
        H, Dh   = self.heads, self.head_dim

        h, gate_attn = self.adaLN1(x, c)
        qkv = self.qkv(h).reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask = attn_mask,
            dropout_p = self.drop if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, N, d)
        x   = x + gate_attn * self.proj(out)

        h, gate_mlp = self.adaLN2(x, c)
        x = x + gate_mlp * self.mlp(h)
        return x
    
    
# class DiT(nn.Module):
#     def __init__(self, cfg: ModelConfig):
#         super().__init__()
#         self.cfg       = cfg
#         self.state_dim = cfg.state_dim
#         d              = cfg.dim

#         self.input_proj = nn.Linear(cfg.state_dim, d)
#         self.pos_embed  = nn.Parameter(torch.zeros(cfg.seq_len, d))
#         nn.init.trunc_normal_(self.pos_embed, std=0.02)

#         self.time_embed = SinusoidalEmbed(d)  # no time_proj needed; Block handles conditioning

#         self.blocks = nn.ModuleList([
#             Block(d, cfg.heads, cfg.dropout, cfg.mlp_ratio)
#             for _ in range(cfg.depth)
#         ])

#         self.norm = nn.LayerNorm(d)
#         self.head = nn.Linear(d, cfg.state_dim)
#         nn.init.zeros_(self.head.weight)
#         nn.init.zeros_(self.head.bias)

#     def forward(
#         self,
#         x:         torch.Tensor,  # [B, T, state_dim]
#         timesteps: torch.Tensor,  # [B, T]  ← per-frame noise level
#     ) -> torch.Tensor:            # [B, T, state_dim]
#         B, T, _ = x.shape

#         tokens = self.input_proj(x) + self.pos_embed[:T]  # [B, T, d]
#         c      = self.time_embed(timesteps)                # [B, T, d]

#         for block in self.blocks:
#             tokens = block(tokens, c)                      # c passed as conditioning

#         return self.head(self.norm(tokens))                # [B, T, state_dim]


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

        self.blocks = nn.ModuleList([
            Block(d, cfg.heads, cfg.dropout, cfg.mlp_ratio)
            for _ in range(cfg.depth)
        ])

        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.state_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # Causal mask cache
        self._mask:     Optional[torch.Tensor] = None
        self._mask_key: Optional[Tuple]        = None

    def _get_causal_mask(
        self, T: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        key = (T, device, dtype)
        if self._mask_key != key:
            i = torch.arange(T, device=device)
            allowed = i[:, None] >= i[None, :]          # [T, T] lower-triangular
            m = torch.zeros(T, T, device=device, dtype=dtype)
            m.masked_fill_(~allowed, float("-inf"))
            self._mask, self._mask_key = m, key
        return self._mask

    def forward(
        self,
        x:         torch.Tensor,  # [B, T, state_dim]
        timesteps: torch.Tensor,  # [B, T]
    ) -> torch.Tensor:            # [B, T, state_dim]
        B, T, _ = x.shape

        tokens = self.input_proj(x) + self.pos_embed[:T]  # [B, T, d]
        c      = self.time_embed(timesteps)                # [B, T, d]

        attn_mask = None
        if self.cfg.causal:
            attn_mask = self._get_causal_mask(T, tokens.device, tokens.dtype)

        for block in self.blocks:
            tokens = block(tokens, c, attn_mask=attn_mask)

        return self.head(self.norm(tokens))