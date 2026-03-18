"""
2D Diffusion Transformer (DiT) — unified sequence model.

The model sees a single flat sequence of T tokens.
Every token = state_proj(x_state) + action_proj(x_action) + pos_embed + t_embed(t_i)

There is no hard context/future distinction in the architecture.
The caller controls which frames are conditioned vs generated solely through
the timestep values it passes in ([B, T] — one per frame).

Causal mask (cfg.causal=True)
  Frame j attends only to frames 0..j.
  Useful for autoregressive-style generation.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    state_dim:  int   = 32
    action_dim: int   = 7
    seq_len:    int   = 18      # T
    dim:        int   = 256
    depth:      int   = 6
    heads:      int   = 8
    dropout:    float = 0.1
    mlp_ratio:  float = 4.0
    causal:     bool  = False


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalEmbed(nn.Module):
    """
    Sinusoidal embed + MLP projection.
    Accepts any integer tensor [...]; returns [..., dim].
    """
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
# Transformer block  (pre-norm, full self-attention)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """
    Pre-norm block: LN → QKV self-attention → residual → LN → MLP → residual.

    attn_mask: additive float [T, T] where 0.0 = attend, −∞ = block.
    SDPA broadcasts over batch and head dims automatically.
    """
    def __init__(self, dim: int, heads: int, dropout: float = 0.0, mlp_ratio: float = 4.0):
        super().__init__()
        assert dim % heads == 0
        self.heads    = heads
        self.head_dim = dim // heads

        self.norm1 = nn.LayerNorm(dim)
        self.qkv   = nn.Linear(dim, dim * 3, bias=True)
        self.proj  = nn.Linear(dim, dim, bias=True)
        self.drop  = dropout

        self.norm2 = nn.LayerNorm(dim)
        hidden     = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(
        self,
        x:         torch.Tensor,                   # [B, T, dim]
        attn_mask: Optional[torch.Tensor] = None,  # [T, T] additive float
    ) -> torch.Tensor:
        B, T, d = x.shape
        H, Dh   = self.heads, self.head_dim

        h   = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # each [B, H, T, Dh]

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask = attn_mask,
            dropout_p = self.drop if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, T, d)
        x   = x + self.proj(out)
        x   = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# DiT
# ---------------------------------------------------------------------------

class DiT(nn.Module):
    """
    Unified sequence DiT.

    Input
    -----
      noisy_seq  : [B, T, state_dim + action_dim]   — all frames, already noised by caller
      timesteps  : [B, T]                            — per-frame noise level (int, 0..N)

    Output
    ------
      [B, T, state_dim + action_dim]  — predicted velocity / epsilon / x0

    Notes
    -----
    • state_proj and action_proj take the per-frame state and action slices.
    • Each token's embedding includes its individual timestep so the model
      knows exactly how noisy each frame is.
    • Positional embedding is learned and frame-index based.
    • The output head is zero-initialised for stable training start.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d        = cfg.dim
        SD, AD   = cfg.state_dim, cfg.action_dim

        # Input projections — applied per-frame
        self.state_proj  = nn.Sequential(
            nn.Linear(SD, d), nn.SiLU(), nn.Linear(d, d)
        )
        self.action_proj = nn.Linear(AD, d)

        # Per-frame timestep embedding
        self.time_embed = SinusoidalEmbed(d)

        # Learned positional embedding over sequence positions
        self.pos_embed = nn.Parameter(torch.zeros(cfg.seq_len, d))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer
        self.blocks = nn.ModuleList([
            Block(d, cfg.heads, cfg.dropout, cfg.mlp_ratio)
            for _ in range(cfg.depth)
        ])

        # Output head — zero-init for stable early training
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, SD + AD)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # Causal mask cache
        self._mask:     Optional[torch.Tensor] = None
        self._mask_key: Optional[Tuple]        = None

    def _get_causal_mask(
        self, T: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """[T, T] lower-triangular additive float mask. Cached by (T, device, dtype)."""
        key = (T, device, dtype)
        if self._mask_key != key:
            m = torch.full((T, T), float("-inf"), device=device, dtype=dtype)
            m.masked_fill_(torch.tril(torch.ones(T, T, device=device, dtype=torch.bool)), 0.0)
            self._mask, self._mask_key = m, key
        return self._mask

    def forward(
        self,
        noisy_seq:  torch.Tensor,   # [B, T, state_dim + action_dim]
        timesteps:  torch.Tensor,   # [B, T]   integer noise levels
    ) -> torch.Tensor:              # [B, T, state_dim + action_dim]
        B, T, _ = noisy_seq.shape
        SD       = self.cfg.state_dim

        # Split packed input back into state / action
        states  = noisy_seq[..., :SD]   # [B, T, SD]
        actions = noisy_seq[..., SD:]   # [B, T, AD]

        # Build per-frame tokens
        # [B, T, d] = state_embed + action_embed + pos_embed + t_embed
        t_emb  = self.time_embed(timesteps)            # [B, T, d]  — per-frame
        tokens = (
            self.state_proj(states)                    # [B, T, d]
            + self.action_proj(actions)                # [B, T, d]
            + self.pos_embed[:T]                       # [T, d]   broadcasts over B
            + t_emb                                    # [B, T, d]
        )

        attn_mask = None
        if self.cfg.causal:
            attn_mask = self._get_causal_mask(T, tokens.device, tokens.dtype)

        for block in self.blocks:
            tokens = block(tokens, attn_mask=attn_mask)

        return self.head(self.norm(tokens))            # [B, T, SD + AD]