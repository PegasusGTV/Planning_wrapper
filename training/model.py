"""
2D Diffusion Transformer (DiT) — separate state and action tokens.

Token layout: [s_0, a_0, s_1, a_1, ..., s_{T-1}, a_{T-1}]  →  length 2T

Attention rules (causal=True):
  - Any token at timestep t can attend to all tokens at timesteps 0..t,
    including both the state and action at the same timestep t.
  - This is a block-lower-triangular mask with 2×2 blocks.

Timestep conditioning uses AdaLN-Zero per block, as before.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    state_dim:  int   = 18
    action_dim: int   = 9
    seq_len:    int   = 100      # T  (number of timesteps, not tokens)
    dim:        int   = 256
    depth:      int   = 6
    heads:      int   = 8
    dropout:    float = 0.1
    mlp_ratio:  float = 4.0
    causal:     bool  = False


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


# ---------------------------------------------------------------------------
# DiT  — separate state / action tokens
# ---------------------------------------------------------------------------

class DiT(nn.Module):
    """
    Separate state and action tokens, interleaved as [s_0, a_0, s_1, a_1, ...].

    Input
    -----
      states     : [B, T, state_dim]
      actions    : [B, T, action_dim]
      timesteps  : [B, T]  — per-frame noise level

    Output
    ------
      state_pred  : [B, T, state_dim]
      action_pred : [B, T, action_dim]
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d        = cfg.dim
        self.state_dim = cfg.state_dim

        # Separate input projections
        self.state_proj  = nn.Sequential(
            nn.Linear(cfg.state_dim, d), nn.SiLU(), nn.Linear(d, d)
        )
        self.action_proj = nn.Linear(cfg.action_dim, d)

        # Positional embedding indexed by timestep (shared across state/action at same t)
        self.pos_embed  = nn.Parameter(torch.zeros(cfg.seq_len, d))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Type embedding: index 0 = state, index 1 = action
        self.type_embed = nn.Parameter(torch.zeros(2, d))
        nn.init.trunc_normal_(self.type_embed, std=0.02)

        # Per-frame timestep embedding — used as conditioning c, not added to tokens
        self.time_embed = SinusoidalEmbed(d)

        # Transformer blocks over the 2T token sequence
        self.blocks = nn.ModuleList([
            Block(d, cfg.heads, cfg.dropout, cfg.mlp_ratio)
            for _ in range(cfg.depth)
        ])

        # Separate output heads for states and actions
        self.state_norm = nn.LayerNorm(d)
        self.state_head = nn.Linear(d, cfg.state_dim)
        nn.init.zeros_(self.state_head.weight)
        nn.init.zeros_(self.state_head.bias)

        self.action_norm = nn.LayerNorm(d)
        self.action_head = nn.Linear(d, cfg.action_dim)
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

        # Causal mask cache
        self._mask:     Optional[torch.Tensor] = None
        self._mask_key: Optional[Tuple]        = None

    def _get_causal_mask(
        self, T: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Block-lower-triangular mask over 2T tokens.

        Token layout: [s_0, a_0, s_1, a_1, ...]
        Token i is at timestep i // 2.
        Token i may attend to token j iff timestep(i) >= timestep(j),
        i.e. i // 2 >= j // 2  — so same-timestep pairs fully see each other.

        Returns additive float mask [2T, 2T]: 0.0 = attend, -inf = block.
        """
        N = 2 * T
        key = (T, device, dtype)
        if self._mask_key != key:
            # timestep index for each of the 2T positions
            ts = torch.arange(N, device=device) // 2          # [2T]
            allowed = ts[:, None] >= ts[None, :]              # [2T, 2T] bool
            m = torch.zeros(N, N, device=device, dtype=dtype)
            m.masked_fill_(~allowed, float("-inf"))
            self._mask, self._mask_key = m, key
        return self._mask

    def forward(
        self,
        states:    torch.Tensor,   # [B, T, state_dim]
        actions:   torch.Tensor,   # [B, T, action_dim]
        timesteps: torch.Tensor,   # [B, T]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = states.shape

        # --- Build token sequence [s_0, a_0, s_1, a_1, ...] ---
        # Each of shape [B, T, d]
        s_tokens = (
            self.state_proj(states)
            + self.pos_embed[:T]          # [T, d] broadcasts over B
            + self.type_embed[0]          # scalar [d] broadcast
        )
        a_tokens = (
            self.action_proj(actions)
            + self.pos_embed[:T]
            + self.type_embed[1]
        )

        # Interleave into [B, 2T, d]:  dim-1 becomes [s_0, a_0, s_1, a_1, ...]
        tokens = torch.stack([s_tokens, a_tokens], dim=2)  # [B, T, 2, d]
        tokens = tokens.reshape(B, 2 * T, -1)              # [B, 2T, d]

        # --- Conditioning: repeat each timestep embedding for its two tokens ---
        c = self.time_embed(timesteps)                     # [B, T, d]
        c = c.unsqueeze(2).expand(-1, -1, 2, -1)          # [B, T, 2, d]
        c = c.reshape(B, 2 * T, -1)                       # [B, 2T, d]

        # --- Attention mask ---
        attn_mask = None
        if self.cfg.causal:
            attn_mask = self._get_causal_mask(T, tokens.device, tokens.dtype)

        # --- Transformer ---
        for block in self.blocks:
            tokens = block(tokens, c, attn_mask=attn_mask)

        # --- Uninterleave and project to outputs ---
        tokens = tokens.reshape(B, T, 2, -1)               # [B, T, 2, d]
        s_out  = tokens[:, :, 0, :]                         # [B, T, d]
        a_out  = tokens[:, :, 1, :]                         # [B, T, d]

        state_pred  = self.state_head(self.state_norm(s_out))    # [B, T, state_dim]
        action_pred = self.action_head(self.action_norm(a_out))  # [B, T, action_dim]

        return state_pred, action_pred