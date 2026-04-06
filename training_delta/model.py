import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    state_dim: int   = 6      # agent_delta(2) + block_delta(2) + relative(2)
    seq_len:   int   = 10
    dim:       int   = 128
    depth:     int   = 3
    heads:     int   = 4
    dropout:   float = 0.1
    mlp_ratio: float = 4.0
    causal:    bool  = False


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

class SinusoidalEmbed(nn.Module):
    def __init__(self, dim: int, max_period: int = 10_000):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        shape  = t.shape
        t_flat = t.reshape(-1).float()
        half   = self.dim // 2
        freqs  = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=t.device).float() / half
        )
        args = t_flat[:, None] * freqs[None]
        emb  = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb).reshape(*shape, self.dim)


# ---------------------------------------------------------------------------
# AdaLN-Zero
# ---------------------------------------------------------------------------

class AdaLNZero(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm       = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim, bias=True))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = self.modulation(c).chunk(3, dim=-1)
        return self.norm(x) * (1 + scale) + shift, gate


# ---------------------------------------------------------------------------
# Transformer block
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
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(
        self,
        x:         torch.Tensor,
        c:         torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, d = x.shape
        H, Dh   = self.heads, self.head_dim

        h, gate_attn = self.adaLN1(x, c)
        qkv = self.qkv(h).reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.drop if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, N, d)
        x   = x + gate_attn * self.proj(out)

        h, gate_mlp = self.adaLN2(x, c)
        x = x + gate_mlp * self.mlp(h)
        return x


# ---------------------------------------------------------------------------
# DiT — two tokens per timestep, each conditioned on relative position
# ---------------------------------------------------------------------------

class DiT(nn.Module):
    """
    Two tokens per timestep, interleaved as:
      [agent_0, block_0, agent_1, block_1, ...]  →  sequence length 2T

    Input  [B, T, 6]:
      x[..., 0:2]  agent_delta
      x[..., 2:4]  block_delta
      x[..., 4:6]  relative  (block_xy - tcp_xy)

    Each token receives its own delta concatenated with the shared relative vector:
      agent token input: concat(agent_delta, relative)  →  Linear(4, d)
      block token input: concat(block_delta, relative)  →  Linear(4, d)

    Output [B, T, 6]: predicted noise for all 6 channels.
      out[..., 0:2]  agent_delta noise   (from agent head)
      out[..., 2:4]  block_delta noise   (from block head)
      out[..., 4:6]  relative noise      (averaged from both heads)
    """

    TOKENS_PER_STEP = 2

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg       = cfg
        self.state_dim = cfg.state_dim   # 6
        d              = cfg.dim

        # Each token: own_delta(2) + relative(2) = 4 input channels
        # self.agent_proj = nn.Linear(4, d)
        # self.block_proj = nn.Linear(4, d)
        self.state_proj = nn.Linear(4, d)
        
        # create learnable agent token of dim size, which is added to the agent token input at every timestep (random init around 0)
        self.agent_id = nn.Parameter(torch.zeros(d))
        self.block_id = nn.Parameter(torch.zeros(d))
        nn.init.trunc_normal_(self.agent_id, std=0.02)
        nn.init.trunc_normal_(self.block_id, std=0.02)
        

        self.pos_embed  = nn.Parameter(torch.zeros(cfg.seq_len * self.TOKENS_PER_STEP, d))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.time_embed = SinusoidalEmbed(d)

        self.blocks = nn.ModuleList([
            Block(d, cfg.heads, cfg.dropout, cfg.mlp_ratio)
            for _ in range(cfg.depth)
        ])

        self.norm = nn.LayerNorm(d)

        # Each head predicts: own_delta(2) + relative(2) = 4 channels
        # The relative prediction from both heads is averaged at the output.
        self.agent_head = nn.Linear(d, 4)   # → agent_delta(2) + relative(2)
        self.block_head = nn.Linear(d, 4)   # → block_delta(2) + relative(2)
        for head in (self.agent_head, self.block_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        self._mask:     Optional[torch.Tensor] = None
        self._mask_key: Optional[Tuple]        = None

    def _get_causal_mask(self, N: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (N, device, dtype)
        if self._mask_key != key:
            i = torch.arange(N, device=device)
            m = torch.zeros(N, N, device=device, dtype=dtype)
            m.masked_fill_(i[:, None] < i[None, :], float("-inf"))
            self._mask, self._mask_key = m, key
        return self._mask

    def forward(
        self,
        x:         torch.Tensor,   # [B, T, 6]
        timesteps: torch.Tensor,   # [B, T]
    ) -> torch.Tensor:             # [B, T, 6]
        B, T, _ = x.shape

        agent_delta = x[..., 0:2]   # [B, T, 2]
        block_delta = x[..., 2:4]   # [B, T, 2]
        relative    = x[..., 4:6]   # [B, T, 2]

        # Build per-token input: own delta + relative
        agent_in = torch.cat([agent_delta, relative], dim=-1)   # [B, T, 4]
        block_in = torch.cat([block_delta, -relative], dim=-1)   # [B, T, 4]

        agent_tok = self.state_proj(agent_in)   # [B, T, d]
        block_tok = self.state_proj(block_in)   # [B, T, d]
        
        # Add learnable token ID embeddings to distinguish agent vs block tokens
        agent_tok = agent_tok + self.agent_id   # [B, T, d]
        block_tok = block_tok + self.block_id   # [B, T, d]

        # Interleave into [B, 2T, d]: [agent_0, block_0, agent_1, ...]
        tokens = torch.stack([agent_tok, block_tok], dim=2).reshape(B, 2 * T, -1)
        tokens = tokens + self.pos_embed[:2 * T]

        # Time conditioning: each timestep t covers both of its tokens
        t_expanded = timesteps.unsqueeze(-1).expand(B, T, 2).reshape(B, 2 * T)
        c = self.time_embed(t_expanded)   # [B, 2T, d]

        attn_mask = None
        if self.cfg.causal:
            attn_mask = self._get_causal_mask(2 * T, tokens.device, tokens.dtype)

        for block in self.blocks:
            tokens = block(tokens, c, attn_mask=attn_mask)

        tokens = self.norm(tokens).reshape(B, T, 2, -1)   # [B, T, 2, d]
        agent_out = tokens[:, :, 0, :]   # [B, T, d]
        block_out = tokens[:, :, 1, :]   # [B, T, d]

        agent_pred = self.agent_head(agent_out)   # [B, T, 4]: agent_delta(2) + relative(2)
        block_pred = self.block_head(block_out)   # [B, T, 4]: block_delta(2) + relative(2)

        # Assemble [B, T, 6]:
        #   agent_delta noise from agent head
        #   block_delta noise from block head
        #   relative noise averaged across both heads (they should agree)
        rel_pred = 0.5 * (agent_pred[..., 2:] - block_pred[..., 2:])

        return torch.cat([
            agent_pred[..., :2],   # agent_delta noise  [B, T, 2]
            block_pred[..., :2],   # block_delta noise  [B, T, 2]
            rel_pred,              # relative noise     [B, T, 2]
        ], dim=-1)                 # [B, T, 6]