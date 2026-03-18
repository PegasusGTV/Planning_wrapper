"""
Diffusion backends: Optimal-Transport Flow Matching and DDIM.

Shared training API
-------------------
  .sample_timesteps(B, device, num_frames=None) → [B] or [B, F]
  .add_noise(x_0, noise, timesteps)             → x_t
  .get_target(x_0, noise, timesteps)            → training target

The .scheduler attribute is a diffusers scheduler used for inference.

NormStats / Stats
-----------------
  Lightweight wrappers for per-dimension mean/std normalisation.
  Defined here so both dataset.py and train.py can import from one place.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, DDIMScheduler


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

@dataclass
class Stats:
    mean: torch.Tensor   # [dim]
    std:  torch.Tensor   # [dim]

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x)) / (self.std.to(x) + 1e-8)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std.to(x) + 1e-8) + self.mean.to(x)


@dataclass
class NormStats:
    state:  Stats
    action: Stats


# ---------------------------------------------------------------------------
# Flow Matching  (OT-CFM)
# ---------------------------------------------------------------------------

class FlowMatching:
    """
    Optimal-Transport Conditional Flow Matching.

      Forward:  x_t = (1-t) · x_0  +  t · noise       t ∈ [0,1]
      Target:   v   = noise − x_0
    """

    prediction_type = "velocity"

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 1.0):
        self.num_train_timesteps = num_train_timesteps
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
        )

    def sample_timesteps(
        self, B: int, device: torch.device, num_frames: Optional[int] = None
    ) -> torch.Tensor:
        shape = (B, num_frames) if num_frames is not None else (B,)
        return torch.randint(0, self.num_train_timesteps, shape, device=device)

    def add_noise(
        self, x_0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Supports [B] or [B, F] timesteps — broadcasts to x_0's shape."""
        t = timesteps.float() / self.num_train_timesteps
        while t.ndim < x_0.ndim:
            t = t.unsqueeze(-1)
        return (1.0 - t) * x_0 + t * noise

    def get_target(
        self,
        x_0:       torch.Tensor,
        noise:     torch.Tensor,
        timesteps: torch.Tensor = None,   # unused, kept for API symmetry
    ) -> torch.Tensor:
        return noise - x_0


# ---------------------------------------------------------------------------
# DDIM  (epsilon prediction)
# ---------------------------------------------------------------------------

class DDIM:
    """
    DDIM with epsilon (noise) prediction.

      Forward:  x_t = √ᾱ_t · x_0  +  √(1−ᾱ_t) · noise
      Target:   ε   = noise
    """

    prediction_type = "epsilon"

    def __init__(
        self,
        num_train_timesteps: int   = 1000,
        beta_schedule:       str   = "squaredcos_cap_v2",
        clip_sample:         bool  = True,
        clip_sample_range:   float = 2.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type="epsilon",
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
        )
        # Keep on CPU; moved to device lazily in _alphas_at
        self._ac_cpu = self.scheduler.alphas_cumprod

    def sample_timesteps(
        self, B: int, device: torch.device, num_frames: Optional[int] = None
    ) -> torch.Tensor:
        shape = (B, num_frames) if num_frames is not None else (B,)
        return torch.randint(0, self.num_train_timesteps, shape, device=device)

    def _alphas_at(self, timesteps: torch.Tensor) -> torch.Tensor:
        """ᾱ_t for given [B] or [B, F] timesteps on the correct device."""
        return self._ac_cpu.to(timesteps.device)[timesteps.long()]

    def add_noise(
        self, x_0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        a = self._alphas_at(timesteps)
        sa, s1a = a.sqrt(), (1.0 - a).sqrt()
        while sa.ndim < x_0.ndim:
            sa  = sa.unsqueeze(-1)
            s1a = s1a.unsqueeze(-1)
        return sa * x_0 + s1a * noise

    def get_target(
        self,
        x_0:       torch.Tensor,
        noise:     torch.Tensor,
        timesteps: torch.Tensor = None,   # unused
    ) -> torch.Tensor:
        return noise   # epsilon prediction always targets the noise