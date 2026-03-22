from dataclasses import dataclass
from typing import Optional
import torch
from diffusers import DDIMScheduler

@dataclass
class Stats:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def _norm(self, v, mn, mx):
        return 2.0 * (v - mn) / (mx - mn + 1e-8) - 1.0

    def _denorm(self, v, mn, mx):
        return (v + 1.0) / 2.0 * (mx - mn + 1e-8) + mn

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., 4] — [tcp_x, tcp_y, block_x, block_y]
        out = x.clone()
        out[..., 0] = self._norm(x[..., 0], self.x_min, self.x_max)  # tcp_x
        out[..., 1] = self._norm(x[..., 1], self.y_min, self.y_max)  # tcp_y
        out[..., 2] = self._norm(x[..., 2], self.x_min, self.x_max)  # block_x  (shared)
        out[..., 3] = self._norm(x[..., 3], self.y_min, self.y_max)  # block_y  (shared)
        return out

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        out[..., 0] = self._denorm(x[..., 0], self.x_min, self.x_max)
        out[..., 1] = self._denorm(x[..., 1], self.y_min, self.y_max)
        out[..., 2] = self._denorm(x[..., 2], self.x_min, self.x_max)
        out[..., 3] = self._denorm(x[..., 3], self.y_min, self.y_max)
        return out


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