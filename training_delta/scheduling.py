"""
Scheduling matrices for Diffusion Forcing inference.

K[m, t] = noise level for frame t at denoising step m
  0               → fully clean
  num_inference_steps → fully noisy

First row = all frames maximally noisy.
Last  row = all frames fully clean (all zeros).
"""

import numpy as np


def generate_scheduling_matrix(
    num_frames:          int,
    num_inference_steps: int,
    schedule:            str   = "pyramid",
    uncertainty_scale:   float = 1.0,
) -> np.ndarray:
    """
    Parameters
    ----------
    num_frames          : F — number of future frames
    num_inference_steps : S — denoising steps
    schedule            : "full_sequence" or "pyramid"
    uncertainty_scale   : pyramid slope; higher → later frames stay noisy longer

    Returns
    -------
    K : int64 array [M, F]  with values in [0, S]
        full_sequence → M = S+1
        pyramid       → M = S + int((F-1)*uncertainty_scale) + 1
    """
    S = num_inference_steps

    if schedule == "full_sequence":
        # All frames share the same noise level — standard DDPM/FM sampling
        col = np.arange(S, -1, -1, dtype=np.int64)
        return np.broadcast_to(col[:, None], (S + 1, num_frames)).copy()

    elif schedule == "pyramid":
        # Earlier frames are denoised first; later frames stay noisy longer
        height = S + int((num_frames - 1) * uncertainty_scale) + 1
        K = np.zeros((height, num_frames), dtype=np.int64)
        for m in range(height):
            for t in range(num_frames):
                K[m, t] = S + int(t * uncertainty_scale) - m
        return np.clip(K, 0, S)

    else:
        raise ValueError(f"Unknown schedule '{schedule}'. Use 'full_sequence' or 'pyramid'.")