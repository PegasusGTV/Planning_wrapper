"""
Shared MPPI infrastructure: base controller class and observation utilities.
Imported by each skill's *_mpc.py file.

MPPI (Model Predictive Path Integral, Williams et al. 2016-2017):
  1. Sample K action sequences around a nominal, perturbed with Gaussian noise.
  2. Roll each out through a dynamics model to get K trajectory costs.
  3. Weight samples by exp(-cost / lambda) -- low-cost trajectories dominate.
  4. Weighted average -> updated nominal sequence.
  5. Execute first action, shift nominal forward, repeat.
"""
from __future__ import annotations

import numpy as np

# Panda pd_ee_delta_pose scales action=1.0 -> 0.1 m of commanded EE delta
# (pos_lower/pos_upper = ±0.1 in the controller config). Subclasses must
# multiply actions by this factor when integrating EE position in rollouts.
EE_POS_ACTION_SCALE = 0.1


class MPPIBase:
    """
    Base MPPI controller. Subclasses implement `rollout_costs()` with a
    task-specific dynamics model and cost function.

    Parameters
    ----------
    horizon : int
        Number of timesteps to plan ahead.
    num_samples : int
        Number of action sequences (K) to sample each step.
    action_dim : int
        Dimensionality of each action (e.g. 3 for XYZ deltas).
    noise_std : float
        Standard deviation of Gaussian perturbations.
    lam : float
        Temperature parameter -- lower values make the weighting sharper.
    """

    def __init__(
        self,
        horizon: int = 8,
        num_samples: int = 256,
        action_dim: int = 3,
        noise_std: float = 0.02,
        lam: float = 0.05,
        action_clip: float | None = 1.0,
    ):
        self.horizon = horizon
        self.K = num_samples
        self.action_dim = action_dim
        self.noise_std = noise_std
        self.lam = lam
        self.action_clip = action_clip
        self.nominal = np.zeros((horizon, action_dim), dtype=np.float32)

    def rollout_costs(self, state: dict, action_seqs: np.ndarray) -> np.ndarray:
        """
        Evaluate K action sequences from the current state.

        Parameters
        ----------
        state : dict
            Arbitrary state dict (ee_pos, cube_pos, etc.) -- defined per skill.
        action_seqs : np.ndarray, shape (K, H, action_dim)
            Perturbed action sequences to evaluate.

        Returns
        -------
        costs : np.ndarray, shape (K,)
        """
        raise NotImplementedError

    def get_action(self, state: dict) -> np.ndarray:
        """
        Run one MPPI step: sample, evaluate, reweight, return first action.
        """
        noise = (
            np.random.randn(self.K, self.horizon, self.action_dim).astype(np.float32)
            * self.noise_std
        )
        action_seqs = self.nominal[None, :, :] + noise  # (K, H, action_dim)
        if self.action_clip is not None:
            action_seqs = np.clip(action_seqs, -self.action_clip, self.action_clip)

        costs = self.rollout_costs(state, action_seqs)

        # MPPI weighting
        costs_shifted = costs - np.min(costs)
        weights = np.exp(-costs_shifted / self.lam)
        weights /= np.sum(weights) + 1e-10

        # Weighted average -> updated nominal
        self.nominal = np.einsum("k,kha->ha", weights, action_seqs)
        action = self.nominal[0].copy()

        # Warm-start: shift nominal forward, zero-pad the last step
        self.nominal = np.roll(self.nominal, -1, axis=0)
        self.nominal[-1] = 0.0

        return action


def get_ee_pos(obs: dict) -> np.ndarray:
    """Extract EE position (3,) from raw env obs or planning wrapper obs."""
    if "extra" in obs:
        extra = obs["extra"]
        if "ee_pos" in extra:
            return np.asarray(extra["ee_pos"], dtype=np.float32).reshape(-1)[:3]
        if "tcp_pose" in extra:
            return np.asarray(extra["tcp_pose"], dtype=np.float32).reshape(-1)[:3]
    if "tcp_pose" in obs:
        return np.asarray(obs["tcp_pose"], dtype=np.float32).reshape(-1)[:3]
    raise KeyError("Cannot find EE position in obs (tried extra.ee_pos, extra.tcp_pose, tcp_pose)")


def step_env(env, action: np.ndarray, render: bool = False):
    """
    Send a full action array to the env and return (obs, term, trunc).
    Handles torch conversion and batch dimension.
    """
    import torch
    action_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
    obs, _, term, trunc, _ = env.step(action_t)
    if render:
        env.render()
    done = np.asarray(term).any() or np.asarray(trunc).any()
    return obs, done
