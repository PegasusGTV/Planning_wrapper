'''
Skill: Move to Goal End Effector Pose

During training, reset to random arm joint config and random goal end effector pose

State space: robot qpose, qvel, ee position, goal position, vector from ee to goal
Action space: delta joint position
Reward: dense negative distance, bonus on success
Success criteria: ee < 2cm from goal

Usage:

Run training:
    python reach_ppo.py

Evaluate:
    python reach_ppo.py --evaluate --checkpoint checkpoints/<run>/final_ckpt.pt
'''
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

_SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SKILL_DIR, "..", ".."))  # POLARIS/ for envs
sys.path.insert(0, os.path.join(_SKILL_DIR, ".."))        # skills/ for ppo_base

from ppo_base import load_agent, train  # noqa: E402


@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    cuda: bool = True

    env_id: str = "Reach-WithObstacles-v1"
    obs_mode: str = "state"
    control_mode: str = "pd_ee_delta_pose"
    num_envs: int = 512
    num_eval_envs: int = 16
    reconfiguration_freq: Optional[int] = None
    eval_reconfiguration_freq: Optional[int] = 1
    num_steps: int = 50
    num_eval_steps: int = 50
    partial_reset: bool = True
    eval_partial_reset: bool = False

    total_timesteps: int = 3_000_000
    learning_rate: float = 3e-4
    anneal_lr: bool = False
    gamma: float = 0.8
    gae_lambda: float = 0.9
    num_minibatches: int = 32
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.1
    reward_scale: float = 1.0
    finite_horizon_gae: bool = False

    eval_freq: int = 25
    save_model: bool = True
    capture_video: bool = True
    save_eval_video_freq: Optional[int] = 1

    evaluate: bool = False
    checkpoint: Optional[str] = None

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


# ---------------------------------------------------------------------------
# Callable skill API
# ---------------------------------------------------------------------------

def _build_flat_obs(obs: dict, goal_xyz: np.ndarray) -> np.ndarray:
    """
    Reconstruct the flat state obs that Reach-WithObstacles produces during training.
    obs_mode='state' flattens: qpos(9), qvel(9), goal_pos(3), ee_pos(3), ee_to_goal(3) = 27.
    """
    qpos   = np.asarray(obs["agent"]["qpos"], dtype=np.float32).reshape(-1)
    qvel   = np.asarray(obs["agent"]["qvel"], dtype=np.float32).reshape(-1)
    ee_pos = np.asarray(obs["extra"]["tcp_pose"], dtype=np.float32).reshape(-1)[:3]
    goal   = goal_xyz.astype(np.float32)
    return np.concatenate([qpos, qvel, goal, ee_pos, goal - ee_pos])


def execute(
    env,
    obs: dict,
    goal_xyz: np.ndarray,
    checkpoint: str,
    max_steps: int = 200,
    render: bool = False,
    device: str = "cpu",
    agent: Any | None = None,
) -> tuple[bool, dict]:
    """
    Run the PPO reach policy on an already-running env to move EE to goal_xyz.
    Requires a checkpoint trained on Reach-WithObstacles-v1 with obs_mode='state'.
    Pass a pre-loaded `agent` to avoid reloading weights (same behavior as load_agent).
    Returns (success, latest_obs).
    """
    if agent is None:
        agent = load_agent(checkpoint, device)
    action_low  = torch.from_numpy(env.action_space.low.reshape(-1)).to(device)
    action_high = torch.from_numpy(env.action_space.high.reshape(-1)).to(device)

    current_obs = obs
    for _ in range(max_steps):
        flat  = _build_flat_obs(current_obs, goal_xyz)
        obs_t = torch.from_numpy(flat).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action = torch.clamp(agent.get_action(obs_t, deterministic=True), action_low, action_high)
        current_obs, _, term, trunc, _ = env.step(action)
        if render:
            env.render()
        ee_pos = np.asarray(current_obs["extra"]["tcp_pose"], dtype=np.float32).reshape(-1)[:3]
        if np.linalg.norm(ee_pos - goal_xyz) < 0.02:
            return True, current_obs
        if np.asarray(term).any() or np.asarray(trunc).any():
            break

    ee_pos = np.asarray(current_obs["extra"]["tcp_pose"], dtype=np.float32).reshape(-1)[:3]
    return bool(np.linalg.norm(ee_pos - goal_xyz) < 0.05), current_obs


if __name__ == "__main__":
    import envs  # noqa: F401 — registers ManiSkill envs
    import tyro
    args = tyro.cli(Args)
    train(args)
