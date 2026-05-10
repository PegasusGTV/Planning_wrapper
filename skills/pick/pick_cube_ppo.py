'''
Skill: Pick Cube

The robot (panda) starts with its EE already near a red cube on the table.
The skill must close the gripper around the cube and lift it slightly off the
table surface. No goal placement is required — the episode ends on grasp.

State space: robot qpos/qvel, EE pose, cube pose, EE-to-cube vector, is_grasped
Action space: pd_ee_delta_pose (6D EE delta + 1D gripper)
Reward: shaped reaching + grasping + lifting, terminal bonus on success
Success: cube is grasped AND lifted > 1.5 cm above table

Usage:

Run training:
    python pick_cube_ppo.py

Evaluate:
    python pick_cube_ppo.py --evaluate --checkpoint checkpoints/<run>/final_ckpt.pt
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

    env_id: str = "PickSkillEnv"
    obs_mode: str = "state"
    control_mode: str = "pd_ee_delta_pose"
    num_envs: int = 1024
    num_eval_envs: int = 16
    reconfiguration_freq: Optional[int] = None
    eval_reconfiguration_freq: Optional[int] = 1
    num_steps: int = 50
    num_eval_steps: int = 50
    partial_reset: bool = True
    eval_partial_reset: bool = False # cannot be used if reconfiguration_freq

    total_timesteps: int = 10_000_000
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

    eval_freq: int = 20
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

def _build_pick_obs(obs: dict, raw_env, obstacle) -> np.ndarray:
    """
    Reconstruct the flat state obs that PickSkillEnv produces during training.
    Layout: qpos(9) + qvel(9) + ee_pos(3) + pick_cube_pos(3) + ee_to_pick_cube(3) + is_grasped(1) = 28
    """
    qpos      = np.asarray(obs["agent"]["qpos"], dtype=np.float32).reshape(-1)
    qvel      = np.asarray(obs["agent"]["qvel"], dtype=np.float32).reshape(-1)
    ee_pos    = raw_env.agent.tcp.pose.p.cpu().numpy().reshape(-1).astype(np.float32)
    cube_pos  = obstacle.pose.p.cpu().numpy().reshape(-1).astype(np.float32)
    is_grasped = np.array(
        [float(raw_env.agent.is_grasping(obstacle).cpu().numpy().any())], dtype=np.float32
    )
    return np.concatenate([qpos, qvel, ee_pos, cube_pos, cube_pos - ee_pos, is_grasped])


def execute(
    env,
    obs: dict,
    block_idx: int,
    checkpoint: str,
    max_steps: int = 200,
    render: bool = False,
    device: str = "cpu",
    agent: Any | None = None,
) -> tuple[bool, dict]:
    """
    Run the PPO pick policy on an already-running PushO env to grasp and lift
    obstacle[block_idx].  Requires a checkpoint trained on PickSkillEnv with
    obs_mode='state'. Pass `agent` to reuse a loaded network.
    Returns (success, latest_obs).
    """
    raw      = env.unwrapped
    obstacle = raw.obstacles[block_idx]

    if agent is None:
        agent = load_agent(checkpoint, device)
    action_low  = torch.from_numpy(env.action_space.low.reshape(-1)).to(device)
    action_high = torch.from_numpy(env.action_space.high.reshape(-1)).to(device)

    LIFT_THRESHOLD = 0.06
    current_obs = obs
    for _ in range(max_steps):
        flat  = _build_pick_obs(current_obs, raw, obstacle)
        obs_t = torch.from_numpy(flat).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action = torch.clamp(agent.get_action(obs_t, deterministic=True), action_low, action_high)
        current_obs, _, term, trunc, _ = env.step(action)
        if render:
            env.render()
        is_grasped = bool(raw.agent.is_grasping(obstacle).cpu().numpy().any())
        cube_z     = float(obstacle.pose.p.cpu().numpy().reshape(-1)[2])
        if is_grasped and cube_z > LIFT_THRESHOLD:
            return True, current_obs
        if np.asarray(term).any() or np.asarray(trunc).any():
            break

    is_grasped = bool(raw.agent.is_grasping(obstacle).cpu().numpy().any())
    cube_z     = float(obstacle.pose.p.cpu().numpy().reshape(-1)[2])
    return bool(is_grasped and cube_z > LIFT_THRESHOLD), current_obs


if __name__ == "__main__":
    import envs  # noqa: F401 — registers ManiSkill envs
    import tyro
    args = tyro.cli(Args)
    train(args)
