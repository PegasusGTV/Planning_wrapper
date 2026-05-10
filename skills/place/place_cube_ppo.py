'''
Skill: Place Cube

The robot (panda) starts every episode with the cube already firmly grasped.
The skill must lower and release the cube onto a randomly sampled goal position
on the table, then retreat the EE away from the placed cube.

State space: robot qpos/qvel, EE pose, cube pose, goal pos,
             EE-to-cube vector, cube-to-goal vector, is_grasped
Action space: pd_ee_delta_pose (6D EE delta + 1D gripper)
Reward: shaped place-toward-goal (gated on grasp) + retreat reward, terminal bonus
Success: cube released near goal AND EE retreated > 10 cm from cube

Usage:

Run training:
    python place_cube_ppo.py

Evaluate:
    python place_cube_ppo.py --evaluate --checkpoint checkpoints/<run>/final_ckpt.pt
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

    env_id: str = "PlaceSkillEnv"
    obs_mode: str = "state"
    control_mode: str = "pd_ee_delta_pose"
    num_envs: int = 1024
    num_eval_envs: int = 16
    reconfiguration_freq: Optional[int] = None
    eval_reconfiguration_freq: Optional[int] = 1
    num_steps: int = 100
    num_eval_steps: int = 100
    partial_reset: bool = True
    eval_partial_reset: bool = False

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

def _build_place_obs(obs: dict, raw_env, obstacle, goal_xyz: np.ndarray) -> np.ndarray:
    """
    Reconstruct the flat state obs that PlaceSkillEnv produces during training.
    Layout: qpos(9) + qvel(9) + is_grasped(1) + tcp_pose(7) + goal_pos(3)
            + obj_pose(7) + tcp_to_obj_pos(3) + obj_to_goal_pos(3) = 42
    """
    qpos      = np.asarray(obs["agent"]["qpos"], dtype=np.float32).reshape(-1)
    qvel      = np.asarray(obs["agent"]["qvel"], dtype=np.float32).reshape(-1)
    is_grasped = np.array(
        [float(raw_env.agent.is_grasping(obstacle).cpu().numpy().any())], dtype=np.float32
    )
    tcp_pose  = raw_env.agent.tcp_pose.raw_pose.cpu().numpy().reshape(-1).astype(np.float32)  # 7
    goal      = goal_xyz.astype(np.float32).reshape(-1)[:3]
    obj_pose  = obstacle.pose.raw_pose.cpu().numpy().reshape(-1).astype(np.float32)           # 7
    tcp_pos   = tcp_pose[:3]
    obj_pos   = obj_pose[:3]
    return np.concatenate([
        qpos, qvel, is_grasped, tcp_pose, goal, obj_pose,
        obj_pos - tcp_pos,  # tcp_to_obj_pos
        goal - obj_pos,     # obj_to_goal_pos
    ])


def execute(
    env,
    obs: dict,
    block_idx: int,
    goal_xyz: np.ndarray,
    checkpoint: str,
    max_steps: int = 200,
    render: bool = False,
    device: str = "cpu",
    agent: Any | None = None,
) -> tuple[bool, dict]:
    """
    Run the PPO place policy on an already-running PushO env to set
    obstacle[block_idx] down at goal_xyz, then retreat.
    Requires a checkpoint trained on PlaceSkillEnv with obs_mode='state'.
    Pass `agent` to reuse a loaded network.
    Returns (success, latest_obs).
    """
    raw      = env.unwrapped
    obstacle = raw.obstacles[block_idx]

    PLACE_THRESH  = 0.05
    RETREAT_DIST  = 0.05
    REST_Z_THRESH = 0.05

    if agent is None:
        agent = load_agent(checkpoint, device)
    action_low  = torch.from_numpy(env.action_space.low.reshape(-1)).to(device)
    action_high = torch.from_numpy(env.action_space.high.reshape(-1)).to(device)

    def _check_success():
        is_grasped   = bool(raw.agent.is_grasping(obstacle).cpu().numpy().any())
        obj_pos      = obstacle.pose.p.cpu().numpy().reshape(-1).astype(np.float32)
        tcp_pos      = raw.agent.tcp_pose.p.cpu().numpy().reshape(-1).astype(np.float32)
        obj_to_goal  = float(np.linalg.norm(obj_pos[:2] - goal_xyz[:2]))
        is_placed    = (not is_grasped) and (obj_to_goal < PLACE_THRESH) and (obj_pos[2] < REST_Z_THRESH)
        is_retreated = float(np.linalg.norm(tcp_pos - obj_pos)) > RETREAT_DIST
        return bool(is_placed and is_retreated)

    # Bail out immediately if the robot isn't holding the cube — nothing to place.
    if not bool(raw.agent.is_grasping(obstacle).cpu().numpy().any()):
        return False, obs

    current_obs = obs
    for _ in range(max_steps):
        flat  = _build_place_obs(current_obs, raw, obstacle, goal_xyz)
        obs_t = torch.from_numpy(flat).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action = torch.clamp(agent.get_action(obs_t, deterministic=True), action_low, action_high)
        current_obs, _, term, trunc, _ = env.step(action)
        if render:
            env.render()
        if _check_success():
            return True, current_obs
        if np.asarray(term).any() or np.asarray(trunc).any():
            break

    return _check_success(), current_obs


if __name__ == "__main__":
    import envs  # noqa: F401 — registers ManiSkill envs
    import tyro
    args = tyro.cli(Args)
    train(args)
