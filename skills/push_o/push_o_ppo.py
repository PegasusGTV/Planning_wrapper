"""
Skill: Push a circular disk to a matching goal region on the table.

The policy learns to close its fingers and use the end-effector as a pusher,
nudging the disk until it overlaps ≥90% with the same-radius goal circle.

Observation layout (state mode, 25-dim flat vector):
    qpos        (9,)   robot joint positions
    qvel        (9,)   robot joint velocities
    ee_pos      (3,)   EE world position
    ee_to_disk  (3,)   disk_pos − ee_pos
    disk_to_goal(3,)   goal_pos − disk_pos
    overlap_frac(1,)   geometric overlap fraction [0, 1]

Reward (dense, see PushOEnv):
    reach   : approach the disk
    push    : move the disk toward the goal
    overlap : geometric overlap fraction
    finger  : reward closing the gripper fingers
    bonus   : 5.0 on success (≥90% overlap)

Usage:
    Train:
        python push_o_ppo.py

    Evaluate:
        python push_o_ppo.py --evaluate --checkpoint checkpoints/<run>/final_ckpt.pt
"""
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


def _circle_overlap_frac_np(disk_xy: np.ndarray, goal_xy: np.ndarray, r: float) -> float:
    """Numpy scalar version of PushOEnv._circle_overlap_frac."""
    d = float(np.linalg.norm(disk_xy - goal_xy))
    if d >= 2.0 * r:
        return 0.0
    cos_arg = np.clip(d / (2.0 * r), -1.0 + 1e-7, 1.0 - 1e-7)
    A = 2.0 * r * r * np.arccos(cos_arg) - 0.5 * d * np.sqrt(max(4.0 * r * r - d * d, 0.0))
    return float(np.clip(A / (np.pi * r * r), 0.0, 1.0))


@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    cuda: bool = True

    env_id: str = "PushO-v1"
    obs_mode: str = "state"
    control_mode: str = "pd_ee_delta_pose"
    num_envs: int = 512
    num_eval_envs: int = 8
    reconfiguration_freq: Optional[int] = None
    eval_reconfiguration_freq: Optional[int] = 1
    num_steps: int = 200
    num_eval_steps: int = 200
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

def _build_flat_obs(obs: dict, raw_env, goal_xyz: np.ndarray) -> np.ndarray:
    """
    Reconstruct the 45-dim flat state vector that PushO-v1 produces during training
    (obs_mode="state" auto-flattens all extra fields; replicate that here for state_dict obs).

    Layout: qpos(9) + qvel(9) + ee_pos(3) + ee_to_disk(3) + disk_to_goal(3)
            + overlap_frac(1) + tcp_pose(7) + obj_pose(7) + goal_pos(3) = 45
    """
    qpos     = np.asarray(obs["agent"]["qpos"], dtype=np.float32).reshape(-1)
    qvel     = np.asarray(obs["agent"]["qvel"], dtype=np.float32).reshape(-1)
    ee_pos   = raw_env.agent.tcp.pose.p.cpu().numpy().reshape(-1).astype(np.float32)
    disk_pos = raw_env.disk.pose.p.cpu().numpy().reshape(-1).astype(np.float32)
    goal     = goal_xyz.astype(np.float32).reshape(3)

    r       = float(raw_env.disk_radius)
    overlap = _circle_overlap_frac_np(disk_pos[:2], goal[:2], r)

    tcp_pose = np.asarray(obs["extra"]["tcp_pose"], dtype=np.float32).reshape(-1)
    obj_pose = np.asarray(obs["extra"]["obj_pose"], dtype=np.float32).reshape(-1)

    return np.concatenate([
        qpos,
        qvel,
        ee_pos,
        disk_pos - ee_pos,   # ee_to_disk
        goal     - disk_pos, # disk_to_goal
        np.array([overlap], dtype=np.float32),
        tcp_pose,            # tcp_pose(7)
        obj_pose,            # obj_pose(7)
        goal,                # goal_pos(3)
    ])


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
    Run the PPO push-O policy on an already-running env to push the disk to goal_xyz.

    Requires a checkpoint trained on PushO-v1 with obs_mode='state'.
    Pass `agent` to reuse a loaded network.
    Returns (success, latest_obs).

    Args:
        env:        The (possibly wrapped) gymnasium environment.
        obs:        Current observation dict from env.
        goal_xyz:   Desired goal position (x, y, z) for the disk centre.
        checkpoint: Path to a .pt checkpoint file.
        max_steps:  Maximum number of steps to run.
        render:     Whether to call env.render() each step.
        device:     Torch device string ('cpu' or 'cuda').
    """
    raw = env.unwrapped

    if agent is None:
        agent = load_agent(checkpoint, device)
    action_low  = torch.from_numpy(env.action_space.low.reshape(-1)).to(device)
    action_high = torch.from_numpy(env.action_space.high.reshape(-1)).to(device)

    SUCCESS_OVERLAP = 0.80

    current_obs = obs
    for _ in range(max_steps):
        flat  = _build_flat_obs(current_obs, raw, goal_xyz)
        obs_t = torch.from_numpy(flat).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action = torch.clamp(
                agent.get_action(obs_t, deterministic=True), action_low, action_high
            )
        current_obs, _, term, trunc, _ = env.step(action)
        if render:
            env.render()

        disk_pos = raw.disk.pose.p.cpu().numpy().reshape(-1).astype(np.float32)
        overlap  = _circle_overlap_frac_np(disk_pos[:2], goal_xyz[:2], float(raw.disk_radius))
        if overlap >= SUCCESS_OVERLAP:
            return True, current_obs
        if np.asarray(term).any() or np.asarray(trunc).any():
            break

    disk_pos = raw.disk.pose.p.cpu().numpy().reshape(-1).astype(np.float32)
    overlap  = _circle_overlap_frac_np(disk_pos[:2], goal_xyz[:2], float(raw.disk_radius))
    return overlap >= SUCCESS_OVERLAP, current_obs


if __name__ == "__main__":
    import mani_skill.envs  # noqa: F401
    import envs              # noqa: F401 — registers PushO-v1
    import tyro
    args = tyro.cli(Args)
    train(args)
