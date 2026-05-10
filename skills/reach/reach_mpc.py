"""
Skill: reach a goal EE position using MPPI in task space.

Control: pd_ee_delta_pose (Cartesian delta position + orientation + gripper).

Usage:
    python skills/reach/reach_mpc.py
    python skills/reach/reach_mpc.py --num_episodes 20 --seed 42
"""
from __future__ import annotations

import argparse
import time

import gymnasium as gym
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import envs  # registers Reach-WithObstacles-v1

from mpc_base import MPPIBase, get_ee_pos, step_env, EE_POS_ACTION_SCALE


class ReachMPPI(MPPIBase):
    """MPPI controller for reaching a Cartesian goal position."""

    def __init__(self, goal_xyz: np.ndarray, **kwargs):
        kwargs.setdefault("horizon", 8)
        kwargs.setdefault("num_samples", 256)
        kwargs.setdefault("noise_std", 0.5)
        kwargs.setdefault("lam", 0.05)
        super().__init__(action_dim=3, **kwargs)
        self.goal_xyz = goal_xyz.copy()

    def rollout_costs(self, state: dict, action_seqs: np.ndarray) -> np.ndarray:
        K, H, _ = action_seqs.shape
        pos = np.broadcast_to(state["ee_pos"], (K, 3)).copy()
        costs = np.zeros(K, dtype=np.float32)

        scaled = action_seqs * EE_POS_ACTION_SCALE
        for t in range(H):
            pos = pos + scaled[:, t, :]
            costs += np.linalg.norm(pos - self.goal_xyz, axis=1)
            costs += 0.01 * np.sum(action_seqs[:, t, :] ** 2, axis=1)

        # Terminal cost
        costs += 5.0 * np.linalg.norm(pos - self.goal_xyz, axis=1)
        return costs


# ---------------------------------------------------------------------------
# Callable skill API
# ---------------------------------------------------------------------------

def execute(
    env,
    obs: dict,
    goal_xyz: np.ndarray,
    max_steps: int = 200,
    success_threshold: float = 0.05,
    render: bool = False,
    **kwargs,
) -> tuple[bool, dict]:
    """
    Move the EE to goal_xyz using MPPI.
    Returns (success, latest_obs).
    """
    controller = ReachMPPI(goal_xyz=goal_xyz, **kwargs)
    act_dim = env.action_space.shape[0]
    current_obs = obs

    for _ in range(max_steps):
        ee_pos = get_ee_pos(current_obs)
        if np.linalg.norm(ee_pos - goal_xyz) < success_threshold:
            return True, current_obs

        delta = controller.get_action({"ee_pos": ee_pos})
        action = np.zeros(act_dim, dtype=np.float32)
        action[:3] = delta
        current_obs, done = step_env(env, action, render)
        if done:
            break

    final_ee = get_ee_pos(current_obs)
    return bool(np.linalg.norm(final_ee - goal_xyz) < success_threshold), current_obs


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_eval(args):
    np.random.seed(args.seed)

    env = gym.make(
        "Reach-WithObstacles-v1",
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        reconfiguration_freq=1,
        sim_backend="physx_cpu",
    )

    successes, final_dists = [], []

    for ep in range(args.num_episodes):
        print(f"\n=== Episode {ep + 1}/{args.num_episodes} ===")
        obs, _ = env.reset()

        ee_pos   = np.asarray(obs["extra"]["ee_pos"], dtype=np.float32).reshape(-1)[:3]
        goal_pos = np.asarray(obs["extra"]["goal_pos"], dtype=np.float32).reshape(-1)
        print(f"  ee_start: {np.round(ee_pos, 3)}")
        print(f"  goal_pos: {np.round(goal_pos, 3)}")

        t0 = time.time()
        success, final_obs = execute(
            env, obs, goal_pos,
            max_steps=args.max_steps,
            horizon=args.horizon,
            num_samples=args.num_samples,
            noise_std=args.noise_std,
            lam=args.lam,
        )
        elapsed = time.time() - t0

        final_ee = np.asarray(final_obs["extra"]["ee_pos"], dtype=np.float32).reshape(-1)[:3]
        dist = float(np.linalg.norm(final_ee - goal_pos))

        successes.append(success)
        final_dists.append(dist)
        print(f"  {'SUCCESS' if success else 'FAIL'}  dist={dist * 100:.1f} cm  ({elapsed:.2f}s)")

    env.close()

    valid = [d for d in final_dists if d < float("inf")]
    print("\n" + "=" * 50)
    print(f"Success rate   : {np.mean(successes) * 100:.1f}%")
    print(f"Mean final dist: {np.mean(valid) * 100:.1f} cm" if valid else "Mean final dist: N/A")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPPI eval for Reach-WithObstacles-v1")
    parser.add_argument("--num_episodes", type=int,   default=10)
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--max_steps",    type=int,   default=200)
    parser.add_argument("--horizon",      type=int,   default=8)
    parser.add_argument("--num_samples",  type=int,   default=256)
    parser.add_argument("--noise_std",    type=float, default=0.5)
    parser.add_argument("--lam",          type=float, default=0.05)
    args = parser.parse_args()
    run_eval(args)