"""
Skill: place a grasped cube at a goal position using MPPI in task space.

Three-phase controller:
  Phase 1 (carry):   move the grasped cube above the goal position.
  Phase 2 (lower):   descend to the goal height, then open gripper.
  Phase 3 (retreat): move EE upward away from the placed cube.

Control: pd_ee_delta_pose (6D EE delta + 1D gripper).

Usage:
    python skills/place/place_cube_mpc.py
    python skills/place/place_cube_mpc.py --num_episodes 20 --seed 42
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
import envs  # registers PlaceSkillEnv

from mpc_base import MPPIBase, get_ee_pos, step_env, EE_POS_ACTION_SCALE


PLACE_THRESHOLD = 0.05
RETREAT_DIST = 0.05
HOVER_HEIGHT = 0.08    # hover above goal before lowering
PLACE_HEIGHT = 0.04    # target z when placing (slightly above table)
RETREAT_HEIGHT = 0.20  # z to retreat to after release


class PlaceMPPI(MPPIBase):
    """MPPI controller for placing. Optimises 3-D EE deltas toward a target waypoint."""

    def __init__(self, **kwargs):
        kwargs.setdefault("horizon", 8)
        kwargs.setdefault("num_samples", 256)
        kwargs.setdefault("noise_std", 0.2)
        kwargs.setdefault("lam", 0.05)
        kwargs.setdefault("action_clip", 0.5)
        super().__init__(action_dim=3, **kwargs)

    def rollout_costs(self, state: dict, action_seqs: np.ndarray) -> np.ndarray:
        K, H, _ = action_seqs.shape
        pos = np.broadcast_to(state["ee_pos"], (K, 3)).copy()
        target = state["target"]
        costs = np.zeros(K, dtype=np.float32)

        scaled = action_seqs * EE_POS_ACTION_SCALE
        for t in range(H):
            pos = pos + scaled[:, t, :]
            costs += np.linalg.norm(pos - target, axis=1)
            costs += 0.01 * np.sum(action_seqs[:, t, :] ** 2, axis=1)

        costs += 5.0 * np.linalg.norm(pos - target, axis=1)
        return costs


# ---------------------------------------------------------------------------
# Callable skill API
# ---------------------------------------------------------------------------

def execute(
    env,
    obs: dict,
    block_idx: int,
    goal_xyz: np.ndarray,
    max_steps: int = 200,
    render: bool = False,
    **kwargs,
) -> tuple[bool, dict]:
    """
    Place obstacle[block_idx] at goal_xyz using MPPI.
    Cube must already be grasped.
    Returns (success, latest_obs).
    """
    raw = env.unwrapped
    obstacle = raw.obstacles[block_idx]
    controller = PlaceMPPI(**kwargs)
    act_dim = env.action_space.shape[0]
    current_obs = obs

    phase = "carry"  # carry -> lower -> release -> retreat
    release_steps = 0

    for step in range(max_steps):
        ee_pos = get_ee_pos(current_obs)
        cube_pos = obstacle.pose.p.cpu().numpy().reshape(-1).astype(np.float32)
        is_grasped = bool(raw.agent.is_grasping(obstacle).cpu().numpy().any())

        # Full success check
        cube_near_goal = float(np.linalg.norm(cube_pos[:2] - goal_xyz[:2])) < PLACE_THRESHOLD
        cube_resting = cube_pos[2] < 0.05
        ee_retreated = float(np.linalg.norm(ee_pos - cube_pos)) > RETREAT_DIST
        if (not is_grasped) and cube_near_goal and cube_resting and ee_retreated:
            return True, current_obs

        if phase == "carry":
            # Move above the goal position
            target = goal_xyz.copy()
            target[2] = HOVER_HEIGHT
            gripper_cmd = -1.0  # closed

            xy_dist = float(np.linalg.norm(ee_pos[:2] - goal_xyz[:2]))
            if xy_dist < 0.02:
                phase = "lower"
                controller.nominal[:] = 0.0

        elif phase == "lower":
            # Descend to place height
            target = goal_xyz.copy()
            target[2] = PLACE_HEIGHT
            gripper_cmd = -1.0  # closed

            if ee_pos[2] < PLACE_HEIGHT + 0.02:
                phase = "release"
                release_steps = 0

        elif phase == "release":
            # Hold position, open gripper
            target = ee_pos.copy()
            gripper_cmd = 1.0  # open
            release_steps += 1

            if release_steps > 15:
                phase = "retreat"
                controller.nominal[:] = 0.0

        elif phase == "retreat":
            # Move up and away
            target = ee_pos.copy()
            target[2] = RETREAT_HEIGHT
            gripper_cmd = 1.0  # open

        delta = controller.get_action({"ee_pos": ee_pos, "target": target})

        action = np.zeros(act_dim, dtype=np.float32)
        action[:3] = delta
        action[-1] = gripper_cmd

        current_obs, done = step_env(env, action, render)
        if done:
            break

    # Final success check
    ee_pos = get_ee_pos(current_obs)
    cube_pos = obstacle.pose.p.cpu().numpy().reshape(-1).astype(np.float32)
    is_grasped = bool(raw.agent.is_grasping(obstacle).cpu().numpy().any())
    cube_near = float(np.linalg.norm(cube_pos[:2] - goal_xyz[:2])) < PLACE_THRESHOLD
    cube_rest = cube_pos[2] < 0.05
    ee_away = float(np.linalg.norm(ee_pos - cube_pos)) > RETREAT_DIST
    return bool((not is_grasped) and cube_near and cube_rest and ee_away), current_obs


class PlaceMPCPreviewSession:
    """One MPPI place step at a time — mirrors `execute` for metric lookahead."""

    def __init__(self, env, block_idx: int, goal_xyz: np.ndarray, **kwargs):
        self.env = env
        self.raw = env.unwrapped
        self.obstacle = self.raw.obstacles[block_idx]
        self.goal_xyz = np.asarray(goal_xyz, dtype=np.float32).reshape(3)
        self.kwargs = kwargs
        self.act_dim = env.action_space.shape[0]
        self.reset()

    def reset(self) -> None:
        self.controller = PlaceMPPI(**self.kwargs)
        self.phase = "carry"
        self.release_steps = 0

    def step_action(self, obs: dict) -> np.ndarray:
        raw = self.raw
        obstacle = self.obstacle
        controller = self.controller
        goal_xyz = self.goal_xyz
        ee_pos = get_ee_pos(obs)
        cube_pos = obstacle.pose.p.cpu().numpy().reshape(-1).astype(np.float32)
        is_grasped = bool(raw.agent.is_grasping(obstacle).cpu().numpy().any())

        if self.phase == "carry":
            target = goal_xyz.copy()
            target[2] = HOVER_HEIGHT
            gripper_cmd = -1.0
            xy_dist = float(np.linalg.norm(ee_pos[:2] - goal_xyz[:2]))
            if xy_dist < 0.02:
                self.phase = "lower"
                controller.nominal[:] = 0.0
        elif self.phase == "lower":
            target = goal_xyz.copy()
            target[2] = PLACE_HEIGHT
            gripper_cmd = -1.0
            if ee_pos[2] < PLACE_HEIGHT + 0.02:
                self.phase = "release"
                self.release_steps = 0
        elif self.phase == "release":
            target = ee_pos.copy()
            gripper_cmd = 1.0
            self.release_steps += 1
            if self.release_steps > 15:
                self.phase = "retreat"
                controller.nominal[:] = 0.0
        elif self.phase == "retreat":
            target = ee_pos.copy()
            target[2] = RETREAT_HEIGHT
            gripper_cmd = 1.0
        else:
            target = goal_xyz.copy()
            gripper_cmd = -1.0

        delta = controller.get_action({"ee_pos": ee_pos, "target": target})
        action = np.zeros(self.act_dim, dtype=np.float32)
        action[:3] = delta
        action[-1] = gripper_cmd
        return action


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_eval(args):
    np.random.seed(args.seed)

    env = gym.make(
        "PlaceSkillEnv",
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        reconfiguration_freq=1,
        sim_backend="physx_cpu",
    )

    successes = []

    for ep in range(args.num_episodes):
        print(f"\n=== Episode {ep + 1}/{args.num_episodes} ===")
        obs, _ = env.reset()

        raw = env.unwrapped
        ee_pos   = np.asarray(obs["extra"]["tcp_pose"], dtype=np.float32).reshape(-1)[:3]
        goal_pos = np.asarray(obs["extra"]["goal_pos"], dtype=np.float32).reshape(-1)
        print(f"  ee_start: {np.round(ee_pos, 3)}")
        print(f"  goal_pos: {np.round(goal_pos, 3)}")

        # PlaceSkillEnv uses a single cube, not obstacles array
        if not hasattr(raw, "obstacles"):
            raw.obstacles = [raw.cube]

        t0 = time.time()
        success, _ = execute(
            env, obs, 0, goal_pos,
            max_steps=args.max_steps,
            horizon=args.horizon,
            num_samples=args.num_samples,
            noise_std=args.noise_std,
            lam=args.lam,
        )
        elapsed = time.time() - t0

        successes.append(success)
        print(f"  {'SUCCESS' if success else 'FAIL'}  ({elapsed:.2f}s)")

    env.close()

    print("\n" + "=" * 50)
    print(f"Success rate: {np.mean(successes) * 100:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPPI eval for PlaceSkillEnv")
    parser.add_argument("--num_episodes", type=int,   default=10)
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--max_steps",    type=int,   default=200)
    parser.add_argument("--horizon",      type=int,   default=8)
    parser.add_argument("--num_samples",  type=int,   default=256)
    parser.add_argument("--noise_std",    type=float, default=0.2)
    parser.add_argument("--lam",          type=float, default=0.05)
    args = parser.parse_args()
    run_eval(args)
