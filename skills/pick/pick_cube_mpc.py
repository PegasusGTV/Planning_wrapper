"""
Skill: pick up a cube using MPPI in task space.

Three-phase controller:
  Phase 1 (approach): drive EE above the cube with gripper open.
  Phase 2 (descend):  lower EE onto the cube, then close gripper.
  Phase 3 (lift):     lift the grasped cube upward.

Control: pd_ee_delta_pose (6D EE delta + 1D gripper).

Usage:
    python skills/pick/pick_cube_mpc.py
    python skills/pick/pick_cube_mpc.py --num_episodes 20 --seed 42
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
import envs  # registers PickSkillEnv

from mpc_base import MPPIBase, get_ee_pos, step_env, EE_POS_ACTION_SCALE


# Phase thresholds
APPROACH_HEIGHT = 0.06   # hover this far above cube before descending
XY_ALIGNED = 0.015       # XY alignment threshold to start descending
DESCEND_Z_TRIGGER = 0.02 # how close in Z to cube before closing gripper
LIFT_Z = 0.12            # target z for lift


class PickMPPI(MPPIBase):
    """MPPI controller for pick. Optimises 3-D EE deltas toward a target waypoint."""

    def __init__(self, **kwargs):
        kwargs.setdefault("horizon", 10)
        kwargs.setdefault("num_samples", 512)
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
    max_steps: int = 200,
    success_threshold_z: float = 0.06,
    render: bool = False,
    **kwargs,
) -> tuple[bool, dict]:
    """
    Pick obstacle[block_idx] using MPPI.
    Returns (success, latest_obs). Success = grasped AND cube_z > threshold.
    """
    raw = env.unwrapped
    obstacle = raw.obstacles[block_idx]
    controller = PickMPPI(**kwargs)
    act_dim = env.action_space.shape[0]
    current_obs = obs

    phase = "approach"  # approach -> descend -> close -> lift

    # After gripper close command, wait this many steps before checking grasp
    close_steps = 0

    for step in range(max_steps):
        ee_pos = get_ee_pos(current_obs)
        cube_pos = obstacle.pose.p.cpu().numpy().reshape(-1).astype(np.float32)
        is_grasped = bool(raw.agent.is_grasping(obstacle).cpu().numpy().any())

        # Success check
        if is_grasped and cube_pos[2] > success_threshold_z:
            return True, current_obs

        # State machine
        if phase == "approach":
            # Target: above the cube
            target = cube_pos.copy()
            target[2] += APPROACH_HEIGHT
            gripper_cmd = 1.0  # open

            # Transition: once XY aligned and above cube, start descending
            xy_dist = float(np.linalg.norm(ee_pos[:2] - cube_pos[:2]))
            if xy_dist < XY_ALIGNED and ee_pos[2] > cube_pos[2] + APPROACH_HEIGHT * 0.5:
                phase = "descend"
                controller.nominal[:] = 0.0  # reset nominal for new phase

        elif phase == "descend":
            # Target: cube centre
            target = cube_pos.copy()
            gripper_cmd = 1.0  # keep open during descent

            # Transition: once EE is close to cube in Z, close gripper
            z_dist = ee_pos[2] - cube_pos[2]
            if z_dist < DESCEND_Z_TRIGGER:
                phase = "close"
                close_steps = 0

        elif phase == "close":
            # Hold position near cube, close gripper
            target = cube_pos.copy()
            gripper_cmd = -1.0  # close
            close_steps += 1

            # Give gripper time to close, then transition to lift
            if close_steps > 15:
                phase = "lift"
                controller.nominal[:] = 0.0

        elif phase == "lift":
            if is_grasped:
                target = ee_pos.copy()
                target[2] = LIFT_Z
            else:
                # Lost grasp, try to re-descend
                phase = "descend"
                target = cube_pos.copy()
            gripper_cmd = -1.0  # stay closed

        delta = controller.get_action({"ee_pos": ee_pos, "target": target})

        action = np.zeros(act_dim, dtype=np.float32)
        action[:3] = delta
        action[-1] = gripper_cmd

        current_obs, done = step_env(env, action, render)
        if done:
            break

    is_grasped = bool(raw.agent.is_grasping(obstacle).cpu().numpy().any())
    cube_z = float(obstacle.pose.p.cpu().numpy().reshape(-1)[2])
    return bool(is_grasped and cube_z > success_threshold_z), current_obs


class PickMPCPreviewSession:
    """One MPPI pick step at a time — mirrors `execute` for metric lookahead (executor auto mode)."""

    def __init__(self, env, block_idx: int, **kwargs):
        self.env = env
        self.raw = env.unwrapped
        self.obstacle = self.raw.obstacles[block_idx]
        self.kwargs = kwargs
        self.act_dim = env.action_space.shape[0]
        self.reset()

    def reset(self) -> None:
        self.controller = PickMPPI(**self.kwargs)
        self.phase = "approach"
        self.close_steps = 0

    def step_action(self, obs: dict) -> np.ndarray:
        """Single `execute` iteration: same targets and phase transitions as `execute`."""
        raw = self.raw
        obstacle = self.obstacle
        controller = self.controller
        ee_pos = get_ee_pos(obs)
        cube_pos = obstacle.pose.p.cpu().numpy().reshape(-1).astype(np.float32)
        is_grasped = bool(raw.agent.is_grasping(obstacle).cpu().numpy().any())

        if self.phase == "approach":
            target = cube_pos.copy()
            target[2] += APPROACH_HEIGHT
            gripper_cmd = 1.0
            xy_dist = float(np.linalg.norm(ee_pos[:2] - cube_pos[:2]))
            if xy_dist < XY_ALIGNED and ee_pos[2] > cube_pos[2] + APPROACH_HEIGHT * 0.5:
                self.phase = "descend"
                controller.nominal[:] = 0.0
        elif self.phase == "descend":
            target = cube_pos.copy()
            gripper_cmd = 1.0
            z_dist = ee_pos[2] - cube_pos[2]
            if z_dist < DESCEND_Z_TRIGGER:
                self.phase = "close"
                self.close_steps = 0
        elif self.phase == "close":
            target = cube_pos.copy()
            gripper_cmd = -1.0
            self.close_steps += 1
            if self.close_steps > 15:
                self.phase = "lift"
                controller.nominal[:] = 0.0
        elif self.phase == "lift":
            if is_grasped:
                target = ee_pos.copy()
                target[2] = LIFT_Z
            else:
                self.phase = "descend"
                target = cube_pos.copy()
            gripper_cmd = -1.0
        else:
            target = cube_pos.copy()
            gripper_cmd = 1.0

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
        "PickSkillEnv",
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
        ee_pos   = np.asarray(obs["extra"]["ee_pos"], dtype=np.float32).reshape(-1)[:3]
        cube_pos = np.asarray(obs["extra"]["pick_cube_pos"], dtype=np.float32).reshape(-1)
        block_idx = int(raw.pick_obstacle_idx[0].item())

        print(f"  ee_start : {np.round(ee_pos, 3)}")
        print(f"  cube_pos : {np.round(cube_pos, 3)}")
        print(f"  block_idx: {block_idx}")

        t0 = time.time()
        success, _ = execute(
            env, obs, block_idx,
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
    parser = argparse.ArgumentParser(description="MPPI eval for PickSkillEnv")
    parser.add_argument("--num_episodes", type=int,   default=10)
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--max_steps",    type=int,   default=200)
    parser.add_argument("--horizon",      type=int,   default=8)
    parser.add_argument("--num_samples",  type=int,   default=256)
    parser.add_argument("--noise_std",    type=float, default=0.2)
    parser.add_argument("--lam",          type=float, default=0.05)
    args = parser.parse_args()
    run_eval(args)
