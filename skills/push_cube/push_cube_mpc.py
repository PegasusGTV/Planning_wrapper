"""
Skill: push a cube to a goal position using MPPI in task space.

The rollout model simulates contact: when the EE is close to the cube,
the cube moves with the push direction.

Control: pd_ee_delta_pose (Cartesian delta position + orientation + gripper).

Usage:
    python skills/push_cube/push_cube_mpc.py
    python skills/push_cube/push_cube_mpc.py --num_episodes 20 --seed 42
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
import envs  # registers PushCube-WithObstacles-v1

from mpc_base import MPPIBase, get_ee_pos, step_env, EE_POS_ACTION_SCALE


# Staging: line EE up behind the cube (opposite goal), approaching from above
# to avoid knocking the cube while moving into position.
STAGE_OFFSET = 0.05      # m behind cube (along -push_dir)
STAGE_HIGH_Z = 0.15      # travel height while repositioning in XY
PUSH_Z = 0.03            # push at cube centre height (cube half_size ~0.02)
STAGE_XY_ALIGN = 0.015   # m — switch from hover to descend when XY aligned
STAGE_ALIGN_DIST = 0.02  # m — switch to push when fully at stage target


class PushCubeMPPI(MPPIBase):
    """MPPI controller for pushing a cube to a goal XY position."""

    def __init__(self, goal_xyz: np.ndarray, contact_radius: float = 0.04, **kwargs):
        kwargs.setdefault("horizon", 10)
        kwargs.setdefault("num_samples", 512)
        kwargs.setdefault("noise_std", 0.4)
        kwargs.setdefault("lam", 0.04)
        kwargs.setdefault("action_clip", 0.5)
        super().__init__(action_dim=3, **kwargs)
        self.goal_xy = goal_xyz[:2].copy()
        self.contact_radius = contact_radius

    def rollout_costs(self, state: dict, action_seqs: np.ndarray) -> np.ndarray:
        K, H, _ = action_seqs.shape
        ee = np.broadcast_to(state["ee_pos"], (K, 3)).copy()
        cube_xy = np.broadcast_to(state["cube_pos"][:2], (K, 2)).copy()
        target_z = float(state["target"][2])
        costs = np.zeros(K, dtype=np.float32)

        scaled = action_seqs * EE_POS_ACTION_SCALE
        for t in range(H):
            ee = ee + scaled[:, t, :]
            ee_xy = ee[:, :2]

            dist_to_cube = np.linalg.norm(ee_xy - cube_xy, axis=1)
            in_contact = dist_to_cube < self.contact_radius

            push_dir = scaled[:, t, :2]
            push_mag = np.linalg.norm(push_dir, axis=1, keepdims=True)
            push_dir_norm = np.where(push_mag > 1e-6, push_dir / push_mag, 0.0)
            cube_xy = np.where(
                in_contact[:, None],
                cube_xy + push_dir_norm * np.minimum(push_mag, dist_to_cube[:, None] + 0.01),
                cube_xy,
            )

            cube_to_goal = np.linalg.norm(cube_xy - self.goal_xy, axis=1)
            ee_to_cube = np.linalg.norm(ee_xy - cube_xy, axis=1)
            z_dev = (ee[:, 2] - target_z) ** 2
            ctrl = np.sum(action_seqs[:, t, :] ** 2, axis=1)
            costs += 2.0 * cube_to_goal + 0.5 * ee_to_cube + 5.0 * z_dev + 0.01 * ctrl

        costs += 10.0 * np.linalg.norm(cube_xy - self.goal_xy, axis=1)
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
    success_threshold: float = 0.05,
    render: bool = False,
    **kwargs,
) -> tuple[bool, dict]:
    """
    Push obstacle[block_idx] to goal_xyz using MPPI.
    Returns (success, latest_obs).
    """
    raw = env.unwrapped
    obstacle = raw.obstacles[block_idx]
    controller = PushCubeMPPI(goal_xyz=goal_xyz, **kwargs)
    act_dim = env.action_space.shape[0]
    current_obs = obs

    phase = "hover"  # hover -> descend -> push

    for _ in range(max_steps):
        ee_pos = get_ee_pos(current_obs)
        cube_pos = obstacle.pose.p.cpu().numpy().reshape(-1).astype(np.float32)

        if float(np.linalg.norm(cube_pos[:2] - goal_xyz[:2])) < success_threshold:
            return True, current_obs

        push_vec = goal_xyz[:2] - cube_pos[:2]
        push_dir = push_vec / (np.linalg.norm(push_vec) + 1e-6)
        stage_xy = cube_pos[:2] - push_dir * STAGE_OFFSET

        if phase == "hover":
            # Travel at safe height to staging XY (avoids knocking the cube).
            target = np.array([stage_xy[0], stage_xy[1], STAGE_HIGH_Z], dtype=np.float32)
            diff = target - ee_pos
            delta = np.clip(diff / EE_POS_ACTION_SCALE, -1.0, 1.0).astype(np.float32)

            if float(np.linalg.norm(ee_pos[:2] - stage_xy)) < STAGE_XY_ALIGN:
                phase = "descend"

        elif phase == "descend":
            # Drop down to push height at the staging XY.
            target = np.array([stage_xy[0], stage_xy[1], PUSH_Z], dtype=np.float32)
            diff = target - ee_pos
            delta = np.clip(diff / EE_POS_ACTION_SCALE, -1.0, 1.0).astype(np.float32)

            if float(np.linalg.norm(ee_pos - target)) < STAGE_ALIGN_DIST:
                phase = "push"
                controller.nominal[:] = 0.0

        else:  # push — drive EE toward goal at push height; cube rides along in contact
            target = np.array([goal_xyz[0], goal_xyz[1], PUSH_Z], dtype=np.float32)
            delta = controller.get_action({"ee_pos": ee_pos, "cube_pos": cube_pos, "target": target})

        action = np.zeros(act_dim, dtype=np.float32)
        action[:3] = delta
        current_obs, done = step_env(env, action, render)
        if done:
            break

    cube_pos = obstacle.pose.p.cpu().numpy().reshape(-1).astype(np.float32)
    return bool(np.linalg.norm(cube_pos[:2] - goal_xyz[:2]) < success_threshold), current_obs


class PushCubeMPCPreviewSession:
    """One MPPI push-cube step at a time — mirrors `execute` for metric lookahead."""

    def __init__(self, env, block_idx: int, goal_xyz: np.ndarray, **kwargs):
        self.env = env
        self.raw = env.unwrapped
        self.obstacle = self.raw.obstacles[block_idx]
        self.goal_xyz = np.asarray(goal_xyz, dtype=np.float32).reshape(3)
        self.kwargs = kwargs
        self.act_dim = env.action_space.shape[0]
        self.reset()

    def reset(self) -> None:
        self.controller = PushCubeMPPI(goal_xyz=self.goal_xyz, **self.kwargs)
        self.phase = "hover"

    def step_action(self, obs: dict) -> np.ndarray:
        obstacle = self.obstacle
        goal_xyz = self.goal_xyz
        ee_pos = get_ee_pos(obs)
        cube_pos = obstacle.pose.p.cpu().numpy().reshape(-1).astype(np.float32)
        push_vec = goal_xyz[:2] - cube_pos[:2]
        push_dir = push_vec / (np.linalg.norm(push_vec) + 1e-6)
        stage_xy = cube_pos[:2] - push_dir * STAGE_OFFSET

        if self.phase == "hover":
            target = np.array([stage_xy[0], stage_xy[1], STAGE_HIGH_Z], dtype=np.float32)
            diff = target - ee_pos
            delta = np.clip(diff / EE_POS_ACTION_SCALE, -1.0, 1.0).astype(np.float32)
            if float(np.linalg.norm(ee_pos[:2] - stage_xy)) < STAGE_XY_ALIGN:
                self.phase = "descend"
        elif self.phase == "descend":
            target = np.array([stage_xy[0], stage_xy[1], PUSH_Z], dtype=np.float32)
            diff = target - ee_pos
            delta = np.clip(diff / EE_POS_ACTION_SCALE, -1.0, 1.0).astype(np.float32)
            if float(np.linalg.norm(ee_pos - target)) < STAGE_ALIGN_DIST:
                self.phase = "push"
                self.controller.nominal[:] = 0.0
        else:
            target = np.array([goal_xyz[0], goal_xyz[1], PUSH_Z], dtype=np.float32)
            delta = self.controller.get_action({"ee_pos": ee_pos, "cube_pos": cube_pos, "target": target})

        action = np.zeros(self.act_dim, dtype=np.float32)
        action[:3] = delta
        return action


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_eval(args):
    np.random.seed(args.seed)

    env = gym.make(
        "PushCube-WithObstacles-v1",
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

        raw = env.unwrapped
        goal_cube_pos = np.asarray(obs["extra"]["goal_cube_pos"], dtype=np.float32).reshape(-1)
        goal_pos      = np.asarray(obs["extra"]["goal_pos"],      dtype=np.float32).reshape(-1)
        ee_pos        = np.asarray(obs["extra"]["ee_pos"],        dtype=np.float32).reshape(-1)[:3]
        block_idx = int(raw.goal_obstacle_idx[0].item())

        print(f"  ee_start     : {np.round(ee_pos, 3)}")
        print(f"  goal_cube_pos: {np.round(goal_cube_pos, 3)}")
        print(f"  goal_pos     : {np.round(goal_pos, 3)}")
        print(f"  block_idx    : {block_idx}")

        t0 = time.time()
        success, final_obs = execute(
            env, obs, block_idx, goal_pos,
            max_steps=args.max_steps,
            horizon=args.horizon,
            num_samples=args.num_samples,
            noise_std=args.noise_std,
            lam=args.lam,
        )
        elapsed = time.time() - t0

        cube_pos = raw.obstacles[block_idx].pose.p.cpu().numpy().reshape(-1).astype(np.float32)
        dist = float(np.linalg.norm(cube_pos[:2] - goal_pos[:2]))

        successes.append(success)
        final_dists.append(dist)
        print(f"  {'SUCCESS' if success else 'FAIL'}  dist={dist * 100:.1f} cm  ({elapsed:.2f}s)")

    env.close()

    print("\n" + "=" * 50)
    print(f"Success rate   : {np.mean(successes) * 100:.1f}%")
    print(f"Mean final dist: {np.mean(final_dists) * 100:.1f} cm")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPPI eval for PushCube-WithObstacles-v1")
    parser.add_argument("--num_episodes", type=int,   default=10)
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--max_steps",    type=int,   default=200)
    parser.add_argument("--horizon",      type=int,   default=10)
    parser.add_argument("--num_samples",  type=int,   default=512)
    parser.add_argument("--noise_std",    type=float, default=0.4)
    parser.add_argument("--lam",          type=float, default=0.04)
    args = parser.parse_args()
    run_eval(args)
