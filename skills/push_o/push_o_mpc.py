"""
Skill: push a circular disk to a goal position using MPPI in task space.

The rollout model simulates contact: when the EE XY is within the disk radius,
the disk moves with the push direction.

Control: pd_ee_delta_pose (Cartesian delta position + orientation + gripper).

Usage:
    python skills/push_o/push_o_mpc.py
    python skills/push_o/push_o_mpc.py --num_episodes 20 --seed 42
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
import envs  # noqa: F401 — registers PushO-v1

from mpc_base import MPPIBase, get_ee_pos, step_env, EE_POS_ACTION_SCALE


# Staging constants — disk is wider (radius 0.05 m) so offsets are larger.
STAGE_OFFSET   = 0.08   # m behind disk centre (along -push_dir)
STAGE_HIGH_Z   = 0.15   # travel height while repositioning in XY
PUSH_Z         = 0.02   # push at disk centre height (disk_half_thickness = 0.02)
STAGE_XY_ALIGN = 0.015  # m — switch from hover to descend when XY aligned
STAGE_ALIGN_DIST = 0.02 # m — switch to push when fully at stage target

# Disk geometry (must match PushOEnv defaults)
DISK_RADIUS = 0.05


def _circle_overlap_frac(disk_xy: np.ndarray, goal_xy: np.ndarray, r: float) -> float:
    """Analytical circle-circle overlap fraction (same formula as PushOEnv)."""
    d = float(np.linalg.norm(disk_xy - goal_xy))
    if d >= 2.0 * r:
        return 0.0
    cos_arg = np.clip(d / (2.0 * r), -1.0 + 1e-7, 1.0 - 1e-7)
    A = 2.0 * r * r * np.arccos(cos_arg) - 0.5 * d * np.sqrt(max(4.0 * r * r - d * d, 0.0))
    return float(np.clip(A / (np.pi * r * r), 0.0, 1.0))


class PushOMPPI(MPPIBase):
    """MPPI controller for pushing a disk to a goal XY position."""

    def __init__(self, goal_xyz: np.ndarray, contact_radius: float = 0.06, **kwargs):
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
        disk_xy = np.broadcast_to(state["disk_pos"][:2], (K, 2)).copy()
        target_z = float(state["target"][2])
        costs = np.zeros(K, dtype=np.float32)

        scaled = action_seqs * EE_POS_ACTION_SCALE
        for t in range(H):
            ee = ee + scaled[:, t, :]
            ee_xy = ee[:, :2]

            dist_to_disk = np.linalg.norm(ee_xy - disk_xy, axis=1)
            in_contact = dist_to_disk < self.contact_radius

            push_dir = scaled[:, t, :2]
            push_mag = np.linalg.norm(push_dir, axis=1, keepdims=True)
            push_dir_norm = np.where(push_mag > 1e-6, push_dir / push_mag, 0.0)
            disk_xy = np.where(
                in_contact[:, None],
                disk_xy + push_dir_norm * np.minimum(push_mag, dist_to_disk[:, None] + 0.01),
                disk_xy,
            )

            disk_to_goal = np.linalg.norm(disk_xy - self.goal_xy, axis=1)
            ee_to_disk   = np.linalg.norm(ee_xy - disk_xy, axis=1)
            z_dev  = (ee[:, 2] - target_z) ** 2
            ctrl   = np.sum(action_seqs[:, t, :] ** 2, axis=1)
            costs += 2.0 * disk_to_goal + 0.5 * ee_to_disk + 5.0 * z_dev + 0.01 * ctrl

        costs += 10.0 * np.linalg.norm(disk_xy - self.goal_xy, axis=1)
        return costs


# ---------------------------------------------------------------------------
# Callable skill API
# ---------------------------------------------------------------------------

def execute(
    env,
    obs: dict,
    goal_xyz: np.ndarray,
    max_steps: int = 200,
    success_overlap: float = 0.80,
    render: bool = False,
    disk_radius: float | None = None,
    **kwargs,
) -> tuple[bool, dict]:
    """
    Push the disk to goal_xyz using MPPI.
    Returns (success, latest_obs).
    """
    raw = env.unwrapped
    r = disk_radius if disk_radius is not None else float(getattr(raw, "disk_radius", DISK_RADIUS))
    controller = PushOMPPI(goal_xyz=goal_xyz, **kwargs)
    act_dim = env.action_space.shape[0]
    current_obs = obs

    phase = "hover"  # hover -> descend -> push

    for _ in range(max_steps):
        ee_pos   = get_ee_pos(current_obs)
        disk_pos = raw.disk.pose.p.cpu().numpy().reshape(-1).astype(np.float32)

        push_vec = goal_xyz[:2] - disk_pos[:2]
        push_dir = push_vec / (np.linalg.norm(push_vec) + 1e-6)
        stage_xy = disk_pos[:2] - push_dir * STAGE_OFFSET

        if phase == "hover":
            target = np.array([stage_xy[0], stage_xy[1], STAGE_HIGH_Z], dtype=np.float32)
            diff   = target - ee_pos
            delta  = np.clip(diff / EE_POS_ACTION_SCALE, -1.0, 1.0).astype(np.float32)
            if float(np.linalg.norm(ee_pos[:2] - stage_xy)) < STAGE_XY_ALIGN:
                phase = "descend"

        elif phase == "descend":
            target = np.array([stage_xy[0], stage_xy[1], PUSH_Z], dtype=np.float32)
            diff   = target - ee_pos
            delta  = np.clip(diff / EE_POS_ACTION_SCALE, -1.0, 1.0).astype(np.float32)
            if float(np.linalg.norm(ee_pos - target)) < STAGE_ALIGN_DIST:
                phase = "push"
                controller.nominal[:] = 0.0

        else:  # push — drive EE toward goal at push height; disk slides along in contact
            target = np.array([goal_xyz[0], goal_xyz[1], PUSH_Z], dtype=np.float32)
            delta  = controller.get_action({"ee_pos": ee_pos, "disk_pos": disk_pos, "target": target})

            # If the EE has arrived at the goal XY but the disk wasn't carried along
            # (overshoot or sideways knock), re-stage from the disk's current position.
            ee_at_goal = float(np.linalg.norm(ee_pos[:2] - goal_xyz[:2])) < 0.03
            if ee_at_goal and _circle_overlap_frac(disk_pos[:2], goal_xyz[:2], r) < success_overlap:
                phase = "hover"
                controller.nominal[:] = 0.0

        action = np.zeros(act_dim, dtype=np.float32)
        action[:3] = delta
        current_obs, done = step_env(env, action, render)

        # Check success immediately after the step — before a potential done-break
        # so that a terminal step that moves the disk onto the goal is caught here
        # rather than relying on the post-loop read (which may see a reset state).
        disk_pos = raw.disk.pose.p.cpu().numpy().reshape(-1).astype(np.float32)
        if _circle_overlap_frac(disk_pos[:2], goal_xyz[:2], r) >= success_overlap:
            return True, current_obs

        if done:
            break

    disk_pos = raw.disk.pose.p.cpu().numpy().reshape(-1).astype(np.float32)
    return _circle_overlap_frac(disk_pos[:2], goal_xyz[:2], r) >= success_overlap, current_obs


class PushOMPCPreviewSession:
    """One MPPI push-O step at a time — mirrors `execute` for metric lookahead."""

    def __init__(self, env, goal_xyz: np.ndarray, **kwargs):
        self.env      = env
        self.raw      = env.unwrapped
        self.goal_xyz = np.asarray(goal_xyz, dtype=np.float32).reshape(3)
        self.r        = float(getattr(self.raw, "disk_radius", DISK_RADIUS))
        self.kwargs   = kwargs
        self.act_dim  = env.action_space.shape[0]
        self.reset()

    def reset(self) -> None:
        self.controller = PushOMPPI(goal_xyz=self.goal_xyz, **self.kwargs)
        self.phase = "hover"

    def step_action(self, obs: dict) -> np.ndarray:
        goal_xyz = self.goal_xyz
        ee_pos   = get_ee_pos(obs)
        disk_pos = self.raw.disk.pose.p.cpu().numpy().reshape(-1).astype(np.float32)
        push_vec = goal_xyz[:2] - disk_pos[:2]
        push_dir = push_vec / (np.linalg.norm(push_vec) + 1e-6)
        stage_xy = disk_pos[:2] - push_dir * STAGE_OFFSET

        if self.phase == "hover":
            target = np.array([stage_xy[0], stage_xy[1], STAGE_HIGH_Z], dtype=np.float32)
            diff   = target - ee_pos
            delta  = np.clip(diff / EE_POS_ACTION_SCALE, -1.0, 1.0).astype(np.float32)
            if float(np.linalg.norm(ee_pos[:2] - stage_xy)) < STAGE_XY_ALIGN:
                self.phase = "descend"
        elif self.phase == "descend":
            target = np.array([stage_xy[0], stage_xy[1], PUSH_Z], dtype=np.float32)
            diff   = target - ee_pos
            delta  = np.clip(diff / EE_POS_ACTION_SCALE, -1.0, 1.0).astype(np.float32)
            if float(np.linalg.norm(ee_pos - target)) < STAGE_ALIGN_DIST:
                self.phase = "push"
                self.controller.nominal[:] = 0.0
        else:
            target = np.array([goal_xyz[0], goal_xyz[1], PUSH_Z], dtype=np.float32)
            delta  = self.controller.get_action({"ee_pos": ee_pos, "disk_pos": disk_pos, "target": target})
            ee_at_goal = float(np.linalg.norm(ee_pos[:2] - goal_xyz[:2])) < 0.03
            if ee_at_goal and _circle_overlap_frac(disk_pos[:2], goal_xyz[:2], self.r) < 0.90:
                self.phase = "hover"
                self.controller.nominal[:] = 0.0

        action = np.zeros(self.act_dim, dtype=np.float32)
        action[:3] = delta
        return action


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_eval(args):
    np.random.seed(args.seed)

    env = gym.make(
        "PushO-v1",
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        reconfiguration_freq=1,
        sim_backend="physx_cpu",
    )

    successes, final_overlaps = [], []

    for ep in range(args.num_episodes):
        print(f"\n=== Episode {ep + 1}/{args.num_episodes} ===")
        obs, _ = env.reset()

        raw      = env.unwrapped
        goal_pos = np.asarray(obs["extra"]["goal_pos"], dtype=np.float32).reshape(-1)
        ee_pos   = np.asarray(obs["extra"]["ee_pos"],   dtype=np.float32).reshape(-1)[:3]

        print(f"  ee_start : {np.round(ee_pos, 3)}")
        print(f"  goal_pos : {np.round(goal_pos, 3)}")

        t0 = time.time()
        success, _ = execute(
            env, obs, goal_pos,
            max_steps=args.max_steps,
            horizon=args.horizon,
            num_samples=args.num_samples,
            noise_std=args.noise_std,
            lam=args.lam,
        )
        elapsed = time.time() - t0

        disk_pos = raw.disk.pose.p.cpu().numpy().reshape(-1).astype(np.float32)
        r        = float(raw.disk_radius)
        overlap  = _circle_overlap_frac(disk_pos[:2], goal_pos[:2], r)

        successes.append(success)
        final_overlaps.append(overlap)
        print(f"  {'SUCCESS' if success else 'FAIL'}  overlap={overlap * 100:.1f}%  ({elapsed:.2f}s)")

    env.close()

    print("\n" + "=" * 50)
    print(f"Success rate    : {np.mean(successes) * 100:.1f}%")
    print(f"Mean overlap    : {np.mean(final_overlaps) * 100:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPPI eval for PushO-v1")
    parser.add_argument("--num_episodes", type=int,   default=10)
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--max_steps",    type=int,   default=3000)
    parser.add_argument("--horizon",      type=int,   default=10)
    parser.add_argument("--num_samples",  type=int,   default=512)
    parser.add_argument("--noise_std",    type=float, default=0.4)
    parser.add_argument("--lam",          type=float, default=0.04)
    args = parser.parse_args()
    run_eval(args)
