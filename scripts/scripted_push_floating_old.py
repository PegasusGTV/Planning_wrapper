#!/usr/bin/env python3
"""
Scripted autonomous policy for the unified Push environment (PushBoundary).
Uses the FloatingGripper agent with a 3-DOF velocity action: [vx, vy, omega_z].
Cube shape only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import gymnasium as gym
from mani_skill.utils.wrappers.record import RecordEpisode

import packages.Planning_wrapper.envs.floating_gripper_old  # noqa: F401 — triggers @register_agent()

from envs.push_boundary import (
    BOUNDARY_CENTER_X as BCX,
    BOUNDARY_CENTER_Y as BCY,
    BOUNDARY_HALF_X   as BHX,
    BOUNDARY_HALF_Y   as BHY,
)

MIN_X = BCX - BHX - 0.025
MAX_X = BCX + BHX + 0.025
MIN_Y = BCY - BHY - 0.025
MAX_Y = BCY + BHY + 0.025


# -----------------------------------------------------------------------------
# Observation helpers
# -----------------------------------------------------------------------------

def _get_xyz(pose_p) -> np.ndarray:
    try:
        import torch
        if isinstance(pose_p, torch.Tensor):
            pose_p = pose_p.cpu().numpy()
    except ImportError:
        pass
    return np.asarray(pose_p, dtype=np.float64).reshape(-1)[:3].copy()


def _get_yaw(pose_q) -> float:
    """Extract yaw from a SAPIEN quaternion [w, x, y, z]."""
    try:
        import torch
        if isinstance(pose_q, torch.Tensor):
            pose_q = pose_q.cpu().numpy()
    except ImportError:
        pass
    q = np.asarray(pose_q, dtype=np.float64).reshape(-1)
    r = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # scipy: [x, y, z, w]
    return float(r.as_euler("xyz")[2])


def _unwrap(env):
    e = env
    while hasattr(e, "env"):
        e = e.env
    return e


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def _cube_face_normals(angle: float) -> list[np.ndarray]:
    c, s = np.cos(angle), np.sin(angle)
    return [
        np.array([ c,  s]),
        np.array([-c, -s]),
        np.array([-s,  c]),
        np.array([ s, -c]),
    ]


def _cube_corners(
    center: np.ndarray, half_x: float, half_y: float, angle: float
) -> list[np.ndarray]:
    c, s = np.cos(angle), np.sin(angle)
    ax = np.array([ c,  s]) * half_x
    ay = np.array([-s,  c]) * half_y
    return [
        center + ax + ay,
        center - ax + ay,
        center + ax - ay,
        center - ax - ay,
    ]


def _segment_hits_obb(
    p1: np.ndarray, p2: np.ndarray,
    center: np.ndarray, half_x: float, half_y: float, angle: float,
) -> bool:
    c, s = np.cos(angle), np.sin(angle)
    R_inv = np.array([[c, s], [-s, c]])
    p1_l = R_inv @ (p1 - center)
    p2_l = R_inv @ (p2 - center)
    d = p2_l - p1_l
    tmin, tmax = 0.0, 1.0
    for i, half in enumerate([half_x, half_y]):
        lo, hi = -half, half
        if abs(d[i]) < 1e-9:
            if p1_l[i] <= lo or p1_l[i] >= hi:
                return False
        else:
            t1 = (lo - p1_l[i]) / d[i]
            t2 = (hi - p1_l[i]) / d[i]
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin >= tmax:
                return False
    return tmin < 1.0 and tmax > 0.0


# -----------------------------------------------------------------------------
# Policy
# -----------------------------------------------------------------------------

class ScriptedPushPolicy:
    """
    Produces 3-DOF velocity commands: [vx, vy, omega_z].

    omega_z is always 0 here — the gripper is a cylinder so rotation
    doesn't matter for pushing.  A future version could use it to align
    a non-circular gripper to the block face.
    """

    PUSH_STEPS          = 20
    ARRIVE_THRESH       = 0.018
    APPROACH_TIMEOUT    = 120
    XY_SPEED            = 0.20      # m/s  — velocity command magnitude
    PUSH_SPEED          = 0.20      # m/s
    PUSH_CONTACT_DIST   = 0.05
    PUSH_TIMEOUT        = 150
    PUSH_TARGET_EPS     = 0.015
    STANDOFF_DIST_RANGE = (0.06, 0.10)
    CUBE_CLEARANCE      = 0.035

    def __init__(
        self,
        rng: np.random.Generator | None = None,
        block_half_x: float = 0.025,
        block_half_y: float = 0.025,
        shape: str = "cube",
    ):
        if shape != "cube":
            raise ValueError(f"ScriptedPushPolicy only supports shape='cube', got '{shape}'")

        self.rng          = rng or np.random.default_rng()
        self.block_half_x = block_half_x
        self.block_half_y = block_half_y

        self._phase              = "pick_new"
        self._phase_step         = 0
        self._push_dir           = np.array([1.0, 0.0])
        self._push_contact_steps = 0
        self._move_tgt           = np.array([BCX, BCY])
        self._waypoint_queue: list[np.ndarray] = []
        self._standoff_dist      = 0.07
        self._block_angle        = 0.0

    def reset(self):
        self._phase              = "pick_new"
        self._phase_step         = 0
        self._push_contact_steps = 0
        self._waypoint_queue     = []
        self._block_angle        = 0.0

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------
    # def act(self, ee_xy, block_xy, block_angle=0.0):
    #     self._block_angle = block_angle
    #     self._phase_step += 1

    #     if self._phase == "pick_new":
    #         self._reposition(ee_xy, block_xy)

    #     elif self._phase == "approach":
    #         # Drain waypoints until we have one we haven't reached yet
    #         while self._waypoint_queue:
    #             self._move_tgt = self._waypoint_queue[0]
    #             if np.linalg.norm(ee_xy - self._move_tgt) < self.ARRIVE_THRESH:
    #                 self._waypoint_queue.pop(0)
    #             else:
    #                 break
    #         else:
    #             # Queue empty — head for standoff
    #             self._move_tgt = self._standoff(block_xy)

    #         at_standoff = (
    #             not self._waypoint_queue
    #             and np.linalg.norm(ee_xy - self._standoff(block_xy)) < self.ARRIVE_THRESH
    #         )
    #         if at_standoff or self._phase_step >= self.APPROACH_TIMEOUT:
    #             self._phase              = "push"
    #             self._phase_step         = 0
    #             self._push_contact_steps = 0

    #     elif self._phase == "push":
    #         if np.linalg.norm(ee_xy - block_xy) < self.PUSH_CONTACT_DIST:
    #             self._push_contact_steps += 1
    #         if self._push_contact_steps >= self.PUSH_STEPS or self._phase_step >= self.PUSH_TIMEOUT:
    #             self._reposition(ee_xy, block_xy)

    #     # # ── velocity command ──────────────────────────────────────────────────────
    #     # if self._phase == "push":
    #     #     # Always drive toward block — don't call _reposition mid-velocity-compute
    #     #     vel_xy = self._toward(ee_xy, block_xy + self._push_dir * 0.03, self.PUSH_SPEED)
    #     # else:
    #     #     vel_xy = self._toward(ee_xy, self._move_tgt, self.XY_SPEED)

    #     # # Workspace clamp
    #     # next_xy = ee_xy + vel_xy * (1.0 / 20.0)
    #     # if not (MIN_X < next_xy[0] < MAX_X):
    #     #     vel_xy[0] = 0.0
    #     # if not (MIN_Y < next_xy[1] < MAX_Y):
    #     #     vel_xy[1] = 0.0

    #     # return np.array([vel_xy[0], vel_xy[1], 0.0], dtype=np.float32)
    #     # Instead of velocity, just return the target position directly
    #     if self._phase == "push":
    #         target_xy = block_xy + self._push_dir * 0.03
    #     else:
    #         target_xy = self._move_tgt

    #     return np.array([target_xy[0], target_xy[1]], dtype=np.float32)
    
    def _toward(src: np.ndarray, dst: np.ndarray, max_delta: float) -> np.ndarray:
        d = dst - src
        dist = np.linalg.norm(d)
        if dist < 1e-6:
            return np.zeros(2)
        return d / dist * min(dist, max_delta)   # never overshoot
    
    def act(self, ee_xy, block_xy, block_angle=0.0):
        self._block_angle = block_angle
        self._phase_step += 1

        if self._phase == "pick_new":
            self._reposition(ee_xy, block_xy)

        elif self._phase == "approach":
            while self._waypoint_queue:
                self._move_tgt = self._waypoint_queue[0]
                if np.linalg.norm(ee_xy - self._move_tgt) < self.ARRIVE_THRESH:
                    self._waypoint_queue.pop(0)
                else:
                    break
            else:
                self._move_tgt = self._standoff(block_xy)

            at_standoff = (
                not self._waypoint_queue
                and np.linalg.norm(ee_xy - self._standoff(block_xy)) < self.ARRIVE_THRESH
            )
            if at_standoff or self._phase_step >= self.APPROACH_TIMEOUT:
                self._phase              = "push"
                self._phase_step         = 0
                self._push_contact_steps = 0

        elif self._phase == "push":
            if np.linalg.norm(ee_xy - block_xy) < self.PUSH_CONTACT_DIST:
                self._push_contact_steps += 1
            if self._push_contact_steps >= self.PUSH_STEPS or self._phase_step >= self.PUSH_TIMEOUT:
                self._reposition(ee_xy, block_xy)

        # ── delta position command ─────────────────────────────────────────────
        if self._phase == "push":
            target = block_xy + self._push_dir * 0.03
        else:
            target = self._move_tgt

        delta = self._toward(ee_xy, target, 0.01)
        print(f"phase={self._phase} step={self._phase_step} contact_steps={self._push_contact_steps} dist_to_block={np.linalg.norm(ee_xy - block_xy):.3f}")
        return np.array([delta[0], delta[1]], dtype=np.float32)

    # -------------------------------------------------------------------------
    # Cube repositioning
    # -------------------------------------------------------------------------

    def _reposition(self, ee_xy: np.ndarray, block_xy: np.ndarray):
        self._push_dir           = self._best_push_dir(block_xy)
        self._push_contact_steps = 0
        self._standoff_dist      = self.rng.uniform(*self.STANDOFF_DIST_RANGE)

        standoff  = self._standoff(block_xy)
        waypoints = self._safe_path(ee_xy, block_xy, standoff)

        self._phase          = "approach"
        self._phase_step     = 0
        self._waypoint_queue = waypoints
        self._move_tgt       = waypoints[0] if waypoints else standoff

    def _best_push_dir(self, block_xy: np.ndarray) -> np.ndarray:
        toward_center = np.array([BCX, BCY]) - block_xy
        normals = _cube_face_normals(self._block_angle)
        scores  = [float(np.dot(n, toward_center)) for n in normals]
        return normals[int(np.argmax(scores))].copy()

    def _standoff(self, block_xy: np.ndarray) -> np.ndarray:
        return self._clamp_xy(block_xy - self._push_dir * self._standoff_dist)

    def _safe_path(
        self, ee_xy: np.ndarray, block_xy: np.ndarray, standoff: np.ndarray
    ) -> list[np.ndarray]:
        hw_x = self.block_half_x + self.CUBE_CLEARANCE
        hw_y = self.block_half_y + self.CUBE_CLEARANCE

        def hits(a: np.ndarray, b: np.ndarray) -> bool:
            return _segment_hits_obb(a, b, block_xy, hw_x, hw_y, self._block_angle)

        if not hits(ee_xy, standoff):
            return []

        corners = [
            self._clamp_xy(c)
            for c in _cube_corners(block_xy, hw_x, hw_y, self._block_angle)
        ]

        best: list[np.ndarray] | None = None
        best_cost = float("inf")

        for c in corners:
            if not hits(ee_xy, c) and not hits(c, standoff):
                cost = float(np.linalg.norm(c - ee_xy) + np.linalg.norm(standoff - c))
                if cost < best_cost:
                    best_cost, best = cost, [c]

        if best is not None:
            return best

        for i, c1 in enumerate(corners):
            for j, c2 in enumerate(corners):
                if i == j:
                    continue
                if not hits(ee_xy, c1) and not hits(c1, c2) and not hits(c2, standoff):
                    cost = float(
                        np.linalg.norm(c1 - ee_xy)
                        + np.linalg.norm(c2 - c1)
                        + np.linalg.norm(standoff - c2)
                    )
                    if cost < best_cost:
                        best_cost, best = cost, [c1, c2]

        return best or []

    def _clamp_xy(self, xy: np.ndarray) -> np.ndarray:
        return np.array([np.clip(xy[0], MIN_X, MAX_X), np.clip(xy[1], MIN_Y, MAX_Y)])

    @staticmethod
    def _toward(src: np.ndarray, dst: np.ndarray, speed: float) -> np.ndarray:
        d = dst - src
        dist = np.linalg.norm(d)
        return d / dist * speed if dist > 1e-6 else np.zeros(2)


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

def run(args: argparse.Namespace):
    env = gym.make(
        "PushBoundary",
        obs_mode="state",
        control_mode="floating_vel",
        render_mode="all",
        sim_backend="auto",
        shape=args.shape,
        num_extra_blocks=args.num_extra_blocks,
        robot_uids="floating_gripper",
    )
    env = RecordEpisode(
        env,
        output_dir=args.record_dir,
        save_trajectory=True,
        save_video=True,
        video_fps=20,
        trajectory_name="scripted",
    )
    base_env = _unwrap(env)

    policy = ScriptedPushPolicy(
        rng=np.random.default_rng(args.seed),
        block_half_x=base_env.block_dims.half_x,
        block_half_y=base_env.block_dims.half_y,
        shape=args.shape,
    )

    obs, _ = env.reset(seed=args.seed)
    policy.reset()

    total_steps = 0
    episode     = 0
    alive_steps = 0

    print(
        f"Running  shape={args.shape}  extra_blocks={args.num_extra_blocks}  "
        f"steps={args.num_steps}"
    )
    print(f"  block_dims = {base_env.block_dims}")
    print(f"  action space = {env.action_space}")

    while total_steps < args.num_steps:
        ee_xyz    = _get_xyz(base_env.agent.tcp.pose.p)
        block_xyz = _get_xyz(base_env.block.pose.p)
        block_yaw = _get_yaw(base_env.block.pose.q)

        # 3-element action: [vx, vy, omega_z]
        # No z correction needed — the velocity controller holds z fixed
        action = policy.act(ee_xyz[:2], block_xyz[:2], block_yaw)

        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1
        alive_steps += 1

        if terminated or truncated:
            episode += 1
            reason = "block escaped" if terminated else "truncated"
            print(f"  Episode {episode:3d}  {alive_steps:5d} steps  ({reason})")
            obs, _ = env.reset()
            policy.reset()
            alive_steps = 0

    env.close()
    print("Done.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--shape",            type=str, default="cube", choices=["cube"])
    p.add_argument("--num_extra_blocks", type=int, default=0)
    p.add_argument("--num_steps",        type=int, default=500)
    p.add_argument("--seed",             type=int, default=None)
    p.add_argument("--record_dir",       type=str, default="demos/PushBoundary/scripted")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())