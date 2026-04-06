#!/usr/bin/env python3
"""
Demonstration collection: face-approach push with collision-free repositioning.

Gripper geometry (from URDF)
-----------------------------
  Cylinder: radius = 0.01 m, length = 0.12 m (vertical).
  In the 2-D XY plane this is a circle of radius GRIPPER_RADIUS = 0.01 m.

Block geometry
--------------
  Cube OBB: half-extents (hx, hy) with arbitrary yaw angle.

Collision model
---------------
  Moving gripper centre along a segment A→B sweeps a capsule of radius r.
  We conservatively approximate this as checking whether the segment
  intersects an OBB inflated by r (i.e. half-extents hx+r, hy+r).
  This is exact along the flat faces of the Minkowski sum and conservative
  (over-estimates) only at the four corners — always safe, never misses a hit.

Repositioning path planning
----------------------------
  When the direct path gripper→standoff would clip the block the policy
  routes via "expanded corners": the four corners of the OBB inflated by
  (GRIPPER_RADIUS + CORNER_CLEARANCE).  These are guaranteed collision-free
  waypoints.  We try every 1-corner and 2-corner combination and pick the
  shortest valid route.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import sapien
import gymnasium as gym
from mani_skill.utils.wrappers.record import RecordEpisode

import envs.floating_gripper  # noqa: F401

from envs.push_boundary import (
    BOUNDARY_CENTER_X as BCX,
    BOUNDARY_CENTER_Y as BCY,
    BOUNDARY_HALF_X   as BHX,
    BOUNDARY_HALF_Y   as BHY,
)

# ---------------------------------------------------------------------------
# Shape constants
# ---------------------------------------------------------------------------

GRIPPER_RADIUS   = 0.01
CORNER_CLEARANCE = 0.015

# Standoff distance: randomised each time a face is selected
STANDOFF_DIST_MIN = 0.03
STANDOFF_DIST_MAX = 0.07
STANDOFF_DIST_MU  = 0.05   # Gaussian centre
STANDOFF_DIST_STD = 0.01

# Corner jitter: each corner's pad radius is the base pad + U[0, CORNER_JITTER]
CORNER_JITTER     = 0.01

# ---------------------------------------------------------------------------
# Gripper spawn workspace
# ---------------------------------------------------------------------------

WALL_MARGIN = 0.1
SPAWN_MIN_X = BCX - BHX + WALL_MARGIN
SPAWN_MAX_X = BCX + BHX - WALL_MARGIN
SPAWN_MIN_Y = BCY - BHY + WALL_MARGIN
SPAWN_MAX_Y = BCY + BHY - WALL_MARGIN

GRIPPER_SPAWN_STD = 0.045
SPAWN_CLEARANCE   = GRIPPER_RADIUS + 0.005   # never spawn touching block
SPAWN_MAX_DIST    = 0.1                      # max distance from block surface at spawn


# ---------------------------------------------------------------------------
# 2-D collision primitives
# ---------------------------------------------------------------------------

def _circle_hits_obb(circle_xy: np.ndarray, r: float,
                     block_xy: np.ndarray, hx: float, hy: float,
                     angle: float) -> bool:
    """
    True if a circle (centre, radius r) overlaps an OBB.
    Uses the standard closest-point-on-OBB test.
    """
    c, s = np.cos(angle), np.sin(angle)
    dx   = circle_xy[0] - block_xy[0]
    dy   = circle_xy[1] - block_xy[1]
    # Project into OBB local frame (R^T · delta)
    lx   =  c * dx + s * dy
    ly   = -s * dx + c * dy
    # Closest point on OBB to circle centre
    cx   = np.clip(lx, -hx, hx)
    cy   = np.clip(ly, -hy, hy)
    return (lx - cx) ** 2 + (ly - cy) ** 2 < r * r


def _segment_hits_obb(p1: np.ndarray, p2: np.ndarray,
                      center: np.ndarray, hx: float, hy: float,
                      angle: float) -> bool:
    """
    Slab-method segment vs OBB intersection test (exact for the OBB rectangle).
    """
    c, s   = np.cos(angle), np.sin(angle)
    R_inv  = np.array([[c, s], [-s, c]])
    p1_l   = R_inv @ (p1 - center)
    p2_l   = R_inv @ (p2 - center)
    d      = p2_l - p1_l
    tmin, tmax = 0.0, 1.0
    for i, half in enumerate([hx, hy]):
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


def _capsule_hits_obb(p1: np.ndarray, p2: np.ndarray, r: float,
                      block_xy: np.ndarray, hx: float, hy: float,
                      angle: float) -> bool:
    """
    True if a capsule (segment p1→p2, radius r) overlaps the OBB.

    Implementation: check the segment against the Minkowski-expanded OBB
    (half-extents hx+r, hy+r).  Conservative at corners but never misses
    a real collision along the flat faces.  Also verify both endpoints
    directly with _circle_hits_obb for extra safety at corners.
    """
    # Expanded OBB covers the flat-face part of the Minkowski sum
    hits_exp = _segment_hits_obb(p1, p2, block_xy, hx + r, hy + r, angle)
    # Endpoint checks catch corner regions the slab test might miss
    ep1 = _circle_hits_obb(p1, r, block_xy, hx, hy, angle)
    ep2 = _circle_hits_obb(p2, r, block_xy, hx, hy, angle)
    return hits_exp or ep1 or ep2


# ---------------------------------------------------------------------------
# Path-planning helpers
# ---------------------------------------------------------------------------

def _jittered_corners(block_xy: np.ndarray, hx: float, hy: float,
                      angle: float, base_pad: float,
                      rng: np.random.Generator) -> list[np.ndarray]:
    """
    Four corners of the OBB, each displaced outward by a randomly sampled
    pad (base_pad + U[0, CORNER_JITTER]).  Guarantees each corner clears the
    block while adding variety to the repositioning paths.
    """
    c, s = np.cos(angle), np.sin(angle)
    corners = []
    for sx in (+1, -1):
        for sy in (+1, -1):
            pad = base_pad + rng.uniform(0.0, CORNER_JITTER)
            lx  = (hx + pad) * sx
            ly  = (hy + pad) * sy
            wx  = block_xy[0] + c * lx - s * ly
            wy  = block_xy[1] + s * lx + c * ly
            corners.append(np.array([wx, wy]))
    return corners


def _plan_approach(ee_xy: np.ndarray, standoff: np.ndarray,
                   block_xy: np.ndarray, hx: float, hy: float,
                   angle: float,
                   rng: np.random.Generator,
                   max_attempts: int = 8) -> list[np.ndarray]:
    """
    Return a collision-free waypoint list: ee_xy → waypoints → standoff.

    Corners are jittered randomly each attempt.  If no valid path is found
    after *max_attempts* corner resamples, returns the nearest-corner fallback.

    The nudged start skips past the contact zone when ee_xy is touching the block.
    """
    r        = GRIPPER_RADIUS
    base_pad = GRIPPER_RADIUS + CORNER_CLEARANCE
    NUDGE    = r * 2.5

    def hits_from_ee(a: np.ndarray, b: np.ndarray) -> bool:
        d    = b - a
        dist = np.linalg.norm(d)
        if dist < 1e-6:
            return False
        a_check = a + d / dist * min(NUDGE, dist * 0.4)
        return _capsule_hits_obb(a_check, b, r, block_xy, hx, hy, angle)

    def hits(a: np.ndarray, b: np.ndarray) -> bool:
        return _capsule_hits_obb(a, b, r, block_xy, hx, hy, angle)

    # Direct path — no corners needed
    if not hits_from_ee(ee_xy, standoff):
        return []

    for attempt in range(max_attempts):
        corners   = _jittered_corners(block_xy, hx, hy, angle, base_pad, rng)
        best: list[np.ndarray] | None = None
        best_cost = float("inf")

        # 1-corner paths
        for c in corners:
            if not hits_from_ee(ee_xy, c) and not hits(c, standoff):
                cost = np.linalg.norm(c - ee_xy) + np.linalg.norm(standoff - c)
                if cost < best_cost:
                    best_cost, best = cost, [c]

        if best is not None:
            return best

        # 2-corner paths
        for i, c1 in enumerate(corners):
            for j, c2 in enumerate(corners):
                if i == j:
                    continue
                if (not hits_from_ee(ee_xy, c1)
                        and not hits(c1, c2)
                        and not hits(c2, standoff)):
                    cost = (np.linalg.norm(c1 - ee_xy)
                          + np.linalg.norm(c2 - c1)
                          + np.linalg.norm(standoff - c2))
                    if cost < best_cost:
                        best_cost, best = cost, [c1, c2]

        if best is not None:
            return best

    # Fallback after all attempts: go via nearest corner
    corners  = _jittered_corners(block_xy, hx, hy, angle, base_pad, rng)
    nearest  = min(corners, key=lambda c: np.linalg.norm(c - ee_xy))
    return [nearest]


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _obb_support(hx: float, hy: float, angle: float,
                 direction: np.ndarray) -> float:
    ax = np.array([ np.cos(angle),  np.sin(angle)])
    ay = np.array([-np.sin(angle),  np.cos(angle)])
    return (hx * abs(float(np.dot(direction, ax)))
          + hy * abs(float(np.dot(direction, ay))))


def _wall_dist(direction: np.ndarray) -> float:
    dx, dy = float(direction[0]), float(direction[1])
    candidates = []
    if dx >  1e-9: candidates.append((SPAWN_MAX_X - BCX) / dx)
    if dx < -1e-9: candidates.append((SPAWN_MIN_X - BCX) / dx)
    if dy >  1e-9: candidates.append((SPAWN_MAX_Y - BCY) / dy)
    if dy < -1e-9: candidates.append((SPAWN_MIN_Y - BCY) / dy)
    return min(c for c in candidates if c > 0)


def _truncated_gaussian(rng: np.random.Generator,
                        mu: float, sigma: float,
                        lo: float, hi: float) -> float:
    from scipy.stats import truncnorm
    a, b = (lo - mu) / sigma, (hi - mu) / sigma
    if a >= b:
        return rng.uniform(lo, hi)
    return float(truncnorm.rvs(a, b, loc=mu, scale=sigma,
                               random_state=rng.integers(2**31)))


def _yaw_to_sapien_quat(yaw: float):
    q = Rotation.from_euler("z", yaw).as_quat()
    return [q[3], q[0], q[1], q[2]]


def _get_xyz(pose_p) -> np.ndarray:
    try:
        import torch
        if isinstance(pose_p, torch.Tensor):
            pose_p = pose_p.cpu().numpy()
    except ImportError:
        pass
    return np.asarray(pose_p, dtype=np.float64).reshape(-1)[:3].copy()


def _get_xyzw_quat(pose_q) -> np.ndarray:
    """Convert SAPIEN [w,x,y,z] quaternion to scipy [x,y,z,w]."""
    try:
        import torch
        if isinstance(pose_q, torch.Tensor):
            pose_q = pose_q.cpu().numpy()
    except ImportError:
        pass
    q = np.asarray(pose_q, dtype=np.float64).reshape(-1)
    return np.array([q[1], q[2], q[3], q[0]])  # [x,y,z,w] for scipy


def _unwrap(env):
    e = env
    while hasattr(e, "env"):
        e = e.env
    return e


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class FaceApproachPushPolicy:
    """
    Two-phase policy per episode:

      APPROACH  Follow a collision-free waypoint path to a standoff point
                placed behind the chosen block face.

      PUSH      Drive in the direction standoff → contact_point, which
                defines the push angle.  Standoff and contact are sampled
                independently, so the push is rarely perpendicular to the
                face — it naturally varies in angle.

    Geometry
    --------
    For a chosen face:
      contact_point = face_surface_centre + tangent * contact_offset
      standoff      = block_centre - normal * standoff_dist
                      + tangent * standoff_offset          (independent offset)
      push_dir      = normalise(contact_point - standoff)

    Because standoff_offset and contact_offset are sampled independently,
    the line standoff→contact is generally NOT parallel to the face normal,
    giving a spread of push angles without any extra parameters.
    """

    MAX_STEP              = 0.015
    ARRIVE_THRESH         = 0.015
    RESAMPLE_INTERVAL_MIN = 20
    RESAMPLE_INTERVAL_MAX = 40
    CONTACT_DIST          = 0.04   # metres — gripper considered in contact below this

    def __init__(self):
        self._phase             : str              = "approach"
        self._push_dir          : np.ndarray       = np.array([1.0, 0.0])
        self._face_index        : int              = 0
        self._contact_offset    : float            = 0.0  # lateral on face surface
        self._standoff_offset   : float            = 0.0  # lateral behind face
        self._standoff_dist     : float            = STANDOFF_DIST_MU
        self._standoff_pt       : np.ndarray       = np.zeros(2)
        self._contact_pt        : np.ndarray       = np.zeros(2)
        self._waypoints         : list[np.ndarray] = []
        self._hx                : float            = 0.025
        self._hy                : float            = 0.025
        self._block_yaw         : float            = 0.0
        self._rng               : np.random.Generator | None = None
        self._step              : int              = 0
        self._next_resample     : int              = 0
        self._style             : str              = "face"

    def _draw_resample_interval(self) -> int:
        return int(self._rng.integers(self.RESAMPLE_INTERVAL_MIN,
                                      self.RESAMPLE_INTERVAL_MAX + 1))

    def reset(self, ee_xy: np.ndarray, block_xy: np.ndarray,
              block_yaw: float, hx: float, hy: float,
              rng: np.random.Generator,
              initial_face: int | None = None) -> np.ndarray:
        self._hx, self._hy = hx, hy
        self._block_yaw    = block_yaw
        self._rng          = rng
        self._step         = 0
        push_dir = self._new_face(ee_xy, block_xy, block_yaw,
                                  prev_face=None, forced_face=initial_face)
        self._next_resample = self._draw_resample_interval()
        return push_dir

    # ── geometry helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _face_normal(face_index: int, yaw: float) -> np.ndarray:
        c, s = np.cos(yaw), np.sin(yaw)
        return [np.array([ c,  s]),
                np.array([-c, -s]),
                np.array([-s,  c]),
                np.array([ s, -c])][face_index]

    @staticmethod
    def _face_tangent(face_index: int, yaw: float) -> np.ndarray:
        c, s = np.cos(yaw), np.sin(yaw)
        return [np.array([-s,  c]),
                np.array([-s,  c]),
                np.array([ c,  s]),
                np.array([ c,  s])][face_index]

    def _face_normal_extent(self, face_index: int) -> float:
        """Half-extent of the block in the face-normal direction."""
        return self._hx if face_index < 2 else self._hy

    def _face_tangent_extent(self, face_index: int) -> float:
        """Half-extent of the block along the face (tangent direction)."""
        return self._hy if face_index < 2 else self._hx

    def _in_contact(self, ee_xy: np.ndarray, block_xy: np.ndarray) -> bool:
        return bool(np.linalg.norm(ee_xy - block_xy) < self.CONTACT_DIST)

    def _compute_geometry(self, block_xy: np.ndarray,
                          face_index: int, yaw: float):
        """
        Compute standoff point, contact point, and push direction for the
        current face_index, offsets, and block pose.

        Returns (standoff_pt, contact_pt, push_dir).
        """
        normal  = self._face_normal(face_index, yaw)
        tangent = self._face_tangent(face_index, yaw)
        n_ext   = self._face_normal_extent(face_index)

        # Contact point: on the face surface, shifted laterally
        contact_pt = (block_xy
                      + normal  * n_ext
                      + tangent * self._contact_offset)

        # Standoff point: behind the face, independently shifted laterally
        standoff_pt = (block_xy
                       - normal  * self._standoff_dist
                       + tangent * self._standoff_offset)

        # Push direction: the line from standoff → contact defines the angle
        d    = contact_pt - standoff_pt
        dist = np.linalg.norm(d)
        push_dir = d / dist if dist > 1e-6 else normal.copy()

        return standoff_pt, contact_pt, push_dir

    # ── face selection ───────────────────────────────────────────────────────

    def _sample_contact_offset(self, face_index: int) -> float:
        """Lateral offset of the contact point on the face surface."""
        half_ext   = self._face_tangent_extent(face_index)
        max_offset = half_ext * 0.65
        return _truncated_gaussian(self._rng, mu=0.0, sigma=max_offset * 0.45,
                                   lo=-max_offset, hi=max_offset)

    def _sample_standoff_offset(self, face_index: int,
                                contact_offset: float) -> float:
        """
        Sample the standoff's lateral offset as contact_offset + delta, where
        delta ~ N(0, σ) truncated to keep the standoff within ±1.5× half-extent.
        When delta=0 the standoff is directly behind the contact point, giving
        a perfectly straight (face-normal) push.  σ controls angle spread.
        """
        half_ext  = self._face_tangent_extent(face_index)
        max_total = half_ext * 1.5
        lo        = -max_total - contact_offset
        hi        =  max_total - contact_offset
        sigma     = half_ext * 0.4   # tighter → straighter on average
        delta     = _truncated_gaussian(self._rng, mu=0.0, sigma=sigma,
                                        lo=lo, hi=hi)
        return contact_offset + delta

    def _new_face(self, ee_xy: np.ndarray, block_xy: np.ndarray,
                  block_yaw: float, prev_face: int | None,
                  forced_face: int | None = None) -> np.ndarray:
        self._style      = self._rng.choice(["face", "directed"])
        new_face_index   = forced_face if forced_face is not None \
                           else int(self._rng.integers(4))
        same_face        = (new_face_index == prev_face)
        in_contact       = self._in_contact(ee_xy, block_xy)
        self._face_index = new_face_index

        # Sample all random parameters
        self._standoff_dist   = _truncated_gaussian(
            self._rng, mu=STANDOFF_DIST_MU, sigma=STANDOFF_DIST_STD,
            lo=STANDOFF_DIST_MIN, hi=STANDOFF_DIST_MAX)
        self._contact_offset  = self._sample_contact_offset(self._face_index)
        self._standoff_offset = self._sample_standoff_offset(self._face_index,
                                                             self._contact_offset)

        # Compute geometry at current block pose
        standoff_pt, contact_pt, push_dir = self._compute_geometry(
            block_xy, self._face_index, block_yaw)
        self._standoff_pt = standoff_pt
        self._contact_pt  = contact_pt
        self._push_dir    = push_dir.copy()

        push_angle = float(np.degrees(np.arctan2(push_dir[1], push_dir[0])))

        if same_face and in_contact:
            self._phase     = "push"
            self._waypoints = []
            print(f"  [{self._style:8s}] face {self._face_index} (same+contact) "
                  f"→ push  angle={push_angle:+.1f}°  "
                  f"c_off={self._contact_offset:+.3f}  "
                  f"s_off={self._standoff_offset:+.3f}")
        else:
            waypoints = _plan_approach(ee_xy, standoff_pt,
                                       block_xy, self._hx, self._hy, block_yaw,
                                       self._rng)
            self._waypoints = waypoints + [standoff_pt]
            self._phase = "approach"
            n      = len(self._waypoints)
            reason = "same/no-contact" if same_face else "new face"
            print(f"  [{self._style:8s}] face {self._face_index} ({reason})  "
                  f"angle={push_angle:+.1f}°  "
                  f"c_off={self._contact_offset:+.3f}  "
                  f"s_off={self._standoff_offset:+.3f}  "
                  f"standoff={self._standoff_dist:.3f}m  "
                  f"approach: {'direct' if n == 1 else f'{n-1} corner(s)'}")

        return self._push_dir

    # ── step ─────────────────────────────────────────────────────────────────

    def act(self, ee_xy: np.ndarray, block_xy: np.ndarray,
            block_yaw: float | None = None) -> np.ndarray:
        if block_yaw is not None:
            self._block_yaw = block_yaw

        # Timer only advances during push — prevents resample mid-approach
        if self._phase == "push":
            self._step += 1

            # Lost contact: gripper drifted more than 2 full side-lengths from
            # the block centre while actively pushing → reposition immediately
            box_length    = 2.0 * max(self._hx, self._hy)
            lost_contact  = np.linalg.norm(ee_xy - block_xy) > 1.5 * box_length
            if lost_contact:
                print(f"    [lost contact] dist={np.linalg.norm(ee_xy - block_xy):.3f}m "
                      f"> {1.5 * box_length:.3f}m → reposition")
                self._new_face(ee_xy, block_xy, self._block_yaw,
                               prev_face=self._face_index)
                self._next_resample = self._step + self._draw_resample_interval()
            elif self._step >= self._next_resample:
                self._new_face(ee_xy, block_xy, self._block_yaw,
                               prev_face=self._face_index)
                self._next_resample = self._step + self._draw_resample_interval()

        # Style B: recompute push geometry from live block yaw each step
        if self._style == "directed":
            _, _, push_dir = self._compute_geometry(
                block_xy, self._face_index, self._block_yaw)
        else:
            push_dir = self._push_dir

        if self._phase == "approach":
            while (self._waypoints
                   and np.linalg.norm(ee_xy - self._waypoints[0]) < self.ARRIVE_THRESH):
                self._waypoints.pop(0)

            if not self._waypoints:
                self._phase = "push"
                print("    → push phase")
                target = ee_xy
            else:
                target = self._waypoints[0]
        else:
            target = ee_xy + push_dir * self.MAX_STEP

        delta = target - ee_xy
        dist  = np.linalg.norm(delta)
        if dist < 1e-6:
            return np.zeros(2, dtype=np.float32)
        return (delta / dist * min(dist, self.MAX_STEP)).astype(np.float32)


# ---------------------------------------------------------------------------
# Gripper teleport
# ---------------------------------------------------------------------------

def _set_gripper_xy(base_env, gx: float, gy: float, gripper_z: float) -> None:
    """
    Teleport the floating gripper to world position (gx, gy).
    Tries qpos (joint-space) first, falls back to set_pose.
    """
    robot = base_env.agent.robot
    try:
        import torch
        n_dof = robot.dof if hasattr(robot, "dof") else len(robot.get_qpos())
        if hasattr(n_dof, "item"):
            n_dof = int(n_dof.item())
        if n_dof > 0:
            qpos    = torch.zeros(n_dof, dtype=torch.float32)
            qpos[0] = gx - BCX
            qpos[1] = gy - BCY
            robot.set_qpos(qpos.unsqueeze(0).expand(base_env.num_envs, -1))
            robot.set_qvel(torch.zeros_like(qpos).unsqueeze(0)
                           .expand(base_env.num_envs, -1))
            return
    except Exception as e:
        print(f"  [INFO] qpos teleport failed ({e}), falling back to set_pose")

    robot.set_pose(sapien.Pose(p=[gx, gy, gripper_z], q=[1, 0, 0, 0]))
    try:
        robot.set_velocity([0, 0, 0])
        robot.set_angular_velocity([0, 0, 0])
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Custom initialisation
# ---------------------------------------------------------------------------

def custom_init(base_env, rng: np.random.Generator,
                gripper_z: float, block_z: float) -> float:
    """
    Spawn block at centre with random yaw; spawn gripper at a collision-free
    position drawn from a truncated Gaussian.  Returns block yaw (radians).
    """
    hx = base_env.block_dims.half_x
    hy = base_env.block_dims.half_y

    yaw = rng.uniform(-np.pi, np.pi)
    base_env.block.set_pose(
        sapien.Pose(p=[BCX, BCY, block_z], q=_yaw_to_sapien_quat(yaw))
    )

    for _ in range(20):
        angle     = rng.uniform(0.0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])

        support = _obb_support(hx, hy, yaw, direction)
        wall_d  = _wall_dist(direction)
        # Ensure gripper circle clears the block surface
        lo = GRIPPER_RADIUS + SPAWN_CLEARANCE
        hi = min(wall_d - support - WALL_MARGIN - GRIPPER_RADIUS, SPAWN_MAX_DIST)
        if hi <= lo:
            continue

        extra = _truncated_gaussian(rng, mu=0.0, sigma=GRIPPER_SPAWN_STD,
                                    lo=lo, hi=hi)
        gx = BCX + direction[0] * (support + extra)
        gy = BCY + direction[1] * (support + extra)

        _set_gripper_xy(base_env, gx, gy, gripper_z)

        actual = _get_xyz(base_env.agent.tcp.pose.p)
        err    = np.linalg.norm(actual[:2] - np.array([gx, gy]))
        if err > 0.005:
            print(f"  [WARN] teleport err={err:.4f}m  "
                  f"intended=({gx:.3f},{gy:.3f})  "
                  f"actual=({actual[0]:.3f},{actual[1]:.3f})")

        print(f"  gripper: intended=({gx:.3f},{gy:.3f})  "
              f"actual=({actual[0]:.3f},{actual[1]:.3f})  "
              f"block_yaw={np.degrees(yaw):+.1f}°")
        return yaw

    raise RuntimeError("custom_init: no valid spawn in 20 attempts")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace):
    env = gym.make(
        "PushBoundary",
        obs_mode="state",
        control_mode="floating_vel",
        render_mode="all",
        sim_backend="auto",
        shape="cube",
        num_extra_blocks=0,
        robot_uids="floating_gripper",
    )
    env = RecordEpisode(
        env,
        output_dir=args.record_dir,
        save_trajectory=True,
        save_video=True,
        video_fps=20,
        trajectory_name="face_push",
    )
    base_env = _unwrap(env)
    policy   = FaceApproachPushPolicy()
    rng      = np.random.default_rng(args.seed)

    obs, _ = env.reset(seed=args.seed)
    gripper_z = float(_get_xyz(base_env.agent.tcp.pose.p)[2])
    block_z   = float(_get_xyz(base_env.block.pose.p)[2])
    hx        = base_env.block_dims.half_x
    hy        = base_env.block_dims.half_y

    print(
        f"Collecting face-approach-push demos\n"
        f"  gripper_z={gripper_z:.4f}  block_z={block_z:.4f}\n"
        f"  block_dims={base_env.block_dims}\n"
        f"  gripper_radius={GRIPPER_RADIUS} m\n"
        f"  action_space={env.action_space}\n"
        f"  max_episode_steps={args.max_episode_steps}\n"
        f"  target episodes: {args.num_episodes}"
    )

    def init_episode():
        yaw      = custom_init(base_env, rng, gripper_z, block_z)
        ee_xy    = _get_xyz(base_env.agent.tcp.pose.p)[:2]
        block_xy = np.array([BCX, BCY])
        push_dir = policy.reset(ee_xy, block_xy, yaw, hx, hy, rng)
        print(f"  initial push face: ({push_dir[0]:+.2f}, {push_dir[1]:+.2f})")

    init_episode()

    episode = alive_steps = total_steps = 0

    while episode < args.num_episodes:
        ee_xyz    = _get_xyz(base_env.agent.tcp.pose.p)
        block_xyz = _get_xyz(base_env.block.pose.p)
        block_yaw = float(Rotation.from_quat(
            _get_xyzw_quat(base_env.block.pose.q)
        ).as_euler("xyz")[2])

        action = policy.act(ee_xyz[:2], block_xyz[:2], block_yaw)

        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1
        alive_steps += 1

        if alive_steps >= args.max_episode_steps:
            truncated = True

        if terminated or truncated:
            episode += 1
            reason = "boundary" if terminated else "timeout"
            print(f"  ep {episode:4d}  {alive_steps:5d} steps  ({reason})")

            if episode >= args.num_episodes:
                break

            obs, _ = env.reset()
            init_episode()
            alive_steps = 0

    env.close()
    print(f"\nDone — {episode} episodes, {total_steps} total steps.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num_episodes",      type=int, default=50)
    p.add_argument("--max_episode_steps", type=int, default=400,
                   help="Hard per-episode step cap (default: 400).")
    p.add_argument("--seed",              type=int, default=None)
    p.add_argument("--record_dir",        type=str,
                   default="demos/PushBoundary/face_push_v3")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())