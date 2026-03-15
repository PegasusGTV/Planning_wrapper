#!/usr/bin/env python3
"""
Scripted autonomous policy for the unified Push environment (PushBoundary).

Modes
-----
standard  clear -> approach -> push along a random chosen axis.
direct    Draws a vector from the EE through the block and pushes directly.
mixed     Randomly alternates between standard and direct pushes.

Usage
-----
    python scripts/scripted_push.py --shape cube
    python scripts/scripted_push.py --shape T --num_extra_blocks 5
    python scripts/scripted_push.py --mode direct --shape T --num_extra_blocks 3 --render
    python scripts/scripted_push.py --record_dir demos/scripted --shape cube --mode mixed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import gymnasium as gym
from mani_skill.utils.wrappers.record import RecordEpisode


from envs.push_boundary import (
    BOUNDARY_CENTER_X as BCX,
    BOUNDARY_CENTER_Y as BCY,
    BOUNDARY_HALF_X   as BHX,
    BOUNDARY_HALF_Y   as BHY,
)

EE_Z = 0.024

MIN_X = BCX - BHX - 0.025
MAX_X = BCX + BHX + 0.025
MIN_Y = BCY - BHY - 0.025
MAX_Y = BCY + BHY + 0.025

EDGE_THRESHOLD = 0.075


def _get_xyz(pose_p) -> np.ndarray:
    try:
        import torch
        if isinstance(pose_p, torch.Tensor):
            pose_p = pose_p.cpu().numpy()
    except ImportError:
        pass
    arr = np.asarray(pose_p, dtype=np.float64).reshape(-1)
    return arr[:3].copy()


def _unwrap(env):
    e = env
    while hasattr(e, "env"):
        e = e.env
    return e


# ─────────────────────────────────────────────────────────────────────────────
# Policy
# ─────────────────────────────────────────────────────────────────────────────

class ScriptedPushPolicy:

    PUSH_STEPS       = 100
    ARRIVE_THRESH    = 0.018
    APPROACH_TIMEOUT = 120
    XY_SPEED         = 0.1
    PUSH_SPEED       = 0.08

    CLEAR_DIST = 0.05

    PUSH_CONTACT_DIST = 0.05
    PUSH_TIMEOUT      = 150
    PUSH_TARGET_EPS   = 0.015

    SAME_SIDE_PROB = 0.60
    DIRECT_PROB    = 0.50

    NOISE_STD_STANDARD = 0.10
    NOISE_STD_DIRECT   = 0.02

    # ── Randomisation knobs ───────────────────────────────────────────────────
    # Rescue direction: angular spread when sampling candidate rescue directions.
    # 0 = always point straight toward boundary centre.
    RESCUE_DIR_JITTER_STD    = 0.25   # radians
    # How many candidate directions to sample when searching for a short rescue.
    RESCUE_DIR_N_CANDIDATES  = 8
    # Maximum allowed distance from block to rescue standoff point.
    # Candidates exceeding this are rejected; if all do, the closest is used.
    MAX_RESCUE_STANDOFF_DIST = 0.10   # metres

    # Rescue approach: lateral wobble on the rescue standoff point.
    # 0 = always aim through exact block centre.
    RESCUE_LATERAL_STD       = 0.03   # metres

    # Standoff geometry (applies to all non-direct pushes)
    STANDOFF_DIST_RANGE   = (0.05, 0.12)    # metres; how far behind block to stand off
    STANDOFF_LAT_RANGE    = (-0.05, 0.05)   # metres; lateral offset of standoff point

    # Multi-waypoint repositioning (standard mode only)
    MULTI_REPOSITION_PROB = 0.00   # probability of inserting intermediate waypoints
    MAX_EXTRA_WAYPOINTS   = 2      # max number of extra waypoints per reposition
    WAYPOINT_SPREAD       = 0.12   # metres; max perpendicular scatter around direct path

    def __init__(
        self,
        rng: np.random.Generator | None = None,
        mode: str = "standard",
        block_half_width: float = 0.025,
    ):
        self.rng              = rng or np.random.default_rng()
        self.mode             = mode
        self.block_half_width = block_half_width

        self._phase          = "pick_new"
        self._phase_step     = 0
        self._push_dir       = np.array([1.0, 0.0])
        self._committed_dir  = np.array([1.0, 0.0])
        self._clear_dir      = np.array([0.0, 1.0])
        self._clear_perp     = np.array([0.0, 1.0])
        self._use_boundary   = False
        self._move_tgt       = np.array([BCX, BCY])
        self._rescuing            = False
        self._push_contact_steps  = 0
        self._is_direct_push      = False
        self._rescue_lateral      = 0.0
        self._waypoint_queue: list[np.ndarray] = []

        self._standoff_dist = 0.06
        self._standoff_lat  = 0.0

    def reset(self):
        self._phase              = "pick_new"
        self._phase_step         = 0
        self._use_boundary       = False
        self._rescuing           = False
        self._push_contact_steps = 0
        self._is_direct_push     = False
        self._rescue_lateral     = 0.0
        self._waypoint_queue     = []

    # ── Main entry point ──────────────────────────────────────────────────────

    def act(self, ee_xy: np.ndarray, block_xy: np.ndarray) -> np.ndarray:
        near_edge = self._near_edge(block_xy)

        if near_edge and not self._rescuing:
            self._rescuing        = True
            self._is_direct_push  = False
            self._rescue_lateral  = self.rng.normal(0, self.RESCUE_LATERAL_STD)
            self._committed_dir   = self._rescue_dir(block_xy)
            self._push_dir        = self._committed_dir.copy()
            self._start_clear(ee_xy, block_xy, use_boundary=True)

        self._phase_step += 1

        if self._phase == "pick_new":
            self._pick_next_action(ee_xy, block_xy)

        if self._phase == "clear":
            lateral = np.dot(ee_xy - block_xy, self._clear_perp)
            if abs(lateral) >= self.CLEAR_DIST:
                self._phase      = "approach"
                self._phase_step = 0
                if self._waypoint_queue:
                    self._move_tgt = self._waypoint_queue[0]
                else:
                    self._move_tgt = (
                        self._boundary_standoff(block_xy)
                        if self._use_boundary
                        else self._standoff_point(block_xy)
                    )

        elif self._phase == "approach":
            if self._waypoint_queue:
                self._move_tgt = self._waypoint_queue[0]
                if np.linalg.norm(ee_xy - self._move_tgt) < self.ARRIVE_THRESH:
                    self._waypoint_queue.pop(0)
                    if self._waypoint_queue:
                        self._move_tgt = self._waypoint_queue[0]
                    else:
                        self._move_tgt = (
                            self._boundary_standoff(block_xy)
                            if self._use_boundary
                            else self._standoff_point(block_xy)
                        )
            else:
                self._move_tgt = (
                    self._boundary_standoff(block_xy)
                    if self._use_boundary
                    else self._standoff_point(block_xy)
                )

            final_tgt = (
                self._boundary_standoff(block_xy)
                if self._use_boundary
                else self._standoff_point(block_xy)
            )
            at_final = (
                not self._waypoint_queue
                and np.linalg.norm(ee_xy - final_tgt) < self.ARRIVE_THRESH
            )
            if at_final or self._phase_step >= self.APPROACH_TIMEOUT:
                self._phase              = "push"
                self._phase_step         = 0
                self._push_contact_steps = 0

        elif self._phase == "push":
            dist_to_block = np.linalg.norm(ee_xy - block_xy)
            if dist_to_block < self.PUSH_CONTACT_DIST:
                self._push_contact_steps += 1

            timed_out    = self._phase_step >= self.PUSH_TIMEOUT
            done_pushing = self._push_contact_steps >= self.PUSH_STEPS

            if done_pushing or timed_out:
                was_rescuing   = self._rescuing
                self._rescuing = False
                self._phase_step = 0

                if was_rescuing and self._near_edge(block_xy):
                    self._rescuing           = True
                    self._is_direct_push     = False
                    self._rescue_lateral     = self.rng.normal(0, self.RESCUE_LATERAL_STD)
                    self._push_contact_steps = 0
                    self._committed_dir      = self._rescue_dir(block_xy)
                    self._push_dir           = self._committed_dir.copy()
                    self._start_clear(ee_xy, block_xy, use_boundary=True)
                elif self.rng.random() < self.SAME_SIDE_PROB:
                    self._phase              = "push"
                    self._phase_step         = 0
                    self._push_contact_steps = 0
                    if self._is_direct_push:
                        vec = block_xy - ee_xy
                        n   = np.linalg.norm(vec)
                        self._push_dir = vec / n if n > 1e-6 else np.array([1.0, 0.0])
                else:
                    self._pick_next_action(ee_xy, block_xy)

        # ── Compute delta ─────────────────────────────────────────────────────
        if self._phase == "push":
            push_target = block_xy + self._push_dir * 0.05
            if np.linalg.norm(ee_xy - push_target) < self.PUSH_TARGET_EPS:
                self._rescuing = False
                self._pick_next_action(ee_xy, block_xy)
                if self._phase == "clear":
                    dxy = self._clear_dir * self.XY_SPEED
                else:
                    pt  = block_xy + self._push_dir * 0.05
                    dxy = self._toward(ee_xy, pt, self.PUSH_SPEED)
            else:
                dxy = self._toward(ee_xy, push_target, self.PUSH_SPEED)
                if not self._rescuing:
                    noise_std   = self.NOISE_STD_DIRECT if self._is_direct_push else self.NOISE_STD_STANDARD
                    angle_noise = self.rng.normal(0, noise_std)
                    c, s        = np.cos(angle_noise), np.sin(angle_noise)
                    dxy         = np.array([[c, -s], [s, c]]).dot(dxy)

        elif self._phase == "clear":
            dxy = self._clear_dir * self.XY_SPEED
        else:
            dxy = self._toward(ee_xy, self._move_tgt, self.XY_SPEED)

        # Safety clamp
        next_xy    = ee_xy + dxy
        next_xy[0] = np.clip(next_xy[0], MIN_X, MAX_X)
        next_xy[1] = np.clip(next_xy[1], MIN_Y, MAX_Y)
        clipped_dxy = next_xy - ee_xy

        intended_dist = np.linalg.norm(dxy)
        actual_dist   = np.linalg.norm(clipped_dxy)
        mostly_clipped = intended_dist > 1e-6 and (actual_dist / intended_dist) < 0.3

        if mostly_clipped:
            if self._phase == "clear":
                self._phase          = "approach"
                self._phase_step     = 0
                self._waypoint_queue = []
                self._move_tgt       = (
                    self._boundary_standoff(block_xy)
                    if self._use_boundary
                    else self._standoff_point(block_xy)
                )
            elif self._phase == "approach":
                self._waypoint_queue     = []
                self._phase              = "push"
                self._phase_step         = 0
                self._push_contact_steps = 0
            elif self._phase == "push":
                self._rescuing = False
                self._pick_next_action(ee_xy, block_xy)

        dxy = clipped_dxy

        action         = np.zeros(6, dtype=np.float32)
        action[0], action[1] = float(dxy[0]), float(dxy[1])
        return action

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _pick_next_action(self, ee_xy: np.ndarray, block_xy: np.ndarray):
        use_direct = (
            self.mode == "direct"
            or (self.mode == "mixed" and self.rng.random() < self.DIRECT_PROB)
        )
        self._is_direct_push = use_direct

        if use_direct:
            self._phase              = "push"
            self._phase_step         = 0
            self._push_contact_steps = 0
            vec = block_xy - ee_xy
            n   = np.linalg.norm(vec)
            self._push_dir      = vec / n if n > 1e-6 else np.array([1.0, 0.0])
            self._committed_dir = self._push_dir.copy()
        else:
            self._committed_dir      = self._random_push_dir(block_xy)
            self._push_dir           = self._committed_dir.copy()
            self._push_contact_steps = 0
            self._start_clear(ee_xy, block_xy)

    def _start_clear(self, ee_xy, block_xy, use_boundary=False):
        self._phase        = "clear"
        self._phase_step   = 0
        self._use_boundary = use_boundary

        self._standoff_dist = self.rng.uniform(*self.STANDOFF_DIST_RANGE)
        self._standoff_lat  = self.rng.uniform(*self.STANDOFF_LAT_RANGE)

        # Perpendicular to the EE->block vector
        eb = block_xy - ee_xy
        n  = np.linalg.norm(eb)
        eb_norm = eb / n if n > 1e-6 else np.array([1.0, 0.0])
        perp = np.array([-eb_norm[1], eb_norm[0]])

        # Pick whichever side points toward the standoff/approach point
        target    = (self._boundary_standoff(block_xy) if use_boundary
                     else self._standoff_point(block_xy))
        to_target = target - ee_xy
        if np.dot(perp, to_target) < 0:
            perp = -perp

        self._clear_dir  = perp
        self._clear_perp = perp

        # Optionally generate intermediate waypoints (standard repositioning only)
        self._waypoint_queue = []
        if not use_boundary and self.rng.random() < self.MULTI_REPOSITION_PROB:
            n_wps = int(self.rng.integers(1, self.MAX_EXTRA_WAYPOINTS + 1))
            path     = target - ee_xy
            path_len = np.linalg.norm(path)
            if path_len > 1e-6:
                path_unit = path / path_len
                path_perp = np.array([-path_unit[1], path_unit[0]])
                for i in range(n_wps):
                    t      = (i + 1) / (n_wps + 1)
                    base   = ee_xy + t * path
                    offset = self.rng.uniform(-self.WAYPOINT_SPREAD, self.WAYPOINT_SPREAD)
                    wp     = self._clamp_xy(base + path_perp * offset)
                    self._waypoint_queue.append(wp)

    def _standoff_point(self, block_xy: np.ndarray) -> np.ndarray:
        perp = np.array([-self._push_dir[1], self._push_dir[0]])
        return self._clamp_xy(
            block_xy - self._push_dir * self._standoff_dist + perp * self._standoff_lat
        )

    def _boundary_standoff(self, block_xy: np.ndarray) -> np.ndarray:
        """Rescue standoff using the current push_dir and rescue_lateral."""
        return self._standoff_for_dir(block_xy, self._push_dir,
                                      rescue_lateral=self._rescue_lateral)

    def _standoff_for_dir(
        self,
        block_xy: np.ndarray,
        push_dir: np.ndarray,
        rescue_lateral: float = 0.0,
    ) -> np.ndarray:
        """Ray-cast from block_xy opposite push_dir to the boundary wall,
        then apply rescue_lateral as a perpendicular wobble.
        Pure function — does not read or mutate any instance state."""
        d      = -push_dir
        t_min  = float("inf")
        for axis, lo, hi in [(0, MIN_X, MAX_X), (1, MIN_Y, MAX_Y)]:
            if abs(d[axis]) > 1e-9:
                for wall in (lo, hi):
                    t = (wall - block_xy[axis]) / d[axis]
                    if 1e-6 < t < t_min:
                        t_min = t
        if t_min == float("inf"):
            t_min = 0.3
        standoff = block_xy + t_min * d
        perp     = np.array([-push_dir[1], push_dir[0]])
        return self._clamp_xy(standoff + perp * rescue_lateral)

    def _near_edge(self, bxy: np.ndarray) -> bool:
        return (
            bxy[0] < BCX - BHX + EDGE_THRESHOLD
            or bxy[0] > BCX + BHX - EDGE_THRESHOLD
            or bxy[1] < BCY - BHY + EDGE_THRESHOLD
            or bxy[1] > BCY + BHY - EDGE_THRESHOLD
        )

    def _clamp_xy(self, xy: np.ndarray) -> np.ndarray:
        return np.array([np.clip(xy[0], MIN_X, MAX_X),
                         np.clip(xy[1], MIN_Y, MAX_Y)])

    def _rescue_dir(self, bxy: np.ndarray) -> np.ndarray:
        """Sample RESCUE_DIR_N_CANDIDATES jittered directions around toward-center.
        Keep those whose standoff distance is within MAX_RESCUE_STANDOFF_DIST and
        pick the shortest among them.  If none qualify, fall back to the candidate
        with the shortest standoff overall (guaranteed to rescue eventually)."""
        v = np.array([BCX, BCY]) - bxy
        n = np.linalg.norm(v)
        base_dir = v / n if n > 1e-6 else np.array([1.0, 0.0])

        # Always include the exact toward-center direction so there is always
        # at least one well-behaved fallback candidate.
        angles = np.concatenate([[0.0],
                                  self.rng.normal(0, self.RESCUE_DIR_JITTER_STD,
                                                  size=self.RESCUE_DIR_N_CANDIDATES)])

        candidates: list[tuple[float, np.ndarray]] = []
        for angle in angles:
            c, s = np.cos(angle), np.sin(angle)
            d    = np.array([[c, -s], [s, c]]).dot(base_dir)
            pt   = self._standoff_for_dir(bxy, d, rescue_lateral=self._rescue_lateral)
            dist = float(np.linalg.norm(pt - bxy))
            candidates.append((dist, d))

        valid = [(dist, d) for dist, d in candidates
                 if dist <= self.MAX_RESCUE_STANDOFF_DIST]

        if valid:
            return min(valid, key=lambda x: x[0])[1]
        else:
            # Nothing within budget — use the direction with the shortest standoff
            return min(candidates, key=lambda x: x[0])[1]

    def _random_push_dir(self, bxy: np.ndarray) -> np.ndarray:
        angle = self.rng.uniform(0, 2 * np.pi)
        rd    = np.array([np.cos(angle), np.sin(angle)])
        cv    = np.array([BCX, BCY]) - bxy
        cn    = np.linalg.norm(cv)
        if cn > 1e-6:
            cv /= cn
        blended = 0.70 * rd + 0.30 * cv
        n = np.linalg.norm(blended)
        return blended / n if n > 1e-6 else rd

    @staticmethod
    def _toward(src: np.ndarray, dst: np.ndarray, speed: float) -> np.ndarray:
        d    = dst - src
        dist = np.linalg.norm(d)
        return d / dist * speed if dist > 1e-6 else np.zeros(2)
        # return d / dist * min(speed, dist) if dist > 1e-6 else np.zeros(2)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace):
    render_mode = "all"
    env = gym.make(
        "PushBoundary",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode=render_mode,
        sim_backend="auto",
        shape=args.shape,
        num_extra_blocks=args.num_extra_blocks,
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

    block_half_width = max(
        base_env.block_dims.half_x,
        base_env.block_dims.half_y,
    )

    rng    = np.random.default_rng(args.seed)
    policy = ScriptedPushPolicy(rng=rng, mode=args.mode, block_half_width=block_half_width)

    obs, _  = env.reset(seed=args.seed)
    policy.reset()

    total_steps = 0
    episode     = 0
    alive_steps = 0

    print(
        f"Running scripted policy  shape={args.shape}  "
        f"extra_blocks={args.num_extra_blocks}  mode={args.mode}  "
        f"steps={args.num_steps}"
    )
    print(f"  block_dims = {base_env.block_dims}")

    while total_steps < args.num_steps:
        ee_xyz    = _get_xyz(base_env.agent.tcp.pose.p)
        block_xyz = _get_xyz(base_env.block.pose.p)

        action     = policy.act(ee_xyz[:2], block_xyz[:2])
        action[2]  = -(ee_xyz[2] - EE_Z)

        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1
        alive_steps += 1

        if terminated or truncated:
            episode += 1
            reason  = "block escaped" if terminated else "truncated"
            print(f"  Episode {episode:3d} ended after {alive_steps:5d} steps ({reason})")
            obs, _ = env.reset()
            policy.reset()
            alive_steps = 0

    env.close()
    print("Done.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scripted push policy (PushBoundary)")

    p.add_argument("--shape",            type=str, default="cube", choices=["cube", "T"])
    p.add_argument("--num_extra_blocks", type=int, default=0)
    p.add_argument("--mode",       type=str, default="mixed",
                   choices=["standard", "direct", "mixed"])
    p.add_argument("--num_steps",  type=int, default=500)
    p.add_argument("--seed",       type=int, default=None)
    p.add_argument("--record_dir", type=str, default="demos/PushBoundary/scripted")

    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())