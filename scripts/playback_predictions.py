#!/usr/bin/env python3
"""
scripts/playback_predictions.py
================================
Load a playback NPZ (created by inference_guided.py --save_playback) and
replay the predicted gripper trajectory in the PushBoundary simulator.

What it does
------------
1. Spawns the block at its ground-truth start pose (start_state[:7]).
2. Initialises the robot to its ground-truth start joint configuration
   (start_state[7:14]).
3. Draws a green square on the ground at the ground-truth end block position
   (end_state[:3]) — the target the model should push toward.
4. Converts the predicted TCP trajectory from the centered preprocessing
   frame to world frame and replays it via a P-controller.
5. Optionally saves a video via ManiSkill's RecordEpisode wrapper.

State layout (from h5_to_wds.py)
---------------------------------
  start_state / end_state : (14,)
    [0:3]   block position      (world frame, metres)
    [3:7]   block quaternion    [w, x, y, z]
    [7:14]  robot joint angles  (7 DOF, radians)

Quaternion conventions used throughout
---------------------------------------
  - All quaternions in this file are [w, x, y, z].
  - sapien.Pose(q=...) takes [w, x, y, z]  (confirmed by euler2quat usage in env).
  - ManiSkill Pose.q also returns [w, x, y, z].
  - The HDF5 ACTOR_QUAT stores [w, x, y, z] (ManiSkill actor state format).
  - No reordering is needed anywhere.

Usage
-----
    python scripts/playback_predictions.py \
        --npz inference_results/my_run/playback/batch000_b00.npz \
        --shape cube

    python scripts/playback_predictions.py \
        --npz_dir inference_results/my_run/playback/ \
        --shape cube --no_render --save_video --video_dir videos/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import envs  # noqa: F401  — registers PushBoundary

import sapien
import sapien.render
import torch

from envs.push_boundary import (
    PushBoundaryEnv,
    BOUNDARY_CENTER_X,
    BOUNDARY_CENTER_Y,
    CUBE_Z_SPAWN,
    T_Z_SPAWN,
    CUBE_HALF
)
from mani_skill.utils.structs import Pose
from utils.playback_utils import load_playback_npz
from utils.rigid_fit import delta_quaternion
from shapely.geometry import Polygon


EE_Z = 0.035   # push height — match scripted_push.py


# ─────────────────────────────────────────────────────────────────────────────
# State unpacking
# ─────────────────────────────────────────────────────────────────────────────

def unpack_state(state: np.ndarray):
    """
    Unpack a (14,) start_state or end_state vector.

    Layout (from h5_to_wds.py):
        [0:3]   block position   (world frame, metres)
        [3:7]   block quaternion [w, x, y, z]
        [7:14]  robot joint angles (7 DOF, radians)
    """
    block_pos    = state[0:3].astype(np.float32)
    block_quat   = state[3:7].astype(np.float32)   # [w, x, y, z]
    joint_angles = state[7:14].astype(np.float32)
    return block_pos, block_quat, joint_angles

from mani_skill.utils.structs import Pose

def tcp_xyz_to_qpos(base_env, target_xyz, target_quat):
    """
    Convert a desired TCP world-frame position to robot joint angles via IK.
    Returns qpos (7,) or None if IK failed.
    """
    ctrl = base_env.agent.controller   # PDEEPoseController (CombinedController wrapping it)
    
    # Get the actual ee pose controller out of the combined controller
    ee_ctrl = ctrl.controllers["arm"]   # check the key name — might be "arm", "ee", etc.
    #   print(list(ctrl.controllers.keys())) to find the right key
    
    # Build target pose in the ROOT frame (base of the robot)
    # The controller works in root frame, so convert world → root
    root_pose = ee_ctrl.root_link.pose   # world-frame pose of robot base
    root_pose_inv = root_pose.inv()
    
    target_pos_t  = torch.tensor(target_xyz,  dtype=torch.float32, device=base_env.device).unsqueeze(0)
    target_quat_t = torch.tensor(target_quat, dtype=torch.float32, device=base_env.device).unsqueeze(0)
    target_world  = Pose.create_from_pq(p=target_pos_t, q=target_quat_t)
    target_root   = root_pose_inv * target_world   # world → root frame

    qpos = ee_ctrl.kinematics.compute_ik(
        pose=target_root,
        q0=base_env.agent.robot.get_qpos(),
        is_delta_pose=False,
        current_pose=ee_ctrl.ee_pose_at_base,
        solver_config=ee_ctrl.config.delta_solver_config,
    )
    return qpos   # (1, 7) tensor or None if IK failed


# ─────────────────────────────────────────────────────────────────────────────
# Env subclass
# ─────────────────────────────────────────────────────────────────────────────

class PlaybackEnv(PushBoundaryEnv):
    """
    PushBoundaryEnv with a green target square and post-reset pose setters.
    """

    def __init__(self, *args, target_xy=(BOUNDARY_CENTER_X, BOUNDARY_CENTER_Y), target_quat=(1, 0, 0, 0), **kwargs):
        self._target_xy    = target_xy
        self._target_quat = target_quat
        self._target_actor = None
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict) -> None:
        super()._load_scene(options)

        green = sapien.render.RenderMaterial(
            base_color=np.array([0.08, 0.85, 0.15, 0.90])
        )
        bldr = self.scene.create_actor_builder()
        bldr.add_box_visual(half_size=[0.033, 0.033, 0.0005], material=green)
        bldr.initial_pose = sapien.Pose(
            p=[float(self._target_xy[0]), float(self._target_xy[1]), 0.001], q =self._target_quat
        )
        self._target_actor = bldr.build_static(name="target_square")

    def set_block_pose_world(
        self,
        position: np.ndarray,                     # (3,)  world-frame XYZ
        quaternion: Optional[np.ndarray] = None,  # [w, x, y, z]
    ) -> None:
        """
        Set block pose in world frame.

        sapien.Pose(q=...) takes [w, x, y, z] — same as what the HDF5 stores
        and what this function receives.  No reordering needed.
        """
        if quaternion is None:
            quaternion = np.array([1., 0., 0., 0.], dtype=np.float32)
        self.block.set_pose(
            sapien.Pose(p=position.tolist(), q=quaternion.tolist())
        )

    def set_robot_joints(self, joint_angles: np.ndarray) -> None:
        """Set the robot's 7-DOF joint configuration after reset."""
        qpos = (
            torch.tensor(joint_angles, dtype=torch.float32, device=self.device)
            .unsqueeze(0)   # (1, 7) — single env
        )
        self.agent.reset(qpos)


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate frame helper
# ─────────────────────────────────────────────────────────────────────────────

def centered_to_world(
    pos_centered: np.ndarray,                   # (T, 3) or (3,)
    initial_block_pos_centered: np.ndarray,     # (3,)
    spawn_xyz_world: np.ndarray,                # (3,)
) -> np.ndarray:
    """
    Shift positions from the centered preprocessing frame to world frame by
    anchoring the initial predicted block centroid to *spawn_xyz_world*.
    """
    return pos_centered + (spawn_xyz_world - initial_block_pos_centered)


# ─────────────────────────────────────────────────────────────────────────────
# Controller
# ─────────────────────────────────────────────────────────────────────────────

def compute_success(
    block_pos_world: np.ndarray,    # (3,)
    block_quat_world: np.ndarray,   # (4,)  [w, x, y, z]
    target_xy: np.ndarray,          # (2,)
    target_half_size: float = 0.033,
    cube_half: float = CUBE_HALF,
    threshold: float = 0.95,
) -> bool:
    """
    Returns True if >= threshold fraction of the (rotated) cube's XY footprint
    overlaps with the target square.
    """
    # Extract yaw from quaternion [w, x, y, z]
    w, x, y, z = block_quat_world
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    # Rotated block corners in world XY
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    corners_local = np.array([
        [ cube_half,  cube_half],
        [-cube_half,  cube_half],
        [-cube_half, -cube_half],
        [ cube_half, -cube_half],
    ])
    R = np.array([[cos_y, -sin_y],
                  [sin_y,  cos_y]])
    corners_world = (R @ corners_local.T).T + block_pos_world[:2]

    block_poly  = Polygon(corners_world)
    tx, ty      = target_xy
    target_poly = Polygon([
        [tx - target_half_size, ty - target_half_size],
        [tx + target_half_size, ty - target_half_size],
        [tx + target_half_size, ty + target_half_size],
        [tx - target_half_size, ty + target_half_size],
    ])

    overlap_area = block_poly.intersection(target_poly).area
    cube_area    = (2 * cube_half) ** 2

    return bool((overlap_area / cube_area) >= threshold)

def compute_action(
    current_xyz: np.ndarray,            # (3,)
    current_quat: np.ndarray,           # [w, x, y, z]
    target_xyz: np.ndarray,             # (3,)
    target_quat: Optional[np.ndarray],  # [w, x, y, z] or None to skip rotation
    gain_xyz: float    = 1.0,
    gain_rot: float    = 1.0,
    max_xyz: float     = 0.10,
    max_rot: float     = 0.50,
    ee_z_target: float = EE_Z,
) -> np.ndarray:
    """
    Proportional P-controller → 6-DOF delta-pose action.

    Z is locked to ee_z_target to keep the gripper at push height.
    Rotation tracking is active only when target_quat is not None.
    """
    action = np.zeros(6, dtype=np.float32)

    # XY: proportional toward target
    delta_xy  = np.clip(gain_xyz * (target_xyz[:2] - current_xyz[:2]), -max_xyz, max_xyz)
    action[0] = delta_xy[0]
    action[1] = delta_xy[1]

    # Z: maintain push height
    action[2] = float(ee_z_target - current_xyz[2])

    # # Rotation: proportional toward target orientation
    # if target_quat is not None:
    #     aa          = delta_quaternion(current_quat, target_quat)
    #     action[3:6] = np.clip(gain_rot * aa, -max_rot, max_rot)

    return action


# ─────────────────────────────────────────────────────────────────────────────
# Main playback loop
# ─────────────────────────────────────────────────────────────────────────────

def run_playback(
    npz_path: Path,
    shape: str               = "cube",
    render: bool             = True,
    save_video: bool         = False,
    video_dir: Optional[str] = None,
    gain_xyz: float          = 1.0,
    gain_rot: float          = 1.0,
    substeps_per_frame: int  = 4,
    num_extra_blocks: int    = 0,
    valid_starts_file: str   = "utils/valid_starts.npy",
    pause_on_done: float     = 2.0,
) -> None:
    data = load_playback_npz(npz_path)

    n_hand      = data["n_hand_int"]
    n_block     = data["n_block_int"]
    context_len = data["context_len_int"]

    tcp_pos_cen  = data["tcp_positions"]      # (T, 3)  centered frame
    tcp_quat     = data["tcp_quaternions"]    # (T, 4)  [w, x, y, z]
    init_blk_cen = data["initial_block_pos"]  # (3,)    centering anchor

    T = tcp_pos_cen.shape[0]

    # ── Unpack ground-truth states (world frame, no centering) ────────────────
    start_block_pos, start_block_quat, start_joint_angles = unpack_state(data["start_state"])
    end_block_pos,   end_block_quat,   _                  = unpack_state(data["end_state"])

    # ── Shift predicted TCP trajectory into world frame ───────────────────────
    z_block     = CUBE_Z_SPAWN if shape == "cube" else T_Z_SPAWN
    spawn_world = np.array([start_block_pos[0], start_block_pos[1], z_block],
                            dtype=np.float32)
    tcp_world   = centered_to_world(tcp_pos_cen, init_blk_cen, spawn_world)

    # Green square uses ground-truth end block XY directly (world frame)
    target_xy_world = end_block_pos[:2]
    target_quat_world = end_block_quat  # [w, x, y, z]

    print(f"\n{'='*60}")
    print(f"Playback: {npz_path.name}")
    print(f"  shape={shape}  T={T}  context_len={context_len}")
    print(f"  block start  (world): {start_block_pos.tolist()}")
    print(f"  block target (world): {end_block_pos.tolist()}")
    print(f"  joint init          : {np.round(start_joint_angles, 3).tolist()}")
    print(f"  TCP @ t=ctx (world) : {tcp_world[context_len].tolist()}")
    print(f"{'='*60}\n")

    # ── Build env ─────────────────────────────────────────────────────────────
    render_mode = "human" if render else None
    if save_video:
        render_mode = "rgb_array"

    env = PlaybackEnv(
        robot_uids="panda_stick",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode=render_mode,
        sim_backend="cpu",
        num_envs=1,
        shape=shape,
        num_extra_blocks=num_extra_blocks,
        target_xy=target_xy_world.tolist(),
        target_quat=target_quat_world.tolist(),
        valid_starts_file=valid_starts_file,
    )

    if save_video and video_dir is not None:
        from mani_skill.utils.wrappers.record import RecordEpisode
        Path(video_dir).mkdir(parents=True, exist_ok=True)
        env = RecordEpisode(
            env,
            output_dir=video_dir,
            save_trajectory=False,
            save_video=True,
            video_fps=20,
            trajectory_name=npz_path.stem,
        )

    obs, _info = env.reset()
    base_env   = env.unwrapped

    # ── Set ground-truth initial poses ────────────────────────────────────────
    # Block: start_block_quat is [w,x,y,z]; sapien.Pose also takes [w,x,y,z].
    base_env.set_block_pose_world(
        position=start_block_pos,
        quaternion=start_block_quat,
    )
    # Robot: ground-truth joint angles
    base_env.set_robot_joints(start_joint_angles)

    # Let physics settle
    for _ in range(10):
        env.step(np.zeros(6, dtype=np.float32))

    # ── Playback loop ─────────────────────────────────────────────────────────
    fail_count = 0
    cur_xyz    = np.zeros(3, dtype=np.float32)   # will be overwritten immediately
    
    for t in range(context_len, T):
        target_tcp_xyz  = tcp_world[t]
        target_tcp_xyz[2] = EE_Z  # ensure target Z is at push height
        tcp_pose = base_env.agent.tcp.pose

        # pose.q returns [w, x, y, z] — use directly, no reordering
        cur_quat = tcp_pose.q[0].cpu().numpy().astype(np.float32)
        
        qpos = tcp_xyz_to_qpos(base_env, target_tcp_xyz, cur_quat)
        base_env.set_robot_joints(qpos[0])
        obs, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))

        if info.get("fail", torch.zeros(1)).any():
            fail_count += 1

        if terminated or truncated:
            print(f"  Episode ended at t={t} "
                    f"(terminated={terminated}, truncated={truncated})")
            break

        if render and t % 5 == 0:
            print(f"  t={t:3d}/{T-1}  "
                  f"tcp=({cur_xyz[0]:.3f},{cur_xyz[1]:.3f})  "
                  f"tgt=({target_tcp_xyz[0]:.3f},{target_tcp_xyz[1]:.3f})  "
                  f"fails={fail_count}")
    if render and pause_on_done > 0:
        time.sleep(pause_on_done)

    # Read block pose before closing the env
    final_block_pos  = base_env.block.pose.p[0].cpu().numpy().astype(np.float32)
    final_block_quat = base_env.block.pose.q[0].cpu().numpy().astype(np.float32)  # [w,x,y,z]
    
    success = compute_success(
        block_pos_world=final_block_pos,
        block_quat_world=final_block_quat,
        target_xy=target_xy_world,
    )
    
    save_file_name = f"{npz_path.stem}"
    if success:
        save_file_name += "_success"

    if save_video and video_dir is not None:
        env.flush_video(name=save_file_name)

    env.close()

    
    print(f"\nPlayback done.  fail_frames={fail_count}  |  Success: {success}")
    return success

# ─────────────────────────────────────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_batch_playback(
    npz_dir: Path,
    shape: str,
    render: bool,
    save_video: bool,
    video_dir: Optional[str],
    gain_xyz: float,
    gain_rot: float,
    substeps: int,
    num_extra_blocks: int,
    valid_starts_file: str,
    max_files: Optional[int] = None,
) -> None:
    npz_files = sorted(npz_dir.glob("*.npz"))
    if max_files is not None:
        npz_files = npz_files[:max_files]
    print(f"Found {len(npz_files)} NPZ files in {npz_dir}")

    for i, npz_path in enumerate(npz_files):
        print(f"\n[{i+1}/{len(npz_files)}]  {npz_path.name}")
        try:
            run_playback(
                npz_path=npz_path,
                shape=shape,
                render=render,
                save_video=save_video,
                video_dir=video_dir,
                gain_xyz=gain_xyz,
                gain_rot=gain_rot,
                substeps_per_frame=substeps,
                num_extra_blocks=num_extra_blocks,
                valid_starts_file=valid_starts_file,
                pause_on_done=1.0 if render else 0.0,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback; traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--npz",     type=str, default=None)
    src.add_argument("--npz_dir", type=str, default=None)

    p.add_argument("--shape",             type=str,   default="cube",
                   choices=["cube", "T", "circle"])
    p.add_argument("--num_extra_blocks",  type=int,   default=0)
    p.add_argument("--valid_starts_file", type=str,   default="utils/valid_starts.npy")
    p.add_argument("--no_render",         action="store_true")
    p.add_argument("--save_video",        action="store_true")
    p.add_argument("--video_dir",         type=str,   default="playback_videos/")
    p.add_argument("--max_files",         type=int,   default=None)
    p.add_argument("--gain_xyz",          type=float, default=1.0)
    p.add_argument("--gain_rot",          type=float, default=1.0)
    p.add_argument("--substeps",          type=int,   default=4)
    return p.parse_args()


def main():
    args   = parse_args()
    render = not args.no_render

    if args.npz is not None:
        run_playback(
            npz_path=Path(args.npz),
            shape=args.shape,
            render=render,
            save_video=args.save_video,
            video_dir=args.video_dir if args.save_video else None,
            gain_xyz=args.gain_xyz,
            gain_rot=args.gain_rot,
            substeps_per_frame=args.substeps,
            num_extra_blocks=args.num_extra_blocks,
            valid_starts_file=args.valid_starts_file,
        )
    else:
        run_batch_playback(
            npz_dir=Path(args.npz_dir),
            shape=args.shape,
            render=render,
            save_video=args.save_video,
            video_dir=args.video_dir if args.save_video else None,
            gain_xyz=args.gain_xyz,
            gain_rot=args.gain_rot,
            substeps=args.substeps,
            num_extra_blocks=args.num_extra_blocks,
            valid_starts_file=args.valid_starts_file,
            max_files=args.max_files,
        )


if __name__ == "__main__":
    main()