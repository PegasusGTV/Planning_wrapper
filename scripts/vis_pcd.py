#!/usr/bin/env python3
"""
scripts/visualize_pointclouds.py

Load a Push-v1/v2 trajectory .h5 and either:
  - Save a GIF of 3D point clouds (default)
  - Launch an interactive live viewer (--render)

Usage
-----
    # Save a GIF
    python scripts/visualize_pointclouds.py \
        --h5 demos/trajectory.h5 \
        --urdf /path/to/panda_arm.urdf \
        --gif out.gif

    # Interactive viewer
    python scripts/visualize_pointclouds.py \
        --h5 demos/trajectory.h5 \
        --urdf /path/to/panda_arm.urdf \
        --render

    # Push-v2, faster playback, start paused
    python scripts/visualize_pointclouds.py \
        --h5 demos/trajectory.h5 --urdf /path/to/panda_arm.urdf \
        --env v2 --fps 20 --render --paused

Controls (--render mode)
--------
    SPACE         pause / resume
    LEFT / RIGHT  step one frame (when paused)
    R             reset camera
    Q / ESC       quit
"""

from __future__ import annotations

import argparse
import time

import h5py
import numpy as np
import fpsample


# ──────────────────────────────────────────────────────────────────────────────
# H5 state layout
# actor (13,):        pos(3)  quat_wxyz(4)  lin_vel(3)  ang_vel(3)
# articulation (27,): pos(3)  quat_wxyz(4)  joint_pos(7)  joint_vel(7)  vel(6)
# ──────────────────────────────────────────────────────────────────────────────
ACTOR_POS  = slice(0, 3)
ACTOR_QUAT = slice(3, 7)
ART_JPOS   = slice(13, 20)


# ──────────────────────────────────────────────────────────────────────────────
# Object geometry (must match push_v1.py / push_v2.py)
# ──────────────────────────────────────────────────────────────────────────────
BLOCK_HALF = 0.025

_T_box1_hw = 0.10 / 2
_T_box1_hh = 0.025 / 2
_T_com_y   = 0.0375 / 2
_T_half_t  = 0.02
T_BOXES = [
    (np.array([0.0, -_T_com_y, 0.0]),                       np.array([_T_box1_hw, _T_box1_hh, _T_half_t])),
    (np.array([0.0, 4*_T_box1_hh - _T_com_y, 0.0]),         np.array([_T_box1_hh, (3/4)*_T_box1_hw, _T_half_t])),
]

WRIST_OFFSET = np.array([0.0, 0.0, 0.02])
WRIST_HALF   = np.array([0.04, 0.04, 0.03])
STICK_RADIUS   = 0.008
STICK_LENGTH   = 0.10
STICK_OFFSET_Z = 0.10

# Viewer colours
BLOCK_COLOR = np.array([0.20, 0.47, 0.96])   # blue
HAND_COLOR  = np.array([0.91, 0.47, 0.10])   # orange


# ──────────────────────────────────────────────────────────────────────────────
# Local-frame surface samplers
# ──────────────────────────────────────────────────────────────────────────────

def _sample_box(offset, half, n, rng):
    n_initial = n * 20
    hx, hy, hz = half
    areas  = np.array([4*hy*hz, 4*hx*hz, 4*hx*hy], dtype=np.float64)
    counts = np.round(areas / areas.sum() * n_initial).astype(int)
    counts[-1] += n_initial - counts.sum()
    pts = []
    if counts[0] > 0:
        signs = rng.choice([-1.0, 1.0], counts[0])
        pts.append(np.stack([signs*hx, rng.uniform(-hy, hy, counts[0]), rng.uniform(-hz, hz, counts[0])], 1))
    if counts[1] > 0:
        signs = rng.choice([-1.0, 1.0], counts[1])
        pts.append(np.stack([rng.uniform(-hx, hx, counts[1]), signs*hy, rng.uniform(-hz, hz, counts[1])], 1))
    if counts[2] > 0:
        signs = rng.choice([-1.0, 1.0], counts[2])
        pts.append(np.stack([rng.uniform(-hx, hx, counts[2]), rng.uniform(-hy, hy, counts[2]), signs*hz], 1))
    initial_points = np.concatenate(pts).astype(np.float32) + offset.astype(np.float32)
    return initial_points[fpsample.fps_sampling(initial_points, n)]


def _sample_cylinder(offset_z, radius, length, n, rng):
    n_initial = n * 20
    lat_a = 2 * np.pi * radius * length
    cap_a = 2 * np.pi * radius**2
    total_a = lat_a + cap_a
    n_lat = max(1, int(n_initial * lat_a / total_a)) if total_a > 0 else 0
    n_cap = n_initial - n_lat
    n_top, n_bot = n_cap // 2, n_cap - n_cap // 2
    pts = []
    if n_lat > 0:
        th = rng.uniform(0, 2*np.pi, n_lat)
        pts.append(np.stack([radius*np.cos(th), radius*np.sin(th),
                             rng.uniform(-length/2, length/2, n_lat) + offset_z], 1))
    for nz, z0 in [(n_top, length/2 + offset_z), (n_bot, -length/2 + offset_z)]:
        if nz > 0:
            r  = radius * np.sqrt(rng.uniform(0, 1, nz))
            th = rng.uniform(0, 2*np.pi, nz)
            pts.append(np.stack([r*np.cos(th), r*np.sin(th), np.full(nz, z0)], 1))
    initial_points = np.concatenate(pts).astype(np.float32)
    return initial_points[fpsample.fps_sampling(initial_points, n)]


# ──────────────────────────────────────────────────────────────────────────────
# Template builder — called ONCE, reused every frame to eliminate jitter
# ──────────────────────────────────────────────────────────────────────────────

def build_templates(n_block: int, n_hand: int, rng) -> dict[str, np.ndarray]:
    cube_tpl = _sample_box(np.zeros(3), np.full(3, BLOCK_HALF), n_block, rng)

    areas = [2 * (4*hy*hz + 4*hx*hz + 4*hx*hy) for _, (hx, hy, hz) in T_BOXES]
    total = sum(areas)
    t_parts = [_sample_box(off, half, max(1, int(n_block * a / total)), rng)
               for (off, half), a in zip(T_BOXES, areas)]
    t_tpl = np.concatenate(t_parts)

    hand_tpl = _sample_cylinder(STICK_OFFSET_Z, STICK_RADIUS, STICK_LENGTH, n_hand, rng)

    return {"cube": cube_tpl, "T": t_tpl, "hand": hand_tpl}


# ──────────────────────────────────────────────────────────────────────────────
# Pose math
# ──────────────────────────────────────────────────────────────────────────────

def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """(w,x,y,z) → (3,3) float64"""
    w, x, y, z = q.astype(np.float64)
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """(3,3) rotation matrix → (w,x,y,z)"""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])


def apply_pose(local_pts: np.ndarray, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    R = _quat_to_rot(quat)
    return (local_pts.astype(np.float64) @ R.T + pos.astype(np.float64)).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# FK via pinocchio
# ──────────────────────────────────────────────────────────────────────────────

class PandaFK:
    def __init__(self, urdf_path: str):
        try:
            import pinocchio as pin
        except ImportError:
            raise ImportError("Install pinocchio: conda install pinocchio -c conda-forge")
        self._pin   = pin
        model       = pin.buildModelFromUrdf(urdf_path)
        self._model = model
        self._data  = model.createData()
        self._fid   = model.getFrameId("panda_hand")
        if self._fid >= model.nframes:
            raise ValueError("'panda_hand' not found in URDF")
        print(f"Loaded URDF ({model.nq} DOF), panda_hand frame id={self._fid}")

    def hand_pose(self, joint_angles, robot_base_pos, robot_base_quat):
        pin   = self._pin
        model = self._model
        data  = self._data
        q = pin.neutral(model)
        q[:len(joint_angles)] = joint_angles.astype(np.float64)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        T_local  = data.oMf[self._fid]
        R_base   = _quat_to_rot(robot_base_quat)
        p_base   = robot_base_pos.astype(np.float64)
        R_world  = R_base @ T_local.rotation
        p_world  = R_base @ T_local.translation + p_base
        return p_world.astype(np.float32), _rot_to_quat(R_world).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Per-step extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_step(traj, step, templates, fk, use_T):
    actors = traj["env_states"]["actors"]
    arts   = traj["env_states"]["articulations"]
    bs     = actors["push_block"][step]
    block  = apply_pose(templates["T"] if use_T else templates["cube"],
                        bs[ACTOR_POS], bs[ACTOR_QUAT])
    art    = arts["panda_stick"][step]
    hp, hq = fk.hand_pose(art[ART_JPOS], art[ACTOR_POS], art[ACTOR_QUAT])
    hand   = apply_pose(templates["hand"], hp, hq)
    return {"block": block, "hand": hand}


# ──────────────────────────────────────────────────────────────────────────────
# Shared utility
# ──────────────────────────────────────────────────────────────────────────────

def _compute_limits(all_clouds):
    pts = np.concatenate([c[k] for c in all_clouds for k in ("block", "hand")])
    pad = 0.05
    c   = pts.mean(0)
    r   = max((pts.max(0) - pts.min(0)).max() / 2 + pad, 0.1)
    return [(c[i]-r, c[i]+r) for i in range(3)]


# ──────────────────────────────────────────────────────────────────────────────
# GIF rendering
# ──────────────────────────────────────────────────────────────────────────────

def render_frame(clouds, title, elev, azim, lims, dpi=100):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5, 5), dpi=dpi)
    ax  = fig.add_subplot(111, projection="3d")
    b   = clouds["block"]
    ax.scatter(b[:,0], b[:,1], b[:,2], s=1.5, c="#3478f5", label="block", depthshade=True)
    h   = clouds["hand"]
    ax.scatter(h[:,0], h[:,1], h[:,2], s=1.5, c="#e8781a", label="hand",  depthshade=True)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7)
    ax.set_zlabel("Z", fontsize=7); ax.tick_params(labelsize=6)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(title, fontsize=8)
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    w, h_px = fig.canvas.get_width_height()
    img = buf.reshape(h_px, w, 4).copy()
    plt.close(fig)
    return img


def save_gif(frames, path, fps):
    try:
        import imageio
        imageio.mimsave(path, frames, duration=1.0/fps, loop=0)
        return
    except ImportError:
        pass
    from PIL import Image
    pil = [Image.fromarray(f) for f in frames]
    pil[0].save(path, save_all=True, append_images=pil[1:],
                duration=int(1000/fps), loop=0)


# ──────────────────────────────────────────────────────────────────────────────
# Interactive viewer (open3d)
# ──────────────────────────────────────────────────────────────────────────────

def _make_pcd(pts, color):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pts), 1)))
    return pcd


def _update_pcd(pcd, pts, color):
    import open3d as o3d
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pts), 1)))


def run_viewer(all_clouds, steps, fps=10.0, start_paused=False):
    import open3d as o3d

    n   = len(all_clouds)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Point Cloud Viewer", width=800, height=800)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.12, 0.12, 0.12])
    opt.point_size = 2.5

    pcd_block = _make_pcd(all_clouds[0]["block"], BLOCK_COLOR)
    pcd_hand  = _make_pcd(all_clouds[0]["hand"],  HAND_COLOR)
    vis.add_geometry(pcd_block)
    vis.add_geometry(pcd_hand)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05))

    state = {"frame": 0, "paused": start_paused, "quit": False, "last_t": time.time()}

    def _refresh(vis):
        clouds = all_clouds[state["frame"]]
        _update_pcd(pcd_block, clouds["block"], BLOCK_COLOR)
        _update_pcd(pcd_hand,  clouds["hand"],  HAND_COLOR)
        vis.update_geometry(pcd_block)
        vis.update_geometry(pcd_hand)
        vis.get_view_control()   # keep current camera

    def on_space(vis):
        state["paused"] = not state["paused"]
        print("Paused" if state["paused"] else "Playing")

    def on_left(vis):
        if state["paused"]:
            state["frame"] = (state["frame"] - 1) % n
            _refresh(vis)

    def on_right(vis):
        if state["paused"]:
            state["frame"] = (state["frame"] + 1) % n
            _refresh(vis)

    def on_quit(vis):
        state["quit"] = True

    vis.register_key_callback(32,        on_space)   # SPACE
    vis.register_key_callback(263,       on_left)    # LEFT
    vis.register_key_callback(262,       on_right)   # RIGHT
    vis.register_key_callback(ord("Q"),  on_quit)
    vis.register_key_callback(27,        on_quit)    # ESC

    frame_dt = 1.0 / fps
    print(f"\nViewer open. {n} frames @ {fps} fps")
    print("  SPACE = pause/resume   ← → = step (paused)   Q/ESC = quit\n")

    while not state["quit"]:
        now = time.time()
        if not state["paused"] and (now - state["last_t"]) >= frame_dt:
            state["frame"]  = (state["frame"] + 1) % n
            state["last_t"] = now
            _refresh(vis)
        if not vis.poll_events():
            break
        vis.update_renderer()

    vis.destroy_window()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize Push-v1/v2 point cloud trajectories."
    )
    p.add_argument("--h5",      required=True,  help="Path to trajectory .h5 file")
    p.add_argument("--urdf",    required=False,
                   default="utils/panda_arm.urdf",)
    p.add_argument("--traj",    default="traj_0", help="Trajectory key inside the .h5")
    p.add_argument("--env",     default="v1", choices=["v1", "v2"],
                   help="Push-v1 (cube) or Push-v2 (T block)")
    p.add_argument("--every",   type=int, default=1,
                   help="Use every N-th state (default 1)")
    p.add_argument("--fps",     type=float, default=20.0,
                   help="Playback / GIF speed in frames/sec (default 10)")
    p.add_argument("--n_block", type=int, default=512)
    p.add_argument("--n_hand",  type=int, default=256)

    # ── mode ──────────────────────────────────────────────────────────────────
    p.add_argument("--render",  action="store_true",
                   help="Launch interactive open3d viewer instead of saving a GIF")
    p.add_argument("--paused",  action="store_true",
                   help="(--render only) Start paused; step with arrow keys")

    # ── GIF-only options ──────────────────────────────────────────────────────
    p.add_argument("--gif",     default="pointclouds.gif",
                   help="Output GIF path (ignored with --render)")
    p.add_argument("--elev",    type=float, default=30.0,
                   help="Camera elevation for GIF (ignored with --render)")
    p.add_argument("--azim",    type=float, default=-60.0,
                   help="Camera azimuth for GIF (ignored with --render)")
    p.add_argument("--dpi",     type=int, default=100,
                   help="DPI for GIF frames (ignored with --render)")
    return p.parse_args()


def main():
    args      = parse_args()
    rng       = np.random.default_rng(0)
    fk        = PandaFK(args.urdf)
    templates = build_templates(args.n_block, args.n_hand, rng)

    with h5py.File(args.h5, "r") as f:
        traj     = f[args.traj]
        n_states = traj["env_states"]["actors"]["push_block"].shape[0]
        steps    = list(range(0, n_states, args.every))
        print(f"'{args.traj}': {n_states} states → {len(steps)} frames (every={args.every})")

        print("Extracting point clouds...")
        all_clouds = [
            extract_step(traj, s, templates, fk, args.env == "v2")
            for s in steps
        ]

    if args.render:
        run_viewer(all_clouds, steps, fps=args.fps, start_paused=args.paused)
    else:
        lims = _compute_limits(all_clouds)
        print("Rendering frames...")
        frames = [
            render_frame(c, f"step {s}", args.elev, args.azim, lims, args.dpi)
            for s, c in zip(steps, all_clouds)
        ]
        print(f"Saving → {args.gif}")
        save_gif(frames, args.gif, args.fps)
        print(f"Done. {len(frames)} frames @ {args.fps} fps")


if __name__ == "__main__":
    if "--render" in __import__("sys").argv:
        try:
            import open3d  # noqa: F401
        except ImportError:
            print("open3d is required for --render:  pip install open3d")
            raise
    main()