#!/usr/bin/env python3
"""
scripts/vis_actions.py

Visualize actions as a "ghost" future gripper point cloud overlaid on the
current state. At each frame, the current hand is rendered in orange and the
predicted future hand positions (from FK on state t+lookahead) are rendered in
a different color with alpha transparency.

Usage
-----
    # GIF (every 5th frame, 1-step lookahead)
    python scripts/vis_actions.py \
        --h5 demos/traj.h5 --urdf utils/panda_arm.urdf \
        --gif out.gif --every 5

    # Interactive open3d viewer, 3-step lookahead
    python scripts/vis_actions.py \
        --h5 demos/traj.h5 --urdf utils/panda_arm.urdf \
        --render --lookahead 3

    # Skip block point cloud (faster)
    python scripts/vis_actions.py \
        --h5 demos/traj.h5 --urdf utils/panda_arm.urdf \
        --no_block --render

Controls (--render)
-------------------
    SPACE        pause / resume
    LEFT / RIGHT step one frame (when paused)
    Q / ESC      quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

BLOCK_HALF = 0.025
_T_box1_hw = 0.10 / 2
_T_box1_hh = 0.025 / 2
_T_com_y   = 0.0375 / 2
_T_half_t  = 0.02
T_BOXES = [
    (np.array([0.0, -_T_com_y, 0.0]),               np.array([_T_box1_hw, _T_box1_hh, _T_half_t])),
    (np.array([0.0, 4*_T_box1_hh - _T_com_y, 0.0]), np.array([_T_box1_hh, (3/4)*_T_box1_hw, _T_half_t])),
]

STICK_RADIUS   = 0.008
STICK_LENGTH   = 0.10
STICK_OFFSET_Z = 0.10

ACTOR_POS  = slice(0, 3)
ACTOR_QUAT = slice(3, 7)
ART_JPOS   = slice(13, 20)

BLOCK_COLOR = np.array([0.20, 0.47, 0.96])   # blue
HAND_COLOR  = np.array([0.91, 0.47, 0.10])   # orange  — current hand
GHOST_COLOR = np.array([0.20, 0.95, 0.60])   # green   — future hand


# ─────────────────────────────────────────────────────────────────────────────
# Math
# ─────────────────────────────────────────────────────────────────────────────

def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q.astype(np.float64)
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
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


def apply_pose(pts: np.ndarray, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    R = _quat_to_rot(quat)
    return (pts.astype(np.float64) @ R.T + pos.astype(np.float64)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# FK
# ─────────────────────────────────────────────────────────────────────────────

class PandaFK:
    def __init__(self, urdf_path: str):
        try:
            import pinocchio as pin
        except ImportError:
            raise ImportError("Install pinocchio: conda install pinocchio -c conda-forge")
        self._pin = pin
        model = pin.buildModelFromUrdf(urdf_path)
        self._model = model
        self._data  = model.createData()
        self._fid   = model.getFrameId("panda_hand")
        print(f"Loaded URDF ({model.nq} DOF), panda_hand frame id={self._fid}")

    def hand_pose(self, joint_angles, robot_base_pos, robot_base_quat):
        pin   = self._pin
        model = self._model
        data  = self._data
        q = pin.neutral(model)
        q[:len(joint_angles)] = joint_angles.astype(np.float64)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        T = data.oMf[self._fid]
        R_base  = _quat_to_rot(robot_base_quat)
        p_world = R_base @ T.translation + robot_base_pos.astype(np.float64)
        R_world = R_base @ T.rotation
        return p_world.astype(np.float32), _rot_to_quat(R_world).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Point samplers
# ─────────────────────────────────────────────────────────────────────────────

def _sample_box(offset, half, n, rng):
    try:
        import fpsample
        use_fps = True
    except ImportError:
        use_fps = False
    n_init = n * 20
    hx, hy, hz = half
    areas  = np.array([4*hy*hz, 4*hx*hz, 4*hx*hy], dtype=np.float64)
    counts = np.round(areas / areas.sum() * n_init).astype(int)
    counts[-1] += n_init - counts.sum()
    pts = []
    if counts[0] > 0:
        s = rng.choice([-1.0, 1.0], counts[0])
        pts.append(np.stack([s*hx, rng.uniform(-hy,hy,counts[0]), rng.uniform(-hz,hz,counts[0])], 1))
    if counts[1] > 0:
        s = rng.choice([-1.0, 1.0], counts[1])
        pts.append(np.stack([rng.uniform(-hx,hx,counts[1]), s*hy, rng.uniform(-hz,hz,counts[1])], 1))
    if counts[2] > 0:
        s = rng.choice([-1.0, 1.0], counts[2])
        pts.append(np.stack([rng.uniform(-hx,hx,counts[2]), rng.uniform(-hy,hy,counts[2]), s*hz], 1))
    initial = np.concatenate(pts).astype(np.float32) + offset.astype(np.float32)
    if use_fps and len(initial) > n:
        return initial[fpsample.fps_sampling(initial, n)]
    idx = rng.choice(len(initial), size=min(n, len(initial)), replace=False)
    return initial[idx]


def _sample_cylinder(offset_z, radius, length, n, rng):
    try:
        import fpsample
        use_fps = True
    except ImportError:
        use_fps = False
    n_init = n * 20
    th = rng.uniform(0, 2*np.pi, n_init)
    z  = rng.uniform(-length/2, length/2, n_init) + offset_z
    pts = np.stack([radius*np.cos(th), radius*np.sin(th), z], 1).astype(np.float32)
    if use_fps and len(pts) > n:
        return pts[fpsample.fps_sampling(pts, n)]
    idx = rng.choice(len(pts), size=min(n, len(pts)), replace=False)
    return pts[idx]


def build_templates(n_block: int, n_hand: int, rng) -> dict:
    cube_tpl = _sample_box(np.zeros(3), np.full(3, BLOCK_HALF), n_block, rng)
    areas    = [2*(4*hy*hz+4*hx*hz+4*hx*hy) for _, (hx,hy,hz) in T_BOXES]
    total    = sum(areas)
    t_parts  = [_sample_box(off, half, max(1, int(n_block*a/total)), rng)
                for (off, half), a in zip(T_BOXES, areas)]
    t_tpl    = np.concatenate(t_parts)
    hand_tpl = _sample_cylinder(STICK_OFFSET_Z, STICK_RADIUS, STICK_LENGTH, n_hand, rng)
    return {"cube": cube_tpl, "T": t_tpl, "hand": hand_tpl}


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame extraction
# ─────────────────────────────────────────────────────────────────────────────

def hand_from_state(arts, step, templates, fk):
    art = arts["panda_stick"][step]
    hp, hq = fk.hand_pose(art[ART_JPOS], art[ACTOR_POS], art[ACTOR_QUAT])
    return apply_pose(templates["hand"], hp, hq)


def extract_frame(traj, step, templates, fk, use_T, no_block, lookahead):
    actors   = traj["env_states"]["actors"]
    arts     = traj["env_states"]["articulations"]
    n_states = arts["panda_stick"].shape[0]

    hand_now    = hand_from_state(arts, step, templates, fk)
    future_step = min(step + lookahead, n_states - 1)
    hand_future = hand_from_state(arts, future_step, templates, fk)

    block = None
    if not no_block:
        bs    = actors["push_block"][step]
        tpl   = templates["T"] if use_T else templates["cube"]
        block = apply_pose(tpl, bs[ACTOR_POS], bs[ACTOR_QUAT])

    return {"block": block, "hand_now": hand_now, "hand_future": hand_future}


# ─────────────────────────────────────────────────────────────────────────────
# GIF rendering
# ─────────────────────────────────────────────────────────────────────────────

def _scene_limits(all_frames):
    pts_all = [f["hand_now"] for f in all_frames] + [f["hand_future"] for f in all_frames]
    pts_all += [f["block"] for f in all_frames if f["block"] is not None]
    pts = np.concatenate(pts_all)
    pad = 0.08
    c   = pts.mean(0)
    r   = max((pts.max(0) - pts.min(0)).max() / 2 + pad, 0.1)
    return [(c[i]-r, c[i]+r) for i in range(3)]


def render_frame_gif(frame_data, step, elev, azim, lims, dpi, ghost_alpha):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 6), dpi=dpi)
    ax  = fig.add_subplot(111, projection="3d")

    if frame_data["block"] is not None:
        b = frame_data["block"]
        ax.scatter(b[:,0], b[:,1], b[:,2], s=1.0,
                   c=[BLOCK_COLOR.tolist()], label="block",
                   depthshade=True, alpha=0.55)

    h = frame_data["hand_now"]
    ax.scatter(h[:,0], h[:,1], h[:,2], s=2.5,
               c=[HAND_COLOR.tolist()], label="hand (now)",
               depthshade=True, alpha=0.9)

    g = frame_data["hand_future"]
    ax.scatter(g[:,0], g[:,1], g[:,2], s=2.5,
               c=[GHOST_COLOR.tolist()], label="hand (future)",
               depthshade=True, alpha=ghost_alpha)

    ax.view_init(elev=elev, azim=azim)
    for i, (lo, hi) in enumerate(lims):
        [ax.set_xlim, ax.set_ylim, ax.set_zlim][i](lo, hi)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7)
    ax.set_zlabel("Z", fontsize=7); ax.tick_params(labelsize=6)
    ax.set_title(f"step {step}", fontsize=8)
    ax.legend(fontsize=7, loc="upper right", markerscale=4)
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    w, h_px = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = buf.reshape(h_px, w, 4).copy()
    plt.close(fig)
    return img


def save_gif(frames_img, path, fps):
    try:
        import imageio
        imageio.mimsave(path, frames_img, duration=1.0/fps, loop=0)
        return
    except ImportError:
        pass
    from PIL import Image
    pil = [Image.fromarray(f) for f in frames_img]
    pil[0].save(path, save_all=True, append_images=pil[1:],
                duration=int(1000/fps), loop=0)


# ─────────────────────────────────────────────────────────────────────────────
# Open3D interactive viewer
# ─────────────────────────────────────────────────────────────────────────────

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


def run_viewer(all_frames, steps, fps=10.0, start_paused=False, ghost_alpha=0.4):
    import open3d as o3d

    # open3d has no per-cloud alpha — blend ghost color toward background
    BG = np.array([0.08, 0.08, 0.08])
    ghost_blended = ghost_alpha * GHOST_COLOR + (1 - ghost_alpha) * BG

    n   = len(all_frames)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Hand Ghost Viewer", width=900, height=900)
    opt = vis.get_render_option()
    opt.background_color = BG
    opt.point_size       = 3.5

    f0    = all_frames[0]
    geoms = {}

    if f0["block"] is not None:
        geoms["block"] = _make_pcd(f0["block"], BLOCK_COLOR)
        vis.add_geometry(geoms["block"])

    geoms["hand_now"]    = _make_pcd(f0["hand_now"],    HAND_COLOR)
    geoms["hand_future"] = _make_pcd(f0["hand_future"], ghost_blended)
    vis.add_geometry(geoms["hand_now"])
    vis.add_geometry(geoms["hand_future"])
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05))

    state = {"frame": 0, "paused": start_paused, "quit": False, "last_t": time.time()}

    def _refresh(vis):
        f = all_frames[state["frame"]]
        if "block" in geoms and f["block"] is not None:
            _update_pcd(geoms["block"], f["block"], BLOCK_COLOR)
            vis.update_geometry(geoms["block"])
        _update_pcd(geoms["hand_now"],    f["hand_now"],    HAND_COLOR)
        _update_pcd(geoms["hand_future"], f["hand_future"], ghost_blended)
        vis.update_geometry(geoms["hand_now"])
        vis.update_geometry(geoms["hand_future"])

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

    vis.register_key_callback(32,       on_space)
    vis.register_key_callback(263,      on_left)
    vis.register_key_callback(262,      on_right)
    vis.register_key_callback(ord("Q"), on_quit)
    vis.register_key_callback(27,       on_quit)

    frame_dt = 1.0 / fps
    print(f"\nGhost Viewer: {n} frames @ {fps} fps")
    print(f"  Orange = current hand   Green (faded) = future hand")
    print(f"  SPACE=pause/resume  ←→=step  Q/ESC=quit\n")

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


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Show future gripper state as a ghost point cloud overlay."
    )
    p.add_argument("--h5",          required=True)
    p.add_argument("--urdf",        default="utils/panda_arm.urdf")
    p.add_argument("--traj",        default="traj_0")
    p.add_argument("--env",         default="v1", choices=["v1", "v2"],
                   help="v1=cube block, v2=T block")
    p.add_argument("--lookahead",   type=int,   default=1,
                   help="How many states ahead to render the ghost hand (default 1)")
    p.add_argument("--every",       type=int,   default=5,
                   help="Use every N-th state (default 5)")
    p.add_argument("--fps",         type=float, default=15.0)
    p.add_argument("--ghost_alpha", type=float, default=0.35,
                   help="Opacity of the future hand cloud, 0–1 (default 0.35)")
    p.add_argument("--n_block",     type=int,   default=400)
    p.add_argument("--n_hand",      type=int,   default=250)
    p.add_argument("--no_block",    action="store_true",
                   help="Skip block point cloud")
    p.add_argument("--render",      action="store_true",
                   help="Interactive open3d viewer (default: save GIF)")
    p.add_argument("--paused",      action="store_true")
    p.add_argument("--gif",         default="ghost_hand.gif")
    p.add_argument("--elev",        type=float, default=30.0)
    p.add_argument("--azim",        type=float, default=-60.0)
    p.add_argument("--dpi",         type=int,   default=100)
    return p.parse_args()


def main():
    args = parse_args()
    rng  = np.random.default_rng(0)
    fk   = PandaFK(args.urdf)
    templates = build_templates(args.n_block, args.n_hand, rng)
    use_T = (args.env == "v2")

    with h5py.File(args.h5, "r") as f:
        traj     = f[args.traj]
        n_states = traj["env_states"]["articulations"]["panda_stick"].shape[0]
        steps    = list(range(0, n_states, args.every))
        print(f"'{args.traj}': {n_states} states → {len(steps)} frames "
              f"(every={args.every}, lookahead={args.lookahead})")
        print("Extracting frames ...")
        all_frames = [
            extract_frame(traj, s, templates, fk, use_T, args.no_block, args.lookahead)
            for s in steps
        ]

    if args.render:
        run_viewer(all_frames, steps, fps=args.fps,
                   start_paused=args.paused, ghost_alpha=args.ghost_alpha)
    else:
        lims = _scene_limits(all_frames)
        frames_img = []
        for i, (fd, s) in enumerate(zip(all_frames, steps)):
            img = render_frame_gif(fd, s, args.elev, args.azim,
                                   lims, args.dpi, args.ghost_alpha)
            frames_img.append(img)
            if i % 20 == 0:
                print(f"  rendered {i}/{len(all_frames)} frames")
        print(f"Saving → {args.gif}")
        save_gif(frames_img, args.gif, args.fps)
        print(f"Done. {len(frames_img)} frames @ {args.fps} fps")


if __name__ == "__main__":
    if "--render" in sys.argv:
        try:
            import open3d  # noqa: F401
        except ImportError:
            print("open3d required for --render:  pip install open3d")
            raise
    main()