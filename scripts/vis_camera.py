#!/usr/bin/env python3
"""
scripts/vis_camera_o3d.py

Visualize a single frame from an HDF5 trajectory as a point cloud with
camera visibility coloring, powered by Open3D mesh raycasting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import open3d as o3d

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

# ─────────────────────────────────────────────────────────────────────────────
# Geometry constants
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

STICK_RADIUS            = 0.008
STICK_LENGTH            = 0.10
STICK_OFFSET_Z_FROM_TCP = 0.10

ACTOR_POS  = slice(0, 3)
ACTOR_QUAT = slice(3, 7)
ART_JPOS   = slice(13, 20)

COLOR_VISIBLE  = np.array([0.15, 0.80, 0.30])   # green
COLOR_OCCLUDED = np.array([0.10, 0.10, 0.10])   # near-black

# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
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

def _pose_to_mat(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Convert position and quaternion to a 4x4 transformation matrix."""
    T = np.eye(4)
    T[:3, :3] = _quat_to_rot(quat)
    T[:3, 3] = pos
    return T

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

# ─────────────────────────────────────────────────────────────────────────────
# Camera & FK (Unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def make_camera(pos, target, up=None, fov_deg=60.0):
    forward = _normalize(np.asarray(target, float) - np.asarray(pos, float))
    if up is None:
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(forward, up)) > 0.99:
            up = np.array([0.0, 1.0, 0.0])
    right = _normalize(np.cross(forward, up))
    up    = _normalize(np.cross(right, forward))
    return {
        "pos":     np.asarray(pos, dtype=np.float64),
        "forward": forward,
        "right":   right,
        "up":      up,
        "fov_rad": np.deg2rad(fov_deg) / 2,
    }

class PandaFK:
    def __init__(self, urdf_path: str):
        import pinocchio as pin
        self._pin = pin
        model = pin.buildModelFromUrdf(urdf_path)
        self._model = model
        self._data  = model.createData()
        self._fid   = model.getFrameId("panda_hand")

    def hand_pose(self, joint_angles, robot_base_pos, robot_base_quat):
        pin = self._pin; model = self._model; data = self._data
        q = pin.neutral(model)
        q[:len(joint_angles)] = joint_angles.astype(np.float64)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        T       = data.oMf[self._fid]
        R_base  = _quat_to_rot(robot_base_quat)
        p_world = R_base @ T.translation + robot_base_pos.astype(np.float64)
        R_world = R_base @ T.rotation
        return p_world.astype(np.float32), _rot_to_quat(R_world).astype(np.float32)

def _draw_camera(ax, cam, scale=0.08):
    pos, fwd, right, up, hfov = cam["pos"], cam["forward"], cam["right"], cam["up"], cam["fov_rad"]
    circle_r, axis_len = scale * 0.35, scale
    
    # Circle
    theta  = np.linspace(0, 2*np.pi, 64)
    circle = pos[np.newaxis] + circle_r * np.cos(theta[:, np.newaxis]) * right[np.newaxis] + circle_r * np.sin(theta[:, np.newaxis]) * up[np.newaxis]
    ax.plot(circle[:, 0], circle[:, 1], circle[:, 2], color="red", linewidth=1.8, zorder=5)
    
    # Axis line
    axis_end = pos + fwd * axis_len
    ax.plot([pos[0], axis_end[0]], [pos[1], axis_end[1]], [pos[2], axis_end[2]], color="red", linewidth=2.0, zorder=5)
    
    # FOV cone lines
    cone_len, tan_hfov = axis_len * 1.4, np.tan(hfov)
    corners = [
        fwd * cone_len + right * tan_hfov * cone_len + up * tan_hfov * cone_len,
        fwd * cone_len + right * -tan_hfov * cone_len + up * tan_hfov * cone_len,
        fwd * cone_len + right * -tan_hfov * cone_len + up * -tan_hfov * cone_len,
        fwd * cone_len + right * tan_hfov * cone_len + up * -tan_hfov * cone_len,
    ]
    for c in corners:
        tip = pos + c
        ax.plot([pos[0], tip[0]], [pos[1], tip[1]], [pos[2], tip[2]], color="red", linewidth=0.7, linestyle="--", alpha=0.6, zorder=4)
    
    tips = [pos + c for c in corners]
    for i in range(4):
        a, b = tips[i], tips[(i+1) % 4]
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="red", linewidth=0.7, linestyle="--", alpha=0.5, zorder=4)
    ax.scatter(*pos, color="red", s=30, zorder=6)

# ─────────────────────────────────────────────────────────────────────────────
# Open3D Geometry & Raycasting
# ─────────────────────────────────────────────────────────────────────────────

def build_base_meshes(use_T: bool) -> dict:
    """Create Open3D meshes at the origin before FK is applied."""
    if use_T:
        block_mesh = o3d.geometry.TriangleMesh()
        for (off, half) in T_BOXES:
            hx, hy, hz = half
            box = o3d.geometry.TriangleMesh.create_box(width=hx*2, height=hy*2, depth=hz*2)
            box.translate([-hx, -hy, -hz]) # Center it
            box.translate(off)             # Apply T-shape offset
            block_mesh += box
    else:
        hx = hy = hz = BLOCK_HALF
        block_mesh = o3d.geometry.TriangleMesh.create_box(width=hx*2, height=hy*2, depth=hz*2)
        block_mesh.translate([-hx, -hy, -hz])

    hand_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=STICK_RADIUS, height=STICK_LENGTH)
    # Open3D cylinders are centered at origin and run along the Z axis
    hand_mesh.translate([0, 0, STICK_OFFSET_Z_FROM_TCP])

    return {"block": block_mesh, "hand": hand_mesh}

def compute_visibility_o3d(points: np.ndarray, meshes: list, cam: dict) -> np.ndarray:
    """
    Cast rays from camera to points against solid meshes. 
    Returns boolean array of visibility.
    """
    cam_pos = cam["pos"]
    fwd     = cam["forward"]
    hfov    = cam["fov_rad"]

    # 1. Setup Open3D Raycasting Scene
    scene = o3d.t.geometry.RaycastingScene()
    for mesh in meshes:
        # Convert legacy mesh to tensor mesh for raycasting
        t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene.add_triangles(t_mesh)

    # 2. Setup Rays
    vecs = points.astype(np.float32) - cam_pos.astype(np.float32)[np.newaxis]
    dists = np.linalg.norm(vecs, axis=1)
    
    # Avoid division by zero
    valid = dists > 1e-9
    dirs = np.where(valid[:, np.newaxis], vecs / dists[:, np.newaxis], 0.0)

    # Array shape for Open3D rays is (N, 6) -> [ox, oy, oz, dx, dy, dz]
    rays = np.zeros((len(points), 6), dtype=np.float32)
    rays[:, :3] = cam_pos
    rays[:, 3:] = dirs
    rays_t = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

    # 3. Cast Rays
    ans = scene.cast_rays(rays_t)
    hit_dists = ans['t_hit'].numpy()

    # 4. Check FOV and Occlusion
    cos_angles = dirs @ fwd
    in_fov = cos_angles > np.cos(hfov)

    # Visible if the ray hits the point (or passes it) before hitting a solid mesh.
    # We subtract a tiny epsilon (1e-4) from dists to prevent points self-occluding 
    # against the exact surface they are sampled from.
    not_occluded = hit_dists >= (dists - 1e-4)
    
    return in_fov & not_occluded

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

VIEWS = [
    {"name": "Front-side",  "elev": 20,  "azim": -60},
    {"name": "Top-down",    "elev": 88,  "azim": -90},
    {"name": "Side",        "elev": 10,  "azim":   0},
]

def visualize(
    h5_path, urdf_path, traj_key="traj_0", frame_idx=0, use_T=False,
    cam_pos=None, cam_target=None, fov_deg=60.0,
    n_block=600, n_hand=350, out_path=None, dpi=130
):
    fk = PandaFK(urdf_path)
    
    # ── 1. Create Base Meshes
    base_meshes = build_base_meshes(use_T)

    # ── 2. Load Frame from HDF5
    with h5py.File(h5_path, "r") as f:
        traj = f[traj_key]
        n_states = traj["env_states"]["articulations"]["panda_stick"].shape[0]
        t = min(frame_idx, n_states - 1)

        art = traj["env_states"]["articulations"]["panda_stick"][t]
        hp, hq = fk.hand_pose(art[ART_JPOS], art[ACTOR_POS], art[ACTOR_QUAT])

        bs = traj["env_states"]["actors"]["push_block"][t]
        block_pos, block_quat = bs[ACTOR_POS], bs[ACTOR_QUAT]

    # ── 3. Transform Meshes
    block_mesh = base_meshes["block"].transform(_pose_to_mat(block_pos, block_quat))
    hand_mesh  = base_meshes["hand"].transform(_pose_to_mat(hp, hq))

    # ── 4. Sample Points for Display
    block_pcd = block_mesh.sample_points_uniformly(number_of_points=n_block)
    hand_pcd  = hand_mesh.sample_points_uniformly(number_of_points=n_hand)
    
    block_pts = np.asarray(block_pcd.points)
    hand_pts  = np.asarray(hand_pcd.points)
    all_pts   = np.concatenate([hand_pts, block_pts], axis=0)

    # ── 5. Auto-place camera
    scene_center = all_pts.mean(axis=0)
    scene_extent = (all_pts.max(0) - all_pts.min(0)).max()

    if cam_target is None: cam_target = scene_center
    if cam_pos is None:
        offset = _normalize(np.array([-0.3, -1.0, 0.6])) * scene_extent * 1.5
        cam_pos = scene_center + offset

    cam = make_camera(cam_pos, cam_target, fov_deg=fov_deg)

    # ── 6. Compute Visibility via Raycasting
    print("Casting rays against solid meshes...")
    visible = compute_visibility_o3d(all_pts, [block_mesh, hand_mesh], cam)
    
    n_vis = visible.sum()
    print(f"Visible: {n_vis}/{len(all_pts)} points ({100*n_vis/len(all_pts):.1f}%)")

    # Matplotlib depth sorting fix: Plot ALL points at once
    colors = np.where(visible[:, np.newaxis], COLOR_VISIBLE, COLOR_OCCLUDED)

    # ── 7. Plotting
    all_and_cam = np.vstack([all_pts, cam_pos[np.newaxis]])
    center = all_and_cam.mean(0)
    radius = max((all_and_cam.max(0) - all_and_cam.min(0)).max() / 2 + 0.12, 0.15)
    lims   = [(float(center[i]-radius), float(center[i]+radius)) for i in range(3)]

    # We use a standard backend to allow interactivity if --out is not provided
    if out_path:
        import matplotlib
        matplotlib.use("Agg")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={"projection": "3d"}, dpi=dpi)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_VISIBLE, markersize=6, label=f"visible ({visible.sum()})"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_OCCLUDED, markersize=6, label=f"occluded ({(~visible).sum()})")
    ]

    for ax, view in zip(axes, VIEWS):
        # The single scatter call fixes the depth sorting issue
        ax.scatter(all_pts[:,0], all_pts[:,1], all_pts[:,2],
                   s=2.5, c=colors, depthshade=False, alpha=0.85)

        _draw_camera(ax, cam, scale=radius * 0.35)

        ax.view_init(elev=view["elev"], azim=view["azim"])
        ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(view["name"], fontsize=10, pad=4)
        ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7); ax.set_zlabel("Z", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(handles=legend_elements, fontsize=7, loc="upper right", framealpha=0.5)

    fig.suptitle(f"{traj_key} frame={t} | cam @ {np.round(cam_pos, 2)} FOV={fov_deg:.0f}°", fontsize=9, y=1.01)
    fig.tight_layout(pad=0.5)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved → {out_path}")
    else:
        plt.show()

    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# CLI (Unchanged except removing point_radius argument)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_vec3(s: str) -> np.ndarray:
    return np.array([float(v) for v in s.split(",")], dtype=np.float64)

def parse_args():
    p = argparse.ArgumentParser(description="Visualize HDF5 frame as point cloud with Open3D mesh raycasting.")
    p.add_argument("--h5",          required=True)
    p.add_argument("--urdf",        default="utils/panda_arm.urdf")
    p.add_argument("--traj",        default="traj_0")
    p.add_argument("--frame",       type=int,   default=100)
    p.add_argument("--env",         default="v1", choices=["v1","v2"])
    p.add_argument("--cam_pos",     default=None)
    p.add_argument("--cam_target",  default=None)
    p.add_argument("--fov",         type=float, default=90.0)
    p.add_argument("--n_block",     type=int,   default=600)
    p.add_argument("--n_hand",      type=int,   default=350)
    p.add_argument("--out",         default=None)
    p.add_argument("--dpi",         type=int,   default=300)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    visualize(
        h5_path    = args.h5,
        urdf_path  = args.urdf,
        traj_key   = args.traj,
        frame_idx  = args.frame,
        use_T      = args.env == "v2",
        cam_pos    = _parse_vec3(args.cam_pos)    if args.cam_pos    else None,
        cam_target = _parse_vec3(args.cam_target) if args.cam_target else None,
        fov_deg    = args.fov,
        n_block    = args.n_block,
        n_hand     = args.n_hand,
        out_path   = args.out,
        dpi        = args.dpi,
    )