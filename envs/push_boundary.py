# envs/push_env.py
"""
Push (unified): Push a block and keep it inside a rectangular boundary.

Combines Push-v1 and Push-v2 into a single configurable environment.

Parameters
----------
shape : "cube" | "T"
    Main block geometry.  "cube" reproduces Push-v1; "T" reproduces Push-v2.
num_extra_blocks : int  (default 0)
    Number of small coloured cubes scattered around the main block.
    Set to 5 to reproduce Push-v2 exactly.

Block dimension API
-------------------
After construction, `env.block_dims` is a `BlockDims` namedtuple:

    block_dims.half_x   – bounding-box half-extent along X  (metres)
    block_dims.half_y   – bounding-box half-extent along Y
    block_dims.half_z   – bounding-box half-extent along Z  (= flat half-thickness)

For shape="T" only, two additional fields are set (None for "cube"):

    block_dims.bar_half_thickness  – half the width of each bar of the T
                                     (both bars are the same width here)
    block_dims.flat_half_thickness – alias for half_z, kept for clarity

Example
-------
    env = gym.make("Push-v3",
                   shape="T",
                   num_extra_blocks=5)
    print(env.unwrapped.block_dims)
    # BlockDims(half_x=0.05, half_y=0.05, half_z=0.02,
    #           bar_half_thickness=0.0125, flat_half_thickness=0.02)
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import sapien
import sapien.render
import sapien.pysapien.physx as physx
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs  # noqa: F401
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.push_t import WhiteTableSceneBuilder
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


# ──────────────────────────────────────────────────────────────────────────────
# Shared geometry constants
# ──────────────────────────────────────────────────────────────────────────────
BOUNDARY_CENTER_X = -0.135
BOUNDARY_CENTER_Y =  0.00
BOUNDARY_HALF_X   =  0.18
BOUNDARY_HALF_Y   =  0.30

STRIP_THICKNESS = 0.006
STRIP_HALF_H    = 1e-4

OUT_MARGIN = -0.005

# Cube block
CUBE_HALF       = 0.025
CUBE_Z_SPAWN    = CUBE_HALF + 1e-3

# T block — all geometry lives here so visualize_pointclouds.py can import it
T_BAR_HALF_W    = 0.10 / 2    # half-width  of the horizontal bar
T_BAR_HALF_H    = 0.025 / 2   # half-height of the horizontal bar (= bar_half_thickness)
T_COM_Y         = 0.0375 / 2  # COM offset that centres the shape
T_HALF_THICK    = 0.02        # flat half-thickness (Z)
T_Z_SPAWN       = T_HALF_THICK + 1e-3

# Circle block
CIRCLE_RADIUS   = 0.025

# Extra cubes
SMALL_HALF  = 0.018
SMALL_Z     = SMALL_HALF + 1e-3

_EXTRA_PALETTE = [
    [0.85, 0.25, 0.25, 1.0],
    [0.25, 0.75, 0.25, 1.0],
    [0.90, 0.75, 0.15, 1.0],
    [0.80, 0.35, 0.80, 1.0],
    [0.25, 0.80, 0.80, 1.0],
]


# ──────────────────────────────────────────────────────────────────────────────
# Block dimension descriptor
# ──────────────────────────────────────────────────────────────────────────────

class BlockDims(NamedTuple):
    """Bounding-box half-extents of the main block, in its local frame."""
    half_x: float
    half_y: float
    half_z: float
    # T-shape only; both are None for a cube
    bar_half_thickness: float | None   # half the width of each bar
    flat_half_thickness: float | None  # alias for half_z, explicit for T


def _cube_dims() -> BlockDims:
    return BlockDims(
        half_x=CUBE_HALF,
        half_y=CUBE_HALF,
        half_z=CUBE_HALF,
        bar_half_thickness=None,
        flat_half_thickness=None,
    )


def _tee_dims() -> BlockDims:
    """
    Bounding box of the T in its local frame.

    The horizontal bar runs ±T_BAR_HALF_W in X.
    In Y the shape spans from (-T_COM_Y - T_BAR_HALF_H) at the bottom of the
    horizontal bar to (4*T_BAR_HALF_H - T_COM_Y + stem_half_h) at the top of
    the stem.  Both limits simplify to ±(T_BAR_HALF_W/2 + T_BAR_HALF_H/2)
    once you work through the algebra, which equals T_BAR_HALF_W (same as X).
    """
    stem_half_h = (3 / 4) * T_BAR_HALF_W          # half-height of vertical stem
    y_min = -T_COM_Y - T_BAR_HALF_H
    y_max =  4 * T_BAR_HALF_H - T_COM_Y + stem_half_h
    half_y = (y_max - y_min) / 2                   # = T_BAR_HALF_W exactly
    return BlockDims(
        half_x=T_BAR_HALF_W,
        half_y=half_y,
        half_z=T_HALF_THICK,
        bar_half_thickness=T_BAR_HALF_H,
        flat_half_thickness=T_HALF_THICK,
    )
    
def _circle_dims() -> BlockDims:
    return BlockDims(
        half_x=CIRCLE_RADIUS,
        half_y=CIRCLE_RADIUS,
        half_z=CIRCLE_RADIUS,
        bar_half_thickness=None,
        flat_half_thickness=None,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────────────

@register_env("PushBoundary", max_episode_steps=5_000)
class PushBoundaryEnv(BaseEnv):
    """
    Unified Push environment.  Registered as "Push-Boundary".

    See module docstring for full parameter documentation.
    """

    SUPPORTED_ROBOTS = ["panda_stick"]
    robot_init_qpos_noise: float = 0.00

    def __init__(
        self,
        *args,
        robot_uids: str = "panda_stick",
        robot_init_qpos_noise: float = 0.00,
        shape: str = "cube",
        num_extra_blocks: int = 0,
        valid_starts_file: str = "utils/valid_starts.npy",
        **kwargs,
    ) -> None:
        if shape not in ("cube", "T", "circle"):
            raise ValueError(f"shape must be 'cube' or 'T' or 'circle', got {shape!r}")

        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.shape = shape
        self.num_extra_blocks = num_extra_blocks
        self.block_dims: BlockDims = _cube_dims() if shape == "cube" else _tee_dims()

        self.valid_starts     = np.load(valid_starts_file)
        self.num_valid_starts = len(self.valid_starts)

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ── sim / sensor config ───────────────────────────────────────────────────
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**18,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose=pose, width=128, height=128,
                             fov=np.pi / 2, near=0.01, far=100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return CameraConfig("render_camera", pose=pose, width=512, height=512,
                            fov=1, near=0.01, far=100)

    # ── agent ─────────────────────────────────────────────────────────────────
    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    # ── scene ─────────────────────────────────────────────────────────────────
    def _load_scene(self, options: dict) -> None:
        self.table_scene = WhiteTableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Initial robot pose from valid starts
        b    = self.num_envs
        idx  = torch.randint(0, self.num_valid_starts, (b,), device=self.device)
        qpos = torch.from_numpy(self.valid_starts[idx.cpu().numpy()]).float().to(self.device)
        self.agent.reset(qpos)

        # ── main block ────────────────────────────────────────────────────────
        if self.shape == "circle":
            self.block = self._build_circle()
        elif self.shape =="cube":
            self.block = self._build_cube()
        elif self.shape == "T":
            self.block = self._build_tee()
        else:
            raise ValueError(f"Unsupported shape: {self.shape}")
        

        # ── boundary strips ───────────────────────────────────────────────────
        strip_mat = sapien.render.RenderMaterial(
            base_color=np.array([230, 80, 30, 255]) / 255
        )
        bx, by = BOUNDARY_CENTER_X, BOUNDARY_CENTER_Y
        hx, hy = BOUNDARY_HALF_X, BOUNDARY_HALF_Y
        t, sh   = STRIP_THICKNESS, STRIP_HALF_H

        self.boundary_strips = []
        for name, cx, cy, hlx, hly in [
            ("bound_top",   bx,      by + hy,  hx + t, t ),
            ("bound_bot",   bx,      by - hy,  hx + t, t ),
            ("bound_left",  bx - hx, by,       t,      hy),
            ("bound_right", bx + hx, by,       t,      hy),
        ]:
            bldr = self.scene.create_actor_builder()
            bldr.add_box_visual(half_size=[hlx, hly, sh], material=strip_mat)
            bldr.initial_pose = sapien.Pose(p=[cx, cy, sh])
            self.boundary_strips.append(bldr.build_kinematic(name=name))

        # ── extra cubes ───────────────────────────────────────────────────────
        small_phys = physx.PhysxMaterial(
            static_friction=0.8, dynamic_friction=0.6, restitution=0.05
        )
        self.extra_blocks = []
        for i in range(self.num_extra_blocks):
            color = _EXTRA_PALETTE[i % len(_EXTRA_PALETTE)]
            bldr  = self.scene.create_actor_builder()
            bldr.add_box_collision(
                half_size=[SMALL_HALF, SMALL_HALF, SMALL_HALF],
                material=small_phys,
                density=120,
            )
            bldr.add_box_visual(
                half_size=[SMALL_HALF, SMALL_HALF, SMALL_HALF],
                material=sapien.render.RenderMaterial(base_color=color),
            )
            bldr.initial_pose = sapien.Pose(p=[0.0, 0.0, -1.0])
            self.extra_blocks.append(bldr.build(name=f"extra_block_{i}"))

    # ── block builders ────────────────────────────────────────────────────────
    def _build_cube(self):
        mat = physx.PhysxMaterial(
            static_friction=1.5, dynamic_friction=1.2, restitution=0.0
        )
        blue = np.array([52, 120, 246, 255]) / 255
        bldr = self.scene.create_actor_builder()
        hs   = CUBE_HALF
        bldr.add_box_collision(half_size=[hs, hs, hs], material=mat, density=200)
        bldr.add_box_visual(
            half_size=[hs, hs, hs],
            material=sapien.render.RenderMaterial(
                base_color=blue, metallic=0.0, roughness=0.5
            ),
        )
        bldr.initial_pose = sapien.Pose(p=[0.0, 0.0, CUBE_Z_SPAWN])
        return bldr.build(name="push_block")

    def _build_tee(self):
        mat  = physx.PhysxMaterial(
            static_friction=1.0, dynamic_friction=0.8, restitution=0.0
        )
        blue = np.array([52, 120, 246, 255]) / 255
        bldr = self.scene.create_actor_builder()
        bldr._mass = 0.12

        stem_half_h = (3 / 4) * T_BAR_HALF_W

        # horizontal bar
        bldr.add_box_collision(
            pose=sapien.Pose([0.0, -T_COM_Y, 0.0]),
            half_size=[T_BAR_HALF_W, T_BAR_HALF_H, T_HALF_THICK],
            material=mat,
        )
        bldr.add_box_visual(
            pose=sapien.Pose([0.0, -T_COM_Y, 0.0]),
            half_size=[T_BAR_HALF_W, T_BAR_HALF_H, T_HALF_THICK],
            material=sapien.render.RenderMaterial(base_color=blue),
        )
        # vertical stem
        bldr.add_box_collision(
            pose=sapien.Pose([0.0, 4 * T_BAR_HALF_H - T_COM_Y, 0.0]),
            half_size=[T_BAR_HALF_H, stem_half_h, T_HALF_THICK],
            material=mat,
        )
        bldr.add_box_visual(
            pose=sapien.Pose([0.0, 4 * T_BAR_HALF_H - T_COM_Y, 0.0]),
            half_size=[T_BAR_HALF_H, stem_half_h, T_HALF_THICK],
            material=sapien.render.RenderMaterial(base_color=blue),
        )

        bldr.initial_pose = sapien.Pose(p=[0.0, 0.0, T_Z_SPAWN])
        return bldr.build(name="push_block")
    
    def _build_circle(self):
        mat  = physx.PhysxMaterial(
            static_friction=1.0, dynamic_friction=0.8, restitution=0.0
        )
        blue = np.array([52, 120, 246, 255]) / 255
        bldr = self.scene.create_actor_builder()
        bldr._mass = 0.12

        # main body
        bldr.add_cylinder_collision(
            pose = sapien.Pose(p=[0.0, 0.0, CIRCLE_RADIUS + 1e-3], q=euler2quat(0, np.pi / 2, 0)),
            radius=CIRCLE_RADIUS, half_length=CIRCLE_RADIUS, material=mat, density=200
        )
        bldr.add_cylinder_visual(
            pose = sapien.Pose(p=[0.0, 0.0, CIRCLE_RADIUS + 1e-3], q=euler2quat(0, np.pi / 2, 0)),
            radius=CIRCLE_RADIUS, half_length=CIRCLE_RADIUS,
            material=sapien.render.RenderMaterial(base_color=blue),
        )

        bldr.initial_pose = sapien.Pose(p=[0.0, 0.0, 0.0])
        return bldr.build(name="push_block")

    # ── episode init ──────────────────────────────────────────────────────────
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Robot
            idx  = torch.randint(0, self.num_valid_starts, (b,), device=self.device)
            qpos = torch.from_numpy(self.valid_starts[idx.cpu().numpy()]).float().to(self.device)
            self.agent.reset(qpos)

            # Boundary strips
            bx, by = BOUNDARY_CENTER_X, BOUNDARY_CENTER_Y
            sh     = STRIP_HALF_H
            hx, hy, t = BOUNDARY_HALF_X, BOUNDARY_HALF_Y, STRIP_THICKNESS
            for strip, pos in zip(self.boundary_strips, [
                [bx,      by + hy, sh],
                [bx,      by - hy, sh],
                [bx - hx, by,      sh],
                [bx + hx, by,      sh],
            ]):
                p = torch.tensor([pos], dtype=torch.float32,
                                  device=self.device).expand(b, -1)
                strip.set_pose(Pose.create_from_pq(p=p))

            # Main block — spawn randomly inside boundary with shape-aware margin
            margin   = (0.02 if self.shape == "cube" else 0.12) + max(
                self.block_dims.half_x, self.block_dims.half_y
            )
            spawn_hx = max(BOUNDARY_HALF_X - margin, 0.01)
            spawn_hy = max(BOUNDARY_HALF_Y - margin, 0.01)
            z_spawn  = CUBE_Z_SPAWN if self.shape == "cube" else T_Z_SPAWN

            bx_r = (torch.rand(b, device=self.device) * 2 - 1) * spawn_hx + BOUNDARY_CENTER_X
            by_r = (torch.rand(b, device=self.device) * 2 - 1) * spawn_hy + BOUNDARY_CENTER_Y
            bz_r = torch.full((b,), z_spawn, device=self.device)

            yaw = torch.rand(b, device=self.device) * 2 * np.pi
            c, s = (yaw / 2).cos(), (yaw / 2).sin()
            bq   = torch.stack([c, torch.zeros_like(c), torch.zeros_like(c), s], dim=1)
            self.block.set_pose(Pose.create_from_pq(
                p=torch.stack([bx_r, by_r, bz_r], dim=1), q=bq
            ))

            # Extra cubes
            if self.extra_blocks:
                self._place_extra_blocks(b, bx_r.cpu().numpy(), by_r.cpu().numpy())

    def _place_extra_blocks(
        self,
        b: int,
        main_xs: np.ndarray,
        main_ys: np.ndarray,
    ) -> None:
        rng      = np.random.default_rng()
        spawn_hx = BOUNDARY_HALF_X - SMALL_HALF - 0.01
        spawn_hy = BOUNDARY_HALF_Y - SMALL_HALF - 0.01
        main_xy  = np.array([float(main_xs[0]), float(main_ys[0])])
        placed: list[np.ndarray] = []

        for obj in self.extra_blocks:
            for _ in range(60):
                x  = rng.uniform(BOUNDARY_CENTER_X - spawn_hx,
                                 BOUNDARY_CENTER_X + spawn_hx)
                y  = rng.uniform(BOUNDARY_CENTER_Y - spawn_hy,
                                 BOUNDARY_CENTER_Y + spawn_hy)
                xy = np.array([x, y])
                if np.linalg.norm(xy - main_xy) < 0.09:
                    continue
                if any(np.linalg.norm(xy - p) < 0.07 for p in placed):
                    continue
                break
            placed.append(xy)
            obj.set_pose(sapien.Pose(
                p=[float(xy[0]), float(xy[1]), SMALL_Z],
                q=[1, 0, 0, 0],
            ))

    # ── evaluate ──────────────────────────────────────────────────────────────
    def evaluate(self) -> dict:
        bpos  = self.block.pose.p
        out_x = (bpos[:, 0] < BOUNDARY_CENTER_X - BOUNDARY_HALF_X - OUT_MARGIN) | \
                (bpos[:, 0] > BOUNDARY_CENTER_X + BOUNDARY_HALF_X + OUT_MARGIN)
        out_y = (bpos[:, 1] < BOUNDARY_CENTER_Y - BOUNDARY_HALF_Y - OUT_MARGIN) | \
                (bpos[:, 1] > BOUNDARY_CENTER_Y + BOUNDARY_HALF_Y + OUT_MARGIN)
        false_t = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return {"success": false_t, "fail": out_x | out_y}

    # ── observations ──────────────────────────────────────────────────────────
    def _get_obs_extra(self, info: dict) -> dict:
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                block_pose=self.block.pose.raw_pose,
                boundary_center=torch.tensor(
                    [BOUNDARY_CENTER_X, BOUNDARY_CENTER_Y],
                    dtype=torch.float32, device=self.device,
                ).unsqueeze(0).expand(self.num_envs, -1),
            )
        return obs

    # ── reward ────────────────────────────────────────────────────────────────
    def compute_dense_reward(self, obs: Any, action: Array, info: dict) -> torch.Tensor:
        tcp_dist = torch.linalg.norm(
            self.block.pose.p - self.agent.tcp.pose.p, dim=1
        )
        approach = (1 - torch.tanh(5 * tcp_dist)) * 0.5

        z_spawn = CUBE_Z_SPAWN if self.shape == "cube" else T_Z_SPAWN
        centre  = torch.tensor(
            [BOUNDARY_CENTER_X, BOUNDARY_CENTER_Y, z_spawn],
            dtype=torch.float32, device=self.device,
        )
        block_dist = torch.linalg.norm(
            self.block.pose.p - centre.unsqueeze(0), dim=1
        )
        keep_in = (1 - torch.tanh(5 * block_dist)) * 0.5

        reward = approach + keep_in
        reward[info["fail"]] = 0.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict) -> torch.Tensor:
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 1.0