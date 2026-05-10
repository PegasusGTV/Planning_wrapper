"""PushO with obstacle blocks on the table, plus skill-specific subclasses."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import sapien
import torch
import mani_skill.envs  # noqa: F401
from mani_skill.agents.robots import Panda
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
import sapien.physx as physx
import sapien.render
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.scene_builder.table import TableSceneBuilder

# SAPIEN 3 cylinders have their principal axis along local-x.
# Rotating -90° around y maps x→z so the disk lies flat on the table.
_DISK_QUAT = [0.7071, 0.0, -0.7071, 0.0]


@register_env("PushO-v1", max_episode_steps=200)
class PushOEnv(BaseEnv):
    """Push a circular disk (flat cylinder) to a same-radius goal region on the table.

    Success when the disk overlaps ≥90% with the goal circle, computed analytically
    from the XY distance between their centres.

    Observation extras: ee_pos(3), disk_pos(3), goal_pos(3),
                        ee_to_disk(3), disk_to_goal(3), overlap_frac(1)
    Reward: reach + push + overlap + finger_close; terminal bonus 5.0 on success.
    """

    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    disk_radius: float = 0.05
    disk_half_thickness: float = 0.02
    disk_mass: float = 0.3
    disk_static_friction: float = 0.8
    disk_dynamic_friction: float = 0.8

    GOAL_HALF: float = 0.15
    DISK_SPAWN_HALF: float = 0.12
    MIN_DISK_GOAL_DIST: float = 0.08
    SUCCESS_OVERLAP: float = 0.90

    def __init__(
        self,
        *args,
        robot_uids: str = "panda",
        robot_init_qpos_noise: float = 0.02,
        disk_radius: float = 0.05,
        disk_half_thickness: float = 0.02,
        disk_mass: float = 0.3,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.disk_radius = disk_radius
        self.disk_half_thickness = disk_half_thickness
        self.disk_mass = disk_mass
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        pm = physx.PhysxMaterial(
            static_friction=self.disk_static_friction,
            dynamic_friction=self.disk_dynamic_friction,
            restitution=0.1,
        )
        vm_disk = sapien.render.RenderMaterial(
            base_color=[0.85, 0.2, 0.2, 1.0], metallic=0.0, roughness=0.5
        )
        vm_goal = sapien.render.RenderMaterial(
            base_color=[0.2, 0.85, 0.2, 0.35], metallic=0.0, roughness=1.0
        )

        disk_builder = self.scene.create_actor_builder()
        disk_builder.add_cylinder_collision(
            radius=self.disk_radius, half_length=self.disk_half_thickness, material=pm
        )
        disk_builder.add_cylinder_visual(
            radius=self.disk_radius, half_length=self.disk_half_thickness, material=vm_disk
        )
        disk_builder._mass = self.disk_mass
        disk_builder.initial_pose = sapien.Pose(
            p=[0.0, 0.0, self.disk_half_thickness], q=_DISK_QUAT
        )
        self.disk = disk_builder.build(name="push_disk")

        goal_half_t = self.disk_half_thickness * 0.4
        goal_builder = self.scene.create_actor_builder()
        goal_builder.add_cylinder_visual(
            radius=self.disk_radius, half_length=goal_half_t, material=vm_goal
        )
        goal_builder.initial_pose = sapien.Pose(
            p=[0.1, 0.0, goal_half_t], q=_DISK_QUAT
        )
        self.goal_site = goal_builder.build_kinematic(name="goal_site")
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            q = torch.tensor(_DISK_QUAT, device=self.device).unsqueeze(0).expand(b, 4)

            goal_xy = (torch.rand((b, 2), device=self.device) * 2 - 1) * self.GOAL_HALF
            goal_z  = torch.full((b, 1), self.disk_half_thickness * 0.4, device=self.device)
            if not hasattr(self, "goal_pos") or self.goal_pos.shape[0] != self.num_envs:
                self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self.goal_pos[env_idx] = torch.cat([goal_xy, goal_z], dim=1)

            disk_xy = torch.zeros((b, 2), device=self.device)
            for i in range(b):
                for _ in range(50):
                    candidate = (torch.rand(2, device=self.device) * 2 - 1) * self.DISK_SPAWN_HALF
                    if torch.norm(candidate - goal_xy[i]) >= self.MIN_DISK_GOAL_DIST:
                        disk_xy[i] = candidate
                        break
                else:
                    disk_xy[i] = -goal_xy[i].clamp(-self.DISK_SPAWN_HALF, self.DISK_SPAWN_HALF)

            disk_z   = torch.full((b, 1), self.disk_half_thickness, device=self.device)
            disk_xyz = torch.cat([disk_xy, disk_z], dim=1)

            self.disk.set_pose(Pose.create_from_pq(p=disk_xyz, q=q))
            self.goal_site.set_pose(Pose.create_from_pq(p=self.goal_pos[env_idx], q=q))

    def _circle_overlap_frac(self, disk_xy: torch.Tensor, goal_xy: torch.Tensor) -> torch.Tensor:
        r = self.disk_radius
        d = torch.norm(disk_xy - goal_xy, dim=-1).clamp(min=0.0)
        d_safe  = d.clamp(max=2.0 * r - 1e-6)
        cos_arg = (d_safe / (2.0 * r)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        A = (
            2.0 * r * r * torch.acos(cos_arg)
            - 0.5 * d_safe * torch.sqrt((4.0 * r * r - d_safe * d_safe).clamp(min=0.0))
        )
        frac = (A / (torch.pi * r * r)).clamp(0.0, 1.0)
        return torch.where(d >= 2.0 * r, torch.zeros_like(frac), frac)

    def evaluate(self) -> Dict[str, Any]:
        disk_xy = self.disk.pose.p[:, :2]
        goal_xy = self.goal_pos[:, :2]
        overlap  = self._circle_overlap_frac(disk_xy, goal_xy)
        return {
            "success":      overlap >= self.SUCCESS_OVERLAP,
            "overlap_frac": overlap,
            "dist_to_goal": torch.norm(disk_xy - goal_xy, dim=1),
        }

    def _get_obs_extra(self, info: Dict) -> Dict:
        ee_pos   = self.agent.tcp.pose.p
        disk_pos = self.disk.pose.p
        return {
            "ee_pos":       ee_pos,
            "ee_to_disk":   disk_pos - ee_pos,
            "disk_to_goal": self.goal_pos - disk_pos,
            "overlap_frac": info["overlap_frac"].unsqueeze(-1),
            # Required by the planning wrapper
            "tcp_pose":     self.agent.tcp.pose.raw_pose,
            "obj_pose":     self.disk.pose.raw_pose,
            "goal_pos":     self.goal_pos,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        ee_pos   = self.agent.tcp.pose.p
        disk_pos = self.disk.pose.p

        reach_reward   = 0.5 * (1.0 - torch.tanh(5.0 * torch.norm(ee_pos - disk_pos, dim=1)))
        push_reward    = 1.0 - torch.tanh(5.0 * info["dist_to_goal"])
        overlap_reward = info["overlap_frac"]

        # Reward closing the fingers to form a solid pushing surface.
        # Panda finger joints each range [0, 0.04] m; sum=0 → fully closed.
        finger_qpos        = self.agent.robot.get_qpos()[:, -2:]
        gripper_closedness = 1.0 - finger_qpos.sum(dim=1) / (2 * 0.04)
        finger_reward      = 0#.3 * gripper_closedness

        reward = reach_reward + push_reward + overlap_reward + finger_reward
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # max non-terminal = 3.5, terminal bonus = 5.0 → normalised ceiling = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5.0

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.5, 0.0, 0.8], target=[0.0, 0.0, 0.0])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

@register_env("PushO-WithObstacles-v1", max_episode_steps=200)
class PushOWithObstaclesEnv(PushOEnv):
    """PushO with multiple obstacle cubes of different sizes on the table."""

    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # (half_size, rgba) for each obstacle — up to 10 cubes
    OBSTACLE_SPECS = [
        (0.020, [0.2, 0.6, 0.2, 1.0]),
        (0.015, [0.2, 0.4, 0.8, 1.0]),
        (0.025, [0.8, 0.4, 0.2, 1.0]),
        (0.018, [0.6, 0.2, 0.6, 1.0]),
        (0.022, [0.9, 0.8, 0.1, 1.0]),
        (0.017, [0.1, 0.8, 0.8, 1.0]),
        (0.020, [0.8, 0.2, 0.2, 1.0]),
        (0.015, [0.5, 0.5, 0.9, 1.0]),
        (0.023, [0.9, 0.5, 0.1, 1.0]),
        (0.018, [0.4, 0.9, 0.4, 1.0]),
    ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.6, 0, 0.9], target=[-0.1, 0, 0.1])
        return CameraConfig("render_camera", pose=pose, width=512, height=512,
                            fov=1, near=0.01, far=100)

    def _load_scene(self, options: dict):
        super()._load_scene(options)

        self.obstacles = []
        for i, (half_size, color) in enumerate(self.OBSTACLE_SPECS):
            cube = actors.build_cube(
                self.scene,
                half_size=half_size,
                color=np.array(color, dtype=np.float32),
                name=f"obstacle_{i}",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[0.0, 0.0, half_size]),
            )
            self.obstacles.append(cube)

    # How many obstacles to place each episode: uniformly sampled from this range (inclusive).
    MIN_OBSTACLES: int = 5
    MAX_OBSTACLES: int = 10
    # Table half-extent (must stay within this XY radius of origin)
    TABLE_HALF: float = 0.25
    # Minimum distance between an obstacle and the disk centre
    MIN_DIST_FROM_DISK: float = 0.06
    # Minimum distance between two obstacles
    MIN_DIST_BETWEEN: float = 0.05

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            disk_pos = self.disk.pose.p[env_idx]  # (b, 3)
            q_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0)

            # Randomly choose how many obstacles to place this episode (same count for all envs in batch)
            n_active = int(torch.randint(self.MIN_OBSTACLES, self.MAX_OBSTACLES + 1, (1,)).item())

            placed_xy: list[torch.Tensor] = []  # (b, 2) tensors for collision checking

            for i, cube in enumerate(self.obstacles):
                half = self.OBSTACLE_SPECS[i][0]

                if i < n_active:
                    # Sample random XY positions with rejection for each env independently
                    xy = torch.zeros((b, 2), device=self.device)
                    for env_i in range(b):
                        disk_xy = disk_pos[env_i, :2]
                        for _ in range(50):
                            candidate = (torch.rand(2, device=self.device) * 2 - 1) * self.TABLE_HALF
                            # Reject if too close to disk
                            if torch.norm(candidate - disk_xy) < self.MIN_DIST_FROM_DISK:
                                continue
                            # Reject if too close to any already-placed obstacle
                            too_close = any(
                                torch.norm(candidate - p[env_i]) < self.MIN_DIST_BETWEEN
                                for p in placed_xy
                            )
                            if too_close:
                                continue
                            xy[env_i] = candidate
                            break
                        else:
                            # Fallback: fixed offset if sampling fails
                            fallback = [(0.08, 0.10), (-0.06, 0.12), (0.10, -0.08), (-0.10, -0.06)]
                            dx, dy = fallback[i % len(fallback)]
                            xy[env_i] = disk_xy + torch.tensor([dx, dy], device=self.device)

                    placed_xy.append(xy)
                    xyz = torch.cat([xy, torch.full((b, 1), half, device=self.device)], dim=1)
                else:
                    # Park inactive obstacles off-table so they don't interfere
                    park_x = 0.5 + i * 0.05
                    xyz = torch.tensor([[park_x, 0.0, half]], device=self.device).expand(b, 3)

                pose = Pose.create_from_pq(p=xyz, q=q_id.expand(b, 4))
                cube.set_pose(pose)


@register_env("Reach-WithObstacles-v1", max_episode_steps=50)
class ReachWithObstaclesEnv(PushOWithObstaclesEnv):
    """Move EE to a random goal above the table. Same scene as PushO-WithObstacles.

    Observation extras: goal_pos (3,), ee_pos (3,), ee_to_goal (3,)
    Success: EE within 2 cm of goal.
    """

    def __init__(self, *args, success_threshold: float = 0.02, **kwargs):
        self.success_threshold = success_threshold
        super().__init__(*args, **kwargs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            goal_xy = (torch.rand((b, 2), device=self.device) * 2 - 1) * 0.25
            goal_z  = torch.rand((b, 1), device=self.device) * 0.30 + 0.15
            if not hasattr(self, "goal_pos") or self.goal_pos.shape[0] != self.num_envs:
                self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self.goal_pos[env_idx] = torch.cat([goal_xy, goal_z], dim=1)

    def evaluate(self, **kwargs):
        ee_pos  = self.agent.tcp.pose.p
        dist    = torch.norm(ee_pos - self.goal_pos, dim=1)
        return {"success": dist < self.success_threshold, "dist_to_goal": dist}

    def _get_obs_extra(self, info):
        ee_pos = self.agent.tcp.pose.p
        return {
            "goal_pos":   self.goal_pos,
            "ee_pos":     ee_pos,
            "ee_to_goal": self.goal_pos - ee_pos,
        }

    def compute_dense_reward(self, obs, action, info):
        reach_reward  = 1.0 - torch.tanh(5.0 * info["dist_to_goal"])
        q = self.agent.tcp.pose.q  # (N, 4) wxyz
        tcp_z_world_z = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        upright_reward = -0.2 * (1.0 - tcp_z_world_z ** 2)
        # Per-step action penalties: encourage small, stable motions.
        # action layout for pd_ee_delta_pose: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        # Roll (3) and pitch (4) get a 5× higher penalty — large values destabilize
        # the EE orientation for downstream skills.
        translate_penalty     = 0.02 * torch.linalg.norm(action[:, :3], dim=1)
        yaw_penalty       = 0.02 * action[:, 5].abs()
        roll_pitch_penalty = 0.10 * action[:, 3:5].abs().sum(dim=1)
        action_reg = translate_penalty + yaw_penalty + roll_pitch_penalty
        reward = reach_reward + upright_reward - action_reg

        # Once successful, override reward with a bonus minus a penalty for EE
        # movement. This teaches the policy to hold still after grasping rather
        # than continuing to drive into the table or out of the scene.
        success = info["success"]
        if success.any():
            action_penalty = torch.linalg.norm(action[success, :7], dim=1)
            reward[success] = 5.0 - 0.5 * action_penalty
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 6.0


@register_env("PushCube-WithObstacles-v1", max_episode_steps=200)
class PushCubeWithObstaclesEnv(PushOWithObstaclesEnv):
    """Push a randomly selected obstacle cube to a goal position on the table.

    Each episode a different obstacle is chosen as the push target.
    Observation extras: ee_pos (3,), goal_cube_pos (3,), goal_pos (3,),
                        ee_to_goal_cube (3,), goal_cube_to_goal (3,)
    Success: goal cube XY within 5 cm of goal XY.
    """

    GOAL_THRESHOLD = 0.05

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        # Cyan sphere: marks the goal cube's starting position each episode
        self.goal_cube_start_site = actors.build_sphere(
            self.scene,
            radius=self.GOAL_THRESHOLD,
            color=[0, 0.8, 1, 0.5],
            name="goal_cube_start_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_cube_start_site)
        # Green sphere: marks the goal position the cube must be pushed to
        self.goal_cube_target_site = actors.build_sphere(
            self.scene,
            radius=self.GOAL_THRESHOLD,
            color=[0, 1, 0, 0.5],
            name="goal_cube_target_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_cube_target_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)

            # Park the disk and goal_site off-table — they are unused in this env.
            q_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(b, 4)
            park_p = torch.tensor([[2.0, 0.0, 0.05]], device=self.device).expand(b, 3)
            park_pose = Pose.create_from_pq(p=park_p, q=q_id)
            self.disk.set_pose(park_pose)
            self.goal_site.set_pose(park_pose)

            # Randomly pick which obstacle is the push target for each env
            if not hasattr(self, "goal_obstacle_idx") or self.goal_obstacle_idx.shape[0] != self.num_envs:
                self.goal_obstacle_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.goal_obstacle_idx[env_idx] = torch.randint(
                len(self.obstacles), (b,), device=self.device)

            # Sample goal XY across the full table (matches the planner grid ±0.30 m)
            goal_xy = (torch.rand((b, 2), device=self.device) * 2 - 1) * self.TABLE_HALF
            if not hasattr(self, "goal_pos") or self.goal_pos.shape[0] != self.num_envs:
                self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self.goal_pos[env_idx] = torch.cat(
                [goal_xy, torch.zeros((b, 1), device=self.device)], dim=1)

            # Place cyan start-position marker at the goal cube's initial XY
            all_obs_pos = torch.stack(
                [obs.pose.p for obs in self.obstacles], dim=0
            )  # (num_obstacles, num_envs, 3)
            goal_idx_b = self.goal_obstacle_idx[env_idx]  # (b,)
            start_xyz = all_obs_pos[goal_idx_b, env_idx]  # (b, 3)
            self.goal_cube_start_site.set_pose(Pose.create_from_pq(p=start_xyz, q=q_id))

            # Place green target marker at the sampled goal XY on the table surface
            target_xyz = torch.cat([goal_xy, torch.zeros((b, 1), device=self.device)], dim=1)
            self.goal_cube_target_site.set_pose(Pose.create_from_pq(p=target_xyz, q=q_id))

    def _get_goal_cube_pos(self) -> torch.Tensor:
        """Position of the selected goal obstacle for each env. Shape: (num_envs, 3)."""
        # (num_obstacles, num_envs, 3) → index per env
        all_pos = torch.stack([obs.pose.p for obs in self.obstacles], dim=0)
        return all_pos[self.goal_obstacle_idx, torch.arange(self.num_envs, device=self.device)]

    def evaluate(self, **kwargs):
        goal_cube_pos = self._get_goal_cube_pos()
        dist = torch.norm(goal_cube_pos[:, :2] - self.goal_pos[:, :2], dim=1)
        return {"success": dist < self.GOAL_THRESHOLD, "dist_cube_to_goal": dist}

    def _get_obs_extra(self, info):
        ee_pos        = self.agent.tcp.pose.p
        goal_cube_pos = self._get_goal_cube_pos()
        return {
            "ee_pos":            ee_pos,
            "goal_cube_pos":     goal_cube_pos,
            "goal_pos":          self.goal_pos,
            "ee_to_goal_cube":   goal_cube_pos - ee_pos,
            "goal_cube_to_goal": self.goal_pos - goal_cube_pos,
        }

    def compute_dense_reward(self, obs, action, info):
        ee_pos        = self.agent.tcp.pose.p
        goal_cube_pos = self._get_goal_cube_pos()
        reach_reward  = 1.0 - torch.tanh(5.0 * torch.norm(ee_pos - goal_cube_pos, dim=1))
        push_reward   = 1.0 - torch.tanh(5.0 * info["dist_cube_to_goal"])
        success_bonus = info["success"].float() * 5.0
        q = self.agent.tcp.pose.q  # (N, 4) wxyz
        tcp_z_world_z = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        upright_reward = -0.2 * (1.0 - tcp_z_world_z ** 2)
        return reach_reward + push_reward + success_bonus + upright_reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 7.0


@register_env("PushO-WallObstacles-v1", max_episode_steps=2000)
class PushOWallObstaclesEnv(PushOWithObstaclesEnv):
    """PushO where all 10 obstacle cubes are fixed in a wall across the table.

    All cubes are placed along WALL_ROW (grid row index), one per grid column.
    This guarantees every BFS path from disk to goal that crosses the wall is
    blocked, forcing the high-level planner to emit pick/place or push_cube
    subgoals rather than just routing around obstacles.

    Disk and goal positions are inherited (random from PushOEnv). Seeds where
    both land on the same side of the wall still work — the planner simply
    won't need to clear the wall for those episodes.
    """

    # Grid row for the wall. Row 4 → y ≈ -0.03 m (just below table centre).
    WALL_ROW: int = 4

    # Always use all 10 obstacles so every grid column is filled.
    MIN_OBSTACLES: int = 10
    MAX_OBSTACLES: int = 10

    # Planner constants (must match llm_plan.py)
    _TABLE_BOUND: float = 0.30
    _GRID: int = 10

    # (disk_xy, goal_xy) in world metres. Computed via region_to_xy().
    PRESETS: dict = {
        #              disk (x, y)          goal (x, y)
        "cross_center": ((-0.06, -0.21),  ( 0.06,  0.21)),
        "same_side":    ((-0.09, -0.15),  ( 0.15, -0.21)),
        "straight-1":   ((0, -0.15), (0, 0.15)),
        "straight-2":   ((0.06, -0.15), (0.06, 0.15)),
        "straight-3":   ((0.12, -0.15), (0.12, 0.15)),
        "straight-4":   ((0.18, -0.15), (0.18, 0.15)),
        "straight-5":   ((-0.06, -0.15), (-0.06, 0.15)),
    }

    # Set to a key from PRESETS to fix positions; None → random (default).
    preset: str | None = "straight-5" #"same_side" #fails when placing cube 6


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Let PushOWithObstaclesEnv set up disk, goal, robot positions first,
        # then immediately reposition every obstacle into the wall.
        super()._initialize_episode(env_idx, options)

        if self.preset is not None:
            disk_xy_val, goal_xy_val = self.PRESETS[self.preset]
            with torch.device(self.device):
                b = len(env_idx)
                q = torch.tensor(_DISK_QUAT, device=self.device).unsqueeze(0).expand(b, 4)

                goal_xy = torch.tensor(list(goal_xy_val), device=self.device).unsqueeze(0).expand(b, 2)
                goal_z  = torch.full((b, 1), self.disk_half_thickness * 0.4, device=self.device)
                self.goal_pos[env_idx] = torch.cat([goal_xy, goal_z], dim=1)
                self.goal_site.set_pose(Pose.create_from_pq(
                    p=self.goal_pos[env_idx], q=q,
                ))

                disk_xy = torch.tensor(list(disk_xy_val), device=self.device).unsqueeze(0).expand(b, 2)
                disk_z  = torch.full((b, 1), self.disk_half_thickness, device=self.device)
                self.disk.set_pose(Pose.create_from_pq(
                    p=torch.cat([disk_xy, disk_z], dim=1), q=q,
                ))
        with torch.device(self.device):
            b = len(env_idx)
            q_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0)

            y_center = (self.WALL_ROW + 0.5) / self._GRID * 2 * self._TABLE_BOUND - self._TABLE_BOUND

            for col, cube in enumerate(self.obstacles):
                half = self.OBSTACLE_SPECS[col][0]
                x_center = (col + 0.5) / self._GRID * 2 * self._TABLE_BOUND - self._TABLE_BOUND
                # Clamp to physical table extent so cubes stay on the surface.
                x_center = float(np.clip(x_center, -self.TABLE_HALF, self.TABLE_HALF))
                xyz = torch.tensor(
                    [[x_center, y_center, half]], device=self.device
                ).expand(b, 3)
                pose = Pose.create_from_pq(p=xyz, q=q_id.expand(b, 4))
                cube.set_pose(pose)


@register_env("PickSkillEnv", max_episode_steps=50)
class PickSkillEnv(PushOWithObstaclesEnv):
    """Pick a randomly selected obstacle cube off the table.

    Each episode one of the inherited obstacle cubes is chosen as the pick
    target. The skill must grasp and lift it above `lift_threshold`.

    Observation extras: ee_pos (3,), pick_cube_pos (3,), ee_to_pick_cube (3,),
                        is_grasped (1,)
    Success: is_grasped AND pick_cube_pos z > lift_threshold.
    """

    # Absolute z height (cube centre) that counts as lifted off the table.
    # Obstacle half-sizes range 0.015–0.025, so 0.06 requires ~3–4 cm of lift.
    lift_threshold = 0.06

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        # Make the goal ring visible so it can highlight the pick target.
        if self.goal_site in self._hidden_objects:
            self._hidden_objects.remove(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            q_id   = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(b, 4)
            park_p = torch.tensor([[2.0, 0.0, 0.05]], device=self.device).expand(b, 3)
            self.disk.set_pose(Pose.create_from_pq(p=park_p, q=q_id))

            PICK_XY_LIMIT = 0.25

            if not hasattr(self, "pick_obstacle_idx") or self.pick_obstacle_idx.shape[0] != self.num_envs:
                self.pick_obstacle_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

            all_pos = torch.stack([obs.pose.p for obs in self.obstacles], dim=0)  # (n_obs, num_envs, 3)
            chosen = torch.randint(len(self.obstacles), (b,), device=self.device)
            for _ in range(len(self.obstacles)):
                pos = all_pos[chosen, env_idx]  # (b, 3)
                out_of_reach = (pos[:, :2].abs() > PICK_XY_LIMIT).any(dim=1)
                if not out_of_reach.any():
                    break
                reroll = torch.randint(len(self.obstacles), (b,), device=self.device)
                chosen = torch.where(out_of_reach, reroll, chosen)
            self.pick_obstacle_idx[env_idx] = chosen

            # Place the goal ring flat over the chosen obstacle as a visual highlight.
            chosen_pos = all_pos[self.pick_obstacle_idx[env_idx], env_idx].clone()
            q_disk = torch.tensor(_DISK_QUAT, device=self.device).unsqueeze(0).expand(b, 4)
            self.goal_site.set_pose(Pose.create_from_pq(p=chosen_pos, q=q_disk))

    def _get_pick_cube_pos(self) -> torch.Tensor:
        """Position of the selected pick obstacle for each env. Shape: (num_envs, 3)."""
        all_pos = torch.stack([obs.pose.p for obs in self.obstacles], dim=0)
        return all_pos[self.pick_obstacle_idx, torch.arange(self.num_envs, device=self.device)]

    def evaluate(self) -> Dict[str, Any]:
        pick_cube_pos = self._get_pick_cube_pos()
        # Stack per-obstacle grasping flags then select the target per env
        is_grasped = torch.stack(
            [self.agent.is_grasping(obs) for obs in self.obstacles], dim=0
        )[self.pick_obstacle_idx, torch.arange(self.num_envs, device=self.device)]
        is_lifted = pick_cube_pos[:, 2] > self.lift_threshold
        return {
            "success": is_grasped & is_lifted,
            "is_grasped": is_grasped,
            "is_lifted": is_lifted,
        }

    def _get_obs_extra(self, info: Dict) -> Dict:
        ee_pos        = self.agent.tcp.pose.p
        pick_cube_pos = self._get_pick_cube_pos()
        return {
            "ee_pos":          ee_pos,
            "pick_cube_pos":   pick_cube_pos,
            "ee_to_pick_cube": pick_cube_pos - ee_pos,
            "is_grasped":      info["is_grasped"],
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        ee_pos        = self.agent.tcp.pose.p
        pick_cube_pos = self._get_pick_cube_pos()
        dist          = torch.linalg.norm(pick_cube_pos - ee_pos, dim=1)
        reach_reward  = 1.0 - torch.tanh(5.0 * dist)
        is_grasped    = info["is_grasped"].float()
        lift_reward   = 1.0 - torch.tanh(10.0 * (self.lift_threshold - pick_cube_pos[:, 2]).clamp(min=0.0))

        # Reward opening the gripper while the EE is near the cube, so the
        # policy learns to arrive with an open hand rather than ramming in closed.
        finger_pos  = self.agent.robot.get_qpos()[:, -2:]   # (N, 2) — last two joints are fingers
        gripper_openness = finger_pos.sum(dim=1) / (2 * 0.04)  # 0=closed, 1=fully open
        near_cube   = (dist < 0.06).float()
        pregrasp_reward = near_cube * gripper_openness
        q = self.agent.tcp.pose.q  # (N, 4) wxyz
        tcp_z_world_z = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        upright_reward = -0.5 * (1.0 - tcp_z_world_z ** 2)
        # Per-step action penalties: encourage small, stable motions.
        # action layout for pd_ee_delta_pose: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        # Roll (3) and pitch (4) get a 5× higher penalty — large values destabilize
        # the EE orientation for downstream skills.
        translate_penalty     = 0.02 * torch.linalg.norm(action[:, :3], dim=1)
        yaw_penalty       = 0.02 * action[:, 5].abs()
        roll_pitch_penalty = 0.10 * action[:, 3:5].abs().sum(dim=1)
        action_reg = translate_penalty + yaw_penalty + roll_pitch_penalty

        reward = reach_reward + pregrasp_reward + is_grasped + lift_reward * is_grasped + upright_reward - action_reg

        # Once successful, override reward with a bonus minus a penalty for EE
        # movement. This teaches the policy to hold still after grasping rather
        # than continuing to drive into the table or out of the scene.
        success = info["success"]
        if success.any():
            action_penalty = torch.linalg.norm(action[success, :7], dim=1)
            reward[success] = 5.0 - 0.5 * action_penalty

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 7.0

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return CameraConfig("render_camera", pose=pose, width=512, height=512,
                            fov=1, near=0.01, far=100)

@register_env("PushO-Scattered", max_episode_steps=2000)
class PushOScatteredEnv(PushOWithObstaclesEnv):
    """
    PushO with exactly 10 obstacle cubes randomly scattered 
    on the table using rejection sampling.
    """
    MIN_OBSTACLES: int = 10
    MAX_OBSTACLES: int = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)


@register_env("PushO-TrappedDisk", max_episode_steps=2000)
class PushOTrappedDiskEnv(PushOWithObstaclesEnv):

    MIN_OBSTACLES: int = 10
    MAX_OBSTACLES: int = 10
    
    # Distance from the disk center to the perimeter line
    PERIMETER_OFFSET: float = 0.1

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Initialize disk and goal positions first
        super()._initialize_episode(env_idx, options)
        
        with torch.device(self.device):
            b = len(env_idx)
            disk_pos = self.disk.pose.p[env_idx] # (b, 3)
            q_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(b, 4)

            # Define the 8 relative XY offsets for a square ring around the center
            # [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            offsets = []
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if i == 0 and j == 0: continue
                    offsets.append((i * self.PERIMETER_OFFSET, j * self.PERIMETER_OFFSET))
            
            for i, cube in enumerate(self.obstacles):
                half = self.OBSTACLE_SPECS[i][0]
                
                if i < 8:
                    # Place cubes in the square perimeter
                    rel_offset = torch.tensor(offsets[i], device=self.device)
                    xy = disk_pos[:, :2] + rel_offset
                else:
                    # Place remaining 2 cubes randomly as "noise"
                    xy = (torch.rand((b, 2), device=self.device) * 2 - 1) * self.TABLE_HALF
                
                xyz = torch.cat([xy, torch.full((b, 1), half, device=self.device)], dim=1)
                self.obstacles[i].set_pose(Pose.create_from_pq(p=xyz, q=q_id))