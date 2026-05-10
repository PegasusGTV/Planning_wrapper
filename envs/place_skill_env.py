from __future__ import annotations

from typing import Any, Dict

import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("PlaceSkillEnv", max_episode_steps=100)
class PlaceSkillEnv(BaseEnv):
    """
    Place skill: lower and release a held cube onto a goal position on the table,
    then retreat the EE away.

    The robot starts every episode with the cube already firmly grasped in the
    gripper. Each episode a cube size is sampled uniformly from CUBE_HALF_SIZES;
    one actor per size is pre-built and the inactive ones are parked off-table.

    Success (all must hold simultaneously):
        - Cube is not grasped (fingers have released it).
        - Cube centre is within place_thresh of the goal position.
        - Cube is resting on the table (z ≈ active cube half-size).
        - EE has retreated > retreat_dist from the cube centre.

    Observation (state mode):
        tcp_pose        (7,)  EE world pose (pos + quat)
        obj_pose        (7,)  cube world pose
        goal_pos        (3,)  target position on table
        tcp_to_obj_pos  (3,)  vector from EE to cube centre
        obj_to_goal_pos (3,)  vector from cube to goal
        is_grasped      (1,)  whether cube is currently grasped
    """

    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    # Discrete cube sizes to randomise over each episode.
    CUBE_HALF_SIZES = [0.015, 0.0175, 0.020, 0.0225, 0.025]

    # Arm joint noise at init (rad) — varies the holding pose to prevent overfitting
    qpos_noise = 0.05

    # Finger closure: slightly tighter than cube so physics resolves a firm grasp
    finger_grip_margin = 0.003

    # Goal sampled within this xy radius on the table
    goal_spawn_half_size = 0.25

    # Cube-to-goal XY distance for a successful place
    place_thresh = 0.025

    # EE must retreat at least this far from cube centre to count as "away"
    retreat_dist = 0.10

    def __init__(
        self,
        *args,
        robot_uids: str = "panda",
        robot_init_qpos_noise: float = 0.02,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # One actor per discrete cube size; inactive ones are parked off-table each episode.
        self.cubes = []
        colors = [
            [1.0, 0.2, 0.2, 1.0],
            [1.0, 0.5, 0.1, 1.0],
            [0.9, 0.9, 0.1, 1.0],
            [0.2, 0.8, 0.2, 1.0],
            [0.2, 0.4, 1.0, 1.0],
        ]
        for i, half in enumerate(self.CUBE_HALF_SIZES):
            cube = actors.build_cube(
                self.scene,
                half_size=half,
                color=colors[i % len(colors)],
                name=f"place_cube_{i}",
                initial_pose=sapien.Pose(p=[2.0 + i * 0.15, 0.0, half]),
            )
            self.cubes.append(cube)

        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.place_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # --- Select a cube size per env ---
            if not hasattr(self, "cube_idx") or self.cube_idx.shape[0] != self.num_envs:
                self.cube_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            chosen = torch.randint(len(self.cubes), (b,), device=self.device)
            self.cube_idx[env_idx] = chosen

            sizes_t = torch.tensor(self.CUBE_HALF_SIZES, device=self.device)
            if not hasattr(self, "active_half_size") or self.active_half_size.shape[0] != self.num_envs:
                self.active_half_size = torch.zeros(self.num_envs, device=self.device)
            self.active_half_size[env_idx] = sizes_t[chosen]

            # Park every cube off-table; active ones will be teleported into the gripper
            # at the first evaluate() call via the cube_placed mechanism.
            q_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(b, 4)
            for i, cube in enumerate(self.cubes):
                half = self.CUBE_HALF_SIZES[i]
                park_xyz = torch.tensor([[2.0 + i * 0.15, 0.0, half]], device=self.device).expand(b, 3)
                cube.set_pose(Pose.create_from_pq(p=park_xyz, q=q_id))

            # --- Arm joint noise (holding pose variety) ---
            current_qpos = self.agent.robot.get_qpos()[env_idx]  # (b, 9)
            arm_noise = (torch.rand((b, 7), device=self.device) * 2 - 1) * self.qpos_noise

            # Finger closure matched to the selected cube size
            active_half_b = self.active_half_size[env_idx]  # (b,)
            finger_val = (active_half_b - self.finger_grip_margin).clamp(min=0.005)
            finger_qpos = finger_val.unsqueeze(1).expand(b, 2)

            new_qpos = torch.cat([current_qpos[:, :7] + arm_noise, finger_qpos], dim=1)
            self.agent.robot.set_qpos(new_qpos)
            self.agent.robot.set_qvel(torch.zeros_like(new_qpos))

            # --- Goal position ---
            goal_xy = (torch.rand((b, 2), device=self.device) * 2 - 1) * self.goal_spawn_half_size
            goal_z = active_half_b.unsqueeze(1)  # resting height = cube half-size
            goal_xyz = torch.cat([goal_xy, goal_z], dim=1)
            if not hasattr(self, "goal_pos") or self.goal_pos.shape[0] != self.num_envs:
                self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self.goal_pos[env_idx] = goal_xyz
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

            # cube_placed=False triggers the TCP teleport on the first evaluate() call
            if not hasattr(self, "cube_placed") or self.cube_placed.shape[0] != self.num_envs:
                self.cube_placed = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            self.cube_placed[env_idx] = False

    def _get_active_cube_pos(self) -> torch.Tensor:
        """World position of the active cube for each env. Shape: (num_envs, 3)."""
        all_pos = torch.stack([c.pose.p for c in self.cubes], dim=0)  # (n, num_envs, 3)
        return all_pos[self.cube_idx, torch.arange(self.num_envs, device=self.device)]

    def evaluate(self) -> Dict[str, Any]:
        # Teleport the active cube into the gripper on the first step of each episode.
        cubes_require_placement = ~self.cube_placed
        if cubes_require_placement.any():
            if self.gpu_sim_enabled:
                self.scene._gpu_fetch_all()
            tcp_raw = self.agent.tcp_pose.raw_pose
            for i, cube in enumerate(self.cubes):
                mask = cubes_require_placement & (self.cube_idx == i)
                if mask.any():
                    new_raw = cube.pose.raw_pose.clone()
                    new_raw[mask] = tcp_raw[mask]
                    cube.set_pose(Pose.create_from_pq(new_raw[:, :3], new_raw[:, 3:]))
            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene._gpu_fetch_all()
            self.cube_placed[cubes_require_placement] = True

        # is_grasped: check all cubes, select the active one per env
        all_grasped = torch.stack(
            [self.agent.is_grasping(c) for c in self.cubes], dim=0
        )  # (n_cubes, num_envs)
        is_grasped = all_grasped[self.cube_idx, torch.arange(self.num_envs, device=self.device)]

        cube_pos = self._get_active_cube_pos()  # (N, 3)
        tcp_pos = self.agent.tcp_pose.p

        obj_to_goal_dist = torch.linalg.norm(cube_pos - self.goal_pos, dim=1)

        # Resting threshold is per-env: active cube half-size + 1 cm tolerance
        rest_z_thresh = self.active_half_size + 0.01

        is_placed = (
            (~is_grasped)
            & (obj_to_goal_dist < self.place_thresh)
            & (cube_pos[:, 2] < rest_z_thresh)
        )

        tcp_to_obj_dist = torch.linalg.norm(tcp_pos - cube_pos, dim=1)
        is_retreated = tcp_to_obj_dist > self.retreat_dist

        return {
            "success": is_placed & is_retreated,
            "is_grasped": is_grasped,
            "is_placed": is_placed,
            "is_retreated": is_retreated,
            "obj_to_goal_dist": obj_to_goal_dist,
        }

    def _get_obs_extra(self, info: Dict) -> Dict:
        cube_pos = self._get_active_cube_pos()
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
            goal_pos=self.goal_pos,
        )
        if "state" in self.obs_mode:
            # Pack active cube pose: use position from stack + orientation from active cube raw_pose
            all_raw = torch.stack([c.pose.raw_pose for c in self.cubes], dim=0)
            active_raw = all_raw[self.cube_idx, torch.arange(self.num_envs, device=self.device)]
            obs.update(
                obj_pose=active_raw,
                tcp_to_obj_pos=cube_pos - self.agent.tcp_pose.p,
                obj_to_goal_pos=self.goal_pos - cube_pos,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):  # noqa: ARG002
        is_grasped = info["is_grasped"].float()
        obj_to_goal_dist = info["obj_to_goal_dist"]
        cube_pos = self._get_active_cube_pos()
        rest_z = self.active_half_size  # per-env resting height, shape (num_envs,)

        # Reward 1: carry cube toward goal — only fires while grasped.
        # Ungrasped proximity is not rewarded, so throwing gains nothing here.
        carry_reward = is_grasped * (1 - torch.tanh(5 * obj_to_goal_dist))
        reward = carry_reward

        # Reward 2: penalise mid-air release — proportional to how high the cube is
        # when not grasped. A thrown cube stays airborne for many steps, so the
        # accumulated penalty is large enough to outweigh any goal-proximity gain.
        z_excess = (cube_pos[:, 2] - rest_z).clamp(min=0.0)  # 0 when at/below table
        reward -= 5.0 * (1.0 - is_grasped) * z_excess

        # Reward 3: reward opening fingers only when cube is already near the table
        # AND near the goal (gate prevents rewarding a mid-air release near the goal).
        near_goal = (obj_to_goal_dist < self.place_thresh * 2).float()
        on_table  = (z_excess < 0.02).float()
        reward += near_goal * on_table * (1.0 - is_grasped)

        # Reward 4: once cube is placed (released + at goal + on table), reward retreat.
        tcp_to_obj_dist = torch.linalg.norm(self.agent.tcp_pose.p - cube_pos, dim=1)
        retreat_reward = 1 - torch.tanh(
            5 * (self.retreat_dist - tcp_to_obj_dist).clamp(min=0.0)
        )
        reward += retreat_reward * info["is_placed"].float()

        q = self.agent.tcp_pose.q  # (N, 4) wxyz
        tcp_z_world_z = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        upright_reward = -0.5 * (1.0 - tcp_z_world_z ** 2)
        reward += upright_reward

        # Per-step action penalties: encourage small, stable motions.
        # action layout for pd_ee_delta_pose: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        # Roll (3) and pitch (4) get a 5× higher penalty — large values destabilize EE orientation.
        translate_penalty  = 0.02 * torch.linalg.norm(action[:, :3], dim=1)
        yaw_penalty        = 0.02 * action[:, 5].abs()
        roll_pitch_penalty = 0.10 * action[:, 3:5].abs().sum(dim=1)
        reward -= translate_penalty + yaw_penalty + roll_pitch_penalty

        # Once placed and retreated, override with bonus minus movement penalty so the
        # policy learns to hold still rather than thrashing after a successful place.
        success = info["success"]
        if success.any():
            action_penalty = torch.linalg.norm(action[success, :7], dim=1)
            reward[success] = 5.0 - 0.5 * action_penalty

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5.0

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.6, 0, 1], target=[-0.1, 0, 0.1])
        return CameraConfig("render_camera", pose=pose, width=512, height=512,
                            fov=1, near=0.01, far=100)
