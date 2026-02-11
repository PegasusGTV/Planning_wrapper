"""PushT with obstacle blocks on the table. Uses planning_wrapper and PushTTaskAdapter."""
import numpy as np
import sapien
import torch

import mani_skill.envs  # noqa: F401
from mani_skill.envs.tasks.tabletop.push_t import PushTEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose


@register_env("PushT-WithObstacles-v1", max_episode_steps=200)
class PushTWithObstaclesEnv(PushTEnv):
    """PushT with multiple obstacle cubes of different sizes on the table."""

    # (half_size, rgba) for each obstacle
    OBSTACLE_SPECS = [
        (0.02, [0.2, 0.6, 0.2, 1.0]),
        (0.015, [0.2, 0.4, 0.8, 1.0]),
        (0.025, [0.8, 0.4, 0.2, 1.0]),
        (0.018, [0.6, 0.2, 0.6, 1.0]),
    ]

    def _load_scene(self, options: dict):
        self.T_mass = 0.3
        self.T_dynamic_friction = 0.8
        self.T_static_friction = 0.8
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

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            tee_pos = self.tee.pose.p
            # Place obstacles around the table (offsets from tee so they don't overlap T)
            offsets_xy = [
                (0.08, 0.10),
                (-0.06, 0.12),
                (0.10, -0.08),
                (-0.10, -0.06),
            ]
            for i, cube in enumerate(self.obstacles):
                half = self.OBSTACLE_SPECS[i][0]
                dx, dy = offsets_xy[i] if i < len(offsets_xy) else (0.05, 0.05)
                xyz = tee_pos.clone()
                xyz[:, 0] += dx
                xyz[:, 1] += dy
                xyz[:, 2] = half
                q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(b, 4)
                pose = Pose.create_from_pq(p=xyz, q=q)
                cube.set_pose(pose)
