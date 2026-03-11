# envs/pusht_extra_object.py
"""
PushT-WithExtraObject-v1

PushT environment with an additional cube object on the table.
Use gym.make("PushT-WithExtraObject-v1", ...) after importing this package
(e.g. `import envs` or `from envs import PushTWithExtraObjectEnv`).
"""

import numpy as np
import sapien
import torch

import mani_skill.envs  # noqa: F401
from mani_skill.envs.tasks.tabletop.push_t import PushTEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.building import actors
from mani_skill.utils.structs import Pose


@register_env("PushT-WithExtraObject-v1", max_episode_steps=10000)
class PushTWithExtraObjectEnv(PushTEnv):
    """
    PushT environment with an additional cube object on the table.
    Reduced T block mass and friction for easier pushing.
    """

    def _load_scene(self, options: dict):
        # Reduce T block mass and friction to make pushing easier
        # Original values: T_mass = 0.8, T_dynamic_friction = 3, T_static_friction = 3
        self.T_mass = 0.2  # Reduced from 0.8 to 0.2 (75% reduction)
        self.T_dynamic_friction = 0.5  # Reduced from 3 to 0.5
        self.T_static_friction = 0.5  # Reduced from 3 to 0.5

        # Call parent's _load_scene to set up the T block and goal
        super()._load_scene(options)

        # Add an additional cube object on the table
        self.extra_obj = actors.build_cube(
            self.scene,
            half_size=0.02,  # 4cm cube
            color=np.array([0, 255, 0, 255]) / 255,  # Green color
            name="extra_cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(
                p=[0.1, 0.1, 0.02]
            ),  # Position on table, above surface
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

        # Place the extra object close to the T block
        with torch.device(self.device):
            b = len(env_idx)
            tee_pos = self.tee.pose.p  # Shape: (b, 3)

            extra_obj_xyz = tee_pos.clone()
            extra_obj_xyz[:, 0] += 0.06  # Offset in x direction (to the side)
            extra_obj_xyz[:, 1] += 0.04  # Offset in y direction (slightly behind)
            extra_obj_xyz[:, 2] = 0.02  # Half size of cube on table surface

            q = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(b, 1)
            extra_obj_pose = Pose.create_from_pq(p=extra_obj_xyz, q=q)
            self.extra_obj.set_pose(extra_obj_pose)
