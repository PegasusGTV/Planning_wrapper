"""
agents/floating_gripper.py

A free-floating cylindrical gripper agent for ManiSkill.
Replaces the Panda arm with a single rigid body controlled via
a 3-DOF velocity drive (vx, vy, omega_z).

Usage
-----
    import agents          # triggers registration via agents/__init__.py
    env = gym.make("PushBoundary", robot_uids="floating_gripper")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from gymnasium import spaces

import sapien

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers.base_controller import (
    BaseController,
    ControllerConfig,
)
from mani_skill.agents.registration import register_agent
from mani_skill.utils.structs import Pose

GRIPPER_Z_FIXED = 0.06 + 0.02   # = 0.065

# ──────────────────────────────────────────────────────────────────────────────
# Controller
# ──────────────────────────────────────────────────────────────────────────────

class FloatingVelocityController(BaseController):

    def _initialize_joints(self):
        self.free_joint = self.articulation.joints[0]

    def set_drive_property(self):
        self.free_joint.set_drive_properties(
            stiffness=0.0,
            damping=self.config.damping,
            force_limit=self.config.force_limit,
            mode="force",
        )

    # def _initialize_action_space(self):
    #     lim = np.array(
    #         [
    #             self.config.max_lin_vel,
    #             self.config.max_lin_vel,
    #             self.config.max_ang_vel,
    #         ],
    #         dtype=np.float32,
    #     )
    #     self.single_action_space = spaces.Box(-lim, lim, dtype=np.float32)

    def reset(self, env_idx=None):
        qvel = self.articulation.get_qvel()
        zeros = torch.zeros_like(qvel)
        self.articulation.set_qvel(zeros)

    # def set_drive_property(self):
    #     # No joint drive to configure — velocity is set directly on the root
    #     pass

    # def set_action(self, action: torch.Tensor):
    #     dt = 1.0 / 20.0
    #     current_pose = self.articulation.get_root_pose()
    #     p = current_pose.p.clone()
    #     p[:, 0] += action[:, 0] * dt
    #     p[:, 1] += action[:, 1] * dt
    #     p[:, 2] = GRIPPER_Z_FIXED   # always lock z, never let physics move it

    #     # Always identity quaternion — the upright orientation is baked
    #     # into the shape's local pose, so the link frame stays world-aligned
    #     n = action.shape[0]
    #     q = torch.zeros(n, 4, device=action.device, dtype=action.dtype)
    #     q[:, 0] = 1.0   # [w=1, x=0, y=0, z=0] = no rotation

    #     self.articulation.set_root_pose(Pose.create_from_pq(p=p, q=q))

    #     # Also kill any velocities physics accumulated this step
    #     self.articulation.set_root_linear_velocity(
    #         torch.zeros(n, 3, device=action.device, dtype=action.dtype)
    #     )
    #     self.articulation.set_root_angular_velocity(
    #         torch.zeros(n, 3, device=action.device, dtype=action.dtype)
    #     )
    
    def set_drive_property(self):
        pass

    # def set_action(self, action: torch.Tensor):
    #     # action: (num_envs, 2) — [dx, dy] in meters
    #     current_pose = self.articulation.get_root_pose()
    #     p = current_pose.p.clone()
    #     p[:, 0] += action[:, 0]
    #     p[:, 1] += action[:, 1]
    #     p[:, 2] = GRIPPER_Z_FIXED

    #     n = action.shape[0]
    #     q = torch.zeros(n, 4, device=action.device, dtype=action.dtype)
    #     q[:, 0] = 1.0

    #     self.articulation.set_root_pose(Pose.create_from_pq(p=p, q=q))
    #     self.articulation.set_root_linear_velocity(
    #         torch.zeros(n, 3, device=action.device, dtype=action.dtype)
    #     )
    #     self.articulation.set_root_angular_velocity(
    #         torch.zeros(n, 3, device=action.device, dtype=action.dtype)
    #     )
    
    def set_action(self, action: torch.Tensor):
        current_pose = self.articulation.get_root_pose()
        p = current_pose.p.clone()
        p[:, 0] += action[:, 0]
        p[:, 1] += action[:, 1]
        p[:, 2] = GRIPPER_Z_FIXED

        n = action.shape[0]
        q = torch.zeros(n, 4, device=action.device, dtype=action.dtype)
        q[:, 0] = 1.0

        self.articulation.set_root_pose(Pose.create_from_pq(p=p, q=q))
        # Don't zero linear velocity — let physics handle contact naturally
        # Only kill angular velocity to prevent spinning
        # self.articulation.set_root_angular_velocity(
        #     torch.zeros(n, 3, device=action.device, dtype=action.dtype)
        # )
        
    def set_action(self, action: torch.Tensor):
        # action: (num_envs, 2) — [dx, dy]
        current_pose = self.articulation.get_root_pose()
        p = current_pose.p.clone()
        p[:, 0] += action[:, 0]
        p[:, 1] += action[:, 1]
        # z and rotation are locked by physics — no need to enforce here

        self.articulation.set_root_pose(Pose.create_from_pq(p=p, q=current_pose.q))

    def _initialize_action_space(self):
        lim = np.array([0.02, 0.02], dtype=np.float32)
        self.single_action_space = spaces.Box(-lim, lim, dtype=np.float32)
        
    def before_simulation_step(self):
        # PhysX reads the drive target each sub-step automatically;
        # nothing to do here.
        pass

    def get_state(self) -> dict:
        # Expose current joint velocities for logging / obs if needed
        return {"joint_vel": self.articulation.get_qvel()}


# ──────────────────────────────────────────────────────────────────────────────
# Controller config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
# class FloatingVelocityConfig(ControllerConfig):
#     """
#     Tuning guide
#     ------------
#     damping      N·s/m   Controls how tightly velocity tracks the command.
#                          ~10  → loose, drifty feel
#                          ~50  → good balance for pushing tasks  (start here)
#                          ~200 → nearly kinematic, very stiff
#     force_limit  N       Cap on the drive force PhysX can apply.  Prevents
#                          explosive forces on first contact.  20 N is enough
#                          for a 0.5 kg block; raise if the gripper feels weak.
#     max_lin_vel  m/s     Clips the action space; also fed to the RL policy
#                          as the scale of linear commands.
#     max_ang_vel  rad/s   Same for rotation.
#     """

#     controller_cls = FloatingVelocityController

#     damping:     float = 50.0
#     force_limit: float = 20.0
#     max_lin_vel: float = 0.5
#     max_ang_vel: float = 3.14159

@dataclass
class FloatingVelocityConfig(ControllerConfig):
    controller_cls = FloatingVelocityController
    kp:          float = 15.0   # position gain — higher = snappier response
    kd:          float = 5.0    # velocity damping — higher = less overshoot
    max_lin_vel: float = 0.5    # m/s speed cap


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────

@register_agent()
class FloatingGripper(BaseAgent):
    """
    A single cylindrical body that floats freely over the table.

    The URDF contains one link ("gripper") joined to the world via a
    floating joint, giving the physics engine a real rigid body with
    mass and inertia.  The velocity controller drives it with a
    damped velocity command so contacts with blocks are physically
    realistic rather than teleport-style.

    tcp
    ---
    `self.agent.tcp` resolves to the gripper link itself, which is what
    PushBoundaryEnv already uses for reward and observation computation —
    no changes needed there.
    """

    uid = "floating_gripper"
    urdf_path = "/data/user_data/mbronars/packages/Planning_wrapper/urdfs/floating_gripper.urdf"
    controller_configs = {"floating_vel": FloatingVelocityConfig(joint_names=[])}

    # No joint-position noise needed — the controller handles reset
    urdf_config: dict = {}

    # Keyframe used by the env's _load_agent call to set the initial pose;
    # PushBoundaryEnv passes sapien.Pose(p=[BOUNDARY_CENTER_X, 0, 0.1])
    # so this is mostly a fallback.
    keyframes = {
        "default": Keyframe(
            qpos=np.zeros(0),   # free joint has no qpos entries in SAPIEN
            pose=sapien.Pose(p=[-0.135, 0.0, 0.1]),
        )
    }

    @property
    def tcp(self):
        """Tool-centre-point: the gripper link itself."""
        return self.robot.links_map["gripper"]

    @property
    def _controller_configs(self):
        # Bypass the base class active_joints lookup — we have no named joints
        return {"floating_vel": FloatingVelocityConfig(joint_names=[])}
    
    def _load_articulation(self, initial_pose):
        from transforms3d.euler import euler2quat
        import sapien.pysapien.physx as physx

        builder = self.scene.create_articulation_builder()
        link_builder = builder.create_link_builder()
        link_builder.set_name("gripper")

        upright = sapien.Pose(q=euler2quat(0, np.pi / 2, 0))
        link_builder.add_cylinder_collision(
            pose=upright, radius=0.01, half_length=0.06,
            material=physx.PhysxMaterial(0.8, 0.6, 0.0),
        )
        link_builder.add_cylinder_visual(pose=upright, radius=0.01, half_length=0.06)
        link_builder.set_mass_and_inertia(
            mass=0.5, cmass_local_pose=sapien.Pose(), inertia=[0.001, 0.001, 0.001],
        )

        builder.fix_root_link = False
        builder.initial_pose = sapien.Pose(  # ← set BEFORE build()
            p=[initial_pose.p[0], initial_pose.p[1], GRIPPER_Z_FIXED]
        )
        self.robot = builder.build(name="floating_gripper")