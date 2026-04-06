"""
agents/floating_gripper.py

A free-floating cylindrical gripper agent for ManiSkill.
Replaces the Panda arm with a single rigid body controlled via
a 2-DOF joint velocity drive (dx, dy) using native SAPIEN joints.

Usage
-----
    import agents          # triggers registration via agents/__init__.py
    env = gym.make("PushBoundary", robot_uids="floating_gripper")
"""

from __future__ import annotations

import numpy as np
import sapien
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers.pd_joint_pos import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent
from transforms3d.euler import euler2quat

# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────

@register_agent()
class FloatingGripper(BaseAgent):
    """
    A single cylindrical body that floats freely over the table.

    Instead of a true floating base, this articulation uses two orthogonal
    prismatic (sliding) joints connected to a fixed root. This natively 
    locks the Z-axis and all rotations using the physics engine, while 
    allowing 2D Cartesian control with perfect isotropic mass dynamics.
    """

    uid = "floating_gripper"
    # Keeping your original URDF path for reference, though we are 
    # building the articulation entirely programmatically below.
    urdf_path = "/data/user_data/mbronars/packages/Planning_wrapper/urdfs/floating_gripper.urdf"

    urdf_config: dict = {}

    # Initial states
    keyframes = {
        "default": Keyframe(
            qpos=np.array([0.0, 0.0]), # The 2 sliding joints start at 0
            pose=sapien.Pose(p=[-0.135, 0.0, 0.085]),
        )
    }
    
    @property
    def _controller_configs(self):
        # return {
        #     "floating_vel": PDJointPosControllerConfig(
        #         joint_names=["joint_x", "joint_y"],
        #         lower=-0.1,          # MATCHED TO PANDA: 5x faster max speed per step
        #         upper=0.1,           # MATCHED TO PANDA: 5x faster max speed per step
        #         stiffness=2000.0,    # High stiffness for strict position tracking
        #         damping=100.0,       # Keeps movements crisp without oscillating
        #         force_limit=1000.0,  # CRANKED UP: Plenty of raw linear pushing power
        #         use_delta=True,      
        #         normalize_action=False, 
        #     )
        # }
        return {
            "floating_vel": PDJointPosControllerConfig(
                joint_names=["joint_x", "joint_y"],
                lower=-0.1,          
                upper=0.1,           
                stiffness=500.0,     # Quartered
                damping=50.0,        # Halved
                force_limit=150.0,   # Still 3x the typical max P-term force (500 * 0.1 = 50)
                use_delta=True,      
                normalize_action=False, 
            )
        }

    @property
    def tcp(self):
        """Tool-centre-point: the gripper link itself."""
        return self.robot.links_map["gripper"]