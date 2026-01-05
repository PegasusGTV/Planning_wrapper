# planning_wrapper/envs/shelf_retrieve_v1.py

"""
ObjectRetrieveFromShelf-v1

Minimal MVP task spec for planning + RL on a shelf-retrieval scenario.

Scene
-----
- A fixed table and a single shelf bay in front of the robot.
- One rigid object is initially placed *inside* the shelf bay.
- A single fixed camera (optional) and a 6-DoF end-effector (same control mode as PushT).

Coordinate / geometry conventions
---------------------------------
- World frame: x-axis points "into" the shelf, y-axis is lateral, z-axis is vertical.
- The shelf bay interior is approximated as an axis-aligned box in the world frame.

  Let:
    shelf_center: 3D position of the bay center in world coordinates.
    shelf_size:   (sx, sy, sz) = (width_x, width_y, height_z) of the interior.

  Then:
    x_min = shelf_center.x - sx / 2
    x_max = shelf_center.x + sx / 2

- The object is represented by its center pose in world coordinates:
    obj_pose: (x, y, z, qw, qx, qy, qz)
  where (x, y, z) is the translation and (qw, qx, qy, qz) is the quaternion.

Success condition
-----------------
Goal: "Pull the object out of the shelf."

Let:
  obj_pos = obj_pose[:3]
  obj_x, obj_y, obj_z = obj_pos

The episode is considered successful when BOTH of the following hold:

1. obj_x is OUTSIDE the shelf bay in x:
     obj_x > x_max + x_margin

   where x_margin >= 0 provides a small safety buffer (e.g., 1–2 cm) so the
   object is clearly out of the bay.

2. The object has been lifted slightly above the bay floor:
     obj_z > shelf_floor_z + z_margin

   where shelf_floor_z = (shelf_center.z - sz / 2)
   and z_margin is a small positive threshold (e.g., 1–2 cm).

Intuitively:
- The object must be pulled out along x (beyond the front face of the bay).
- The object must be lifted slightly to avoid grazing the shelf floor.

Observation (state_dict mode)
-----------------------------
We extend ManiSkill's `state_dict` observation with an "extra" dict that
contains all planning-relevant fields:

  obs["extra"] contains:
    - "tcp_pose":   (7,)  end-effector pose in world frame
    - "obj_pose":   (7,)  object pose in world frame
    - "shelf_center": (3,) center of the shelf bay (world)
    - "shelf_size":   (3,) (sx, sy, sz) of the shelf interior (world)
    - "shelf_bounds_x": (2,) [x_min, x_max]
    - optionally: any precomputed convenience scalars used for success checks

The planning wrapper will then build a compact planning observation using:
    tcp_pose, obj_pose, shelf_center, shelf_size, shelf_bounds_x

Control mode
------------
- Use the exact same control mode as PushT for the MVP, e.g.:
    control_mode = "pd_ee_delta_pose"

This ensures that:
  - the planning wrapper's action handling stays identical to PushT,
  - planners / RL algorithms can be reused across PushT and ObjectRetrieveFromShelf.

Episode termination & reward (MVP)
----------------------------------
- max_episode_steps: 200
- Terminate when:
    - success condition is satisfied, OR
    - step count exceeds max_episode_steps.

- Reward:
    - Sparse MVP:
        reward = 1.0 on success, 0.0 otherwise.
    - (Can be extended later with shaped distance-to-goal rewards.)

Summary (single-line spec)
--------------------------
"The robot must pull the object out of the shelf bay along +x and lift it
slightly, such that the object center is beyond the bay's front face and
above the shelf floor, as computed from the object pose and shelf geometry."
"""


# import numpy as np
# from typing import Dict, Any


# def compute_shelf_success(extra: Dict[str, Any]) -> bool:
#     """Compute success for ObjectRetrieveFromShelf-v1 from obs['extra']."""
#     obj_pose = np.asarray(extra["obj_pose"])      # (7,)
#     shelf_center = np.asarray(extra["shelf_center"])  # (3,)
#     shelf_size = np.asarray(extra["shelf_size"])      # (3,)

#     sx, sy, sz = shelf_size
#     x_min = shelf_center[0] - sx / 2.0
#     x_max = shelf_center[0] + sx / 2.0
#     shelf_floor_z = shelf_center[2] - sz / 2.0

#     obj_x, _, obj_z = obj_pose[:3]

#     x_margin = 0.02  # 2 cm
#     z_margin = 0.02  # 2 cm

#     pulled_out = obj_x > (x_max + x_margin)
#     lifted = obj_z > (shelf_floor_z + z_margin)

#     return bool(pulled_out and lifted)
# planning_wrapper/envs/shelf_retrieve_v1.py

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
import sapien.core as sapien
import sapien.physx as physx
import sapien.render

from .shelf_scene_builder import ShelfSceneBuilder


def compute_shelf_success(extra: Dict[str, Any]) -> bool:
    """
    Compute success for ObjectRetrieveFromShelf-v1 from obs['extra'].

    Success when:
      - object is pulled out along +x beyond bay front face, and
      - object is slightly above bay floor.
    """
    obj_pose = np.asarray(extra["obj_pose"])        # (7,)
    shelf_center = np.asarray(extra["shelf_center"])  # (3,)
    shelf_size = np.asarray(extra["shelf_size"])      # (3,)

    sx, sy, sz = shelf_size
    cx, cy, cz = shelf_center

    # Bay bounds in x
    x_min = cx - sx / 2.0
    x_max = cx + sx / 2.0

    # Bay floor height
    shelf_floor_z = cz - sz / 2.0

    obj_x, _, obj_z = obj_pose[:3]

    x_margin = 0.02  # 2 cm beyond front face
    z_margin = 0.02  # 2 cm above floor

    pulled_out = obj_x > (x_max + x_margin)
    lifted = obj_z > (shelf_floor_z + z_margin)

    return bool(pulled_out and lifted)


@register_env("ObjectRetrieveFromShelf-v1", max_episode_steps=200)
class ObjectRetrieveFromShelfEnv(BaseEnv):
    """
    Minimal shelf-retrieval task for planning + RL.

    Scene:
      - Static table.
      - Static open-front shelf bay on top of table (built by ShelfSceneBuilder).
      - One dynamic object initially placed INSIDE the bay.

    Goal:
      - Pull the object out along +x so that its center is beyond the bay front face.
      - Lift it slightly above the shelf floor.

    Obs (state_dict mode) will include extra fields:
      - tcp_pose
      - obj_pose
      - shelf_center
      - shelf_size
      - shelf_bounds_x
    """

    # You can restrict robots if you want; keep same as PushT for now.
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    SUPPORTED_REWARD_MODES = ["sparse", "none"]

    def __init__(
        self,
        *args,
        robot_uids: str | Tuple[str, ...] = "panda",
        spawn_mode: str = "random_small",
        **kwargs,
    ) -> None:
        # Initialize shelf builder BEFORE super().__init__() because
        # super().__init__() triggers reset() which calls _load_scene()
        # Scene / geometry builder for static shelf + table
        self.shelf_builder = ShelfSceneBuilder()

        # Handle to the dynamic object (will be created or reused later)
        self.obj: Any | None = None

        # Cube geometry - must be set before super().__init__() triggers _load_scene()
        self.cube_half_size = np.array([0.03, 0.03, 0.03], dtype=np.float32)

        # Spawn behavior
        self.spawn_mode = spawn_mode  # "fixed" (Phase A) or "random_small" (Phase B)

        # Will be filled after robot is built in _load_scene
        self._default_qpos = None

        # Keep same control/obs mode choices as PushT for compatibility
        super().__init__(
            *args,
            robot_uids=robot_uids,
            **kwargs,
        )
        

    # ------------------------------------------------------------------ #
    # Required hooks: _load_scene, _initialize_episode, evaluate, extras #
    # ------------------------------------------------------------------ #

    def _load_scene(self, options: Dict[str, Any]) -> None:
        """
        Build static scene objects (table + shelf) and robot.

        Static geometry does NOT need to be saved/restored for planning.
        """
        # Typical BaseEnv behavior: build ground, robot, etc.
        # If your other envs call a helper here (like _build_default_scene),
        # do the same.
        super()._load_scene(options)
        
        # self.scene is a ManiSkillScene wrapper; get the underlying SAPIEN scene
        # For single env, use sub_scenes[0]; for parallel envs, build in all sub_scenes
        scene = self.scene.sub_scenes[0]

        # Save default robot configuration for consistent resets
        try:
            self._default_qpos = self.agent.robot.get_qpos().copy()
        except Exception:
            self._default_qpos = None

        # Build shelf + table in all sub_scenes
        for sub_scene in self.scene.sub_scenes:
            self.shelf_builder.build(sub_scene)

        # Create materials using the new API
        pm = physx.PhysxMaterial(static_friction=0.5, dynamic_friction=0.5, restitution=0.01)
        vm = sapien.render.RenderMaterial(
            base_color=[0.9, 0.3, 0.3, 1.0],
            metallic=0.0,
            roughness=0.4,
        )

        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=self.cube_half_size, material=pm)
        builder.add_box_visual(half_size=self.cube_half_size, material=vm)

        bay_center = self.shelf_builder.bay_center_xyz
        initial_pos = bay_center.copy()
        initial_pos[2] += self.cube_half_size[2]

        builder.initial_pose = sapien.Pose(initial_pos, [1, 0, 0, 0])
        self.obj = builder.build(name="shelf_object")
        # NOTE:
        # - Dynamic object spawning will happen in _initialize_episode.
        # - Robot is constructed by BaseEnv (via super()._load_scene).
    def _reset_robot_pose(self) -> None:
        """Reset robot to a consistent start configuration."""
        if self._default_qpos is None:
            # Fallback: let BaseEnv / agent handle it
            try:
                self.agent.reset()
            except Exception:
                pass
            return

        try:
            robot = self.agent.robot
            qpos = self._default_qpos.copy()
            qvel = np.zeros_like(qpos)
            robot.set_qpos(qpos)
            robot.set_qvel(qvel)
        except Exception:
            # If something goes wrong, we don't want the env to crash
            pass
    def _spawn_object_fixed(self) -> None:
        """
        Phase A: fixed spawn.
        Place the cube at a known safe pose in the center of the bay.
        """
        if self.obj is None:
            return

        bay_min = self.shelf_builder.bay_min_xyz
        bay_max = self.shelf_builder.bay_max_xyz
        bay_center = 0.5 * (bay_min + bay_max)

        pos = bay_center.copy()
        # Put cube resting on the bay floor:
        # floor z = bay_min_z
        # cube center z = floor_z + half_height + small epsilon
        floor_z = bay_min[2]
        pos[2] = floor_z + self.cube_half_size[2] + 1e-3

        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.obj.set_pose(sapien.Pose(pos, quat))

    def _spawn_object_random_small(self) -> None:
        """
        Phase B: small randomization.
        Randomize x/y within bay bounds with a margin from walls.
        Keep z = floor + half cube size.
        """
        if self.obj is None:
            return

        bay_min = self.shelf_builder.bay_min_xyz
        bay_max = self.shelf_builder.bay_max_xyz

        # Margins in x/y to avoid intersecting walls
        # (must be >= cube_half_size + a tiny epsilon)
        mx = float(self.cube_half_size[0] + 0.01)
        my = float(self.cube_half_size[1] + 0.01)

        # Keep z exactly as in fixed spawn
        floor_z = bay_min[2]
        z_center = floor_z + self.cube_half_size[2] + 1e-3

        # Allowed range inside bay for cube CENTER
        x_min = bay_min[0] + mx
        x_max = bay_max[0] - mx
        y_min = bay_min[1] + my
        y_max = bay_max[1] - my

        # Safety: if margins over-shrink, clamp so min <= max
        if x_min > x_max:
            mid = 0.5 * (bay_min[0] + bay_max[0])
            x_min = x_max = mid
        if y_min > y_max:
            mid = 0.5 * (bay_min[1] + bay_max[1])
            y_min = y_max = mid

        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        pos = np.array([x, y, z_center], dtype=np.float32)

        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.obj.set_pose(sapien.Pose(pos, quat))

    def _initialize_episode(self, env_idx: int, options: Dict[str, Any]) -> None:
        """
        Reset per-episode state: place object inside shelf bay, randomize pose, etc.

        For now, this is a skeleton; you will flesh out object creation later.
        """
        # Always call super to ensure robot and BaseEnv internals reset properly
         # BaseEnv resets robot agents, controller internals, etc.
        super()._initialize_episode(env_idx, options)

        # 1) Reset robot to consistent configuration
        self._reset_robot_pose()

        # 2) Place the cube in the bay
        if self.spawn_mode == "fixed":
            # Phase A: deterministic pose
            self._spawn_object_fixed()
        else:
            # Default: Phase B small randomization
            self._spawn_object_random_small()
    
    def _get_obs_extra(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Minimal, stable extra observation signals for planning + RL.

        Always returns:
          - "tcp_pose":  (7,) end-effector pose (pos + quat)

        In state_dict mode (and generally, for this env) also returns:
          - "obj_pose":    (7,) object pose (pos + quat); if missing, NaNs
          - "bay_center":  (3,) bay interior center
          - "bay_size":    (3,) bay interior size
        """
        extra: Dict[str, Any] = {}

        # -------- tcp_pose (always) --------
        tcp_pose = np.full((7,), np.nan, dtype=np.float32)
        try:
            tcp = self.agent.tcp  # ManiSkill's robot wrapper
            # Use raw_pose which is a tensor, then convert to numpy
            pose_raw = tcp.pose.raw_pose  # Shape: (1, 7) or (7,)
            if hasattr(pose_raw, 'cpu'):
                pose_array = pose_raw.cpu().numpy()
            else:
                pose_array = np.asarray(pose_raw)
            # Remove batch dimension if present
            if pose_array.ndim > 1:
                pose_array = pose_array[0]
            tcp_pose[:] = pose_array.astype(np.float32)
        except Exception as e:
            # keep NaNs if something goes wrong, but key exists
            pass
        extra["tcp_pose"] = tcp_pose

        # -------- obj_pose (state_dict / this env) --------
        obj_pose = np.full((7,), np.nan, dtype=np.float32)
        if self.obj is not None:
            try:
                pose = self.obj.get_pose()
                pos = np.asarray(pose.p, dtype=np.float32)
                quat = np.asarray(pose.q, dtype=np.float32)
                obj_pose[:] = np.concatenate([pos, quat], axis=0)
            except Exception:
                # keep NaNs, but the shape stays (7,)
                pass
        extra["obj_pose"] = obj_pose

        # -------- shelf bay geometry (static & always valid) --------
        bay_center = self.shelf_builder.bay_center_xyz.astype(np.float32)
        bay_size = self.shelf_builder.bay_size_xyz.astype(np.float32)
        bay_min = self.shelf_builder.bay_min_xyz.astype(np.float32)
        bay_max = self.shelf_builder.bay_max_xyz.astype(np.float32)

        extra["bay_center"] = bay_center  # (3,)
        extra["bay_size"] = bay_size      # (3,)

        # -------- goal_pos: target position for pulling object out of shelf --------
        # Goal is to pull object out along +x beyond the front face of the bay
        # Position: slightly beyond bay_max_x, same y as bay center, slightly above floor
        x_margin = 0.05  # 5 cm beyond front face (more than success margin for planning)
        z_margin = 0.03  # 3 cm above floor
        
        goal_pos = np.array([
            bay_max[0] + x_margin,  # x: beyond front face
            bay_center[1],           # y: same lateral position
            bay_min[2] + z_margin,  # z: slightly above floor
        ], dtype=np.float32)
        extra["goal_pos"] = goal_pos  # (3,)

        return extra

    def _compute_success(self) -> bool:
        """
        Compute whether the object has been successfully retrieved from the shelf.
        
        Success conditions:
        - Object is pulled out along +x beyond bay front face (x > x_max + margin)
        - Object is lifted slightly above bay floor (z > floor_z + margin)
        """
        if self.obj is None:
            return False
        
        try:
            # Get current object pose
            obj_pose = self.obj.get_pose()
            obj_pos = np.asarray(obj_pose.p, dtype=np.float32)
            obj_x, _, obj_z = obj_pos
            
            # Get shelf geometry
            bay_center = self.shelf_builder.bay_center_xyz
            bay_size = self.shelf_builder.bay_size_xyz
            bay_min = self.shelf_builder.bay_min_xyz
            bay_max = self.shelf_builder.bay_max_xyz
            
            sx, sy, sz = bay_size
            cx, cy, cz = bay_center
            
            # Bay bounds in x
            x_max = bay_max[0]
            
            # Bay floor height
            shelf_floor_z = bay_min[2]
            
            # Margins
            x_margin = 0.02  # 2 cm beyond front face
            z_margin = 0.02  # 2 cm above floor
            
            # Check success conditions
            pulled_out = obj_x > (x_max + x_margin)
            lifted = obj_z > (shelf_floor_z + z_margin)
            
            return bool(pulled_out and lifted)
        except Exception:
            return False

    # ManiSkill calls evaluate() at the end of episode to compute metrics.
    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """
        Compute success/failure indicators for the current episode.

        Returns:
            {
                "success": float in {0.0, 1.0},
                "is_success": bool
            }

        For batched / vectorized setups, ManiSkill will handle aggregation.
        """
        success = self._compute_success()

        # Return as tensor for batched environments
        success_tensor = torch.tensor([success], device=self.device, dtype=torch.bool)
        if self.num_envs > 1:
            success_tensor = success_tensor.repeat(self.num_envs)

        # Numeric + boolean versions for downstream logging
        return {
            "success": success_tensor,
            "is_success": success_tensor,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        """Sparse reward: 1.0 on success, 0.0 otherwise."""
        # For sparse mode, this won't be called, but we implement it for completeness
        success = info.get("success", False)
        if isinstance(success, torch.Tensor):
            return success.float()
        return float(success)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        """Normalized dense reward (same as sparse for now)."""
        return self.compute_dense_reward(obs=obs, action=action, info=info)
