"""
PushT touch demo (GT): EE approaches behind object, descends, then pushes object toward goal.
This version includes an additional object on the table.
This is just to visually verify physics/contact.
"""

from typing import Dict, Any
import time
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

try:
    import gymnasium as gym
except ImportError:
    import gym

import mani_skill.envs  # noqa: F401
from mani_skill.envs.tasks.tabletop.push_t import PushTEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.building import actors
from mani_skill.utils.structs import Pose

from planning_wrapper.wrappers.maniskill_planning import ManiSkillPlanningWrapper
from planning_wrapper.adapters import PushTTaskAdapter


# Register a custom environment that extends PushTEnv with an additional object
@register_env("PushT-WithExtraObject-v1", max_episode_steps=200)
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
        self.T_static_friction = 0.5   # Reduced from 3 to 0.5
        
        # Call parent's _load_scene to set up the T block and goal
        super()._load_scene(options)
        
        # Add an additional cube object on the table
        # Create a cube with reasonable size and color
        self.extra_obj = actors.build_cube(
            self.scene,
            half_size=0.02,  # 4cm cube
            color=np.array([0, 255, 0, 255]) / 255,  # Green color
            name="extra_cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0.1, 0.1, 0.02]),  # Position on table, above surface
        )
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Call parent's _initialize_episode to set up the T block
        super()._initialize_episode(env_idx, options)
        
        # Place the extra object close to the T block
        with torch.device(self.device):
            b = len(env_idx)
            # Get the T block's position
            tee_pos = self.tee.pose.p  # Shape: (b, 3)
            
            # Place the extra cube close to the T block
            # Offset it slightly to the side and slightly behind
            extra_obj_xyz = tee_pos.clone()
            extra_obj_xyz[:, 0] += 0.06  # Offset in x direction (to the side)
            extra_obj_xyz[:, 1] += 0.04  # Offset in y direction (slightly behind)
            extra_obj_xyz[:, 2] = 0.02   # Half size of cube (4cm / 2 = 2cm) on table surface
            
            # Set the pose of the extra object
            q = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(b, 1)  # No rotation
            extra_obj_pose = Pose.create_from_pq(p=extra_obj_xyz, q=q)
            self.extra_obj.set_pose(extra_obj_pose)


# ----- Tuning knobs (aggressive enough to make real contact) -----
MAX_STEPS = 520
RENDER_SLEEP = 0.02    # ~50 FPS
STEP_SIZE = 0.003      # slightly smaller step but more controlled
PUSH_STEP_SIZE = 0.08  # larger step size for pushing (increases force)

# Get closer and slightly lower so the stick actually hits the T block.
BACKOFF = 0.010        # closer behind the tee
ABOVE_Z = 0.03         # approach clearly above the tee
# Extra Z offset around the very bottom of the T block (see TEE_HALF_THICKNESS / TABLE_CLEARANCE_Z)
# Positive values push slightly higher than the bottom, negative values slightly lower (but still clamped).
PUSH_Z_OFFSET = -0.01  # default: push ~1 cm lower than the bottom clearance reference
CONTACT_EPS = 0.010    # aim a full 1 cm into the face
GOAL_CLOSE_XY = 0.03   # stop if object is close in XY
PUSH_FORCE_MULTIPLIER = 10.0  # multiplier for push force (higher = more force)
AGGRESSIVE_PUSH_STEPS = 100   # number of steps with extra aggressive pushing
HIDE_OBJ_ORI = True    # match PushT-v1 GT demo

# Geometry of the T block for contact height computation
TEE_HALF_THICKNESS = 0.02      # half of 0.04 m (from PushT env)
# Keep TCP very close to the table; raise if you see table collisions.
TABLE_CLEARANCE_Z = 0.0003     # 0.3 mm above the table


def mask_obj_orientation_in_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Optional: keep obj_pose length the same but wipe quaternion."""
    if not isinstance(obs, dict) or "extra" not in obs or "obj_pose" not in obs["extra"]:
        return obs
    obs2 = dict(obs)
    extra2 = dict(obs["extra"])
    obj_pose = np.asarray(extra2["obj_pose"], dtype=np.float32).copy()
    if obj_pose.shape[-1] >= 7:
        obj_pose[..., 3:7] = np.array([0, 0, 0, 1], dtype=np.float32)
    extra2["obj_pose"] = obj_pose
    obs2["extra"] = extra2
    return obs2


def _as3(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return x[:3].copy()


def _tcp_pos(planning_obs):
    tcp_pose = np.asarray(planning_obs["tcp_pose"], dtype=np.float32).reshape(-1, 7)[0]
    return tcp_pose[:3].copy()


def _obj_pos(planning_obs):
    obj_pose = np.asarray(planning_obs["obj_pose"], dtype=np.float32).reshape(-1, 7)[0]
    return obj_pose[:3].copy()


def _goal_pos(planning_obs):
    gp = np.asarray(planning_obs["goal_pos"], dtype=np.float32).reshape(-1)
    return gp[:3].copy()


def make_delta_action(wrapper: ManiSkillPlanningWrapper, delta_xyz: np.ndarray) -> np.ndarray:
    """
    pd_ee_delta_pose: first 3 dims are delta position.
    The controller normalizes actions, so we need to convert raw delta (in meters)
    to the normalized action space [-1, 1].
    
    Formula: normalized = (delta - 0.5*(high+low)) / (0.5*(high-low))
    For bounds [-0.1, 0.1]: normalized = delta / 0.1
    """
    # Get controller bounds from wrapper
    low, high = wrapper.get_controller_bounds()
    
    sample = wrapper.action_space.sample()
    action = np.zeros_like(sample, dtype=np.float32)
    flat = action.reshape(-1)
    
    # Convert delta from meters to normalized action space
    # Inverse of clip_and_scale_action: normalized = (delta - center) / half_range
    center = 0.5 * (high + low)
    half_range = 0.5 * (high - low)
    if half_range < 1e-6:
        # Avoid division by zero
        normalized_delta = np.zeros_like(delta_xyz, dtype=np.float32)
    else:
        normalized_delta = (delta_xyz - center) / half_range
    
    # Clip to [-1, 1] to respect action space bounds
    normalized_delta = np.clip(normalized_delta, -1.0, 1.0)
    
    flat[:3] = normalized_delta.astype(np.float32)
    return action


def move_towards(curr: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    d = target - curr
    dist = float(np.linalg.norm(d))
    if dist < 1e-6:
        return np.zeros(3, dtype=np.float32)
    step = min(max_step, dist)
    return (d / dist * step).astype(np.float32)


def touch_and_push_policy(wrapper: ManiSkillPlanningWrapper, obs: Dict[str, Any], ctx: Dict[str, Any]) -> np.ndarray:
    """
    Simple state machine (matching the known-good pusht_demo4):
      0) compute push direction, pre-contact point
      1) go above pre-contact
      2) descend to push height
      3) push forward toward goal (keeping height) in a straight line
    """
    if HIDE_OBJ_ORI:
        obs = mask_obj_orientation_in_obs(obs)

    planning_obs = wrapper.get_planning_obs(obs)

    tcp = _tcp_pos(planning_obs)
    obj = _obj_pos(planning_obs)
    goal = _goal_pos(planning_obs)

    # Stop if object already near goal (xy)
    if np.linalg.norm((obj - goal)[:2]) < GOAL_CLOSE_XY:
        ctx["phase"] = 4  # done

    # Compute push direction in XY (goal - object) once; keep it fixed for a straight-line shove.
    if "push_dir_xy" not in ctx:
        d_xy = (goal - obj)[:2]
        n = float(np.linalg.norm(d_xy))
        if n < 1e-6:
            push_dir_xy = np.array([1.0, 0.0], dtype=np.float32)
        else:
            push_dir_xy = (d_xy / n).astype(np.float32)
        ctx["push_dir_xy"] = push_dir_xy
        ctx["pre_xy"] = obj[:2] - push_dir_xy * BACKOFF
        ctx["contact_xy"] = obj[:2] - push_dir_xy * CONTACT_EPS
    else:
        push_dir_xy = ctx["push_dir_xy"]
        pre_xy = ctx["pre_xy"]
        contact_xy = ctx["contact_xy"]

    # Pre-contact point: behind object along push direction (cached once above)
    pre_xy = ctx.get("pre_xy", obj[:2] - push_dir_xy * BACKOFF)
    contact_xy = ctx.get("contact_xy", obj[:2] - push_dir_xy * CONTACT_EPS)  # closer to the block face

    # Heights
    z_above = float(obj[2] + ABOVE_Z)

    # Compute a very low push height: near the bottom of the T, just above the table.
    # obj[2] is approximately the center of mass height; subtract half thickness to get bottom.
    raw_z_push = float(obj[2] - TEE_HALF_THICKNESS + TABLE_CLEARANCE_Z + PUSH_Z_OFFSET)
    # Never go below the table clearance.
    z_push = max(raw_z_push, TABLE_CLEARANCE_Z)

    phase = ctx.get("phase", 0)

    # Phase 0: initialize ctx
    if phase == 0:
        ctx["phase"] = 1
        ctx["push_steps"] = 0

    # Phase 1: go directly down to push height at pre-contact XY
    if ctx["phase"] == 1:
        target = np.array([pre_xy[0], pre_xy[1], z_push], dtype=np.float32)
        delta = move_towards(tcp, target, STEP_SIZE)
        if np.linalg.norm(target - tcp) < 0.008:
            ctx["phase"] = 2
        return make_delta_action(wrapper, delta)

    # Phase 2: go down to push height at the *contact* XY
    if ctx["phase"] == 2:
        target = np.array([contact_xy[0], contact_xy[1], z_push], dtype=np.float32)
        delta = move_towards(tcp, target, STEP_SIZE)
        if np.linalg.norm(target - tcp) < 0.008:
            ctx["phase"] = 3
        return make_delta_action(wrapper, delta)

    # Phase 3: push forward in XY while holding Z
    if ctx["phase"] == 3:
        ctx["push_steps"] += 1

        # Calculate forward step with force multiplier
        # Use larger force for initial aggressive push steps
        if ctx["push_steps"] < AGGRESSIVE_PUSH_STEPS:
            fwd = PUSH_STEP_SIZE * PUSH_FORCE_MULTIPLIER
        else:
            fwd = PUSH_STEP_SIZE

        # move forward along push direction, keep z near push height
        target = np.array(
            [tcp[0] + push_dir_xy[0] * fwd,
             tcp[1] + push_dir_xy[1] * fwd,
             z_push],
            dtype=np.float32
        )
        # Use the calculated fwd as max_step to allow the full movement
        delta = move_towards(tcp, target, fwd)

        # stop pushing after some steps or if goal is close
        if ctx["push_steps"] > 350 or np.linalg.norm((obj - goal)[:2]) < GOAL_CLOSE_XY:
            ctx["phase"] = 4
        return make_delta_action(wrapper, delta)

    # Phase 4: done — hold still
    return make_delta_action(wrapper, np.zeros(3, dtype=np.float32))


def main():
    # Use the custom environment with extra object
    env = gym.make(
        "PushT-WithExtraObject-v1",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        # Avoid truncation at 100 steps while the scripted policy runs.
        max_episode_steps=MAX_STEPS,
        render_mode="human",
        sim_backend="cpu",
        num_envs=1,
    )

    w = ManiSkillPlanningWrapper(env, adapter=PushTTaskAdapter())
    obs, info = w.reset(seed=0)
    print("GT demo reset(seed=0) with extra object on table.")

    ctx: Dict[str, Any] = {"phase": 0}

    prev_obj_pos = None
    for t in range(MAX_STEPS):
        extra = obs["extra"]
        obj_pose = np.asarray(extra["obj_pose"])
        goal_pos = np.asarray(extra["goal_pos"])
        
        # Debug prints: check TCP-obj distance to verify contact
        planning_obs = w.get_planning_obs(obs)
        tcp = _tcp_pos(planning_obs)
        obj = _obj_pos(planning_obs)
        dist_xy = np.linalg.norm((tcp - obj)[:2])
        dz = tcp[2] - obj[2]
        
        if t % 20 == 0:
            print(f"[t={t}] obj_pos={obj_pose[:3]} goal_pos={goal_pos[:3]} "
                  f"phase={ctx.get('phase')} dist_xy={dist_xy:.4f} dz={dz:.4f}")

        try:
            w.render()
        except Exception:
            pass
        time.sleep(RENDER_SLEEP)

        action = touch_and_push_policy(w, obs, ctx)
        obs, reward, terminated, truncated, info = w.step(action)

        if terminated or truncated:
            print(f"Episode ended at t={t}. Resetting.")
            obs, info = w.reset(seed=0)
            ctx = {"phase": 0}

    w.close()
    print("Finished GT touch demo with extra object.")


if __name__ == "__main__":
    main()

