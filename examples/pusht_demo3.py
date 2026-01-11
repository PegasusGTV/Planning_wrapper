"""
PushT touch demo (GT): EE approaches behind object, descends, then pushes object toward goal.
This is just to visually verify physics/contact.
"""

from typing import Dict, Any
import time
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym

import mani_skill.envs  # noqa: F401

from planning_wrapper.wrappers.maniskill_planning import ManiSkillPlanningWrapper
from planning_wrapper.adapters import PushTTaskAdapter


# ----- Tuning knobs (aggressive enough to make real contact) -----
MAX_STEPS = 200
RENDER_SLEEP = 0.02    # ~50 FPS
STEP_SIZE = 0.003      # slightly smaller step but more controlled

# Get closer and slightly lower so the stick actually hits the T block.
BACKOFF = 0.010        # closer behind the tee
ABOVE_Z = 0.03         # approach clearly above the tee
PUSH_Z_OFFSET = -0.005  # push a bit *below* the center (more leverage)
CONTACT_EPS = 0.010    # aim a full 1 cm into the face
GOAL_CLOSE_XY = 0.03   # stop if object is close in XY
HIDE_OBJ_ORI = True    # match PushT-v1 GT demo


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
    z_push = float(obj[2] + PUSH_Z_OFFSET)

    phase = ctx.get("phase", 0)

    # Phase 0: initialize ctx
    if phase == 0:
        ctx["phase"] = 1
        ctx["push_steps"] = 0

    # Phase 1: go above pre-contact
    if ctx["phase"] == 1:
        target = np.array([pre_xy[0], pre_xy[1], z_above], dtype=np.float32)
        delta = move_towards(tcp, target, STEP_SIZE)
        if np.linalg.norm(target - tcp) < 0.01:
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

        # first few steps: slightly more aggressive to ensure contact
        fwd = STEP_SIZE
        if ctx["push_steps"] < 5:
            fwd = STEP_SIZE * 1.5

        # move forward along push direction, keep z near push height
        target = np.array(
            [tcp[0] + push_dir_xy[0] * fwd,
             tcp[1] + push_dir_xy[1] * fwd,
             z_push],
            dtype=np.float32
        )
        delta = move_towards(tcp, target, STEP_SIZE)

        # stop pushing after some steps or if goal is close
        if ctx["push_steps"] > 350 or np.linalg.norm((obj - goal)[:2]) < GOAL_CLOSE_XY:
            ctx["phase"] = 4
        return make_delta_action(wrapper, delta)

    # Phase 4: done — hold still
    return make_delta_action(wrapper, np.zeros(3, dtype=np.float32))


def main():
    # Same env creation pattern as pusht_demo4: PushT-v1, state_dict, pd_ee_delta_pose, human render
    env = gym.make(
        "PushT-v1",
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
    print("GT demo reset(seed=0).")

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
    print("Finished GT touch demo.")


if __name__ == "__main__":
    main()
