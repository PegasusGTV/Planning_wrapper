"""
Simple, not-necessarily-successful "planner" for PushT-v1.

Idea:
- Use the MSPlanningWrapper + PushTTaskAdapter.
- At each step, move the end-effector (tcp_pose) a little bit toward the goal_pos.
- Just for visualization and sanity checking.
"""

from typing import List

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym

import mani_skill.envs  # noqa: F401  # register ManiSkill envs

from planning_wrapper.wrappers.maniskill_planning import ManiSkillPlanningWrapper
from planning_wrapper.adapters import PushTTaskAdapter


MAX_STEPS = 150
STEP_SIZE = 0.01  # meters per step toward the goal


def simple_push_policy(wrapper: ManiSkillPlanningWrapper, obs) -> np.ndarray:
    """
    Given a state_dict obs, compute a very naive action:
    - move tcp position a small step toward goal_pos
    - keep orientation and gripper unchanged

    Assumes control_mode="pd_ee_delta_pose" with action shaped like:
        [dx, dy, dz, d_rx, d_ry, d_rz, gripper]
    but we only touch the first 3 components.
    """
    planning_obs = wrapper.get_planning_obs(obs)
    goal_pos = np.asarray(planning_obs["goal_pos"]).reshape(-1, 3)[0]  # Remove batch dim, get (3,)
    tcp_pose = np.asarray(planning_obs["tcp_pose"]).reshape(-1, 7)[0]  # Remove batch dim, get (7,)
    tcp_pos = tcp_pose[:3]  # Extract position from pose (first 3 elements)

    # Direction from tcp -> goal
    direction = goal_pos - tcp_pos
    dist = np.linalg.norm(direction)

    if dist < 1e-4:
        # Already at goal (ish) -> do nothing
        delta_pos = np.zeros(3, dtype=np.float32)
    else:
        direction = direction / dist
        step = min(STEP_SIZE, dist)
        delta_pos = direction * step

    # Build action with correct shape
    sample = wrapper.action_space.sample()
    action = np.zeros_like(sample, dtype=np.float32)

    # Fill first 3 dims with delta position; leave others as 0
    action_flat = action.reshape(-1)
    action_flat[:3] = delta_pos  # [dx, dy, dz]

    # If there's a gripper dimension, we can keep it slightly closed or open
    # (here we just leave it as whatever 0 means)
    return action


def main():
    # 1) Build env + wrapper + adapter
    env = gym.make(
        "PushT-v1",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        # Override ManiSkill's default 100-step TimeLimit so the episode
        # does not truncate at t=99 while our loop is still running.
        max_episode_steps=MAX_STEPS,
        render_mode="human",  # or "rgb_array" depending on your setup
    )

    adapter = PushTTaskAdapter()
    w = ManiSkillPlanningWrapper(env, adapter=adapter)

    # 2) Reset
    obs, info = w.reset(seed=0)
    print("Reset with seed=0.")

    # 3) Planning loop (really just a naive policy loop)
    obj_traj: List[np.ndarray] = []
    goal_traj: List[np.ndarray] = []

    for t in range(MAX_STEPS):
        # Record positions for debugging/plotting
        extra = obs["extra"]
        obj_pose = np.asarray(extra["obj_pose"])
        goal_pos = np.asarray(extra["goal_pos"])
        obj_traj.append(obj_pose.copy())
        goal_traj.append(goal_pos.copy())

        print(f"Step {t}:")
        print("  obj_pos :", obj_pose[:3])
        print("  goal_pos:", goal_pos[:3])

        # Render for visualization (if supported)
        try:
            w.render()
        except Exception:
            # If render() is not available / fails, just ignore
            pass

        # Compute action from simple policy
        action = simple_push_policy(w, obs)

        # Step env
        obs, reward, terminated, truncated, info = w.step(action)
        print("  reward  :", float(reward))
        if terminated or truncated:
            print(f"Episode ended at step {t} (terminated={terminated}, truncated={truncated})")
            break

    w.close()
    print("Finished simple planning demo.")


if __name__ == "__main__":
    main()
