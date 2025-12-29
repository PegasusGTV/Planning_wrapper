# examples/simple_planning_demo_long.py

"""
Longer / smoother simple "planner" demo for PushT-v1.
"""

from typing import List
import time

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym

import mani_skill.envs  # noqa: F401

from planning_wrapper.wrappers.maniskill_planning import ManiSkillPlanningWrapper
from planning_wrapper.adapters import PushTTaskAdapter


MAX_STEPS = 1000      # longer sim
STEP_SIZE = 0.005    # smaller step for smoother motion
RENDER_SLEEP = 0.02  # ~50 FPS


def simple_push_policy(wrapper: ManiSkillPlanningWrapper, obs) -> np.ndarray:
    planning_obs = wrapper.get_planning_obs(obs)
    goal_pos = np.asarray(planning_obs["goal_pos"]).reshape(-1, 3)[0]  # Remove batch dim, get (3,)
    tcp_pose = np.asarray(planning_obs["tcp_pose"]).reshape(-1, 7)[0]  # Remove batch dim, get (7,)
    tcp_pos = tcp_pose[:3]  # Extract position from pose (first 3 elements)

    direction = goal_pos - tcp_pos
    dist = np.linalg.norm(direction)

    if dist < 1e-4:
        delta_pos = np.zeros(3, dtype=np.float32)
    else:
        direction = direction / dist
        step = min(STEP_SIZE, dist)
        delta_pos = direction * step

    sample = wrapper.action_space.sample()
    action = np.zeros_like(sample, dtype=np.float32)
    action_flat = action.reshape(-1)
    action_flat[:3] = delta_pos
    return action


def main():
    env = gym.make(
        "PushT-v1",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        render_mode="human",
    )

    adapter = PushTTaskAdapter()
    w = ManiSkillPlanningWrapper(env, adapter=adapter)

    obs, info = w.reset(seed=0)
    print("Reset with seed=0.")

    obj_traj: List[np.ndarray] = []
    goal_traj: List[np.ndarray] = []

    for t in range(MAX_STEPS):
        extra = obs["extra"]
        obj_pose = np.asarray(extra["obj_pose"])
        goal_pos = np.asarray(extra["goal_pos"])
        obj_traj.append(obj_pose.copy())
        goal_traj.append(goal_pos.copy())

        print(f"Step {t}: obj_pos={obj_pose[:3]}, goal_pos={goal_pos[:3]}")

        try:
            w.render()
        except Exception:
            pass

        time.sleep(RENDER_SLEEP)

        action = simple_push_policy(w, obs)
        obs, reward, terminated, truncated, info = w.step(action)
        if terminated or truncated:
            print(f"Episode ended at step {t} (terminated={terminated}, truncated={truncated})")
            break

    w.close()
    print("Finished long simple planning demo.")


if __name__ == "__main__":
    main()
