"""
Simple visualization demo for Reach-WithObstacles-v1.

A proportional controller moves the EE toward the sampled goal position.
Control mode: pd_ee_delta_pose (easier to write a simple goal-reaching policy).

Run:
    python examples/reach_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym

import envs  # noqa: F401 — registers Reach-WithObstacles-v1


MAX_STEPS = 200
STEP_SIZE = 0.05   # max delta per step (metres)


def reach_policy(obs) -> np.ndarray:
    """Proportional controller: move EE toward goal."""
    extra     = obs["extra"]
    ee_to_goal = np.asarray(extra["ee_to_goal"]).reshape(3)   # (3,)

    dist = np.linalg.norm(ee_to_goal)
    if dist > 1e-4:
        direction = ee_to_goal / dist
    else:
        direction = np.zeros(3, dtype=np.float32)

    delta_pos = direction * min(STEP_SIZE, dist)

    # pd_ee_delta_pose on panda_stick: [dx, dy, dz, drx, dry, drz] (no gripper)
    action = np.zeros(6, dtype=np.float32)
    action[:3] = delta_pos
    return action


def main():
    env = gym.make(
        "Reach-WithObstacles-v1",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        render_mode="human",
    )

    obs, _ = env.reset(seed=0)
    print("Reset. Goal position:", np.asarray(obs["extra"]["goal_pos"]).reshape(3))

    for t in range(MAX_STEPS):
        extra      = obs["extra"]
        ee_pos     = np.asarray(extra["ee_pos"]).reshape(3)
        goal_pos   = np.asarray(extra["goal_pos"]).reshape(3)
        dist       = np.linalg.norm(goal_pos - ee_pos)

        print(f"step {t:3d} | ee {ee_pos} | goal {goal_pos} | dist {dist:.4f}")

        env.render()

        if dist < env.unwrapped.success_threshold:
            print(f"SUCCESS at step {t}!")
            break

        action = reach_policy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        time.sleep(1 / 30)

        if terminated or truncated:
            print(f"Episode ended at step {t}.")
            break

    env.close()


if __name__ == "__main__":
    main()
