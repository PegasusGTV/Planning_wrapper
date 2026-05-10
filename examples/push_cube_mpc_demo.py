"""
Visualize the MPPI push controller for PushCube-WithObstacles-v1.

Runs a single episode: push the goal cube to the goal position.

Run:
    python examples/push_cube_mpc_demo.py
    python examples/push_cube_mpc_demo.py --seed 42
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, str(Path(__file__).parent.parent / "skills"))
sys.path.insert(0, str(Path(__file__).parent.parent / "skills" / "push_cube"))

import gymnasium as gym
import numpy as np

import envs  # noqa: F401
from push_cube_mpc import execute  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",      type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()

    env = gym.make(
        "PushCube-WithObstacles-v1",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        render_mode="human",
        sim_backend="physx_cpu",
    )

    obs, _ = env.reset(seed=args.seed)
    env.render()

    raw = env.unwrapped
    ee_pos        = np.asarray(obs["extra"]["ee_pos"], dtype=np.float32).reshape(-1)[:3]
    goal_cube_pos = np.asarray(obs["extra"]["goal_cube_pos"], dtype=np.float32).reshape(-1)
    goal_pos      = np.asarray(obs["extra"]["goal_pos"], dtype=np.float32).reshape(-1)
    block_idx = int(raw.goal_obstacle_idx[0].item())

    print(f"Start EE position : {np.round(ee_pos, 3)}")
    print(f"Goal cube position: {np.round(goal_cube_pos, 3)}")
    print(f"Goal position     : {np.round(goal_pos, 3)}")
    print(f"Block index       : {block_idx}")
    print("Running MPPI push controller...")

    success, _ = execute(env, obs, block_idx, goal_pos, max_steps=args.max_steps, render=True)

    print(f"\n{'SUCCESS' if success else 'FAIL'}")
    time.sleep(2.0)
    env.close()


if __name__ == "__main__":
    main()
