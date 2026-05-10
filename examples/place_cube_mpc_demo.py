"""
Visualize the MPPI place controller for PlaceSkillEnv.

The cube starts already grasped. The controller carries it to the goal,
releases, and retreats.

Run:
    python examples/place_cube_mpc_demo.py
    python examples/place_cube_mpc_demo.py --seed 42
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, str(Path(__file__).parent.parent / "skills"))
sys.path.insert(0, str(Path(__file__).parent.parent / "skills" / "place"))

import gymnasium as gym
import numpy as np

import envs  # noqa: F401
from place_cube_mpc import execute  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",       type=int,   default=5)
    parser.add_argument("--max-steps",  type=int,   default=200)
    args = parser.parse_args()

    env = gym.make(
        "PlaceSkillEnv",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        render_mode="human",
        sim_backend="physx_cpu",
    )

    obs, _ = env.reset(seed=args.seed)
    env.render()

    raw = env.unwrapped
    ee_pos   = np.asarray(obs["extra"]["tcp_pose"], dtype=np.float32).reshape(-1)[:3]
    goal_pos = np.asarray(obs["extra"]["goal_pos"], dtype=np.float32).reshape(-1)

    print(f"Start EE position: {np.round(ee_pos, 3)}")
    print(f"Goal position    : {np.round(goal_pos, 3)}")
    print("Running MPPI place controller...")

    # PlaceSkillEnv uses a single cube, not obstacles array
    if not hasattr(raw, "obstacles"):
        raw.obstacles = [raw.cube]

    success, _ = execute(env, obs, 0, goal_pos, max_steps=args.max_steps, render=True)

    print(f"\n{'SUCCESS' if success else 'FAIL'}")
    time.sleep(2.0)
    env.close()


if __name__ == "__main__":
    main()
