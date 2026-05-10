"""
Visualize the MPPI pick controller for PickSkillEnv.

Runs a single episode: approach cube from above, grasp, lift.

Run:
    python examples/pick_cube_mpc_demo.py
    python examples/pick_cube_mpc_demo.py --seed 42
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, str(Path(__file__).parent.parent / "skills"))
sys.path.insert(0, str(Path(__file__).parent.parent / "skills" / "pick"))

import gymnasium as gym
import numpy as np

import envs  # noqa: F401
from pick_cube_mpc import execute  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",       type=int,   default=5)
    parser.add_argument("--max-steps",  type=int,   default=200)
    args = parser.parse_args()

    env = gym.make(
        "PickSkillEnv",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        render_mode="human",
        sim_backend="physx_cpu",
    )

    obs, _ = env.reset(seed=args.seed)
    env.render()

    raw = env.unwrapped
    ee_pos   = np.asarray(obs["extra"]["ee_pos"], dtype=np.float32).reshape(-1)[:3]
    cube_pos = np.asarray(obs["extra"]["pick_cube_pos"], dtype=np.float32).reshape(-1)
    block_idx = int(raw.pick_obstacle_idx[0].item())

    print(f"Start EE position : {np.round(ee_pos, 3)}")
    print(f"Pick cube position: {np.round(cube_pos, 3)}")
    print(f"Block index       : {block_idx}")
    print("Running MPPI pick controller...")

    success, _ = execute(env, obs, block_idx, max_steps=args.max_steps, render=True)

    print(f"\n{'SUCCESS' if success else 'FAIL'}")
    time.sleep(2.0)
    env.close()


if __name__ == "__main__":
    main()
