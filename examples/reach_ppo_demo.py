"""
Visualize the trained PPO policy for Reach-WithObstacles-v1.

Loads the final checkpoint from reach_ppo.py and runs a single episode
with live rendering.

Run:
    python examples/reach_ppo_demo.py
    python examples/reach_ppo_demo.py --checkpoint runs/Reach-WithObstacles-v1__1__<timestamp>/final_ckpt.pt
"""

import argparse
import sys
import os
import time
import types
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym
import numpy as np
import torch

from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper

import envs  # noqa: F401 — registers Reach-WithObstacles-v1

# Import Agent from reach_ppo without running its __main__ block
sys.path.insert(0, str(Path(__file__).parent.parent / "skills" / "reach"))
from reach_ppo import Agent  # noqa: E402


DEFAULT_CHECKPOINT = "runs/Reach-WithObstacles-v1__1__<timestamp>/final_ckpt.pt"
MAX_STEPS = 200


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic (mean) actions (default: True)")
    parser.add_argument("--render-fps", type=float, default=5.0,
                        help="Target rendering speed in frames per second")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(
        "Reach-WithObstacles-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode="human",
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)

    # Agent.__init__ expects an object with single_observation_space / single_action_space
    env_ns = types.SimpleNamespace(
        single_observation_space=env.observation_space,
        single_action_space=env.action_space,
    )
    agent = Agent(env_ns).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    agent.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    action_low  = torch.from_numpy(env.action_space.low).to(device)
    action_high = torch.from_numpy(env.action_space.high).to(device)

    obs, _ = env.reset(seed=args.seed)

    for t in range(MAX_STEPS):
        env.render()

        obs_t = torch.from_numpy(np.asarray(obs)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action = agent.get_action(obs_t, deterministic=args.deterministic)
            action = torch.clamp(action, action_low, action_high)

        obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
        time.sleep(1.0 / args.render_fps)
        success = info.get('success', False)
        if hasattr(success, 'item'):
            success = success.item()
        print(f"step {t:3d} | reward {float(reward):.4f} | success {success}")

        if terminated or truncated:
            print(f"Episode ended at step {t}.")
            break

    env.close()


if __name__ == "__main__":
    main()
