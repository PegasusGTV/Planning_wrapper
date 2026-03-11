import sys
from pathlib import Path

# Add project root so 'utils' and 'envs' are importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tyro
import gymnasium as gym
import numpy as np
import sapien

from mani_skill.utils.wrappers import RecordEpisode

import envs
from utils.teleop_args import Args
from utils.teleop_utils import get_currently_pressed_keys, get_teleop_action


def get_action(env, action_type, pressed_keys=None):
    if action_type == "random":
        action = env.action_space.sample() if env.action_space is not None else None
    elif action_type == "teleop":
        if pressed_keys is None:
            action = np.zeros(6)
        else:
            action = get_teleop_action(pressed_keys)
    else:
        raise NotImplementedError
    return action


def main(args: Args):
    if args.render_mode == "none":
        args.render_mode = None
    np.set_printoptions(suppress=True, precision=3)
    verbose = not args.quiet
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    if args.seed is not None:
        np.random.seed(args.seed[0])
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in [
        "sensor_data",
        "rgb",
        "rgbd",
        "depth",
        "pointcloud",
    ]:
        print(
            "Disabling parallel single scene/GUI render as observation mode is a visual one. "
            "Change observation mode to state or state_dict to see a parallel env render"
        )
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        enable_shadow=True,
        parallel_in_single_scene=parallel_in_single_scene,
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))
        if len(env_kwargs["robot_uids"]) == 1:
            env_kwargs["robot_uids"] = env_kwargs["robot_uids"][0]

    env = gym.make(args.env_id, **env_kwargs)
    record_dir = args.record_dir
    if record_dir:
        record_dir = f"{record_dir}/{args.env_id}/teleop/"
        env = RecordEpisode(
            env,
            record_dir,
            trajectory_name="trajectory_{}".format(args.demo_id),
            save_video=False,
            info_on_video=False,
            save_trajectory=True,
            source_type="teleoperation",
        )

    if verbose:
        print("Observation space", env.observation_space)
        print("Action space", env.action_space)
        if env.unwrapped.agent is not None:
            print("Control mode", env.unwrapped.control_mode)
        print("Reward mode", env.unwrapped.reward_mode)

    obs, _ = env.reset(seed=args.seed, options=dict(reconfigure=True))

    if args.seed is not None and env.action_space is not None:
        env.action_space.seed(args.seed[0])
    if args.render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        env.render()
    else:
        viewer = None

    while True:
        pressed_keys = get_currently_pressed_keys(viewer)
        if "q" in pressed_keys:
            print("Exiting...")
            break
        action = get_action(env, args.action_type, pressed_keys)

        obs, reward, terminated, truncated, info = env.step(action)
        if verbose:
            print("reward", reward)
            print("terminated", terminated)
            print("truncated", truncated)
            print("info", info)
        if args.render_mode is not None:
            env.render()
        # if args.render_mode is None or args.render_mode != "human":
        if (terminated | truncated).any():
            break
    env.close()

    if record_dir:
        print(f"Saving video to {record_dir}")


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
