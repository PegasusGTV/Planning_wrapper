import sys
from pathlib import Path

# Add project root so 'utils' is importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np

import tyro
import json
import h5py

import envs
import mani_skill.trajectory.utils as trajectory_utils
from mani_skill.utils.wrappers import RecordEpisode
from utils.teleop_args import Args


def main(args: Args):
    if args.render_mode == "none":
        args.render_mode = None
    np.set_printoptions(suppress=True, precision=3)
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
        "point_cloud",
    ]:
        print(
            "Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render"
        )
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode="rgb_array",
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        enable_shadow=True,
        parallel_in_single_scene=parallel_in_single_scene,
    )
    env = gym.make(args.env_id, **env_kwargs)
    record_dir = args.record_dir

    h5_file_path = (
        f"{record_dir}/{args.env_id}/teleop_filtered/trajectory_{args.demo_id}.h5"
    )
    json_file_path = (
        f"{record_dir}/{args.env_id}/teleop_filtered/trajectory_{args.demo_id}.json"
    )
    trajectory_data = h5py.File(h5_file_path)
    with open(json_file_path, "r") as f:
        json_data = json.load(f)

    record_dir = f"{record_dir}/{args.env_id}/teleop_filtered_videos/"
    env = RecordEpisode(
        env,
        record_dir,
        trajectory_name="trajectory_{}".format(args.demo_id),
        save_video=True,
        info_on_video=False,
        save_trajectory=False,
        source_type="teleoperation",
    )

    for episode in json_data["episodes"]:
        traj_id = f"traj_{episode['episode_id']}"
        data = trajectory_data[traj_id]
        env.reset(**episode["reset_kwargs"])
        env_states_list = trajectory_utils.dict_to_list_of_dicts(data["env_states"])

        env.base_env.set_state_dict(env_states_list[0])
        raw_step = 0
        saved_step = 0
        acc_zero_step = 0
        print(len(data["actions"]))

        for action in np.array(data["actions"]):
            raw_step += 1
            if True:  # (action != np.zeros_like(action)).any():
                obs, _, _, _, info = env.step(action)
                saved_step += 1
                print(f"Raw step: {raw_step}, Saved step: {saved_step}")
                # if raw_step >= 4148:
                #     break
                # if info['success']:
                #     break

    trajectory_data.close()
    env.close()
    del env


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
