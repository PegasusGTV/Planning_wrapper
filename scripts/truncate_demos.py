import sys
from pathlib import Path

# Add project root so 'utils' is importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tyro
import json
import os
import h5py
import mani_skill.trajectory.utils as trajectory_utils

from utils.teleop_args import Args


def main(args: Args):
    if args.render_mode == "none":
        args.render_mode = None
    np.set_printoptions(suppress=True, precision=3)

    record_dir = args.record_dir

    h5_file_path = f"{record_dir}/{args.env_id}/teleop/trajectory_{args.demo_id}.h5"
    json_file_path = f"{record_dir}/{args.env_id}/teleop/trajectory_{args.demo_id}.json"
    trajectory_data = h5py.File(h5_file_path)
    with open(json_file_path, "r") as f:
        json_data = json.load(f)

    new_record_dir = f"{record_dir}/{args.env_id}/teleop_filtered"
    if not os.path.exists(new_record_dir):
        os.makedirs(new_record_dir)

    filtered_h5 = h5py.File(f"{new_record_dir}/trajectory_{args.demo_id}.h5", "w")
    filtered_json = json_data.copy()

    for episode in json_data["episodes"]:
        traj_id = f"traj_{episode['episode_id']}"
        data = trajectory_data[traj_id]
        actions = np.array(data["actions"])
        env_states = trajectory_utils.dict_to_list_of_dicts(data["env_states"])

        nonzero_indices = [i for i, a in enumerate(actions) if (a != 0).any()]
        # nonzero_indices = nonzero_indices[:733]  # truncate the last few unsuccessful states
        filtered_actions = actions[nonzero_indices]
        filtered_env_states = [env_states[0]] + [env_states[i] for i in nonzero_indices]

        print(len(filtered_env_states), len(filtered_actions))
        if len(filtered_actions) == 0:
            print(f"Skipping empty trajectory {traj_id}")
            continue

        # Write to new h5
        grp = filtered_h5.create_group(traj_id)
        grp.create_dataset("actions", data=filtered_actions)
        grp_env_states = trajectory_utils.list_of_dicts_to_dict(filtered_env_states)

        grp.create_dataset(
            f"env_states/actors/table-workspace",
            data=grp_env_states["actors"]["table-workspace"],
        )
        # grp.create_dataset(f"env_states/actors/goal_site", data=relabel_goal)
        # grp.create_dataset(f"env_states/actors/target", data=target)

        # Copy everything over
        grp.create_dataset(
            f"env_states/actors/Tee", data=grp_env_states["actors"]["Tee"]
        )
        grp.create_dataset(
            f"env_states/actors/goal_Tee", data=grp_env_states["actors"]["goal_Tee"]
        )
        grp.create_dataset(
            f"env_states/actors/goal_ee", data=grp_env_states["actors"]["goal_ee"]
        )
        grp.create_dataset(
            f"env_states/actors/extra_cube", data=grp_env_states["actors"]["extra_cube"]
        )
        grp.create_dataset(
            f"env_states/articulations/panda_stick",
            data=grp_env_states["articulations"]["panda_stick"],
        )
        grp.create_dataset(
            f"env_states/controller/arm/target_pose",
            data=grp_env_states["controller"]["arm"]["target_pose"],
        )

        # Copy episode metadata
        filtered_json["episodes"][episode["episode_id"]]["elapsed_steps"] = len(
            filtered_actions
        )
        filtered_json["episodes"][episode["episode_id"]]["success"] = True  # TODO

    trajectory_data.close()
    filtered_h5.close()

    with open(f"{new_record_dir}/trajectory_{args.demo_id}.json", "w") as f:
        json.dump(filtered_json, f, indent=2)


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
