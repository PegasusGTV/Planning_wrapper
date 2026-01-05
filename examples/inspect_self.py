"""
Quick inspection script for ObjectRetrieveFromShelf-v1 + planning wrapper.

Usage:
    python examples/inspect_shelf.py

What it does:
  - Creates the shelf retrieval env
  - Wraps it with ManiSkillPlanningWrapper (no custom adapter)
  - Prints:
      * action space
      * planning_obs keys
      * joint info (DOF, names, controlled indices, qpos/qvel shapes)
      * initial obj_pose and bay bounds from obs["extra"]
"""

import numpy as np

import gymnasium as gym  # or `import gym` if your ManiSkill build uses old gym

# Make sure env registration side-effects run
import envs  # noqa: F401

from planning_wrapper.wrappers.maniskill_planning import ManiSkillPlanningWrapper
from planning_wrapper.adapters.shelf_retrieve import ShelfRetrieveTaskAdapter


def main() -> None:
    # 1) Create raw ManiSkill env
    env = gym.make(
        "ObjectRetrieveFromShelf-v1",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",  # same as PushT wrapper uses
        spawn_mode="random_small",        # or "fixed" if you prefer
    )

    print("=== Raw env ===")
    print("Env:", env)

    # 2) Wrap with planning wrapper using shelf retrieval adapter
    adapter = ShelfRetrieveTaskAdapter()
    wrapped = ManiSkillPlanningWrapper(env, adapter=adapter)

    print("\n=== Wrapper info ===")
    print("Wrapper:", wrapped)

    # 3) Reset and grab initial observation
    obs, info = wrapped.reset(seed=0)
    assert isinstance(obs, dict), "Expected state_dict obs to be a dict"

    extra = obs.get("extra", {})
    if not isinstance(extra, dict):
        raise RuntimeError("Expected obs['extra'] to be a dict in state_dict mode")

    # 4) Action space
    print("\n=== Action space ===")
    print(wrapped.action_space)

    # 5) Planning observation from wrapper
    planning_obs = wrapped.get_planning_obs(obs)

    print("\n=== Planning observation ===")
    print("Keys:", list(planning_obs.keys()))
    for k, v in planning_obs.items():
        try:
            arr = np.asarray(v)
            print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
        except Exception:
            print(f"  {k}: (non-array value: {type(v)})")

    # 6) Joint / controller info
    print("\n=== Joint / controller info ===")
    try:
        # If your wrapper has this helper (as in your README)
        wrapped.print_controller_summary()
    except AttributeError:
        print("wrapped.print_controller_summary() not found; printing basic joint info instead.")
        try:
            qpos = wrapped.get_qpos()
            qvel = wrapped.get_qvel()
            print("qpos shape:", np.asarray(qpos).shape)
            print("qvel shape:", np.asarray(qvel).shape)
        except Exception as e:
            print("Failed to query qpos/qvel:", e)

    # Attempt to show controlled joints if your wrapper exposes them
    for name in ["controlled_joint_indices", "controlled_qpos", "controlled_qvel"]:
        if hasattr(wrapped, name):
            try:
                value = getattr(wrapped, name)()
                arr = np.asarray(value)
                print(f"{name}(): shape={arr.shape}, value={arr}")
            except Exception as e:
                print(f"{name}() call failed:", e)

    # 7) Initial object pose + bay bounds from obs["extra"]
    print("\n=== Shelf / object info from obs['extra'] ===")

    obj_pose = extra.get("obj_pose", None)
    bay_center = extra.get("bay_center", None)
    bay_size = extra.get("bay_size", None)

    print("obj_pose:", obj_pose)
    print("bay_center:", bay_center)
    print("bay_size:", bay_size)

    # If you also stored bay_min/bay_max in extra, print them; otherwise infer
    bay_min = extra.get("bay_min", None)
    bay_max = extra.get("bay_max", None)

    if bay_min is not None and bay_max is not None:
        print("bay_min:", bay_min)
        print("bay_max:", bay_max)
    elif bay_center is not None and bay_size is not None:
        bay_center = np.asarray(bay_center, dtype=np.float32)
        bay_size = np.asarray(bay_size, dtype=np.float32)
        inferred_min = bay_center - 0.5 * bay_size
        inferred_max = bay_center + 0.5 * bay_size
        print("bay_min (inferred):", inferred_min)
        print("bay_max (inferred):", inferred_max)

    # 8) Take a couple of random steps and ensure everything stays stable
    print("\n=== Stepping a few times to check stability ===")
    for t in range(3):
        action = wrapped.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped.step(action)

        extra = obs.get("extra", {})
        planning_obs = wrapped.get_planning_obs(obs)

        print(f"\nStep {t}:")
        print("  reward:", reward)
        print("  terminated:", terminated, "truncated:", truncated)
        print("  planning_obs keys:", list(planning_obs.keys()))

    wrapped.close()


if __name__ == "__main__":
    main()
