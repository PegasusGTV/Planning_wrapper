try:
    import gymnasium as gym
except ImportError:
    import gym

import mani_skill.envs  # noqa: F401

from planning_wrapper.wrappers.maniskill_planning import ManiSkillPlanningWrapper
from planning_wrapper.adapters import PushTTaskAdapter


def main():
    env = gym.make(
        "PushT-v1",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
    )
    w = ManiSkillPlanningWrapper(env, adapter=PushTTaskAdapter())

    obs, info = w.reset(seed=0)

    # Print controller/robot summary
    w.print_controller_summary()

    # Direct access
    qpos = w.get_qpos()
    qvel = w.get_qvel()
    c_idx = w.controlled_joint_indices()
    cqpos = w.controlled_qpos()
    cqvel = w.controlled_qvel()

    print("Full qpos shape:", qpos.shape)
    print("Full qvel shape:", qvel.shape)
    print("Controlled joint indices:", c_idx)
    print("Controlled qpos shape:", cqpos.shape)
    print("Controlled qvel shape:", cqvel.shape)

    env.close()


if __name__ == "__main__":
    main()
