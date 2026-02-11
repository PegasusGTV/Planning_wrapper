"""Run PushT-WithObstacles with planning wrapper (no policy, just env check)."""
import gymnasium as gym

import pusht_w_obstacles  # noqa: F401 - register env
from planning_wrapper.adapters import PushTTaskAdapter
from planning_wrapper.wrappers.maniskill_planning import ManiSkillPlanningWrapper


def main():
    env = gym.make(
        "PushT-WithObstacles-v1",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
    )
    adapter = PushTTaskAdapter()
    wrapper = ManiSkillPlanningWrapper(env, adapter=adapter)

    obs, info = wrapper.reset(seed=0)
    wrapper.print_controller_summary()

    for step in range(20):
        action = wrapper.action_space.sample()
        obs, reward, terminated, truncated, info = wrapper.step(action)
        if terminated or truncated:
            break

    wrapper.close()
    print("Demo finished.")


if __name__ == "__main__":
    main()
