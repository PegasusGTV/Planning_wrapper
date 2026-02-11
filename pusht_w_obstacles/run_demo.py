"""Run PushT-WithObstacles with planning wrapper (no policy, just env check)."""
import time

import gymnasium as gym

import pusht_w_obstacles  # noqa: F401 - register env
from planning_wrapper.adapters import PushTTaskAdapter
from planning_wrapper.wrappers.maniskill_planning import ManiSkillPlanningWrapper


def main():
    env = gym.make(
        "PushT-WithObstacles-v1",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        render_mode="human",
    )
    adapter = PushTTaskAdapter()
    wrapper = ManiSkillPlanningWrapper(env, adapter=adapter)

    obs, info = wrapper.reset(seed=0)
    wrapper.print_controller_summary()

    for step in range(20):
        try:
            wrapper.render()
        except Exception:
            pass
        action = wrapper.action_space.sample()
        obs, reward, terminated, truncated, info = wrapper.step(action)
        if terminated or truncated:
            break

    # Keep window open briefly so you can see the final state
    for _ in range(60):
        try:
            wrapper.render()
        except Exception:
            pass
        time.sleep(0.1)
    wrapper.close()
    print("Demo finished.")


if __name__ == "__main__":
    main()
