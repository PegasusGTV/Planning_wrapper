# # examples/inspect_pusht.py
# from planning_wrapper import ManiSkillPlanningWrapper

# def main():
#     env = ManiSkillPlanningWrapper(env_id="PushT-v1")
#     obs = env.reset()
#     print("Initial obs:", obs)

# if __name__ == "__main__":
#     main()
from codecs import namereplace_errors
import gymnasium as gym
import mani_skill.envs
import numpy as np

def main():
    env = gym.make(
        "PushT-v1",
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
        )
    
    obs, info = env.reset(seed=0)
    print("\n=== ACTION SPACE ===")
    print(env.action_space)

    print("\n=== OBS TYPE + KEYS ===")
    print("type(obs):", type(obs))

    if not isinstance(obs, dict):
        print("WARNING: expected dict obs, got:", type(obs))
        env.close()
        return

    print("obs keys: ", list(obs.keys()))

    agent = obs.get("agent", {})
    extra = obs.get("extra", {})

    print("\nobs['agent'].keys():", list(agent.keys()))
    print("obs['extra'].keys():", list(extra.keys()))

    print("\n=== AGENT STATE SHAPES ===")
    qpos = agent.get("qpos", None)
    qvel = agent.get("qvel", None)

    if qpos is not None:
        print("agent['qpos'] shape:", np.asarray(qpos).shape)
    else:
        print("agent['qpos'] missing")

    if qvel is not None:
        print("agent['qvel'] shape:", np.asarray(qvel).shape)
    else:
        print("agent['qvel'] missing")

    print("\n=== EXTRA FIELDS ===")
    print("has extra['tcp_pose']:", "tcp_pose" in extra)
    if "tcp_pose" in extra:
        print("extra['tcp_pose'] shape:", np.asarray(extra["tcp_pose"]).shape)

    print("has extra['goal_pos']:", "goal_pos" in extra)
    if "goal_pos" in extra:
        print("extra['goal_pos'] shape:", np.asarray(extra["goal_pos"]).shape)

    print("has extra['obj_pose']:", "obj_pose" in extra)
    if "obj_pose" in extra:
        print("extra['obj_pose'] shape:", np.asarray(extra["obj_pose"]).shape)

    env.close()


if __name__ == "__main__":
    main()
