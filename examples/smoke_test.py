import gymnasium as gym
import envs

env = gym.make(
    "ObjectRetrieveFromShelf-v1",
    obs_mode="state_dict",
    control_mode="pd_ee_delta_pose",
)

obs, info = env.reset()

for t in range(5):
    extra = obs["extra"]
    print(
        f"t={t}",
        "tcp_pose shape:", extra["tcp_pose"].shape,
        "obj_pose shape:", extra["obj_pose"].shape,
        "bay_center:", extra["bay_center"],
        "bay_size:", extra["bay_size"],
    )
    obs, rew, terminated, truncated, info = env.step(env.action_space.sample())

env.close()
