import gymnasium as gym
import mani_skill.envs
from planning_wrapper.wrappers import make_pusht_vis_gt, make_pusht_vis_no_ori

w = make_pusht_vis_no_ori()
obs, info = w.reset(seed=0)
print("obj_pose:", obs["extra"]["obj_pose"])
# should always have quat = [0,0,0,1] (or whatever constant you set)
print(obs["agent"].keys())  # should include qpos, qvel

w.close()