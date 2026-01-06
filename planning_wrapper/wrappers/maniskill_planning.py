from __future__ import annotations
from typing import Any, Dict, Optional
import gymnasium as gym
import mani_skill.envs
import numpy as np
from ..adapters import BaseTaskAdapter

class ManiSkillPlanningWrapper:
    def __init__(self, env: Any, adapter: Optional[BaseTaskAdapter] = None, hide_obj_orientation: bool = False):
        self.env = env
        self.adapter = adapter
        self.hide_obj_orientation = hide_obj_orientation
        self.root = self.env.unwrapped
        self.agent = getattr(self.root, "agent", None)

        self.controller = getattr(self.agent, "controller", None) if self.agent else None
        self.robot = getattr(self.agent, "robot", None) if self.agent else None
    
    def clone_state(self) -> Dict[str, Any]:
        state = self.root.get_state_dict()

        sim_state: Dict[str, Any] = {}
        if "actors" in state:
            sim_state["actors"] = state["actors"]
        if "articulations" in state:
            sim_state["articulations"] = state["articulations"]
        
        controller_state = state.get("controller", None)
        
        if self.adapter is not None:
            task_state = self.adapter.get_task_state(self.root)
        else:
            task_state = None
        snapshot = {
            "sim_state": sim_state,
            "controller_state": controller_state,
            "task_state": task_state,
        }
        return snapshot

    def restore_state(self, snapshot: Dict[str, Any]) -> None:
        
        sim_state = snapshot.get("sim_state", None)
        controller_state = snapshot.get("controller_state", None)
        task_state = snapshot.get("task_state", None)

        if sim_state is not None:
            self.root.set_state_dict(sim_state)
        if controller_state is not None and self.agent is not None:
            if hasattr(self.agent, "set_controller_state"):
                self.agent.set_controller_state(controller_state)
            else:
                pass
        
        if task_state is not None and self.adapter is not None:
            self.adapter.set_task_state(self.root, task_state)
        else:
            pass

    def get_planning_obs(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        if not isinstance(obs, dict):
            raise TypeError(f"get_planning_obs expects dict obs, got {type(obs)}")
        
        agent = obs.get("agent")
        extra = obs.get("extra")

        if agent is None or extra is None:
            raise KeyError("Expected obs to contain 'agent' and 'extra' keys.")
        if "qpos" not in agent or "qvel" not in agent:
            raise KeyError("obs['agent'] must contain 'qpos' and 'qvel' for planning_obs.")

        qpos = np.asarray(agent["qpos"], dtype=np.float32).copy()
        qvel = np.asarray(agent["qvel"], dtype=np.float32).copy()

        # Required keys for planning observation
        required_keys = ["tcp_pose", "goal_pos", "obj_pose"]
        missing_extra = [k for k in required_keys if k not in extra]
        if missing_extra:
            env_name = getattr(self.root, "spec", None)
            env_id = getattr(env_name, "id", None) if env_name else None
            env_hint = f" for {env_id}" if env_id else ""
            raise KeyError(
                f"obs['extra'] missing required keys for planning_obs: {missing_extra}. "
                f"Make sure obs_mode='state_dict'{env_hint}."
            )
        
        tcp_pose = np.asarray(extra["tcp_pose"], dtype=np.float32).copy()
        goal_pos = np.asarray(extra["goal_pos"], dtype=np.float32).copy()
        obj_pose = np.asarray(extra["obj_pose"], dtype=np.float32).copy()
        
        planning_obs = {
            "qpos": qpos,
            "qvel": qvel,
            "tcp_pose": tcp_pose,
            "goal_pos": goal_pos,
            "obj_pose": obj_pose,
        }

        return planning_obs
    
    def flatten_planning_obs(self, planning_obs: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(planning_obs["qpos"], dtype=np.float32).ravel(),
                np.asarray(planning_obs["qvel"], dtype=np.float32).ravel(),
                np.asarray(planning_obs["obj_pose"], dtype=np.float32).ravel(),
                np.asarray(planning_obs["goal_pos"], dtype=np.float32).ravel(),
                np.asarray(planning_obs["tcp_pose"], dtype=np.float32).ravel(),
            ],
            axis=0,
        )
    
    def get_qpos(self) -> np.ndarray:
        if self.robot is None or not hasattr(self.robot, "get_qpos"):
            raise RuntimeError("Robot articulation missing or does not implement get_qpos().")
        qpos = self.robot.get_qpos()
        return np.asarray(qpos, dtype=np.float32).copy()
    
    def get_qvel(self) -> np.ndarray:
        if self.robot is None or not hasattr(self.robot, "get_qvel"):
            raise RuntimeError("Robot articulation missing or does not implement get_qvel().")
        qvel = self.robot.get_qvel()
        return np.asarray(qvel, dtype=np.float32).copy()
    
    def controlled_joint_indices(self) -> np.ndarray:
        if self.controller is None:
            raise RuntimeError("Controller not found on agent.")
        
        idx = getattr(self.controller, "active_joint_indices", None)
        if idx is None:
            raise RuntimeError(
                "Controller does not expose 'active_joint_indices'. "
                "Check the ManiSkill controller API."
            )
        return np.asarray(idx, dtype=np.int64).copy()
    
    def controlled_qpos(self) -> np.ndarray:
        if self.controller is None:
            raise RuntimeError("Controller not found on agent.")
        
        if hasattr(self.controller, "qpos"):
            qpos = np.array(self.controller.qpos, dtype=np.float32)
            return qpos.copy()
        
        full_qpos = self.get_qpos()
        idx = self.controlled_joint_indices()

        return full_qpos[idx]
    
    def controlled_qvel(self) -> np.ndarray:
        if self.controller is None:
            raise RuntimeError("Controller not found on agent.")
        
        if hasattr(self.controller, "qvel"):
            qvel = np.array(self.controller.qvel, dtype=np.float32)
            return qvel.copy()
        
        full_qvel = self.get_qvel()
        idx = self.controlled_joint_indices()
        
        return full_qvel[idx]
    def print_controller_summary(self) -> None:
        print("\n=== Controller / Robot Summary ===")

        if self.robot is None:
            print("Robot: <missing>")
        else:
            # DOF and joint names from articulation
            qpos = self.get_qpos()
            dof = qpos.shape[0]
            print(f"Robot DOF: {dof}")

            joint_names = []
            if hasattr(self.robot, "get_joints"):
                try:
                    joints = self.robot.get_joints()
                    for j in joints:
                        # SAPIEN joints usually have 'get_name()'
                        name = getattr(j, "get_name", None)
                        if callable(name):
                            joint_names.append(name())
                        elif hasattr(j, "name"):
                            joint_names.append(j.name)
                except Exception:
                    pass

            if joint_names:
                print("Joints:")
                for i, n in enumerate(joint_names):
                    print(f"  [{i}] {n}")
            else:
                print("Joints: <could not retrieve joint names>")

        if self.controller is None:
            print("\nController: <missing>")
            return

        # Controlled joints
        try:
            idx = self.controlled_joint_indices()
            print("\nControlled joint indices:", idx.tolist())
        except Exception as e:
            print("\nControlled joint indices: <error>", e)

        # Controller repr usually contains joint + DOF info
        print("\nController repr:")
        print(self.controller)
        print("=== End summary ===\n")

    def _filter_obs(self, obs):
        if not self.hide_obj_orientation:
            return obs
        if not isinstance(obs, dict) or "extra" not in obs:
            return obs
        extra = obs.get("extra", None)
        if not isinstance(extra, dict) or "obj_pose" not in extra:
            return obs

        obj_pose = np.asarray(extra["obj_pose"], dtype=np.float32).copy()
        if obj_pose.shape[-1] >= 7:
            obj_pose[..., 3:7] = np.array([0, 0, 0, 1], dtype=np.float32)

        # write back (copy dicts so you don't mutate shared references)
        obs2 = dict(obs)
        extra2 = dict(extra)
        extra2["obj_pose"] = obj_pose
        obs2["extra"] = extra2
        return obs2
        
    def reset(self, *args, **kwargs):
        # Use the wrapped env to respect outer wrappers (viewer/step hooks, etc.)
        obs, info = self.env.reset(*args, **kwargs)
        return self._filter_obs(obs), info
    
    def step(self, action: np.ndarray):
        # Step through the wrapped env (not the unwrapped root) for wrapper behavior
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._filter_obs(obs), reward, terminated, truncated, info
    
    def close(self):
        # Close via the wrapped env to respect any outer wrapper hooks
        return self.env.close()

    def get_controller_bounds(self) -> tuple:
        """
        Get the original (unnormalized) action bounds for the controller.
        Returns (low, high) tuple for position control bounds.
        For pd_ee_delta_pose, this is typically (-0.1, 0.1) for position.
        """
        if self.controller is None:
            raise RuntimeError("Controller not found on agent.")
        
        # Try to get bounds from the arm controller if it's a CombinedController
        if hasattr(self.controller, "controllers") and "arm" in self.controller.controllers:
            arm_controller = self.controller.controllers["arm"]
            if hasattr(arm_controller, "action_space_low") and hasattr(arm_controller, "action_space_high"):
                # Get position bounds (first 3 dims)
                low = arm_controller.action_space_low[0:3].cpu().numpy()
                high = arm_controller.action_space_high[0:3].cpu().numpy()
                # For pd_ee_delta_pose, bounds should be the same for all 3 dims
                return (float(low[0]), float(high[0]))
        
        # Default bounds for pd_ee_delta_pose (from panda_stick config)
        return (-0.1, 0.1)
    
    def __getattr__(self, name):
        return getattr(self.root, name)
    
def main():
    env = gym.make(
        "PushT-v1",
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
        )
    wrapper = ManiSkillPlanningWrapper(env)
    snapshot = wrapper.clone_state()
    wrapper.restore_state(snapshot)
    # from planning_wrapper.adapters.pusht import PushTTaskAdapter

    # env = gym.make("PushT-v1", obs_mode="state_dict", control_mode="pd_ee_delta_pose")
    # adapter = PushTTaskAdapter()
    # w = MSPlanningWrapper(env, adapter=adapter)

    # snap = w.clone_state()
    # # snap now contains "task_state"
    # print(snap["task_state"].keys())  # e.g. goal_tee_pose, ee_goal_pose

    # w.restore_state(snap)
    # print(wrapper.root.observation_space)
    # print(type(wrapper.env))
    # print(type(wrapper.root))
    # print(type(wrapper.agent))
    # print(type(wrapper.controller))
    # print(type(wrapper.robot))
    # print("================================================")
    # print(wrapper.root)
    # print(wrapper.agent)
    # print(wrapper.controller)
    # print(wrapper.robot)
    # print("================================================")   
    # print("root:", type(wrapper.root))
    # print("agent:", type(wrapper.agent))
    # print("controller:", type(wrapper.controller))
    # print("robot:", type(wrapper.robot))
    # print("================================================")
    # obs, info = wrapper.reset(seed=0)
    # print("reset ok")
    # print("================================================")
    # action = wrapper.action_space.sample()
    # print("action:", action)
    # print("================================================")
    # obs, reward, terminated, truncated, info = wrapper.step(action)
    # print("step ok")
    # print("obs:", obs)
    # print("reward:", reward)
    # print("terminated:", terminated)
    # print("truncated:", truncated)
    # print("info:", info)
    # print("================================================")
    # wrapper.close()
    # print("close ok")
    # print("================================================")
if __name__ == "__main__":
    main()
    