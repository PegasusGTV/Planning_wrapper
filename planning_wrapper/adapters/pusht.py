from __future__ import annotations
from typing import Any, Dict
from .base import BaseTaskAdapter

class PushTTaskAdapter(BaseTaskAdapter):
    def get_task_state(self, env: Any) -> Dict[str, Any]:
        root = getattr(env, "unwrapped", None)
        if not hasattr(root, "goal_tee") or not hasattr(root, "ee_goal_pos"):
            raise RuntimeError(
                "PushTTaskAdapter expects env.unwrapped to have "
                "'goal_tee' and 'ee_goal_pos' attributes."
            )

        goal_tee_pose = root.goal_tee.pose.raw_pose
        ee_goal_pose = root.ee_goal_pos.pose.raw_pose

        task_state: Dict[str, Any] = {
            "goal_tee_pose": goal_tee_pose,
            "ee_goal_pose": ee_goal_pose,
        }

        if hasattr(root, "goal_z_rot"):
            task_state["goal_z_rot"] = root.goal_z_rot
        if hasattr(root, "goal_offset"):
            task_state["goal_offset"] = root.goal_offset
        
        return task_state
    
    def set_task_state(self, env: Any, task_state: Dict[str, Any]) -> None:

        root = getattr(env, "unwrapped", None)

        if not hasattr(root, "goal_tee") or not hasattr(root, "ee_goal_pos"):
            raise RuntimeError(
                "PushTTaskAdapter expects env.unwrapped to have "
                "'goal_tee' and 'ee_goal_pos' attributes."
            )
        
        goal_tee_pose = task_state.get("goal_tee_pose", None)
        ee_goal_pose = task_state.get("ee_goal_pose", None)

        if goal_tee_pose is not None:
            root.goal_tee.pose.raw_pose = goal_tee_pose
        if ee_goal_pose is not None:
            root.ee_goal_pos.pose.raw_pose = ee_goal_pose

        if "goal_z_rot" in task_state and hasattr(root, "goal_z_rot"):
            root.goal_z_rot = task_state["goal_z_rot"]
        if "goal_offset" in task_state and hasattr(root, "goal_offset"):
            root.goal_offset = task_state["goal_offset"]