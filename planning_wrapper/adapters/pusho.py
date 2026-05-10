from __future__ import annotations
from typing import Any, Dict
from .base import BaseTaskAdapter


class PushOTaskAdapter(BaseTaskAdapter):
    """Snapshot task-specific state not fully covered by generic sim state (for clone/restore)."""

    def get_task_state(self, env: Any) -> Dict[str, Any]:
        root = getattr(env, "unwrapped", env)
        task_state: Dict[str, Any] = {}

        # Legacy layout (older envs)
        if hasattr(root, "goal_disk") and hasattr(root, "ee_goal_pos"):
            task_state["goal_disk_pose"] = root.goal_disk.pose.raw_pose
            task_state["ee_goal_pose"] = root.ee_goal_pos.pose.raw_pose
            if hasattr(root, "goal_z_rot"):
                task_state["goal_z_rot"] = root.goal_z_rot
            if hasattr(root, "goal_offset"):
                task_state["goal_offset"] = root.goal_offset
            return task_state

        # Current PushO / PushO-WithObstacles / wall variant
        if hasattr(root, "goal_pos"):
            task_state["goal_pos"] = root.goal_pos.clone()
        if hasattr(root, "goal_site"):
            task_state["goal_site_raw"] = root.goal_site.pose.raw_pose.clone()
        if hasattr(root, "disk"):
            task_state["disk_raw"] = root.disk.pose.raw_pose.clone()

        return task_state

    def set_task_state(self, env: Any, task_state: Dict[str, Any]) -> None:
        root = getattr(env, "unwrapped", env)

        if "goal_disk_pose" in task_state or "ee_goal_pose" in task_state:
            if not hasattr(root, "goal_disk") or not hasattr(root, "ee_goal_pos"):
                return
            if task_state.get("goal_disk_pose") is not None:
                root.goal_disk.pose.raw_pose = task_state["goal_disk_pose"]
            if task_state.get("ee_goal_pose") is not None:
                root.ee_goal_pos.pose.raw_pose = task_state["ee_goal_pose"]
            if "goal_z_rot" in task_state and hasattr(root, "goal_z_rot"):
                root.goal_z_rot = task_state["goal_z_rot"]
            if "goal_offset" in task_state and hasattr(root, "goal_offset"):
                root.goal_offset = task_state["goal_offset"]
            return

        if "goal_pos" in task_state and hasattr(root, "goal_pos"):
            root.goal_pos.copy_(task_state["goal_pos"])
        if "goal_site_raw" in task_state and hasattr(root, "goal_site"):
            root.goal_site.pose.raw_pose = task_state["goal_site_raw"]
        if "disk_raw" in task_state and hasattr(root, "disk"):
            root.disk.pose.raw_pose = task_state["disk_raw"]
