from __future__ import annotations
from typing import Any, Dict
import numpy as np
from .base import BaseTaskAdapter


class ShelfRetrieveTaskAdapter(BaseTaskAdapter):
    """
    Task adapter for ObjectRetrieveFromShelf-v1.
    
    For this task, the main task state is the object's pose, which is already
    captured in the sim_state. However, we can optionally save/restore the
    spawn_mode or other task-specific parameters if needed.
    """
    
    def get_task_state(self, env: Any) -> Dict[str, Any]:
        """
        Extract task-specific state that needs to be saved/restored.
        
        For shelf retrieval, the object pose is already in sim_state,
        but we can save spawn_mode and other task parameters.
        """
        root = getattr(env, "unwrapped", env)
        
        task_state: Dict[str, Any] = {}
        
        # Save spawn mode if it exists
        if hasattr(root, "spawn_mode"):
            task_state["spawn_mode"] = root.spawn_mode
        
        # Optionally save default qpos for robot reset consistency
        if hasattr(root, "_default_qpos") and root._default_qpos is not None:
            task_state["default_qpos"] = np.asarray(root._default_qpos, dtype=np.float32).copy()
        
        return task_state
    
    def set_task_state(self, env: Any, task_state: Dict[str, Any]) -> None:
        """
        Restore task-specific state.
        
        For shelf retrieval, we mainly need to ensure the spawn_mode is consistent.
        The object pose will be restored via sim_state.
        """
        root = getattr(env, "unwrapped", env)
        
        # Restore spawn mode if it was saved
        if "spawn_mode" in task_state and hasattr(root, "spawn_mode"):
            root.spawn_mode = task_state["spawn_mode"]
        
        # Restore default qpos if it was saved
        if "default_qpos" in task_state and hasattr(root, "_default_qpos"):
            root._default_qpos = np.asarray(task_state["default_qpos"], dtype=np.float32).copy()

