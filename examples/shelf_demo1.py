"""
Simple visualization demo for ObjectRetrieveFromShelf-v1.

Idea:
- Use the ManiSkillPlanningWrapper + ShelfRetrieveTaskAdapter.
- At each step, move the end-effector (tcp_pose) toward the object, then toward the goal.
- Visualize the robot pulling the object out of the shelf.
"""

from typing import List

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym

import envs  # noqa: F401  # register shelf retrieval env

from planning_wrapper.wrappers.maniskill_planning import ManiSkillPlanningWrapper
from planning_wrapper.adapters.shelf_retrieve import ShelfRetrieveTaskAdapter


MAX_STEPS = 200
STEP_SIZE = 0.015  # meters per step toward the target
APPROACH_DIST = 0.05  # distance to maintain from object when approaching


def simple_retrieve_policy(wrapper: ManiSkillPlanningWrapper, obs) -> np.ndarray:
    """
    Simple policy for shelf retrieval:
    1. If far from object, approach it
    2. Once close, pull it toward the goal (outside the shelf)
    
    Assumes control_mode="pd_ee_delta_pose" with action shaped like:
        [dx, dy, dz, d_rx, d_ry, d_rz, gripper]
    """
    planning_obs = wrapper.get_planning_obs(obs)
    
    # Extract positions (remove batch dimension if present)
    goal_pos = np.asarray(planning_obs["goal_pos"]).reshape(-1, 3)[0]  # (3,)
    obj_pose = np.asarray(planning_obs["obj_pose"]).reshape(-1, 7)[0]  # (7,)
    tcp_pose = np.asarray(planning_obs["tcp_pose"]).reshape(-1, 7)[0]  # (7,)
    
    obj_pos = obj_pose[:3]  # Object position
    tcp_pos = tcp_pose[:3]  # TCP position
    
    # Handle NaN TCP position (robot might not be initialized yet)
    if np.any(np.isnan(tcp_pos)):
        # If TCP is not available, try to get it directly from the wrapper
        try:
            if wrapper.agent and hasattr(wrapper.agent, 'tcp'):
                tcp_pose_direct = wrapper.agent.tcp.pose
                tcp_pos = np.asarray(tcp_pose_direct.p, dtype=np.float32)
            else:
                # Fallback: move toward object from a default position
                tcp_pos = np.array([0.3, 0.0, 0.6], dtype=np.float32)
        except Exception:
            tcp_pos = np.array([0.3, 0.0, 0.6], dtype=np.float32)
    
    # Check if object is already at goal
    obj_to_goal_dist = np.linalg.norm(obj_pos - goal_pos)
    
    # Distance from TCP to object
    tcp_to_obj_dist = np.linalg.norm(tcp_pos - obj_pos)
    
    # Strategy: 
    # - If object is far from goal, pull it toward goal
    # - If TCP is far from object, approach object first
    if obj_to_goal_dist > 0.1:  # Object not yet at goal
        if tcp_to_obj_dist > APPROACH_DIST:  # Need to approach object first
            # Move TCP toward object
            direction = obj_pos - tcp_pos
            dist = np.linalg.norm(direction)
            if dist > 1e-4:
                direction = direction / dist
                step = min(STEP_SIZE, dist - APPROACH_DIST)
                delta_pos = direction * step
            else:
                delta_pos = np.zeros(3, dtype=np.float32)
        else:
            # Close to object, pull it toward goal
            # Move TCP in direction from object to goal
            direction = goal_pos - obj_pos
            dist = np.linalg.norm(direction)
            if dist > 1e-4:
                direction = direction / dist
                step = min(STEP_SIZE, dist)
                delta_pos = direction * step
            else:
                delta_pos = np.zeros(3, dtype=np.float32)
    else:
        # Object is close to goal, fine-tune position
        direction = goal_pos - tcp_pos
        dist = np.linalg.norm(direction)
        if dist > 1e-4:
            direction = direction / dist
            step = min(STEP_SIZE * 0.5, dist)  # Smaller steps for fine-tuning
            delta_pos = direction * step
        else:
            delta_pos = np.zeros(3, dtype=np.float32)
    
    # Build action with correct shape
    sample = wrapper.action_space.sample()
    action = np.zeros_like(sample, dtype=np.float32)
    
    # Fill first 3 dims with delta position; leave others as 0
    action_flat = action.reshape(-1)
    action_flat[:3] = delta_pos  # [dx, dy, dz]
    
    return action


def main():
    # 1) Build env + wrapper + adapter
    env = gym.make(
        "ObjectRetrieveFromShelf-v1",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        render_mode="human",  # Enable visualization
        spawn_mode="fixed",  # Use fixed spawn for consistent demo
    )

    adapter = ShelfRetrieveTaskAdapter()
    w = ManiSkillPlanningWrapper(env, adapter=adapter)

    # 2) Reset
    obs, info = w.reset(seed=0)
    print("Reset with seed=0.")
    print("Initial state:")
    extra = obs["extra"]
    obj_pose = np.asarray(extra["obj_pose"])
    goal_pos = np.asarray(extra["goal_pos"])
    bay_center = np.asarray(extra["bay_center"])
    bay_size = np.asarray(extra["bay_size"])
    print(f"  Object position: {obj_pose[:3]}")
    print(f"  Goal position: {goal_pos}")
    print(f"  Bay center: {bay_center}, size: {bay_size}")

    # 3) Planning loop
    obj_traj: List[np.ndarray] = []
    tcp_traj: List[np.ndarray] = []
    goal_traj: List[np.ndarray] = []

    for t in range(MAX_STEPS):
        # Record positions for debugging/plotting
        extra = obs["extra"]
        obj_pose = np.asarray(extra["obj_pose"])
        goal_pos = np.asarray(extra["goal_pos"])
        tcp_pose = np.asarray(extra["tcp_pose"])
        
        obj_traj.append(obj_pose[:3].copy())
        tcp_traj.append(tcp_pose[:3].copy())
        goal_traj.append(goal_pos.copy())

        # Compute distances
        obj_pos = obj_pose[:3]
        tcp_pos = tcp_pose[:3]
        obj_to_goal_dist = np.linalg.norm(obj_pos - goal_pos)
        tcp_to_obj_dist = np.linalg.norm(tcp_pos - obj_pos)

        print(f"\nStep {t}:")
        print(f"  Object position: {obj_pos}")
        print(f"  TCP position: {tcp_pos}")
        print(f"  Goal position: {goal_pos}")
        print(f"  Object->Goal distance: {obj_to_goal_dist:.4f}")
        print(f"  TCP->Object distance: {tcp_to_obj_dist:.4f}")

        # Check success
        success = w.unwrapped._compute_success()
        if success:
            print(f"  ✓ SUCCESS! Object retrieved from shelf at step {t}")
            # Render one more time to show final state
            try:
                w.render()
            except Exception:
                pass
            break

        # Render for visualization
        try:
            w.render()
        except Exception:
            # If render() is not available / fails, just ignore
            pass

        # Compute action from simple policy
        action = simple_retrieve_policy(w, obs)

        # Step env
        obs, reward, terminated, truncated, info = w.step(action)
        print(f"  Reward: {float(reward)}")
        
        if terminated or truncated:
            print(f"Episode ended at step {t} (terminated={terminated}, truncated={truncated})")
            # Check final success
            final_success = w.unwrapped._compute_success()
            if final_success:
                print("  ✓ SUCCESS! Object was retrieved.")
            else:
                print("  ✗ Did not succeed.")
            break

    w.close()
    print("\n" + "="*50)
    print("Finished shelf retrieval demo.")
    print(f"Total steps: {len(obj_traj)}")
    if len(obj_traj) > 0:
        final_obj_pos = obj_traj[-1]
        print(f"Final object position: {final_obj_pos}")
        print(f"Goal position: {goal_traj[0] if goal_traj else 'N/A'}")
        final_dist = np.linalg.norm(final_obj_pos - goal_traj[0]) if goal_traj else np.inf
        print(f"Final distance to goal: {final_dist:.4f}")


if __name__ == "__main__":
    main()

