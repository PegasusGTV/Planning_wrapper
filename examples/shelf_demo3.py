"""
Enhanced visualization demo for ObjectRetrieveFromShelf-v1 with multiple objects and layers.

This demo:
- Shows panda robot interacting with multiple objects in a multi-layer shelf
- Demonstrates retrieving a specified target object from among many objects
- Shows objects moving and interacting with each other
- Uses improved camera angle for better visualization
"""

from typing import List, Tuple

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym

import envs  # noqa: F401  # register shelf retrieval env

from planning_wrapper.wrappers.maniskill_planning import ManiSkillPlanningWrapper
from planning_wrapper.adapters.shelf_retrieve import ShelfRetrieveTaskAdapter


MAX_STEPS = 500
APPROACH_STEP_SIZE = 0.02  # meters per step when approaching object
PULL_STEP_SIZE = 0.015  # meters per step when pulling object
APPROACH_DIST = 0.08  # distance to maintain from object


def multi_object_retrieve_policy(
    wrapper: ManiSkillPlanningWrapper, 
    obs,
    phase: str = "approach",
    target_obj_id: int = 0,
) -> Tuple[np.ndarray, str]:
    """
    Policy for retrieving a specific target object from multiple objects.
    
    Phases:
    1. "approach": Move TCP close to target object
    2. "pull": Pull target object toward goal (outside shelf)
    
    Returns: (action, next_phase)
    """
    planning_obs = wrapper.get_planning_obs(obs)
    
    # Extract target object pose - try multiple sources
    target_obj_pose = None
    if "target_obj_pose" in planning_obs:
        target_obj_pose = np.asarray(planning_obs["target_obj_pose"]).reshape(-1, 7)[0]  # (7,)
    elif "obj_poses" in planning_obs:
        obj_poses = np.asarray(planning_obs["obj_poses"])
        if len(obj_poses) > target_obj_id:
            target_obj_pose = obj_poses[target_obj_id]
        elif len(obj_poses) > 0:
            target_obj_pose = obj_poses[0]
        else:
            target_obj_pose = np.full((7,), np.nan, dtype=np.float32)
    elif "obj_pose" in planning_obs:
        # Fallback to single obj_pose
        target_obj_pose = np.asarray(planning_obs["obj_pose"]).reshape(-1, 7)[0]
    else:
        target_obj_pose = np.full((7,), np.nan, dtype=np.float32)
    
    goal_pos = np.asarray(planning_obs["goal_pos"]).reshape(-1, 3)[0]  # (3,)
    tcp_pose = np.asarray(planning_obs["tcp_pose"]).reshape(-1, 7)[0]  # (7,)
    
    # Ensure target_obj_pose is a numpy array
    if target_obj_pose is None:
        target_obj_pose = np.full((7,), np.nan, dtype=np.float32)
    target_obj_pos = target_obj_pose[:3]
    tcp_pos = tcp_pose[:3]
    
    # Handle NaN TCP position
    if np.any(np.isnan(tcp_pos)):
        try:
            if wrapper.agent and hasattr(wrapper.agent, 'tcp'):
                tcp_pose_direct = wrapper.agent.tcp.pose
                tcp_pos = np.asarray(tcp_pose_direct.p, dtype=np.float32)
            else:
                tcp_pos = np.array([0.3, 0.0, 0.6], dtype=np.float32)
        except Exception:
            tcp_pos = np.array([0.3, 0.0, 0.6], dtype=np.float32)
    
    # Handle NaN target object position (object might not exist)
    if np.any(np.isnan(target_obj_pos)):
        # Fallback: try to get from obj_poses array
        if "obj_poses" in planning_obs:
            obj_poses = np.asarray(planning_obs["obj_poses"])
            if len(obj_poses) > target_obj_id:
                target_obj_pose = obj_poses[target_obj_id]
                target_obj_pos = target_obj_pose[:3]
            else:
                target_obj_pos = np.array([0.6, 0.0, 0.8], dtype=np.float32)
        else:
            target_obj_pos = np.array([0.6, 0.0, 0.8], dtype=np.float32)
    
    tcp_to_obj_dist = np.linalg.norm(tcp_pos - target_obj_pos)
    obj_to_goal_dist = np.linalg.norm(target_obj_pos - goal_pos)
    
    # Build action
    sample = wrapper.action_space.sample()
    action = np.zeros_like(sample, dtype=np.float32)
    action_flat = action.reshape(-1)
    
    next_phase = phase
    
    if phase == "approach":
        # Phase 1: Approach the target object
        if tcp_to_obj_dist > APPROACH_DIST:
            direction = target_obj_pos - tcp_pos
            dist = np.linalg.norm(direction)
            if dist > 1e-4:
                direction = direction / dist
                step = min(APPROACH_STEP_SIZE, dist - APPROACH_DIST)
                delta_pos = direction * step
            else:
                delta_pos = np.zeros(3, dtype=np.float32)
        else:
            # Close enough, switch to pull phase
            next_phase = "pull"
            delta_pos = np.zeros(3, dtype=np.float32)
    
    elif phase == "pull":
        # Phase 2: Pull target object toward goal
        if obj_to_goal_dist > 0.05:  # Still need to pull
            # Move TCP in direction from object to goal
            # But maintain some distance from object
            pull_direction = goal_pos - target_obj_pos
            pull_dist = np.linalg.norm(pull_direction)
            
            if pull_dist > 1e-4:
                pull_direction = pull_direction / pull_dist
                
                # Target TCP position: slightly behind object in pull direction
                target_tcp_pos = target_obj_pos - APPROACH_DIST * 0.5 * pull_direction
                tcp_to_target = target_tcp_pos - tcp_pos
                tcp_to_target_dist = np.linalg.norm(tcp_to_target)
                
                if tcp_to_target_dist > 0.01:
                    # Move TCP toward target position
                    direction = tcp_to_target / tcp_to_target_dist
                    step = min(PULL_STEP_SIZE, tcp_to_target_dist)
                    delta_pos = direction * step
                else:
                    # TCP is in good position, move object directly
                    step = min(PULL_STEP_SIZE, pull_dist)
                    delta_pos = pull_direction * step
            else:
                delta_pos = np.zeros(3, dtype=np.float32)
        else:
            # Object is close to goal, fine-tune
            direction = goal_pos - tcp_pos
            dist = np.linalg.norm(direction)
            if dist > 1e-4:
                direction = direction / dist
                step = min(PULL_STEP_SIZE * 0.5, dist)
                delta_pos = direction * step
            else:
                delta_pos = np.zeros(3, dtype=np.float32)
    
    action_flat[:3] = delta_pos
    return action, next_phase


def main():
    # 1) Build env + wrapper + adapter
    # Configure for multiple objects and layers
    env = gym.make(
        "ObjectRetrieveFromShelf-v1",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        render_mode="human",  # Enable visualization
        spawn_mode="fixed",  # Fixed object positions (not random)
        num_objects=16,  # Spawn 16 objects
        target_object_id=0,  # Target the first object (red cube)
    )

    adapter = ShelfRetrieveTaskAdapter()
    w = ManiSkillPlanningWrapper(env, adapter=adapter)

    # 2) Reset
    obs, info = w.reset(seed=42)
    print("="*70)
    print("Multi-Object Shelf Retrieval Demo - Panda Robot")
    print("="*70)
    print("\nInitial state:")
    extra = obs["extra"]
    
    # Get information about all objects
    obj_poses = np.asarray(extra["obj_poses"])  # (N, 7)
    target_obj_id = extra["target_obj_id"]
    target_obj_pose = np.asarray(extra["target_obj_pose"])
    goal_pos = np.asarray(extra["goal_pos"])
    bay_center = np.asarray(extra["bay_center"])
    bay_size = np.asarray(extra["bay_size"])
    
    print(f"  Number of objects: {len(obj_poses)}")
    print(f"  Target object ID: {target_obj_id}")
    print(f"  Target object position: {target_obj_pose[:3]}")
    print(f"  Goal position: {goal_pos}")
    print(f"  Bay center: {bay_center}, size: {bay_size}")
    
    # Print positions of all objects
    print("\n  All object positions:")
    for i, obj_pose in enumerate(obj_poses):
        obj_pos = obj_pose[:3]
        marker = " <-- TARGET" if i == target_obj_id else ""
        print(f"    Object {i}: {obj_pos}{marker}")
    print()

    # 3) Planning loop with phase tracking
    phase = "approach"
    target_obj_traj: List[np.ndarray] = []
    tcp_traj: List[np.ndarray] = []
    phases_traj: List[str] = []
    all_objects_traj: List[List[np.ndarray]] = [[] for _ in range(len(obj_poses))]

    for t in range(MAX_STEPS):
        extra = obs["extra"]
        target_obj_pose = np.asarray(extra["target_obj_pose"])
        tcp_pose = np.asarray(extra["tcp_pose"])
        goal_pos = np.asarray(extra["goal_pos"])
        obj_poses = np.asarray(extra["obj_poses"])
        
        target_obj_pos = target_obj_pose[:3]
        tcp_pos = tcp_pose[:3]
        
        target_obj_traj.append(target_obj_pos.copy())
        tcp_traj.append(tcp_pos.copy())
        phases_traj.append(phase)
        
        # Track all objects
        for i, obj_pose in enumerate(obj_poses):
            if i < len(all_objects_traj):
                all_objects_traj[i].append(obj_pose[:3].copy())

        # Compute metrics
        obj_to_goal_dist = np.linalg.norm(target_obj_pos - goal_pos)
        tcp_to_obj_dist = np.linalg.norm(tcp_pos - target_obj_pos)
        
        # Check if target object is outside shelf
        bay_max_x = bay_center[0] + bay_size[0] / 2.0
        is_outside = target_obj_pos[0] > bay_max_x + 0.02

        if t % 20 == 0 or (len(phases_traj) > 1 and phase != phases_traj[-2]):
            print(f"Step {t:4d} [{phase:8s}]: "
                  f"target_obj=({target_obj_pos[0]:.3f}, {target_obj_pos[1]:.3f}, {target_obj_pos[2]:.3f}), "
                  f"obj->goal={obj_to_goal_dist:.3f}, "
                  f"tcp->obj={tcp_to_obj_dist:.3f}, "
                  f"outside={is_outside}")

        # Check success
        success = w.unwrapped._compute_success()
        if success:
            print(f"\n{'='*70}")
            print(f"✓ SUCCESS! Target object {target_obj_id} retrieved from shelf at step {t}")
            print(f"{'='*70}")
            print(f"  Final target object position: {target_obj_pos}")
            print(f"  Final distance to goal: {obj_to_goal_dist:.4f}")
            
            # Show final positions of all objects
            print("\n  Final positions of all objects:")
            for i, obj_pose in enumerate(obj_poses):
                obj_pos = obj_pose[:3]
                marker = " <-- TARGET (RETRIEVED)" if i == target_obj_id else ""
                print(f"    Object {i}: {obj_pos}{marker}")
            
            try:
                w.render()
            except Exception:
                pass
            break

        # Render for visualization (shows objects moving)
        try:
            w.render()
        except Exception:
            pass

        # Compute action
        action, next_phase = multi_object_retrieve_policy(
            w, obs, phase, target_obj_id=target_obj_id
        )
        if next_phase != phase:
            print(f"  → Phase transition: {phase} → {next_phase}")
        phase = next_phase

        # Step
        obs, reward, terminated, truncated, info = w.step(action)
        
        # Handle termination/truncation
        # Note: With max_episode_steps=2000, truncation should not occur before MAX_STEPS=1500
        if terminated or truncated:
            # Convert tensor to bool if needed
            if hasattr(terminated, 'item'):
                terminated = bool(terminated.item())
            if hasattr(truncated, 'item'):
                truncated = bool(truncated.item())
            
            print(f"\nEpisode ended at step {t} (terminated={terminated}, truncated={truncated})")
            final_success = w.unwrapped._compute_success()
            if final_success:
                print("  ✓ SUCCESS! Target object was retrieved.")
            else:
                print("  ✗ Did not succeed.")
            break

    w.close()
    
    # Summary
    print("\n" + "="*70)
    print("Demo Summary")
    print("="*70)
    print(f"Total steps: {len(target_obj_traj)}")
    if len(target_obj_traj) > 0:
        initial_obj = target_obj_traj[0]
        final_obj = target_obj_traj[-1]
        print(f"Initial target object position: {initial_obj}")
        print(f"Final target object position: {final_obj}")
        print(f"Distance traveled: {np.linalg.norm(final_obj - initial_obj):.4f} m")
        print(f"Final distance to goal: {np.linalg.norm(final_obj - goal_pos):.4f} m")
        
        # Phase statistics
        approach_steps = sum(1 for p in phases_traj if p == "approach")
        pull_steps = sum(1 for p in phases_traj if p == "pull")
        print(f"Phase breakdown: approach={approach_steps}, pull={pull_steps}")
        
        # Show movement of other objects (demonstrates interaction)
        print("\n  Movement of other objects (shows interaction):")
        for i in range(len(all_objects_traj)):
            if i != target_obj_id and len(all_objects_traj[i]) > 0:
                initial = all_objects_traj[i][0]
                final = all_objects_traj[i][-1]
                dist = np.linalg.norm(final - initial)
                if dist > 0.01:  # Only show if moved significantly
                    print(f"    Object {i}: moved {dist:.4f} m "
                          f"({initial} → {final})")


if __name__ == "__main__":
    main()

