"""
More sophisticated visualization demo for ObjectRetrieveFromShelf-v1.

This demo:
- Uses a two-phase approach: approach object, then pull it out
- Shows more detailed trajectory information
- Includes visualization of the shelf geometry
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


MAX_STEPS = 1000
APPROACH_STEP_SIZE = 0.02  # meters per step when approaching object
PULL_STEP_SIZE = 0.015  # meters per step when pulling object
APPROACH_DIST = 0.08  # distance to maintain from object


def two_phase_retrieve_policy(
    wrapper: ManiSkillPlanningWrapper, 
    obs,
    phase: str = "approach"
) -> Tuple[np.ndarray, str]:
    """
    Two-phase policy:
    1. "approach": Move TCP close to object
    2. "pull": Pull object toward goal (outside shelf)
    
    Returns: (action, next_phase)
    """
    planning_obs = wrapper.get_planning_obs(obs)
    
    # Extract positions
    goal_pos = np.asarray(planning_obs["goal_pos"]).reshape(-1, 3)[0]  # (3,)
    obj_pose = np.asarray(planning_obs["obj_pose"]).reshape(-1, 7)[0]  # (7,)
    tcp_pose = np.asarray(planning_obs["tcp_pose"]).reshape(-1, 7)[0]  # (7,)
    
    obj_pos = obj_pose[:3]
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
    
    tcp_to_obj_dist = np.linalg.norm(tcp_pos - obj_pos)
    obj_to_goal_dist = np.linalg.norm(obj_pos - goal_pos)
    
    # Build action
    sample = wrapper.action_space.sample()
    action = np.zeros_like(sample, dtype=np.float32)
    action_flat = action.reshape(-1)
    
    next_phase = phase
    
    if phase == "approach":
        # Phase 1: Approach the object
        if tcp_to_obj_dist > APPROACH_DIST:
            direction = obj_pos - tcp_pos
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
        # Phase 2: Pull object toward goal
        if obj_to_goal_dist > 0.05:  # Still need to pull
            # Move TCP in direction from object to goal
            # But maintain some distance from object
            pull_direction = goal_pos - obj_pos
            pull_dist = np.linalg.norm(pull_direction)
            
            if pull_dist > 1e-4:
                pull_direction = pull_direction / pull_dist
                
                # Target TCP position: slightly behind object in pull direction
                target_tcp_pos = obj_pos - APPROACH_DIST * 0.5 * pull_direction
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
    env = gym.make(
        "ObjectRetrieveFromShelf-v1",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        render_mode="human",
        spawn_mode="fixed",  # Use fixed spawn for consistent demo
    )

    adapter = ShelfRetrieveTaskAdapter()
    w = ManiSkillPlanningWrapper(env, adapter=adapter)

    # 2) Reset
    obs, info = w.reset(seed=0)
    print("="*60)
    print("Shelf Retrieval Demo - Two-Phase Approach")
    print("="*60)
    print("\nInitial state:")
    extra = obs["extra"]
    obj_pose = np.asarray(extra["obj_pose"])
    goal_pos = np.asarray(extra["goal_pos"])
    bay_center = np.asarray(extra["bay_center"])
    bay_size = np.asarray(extra["bay_size"])
    bay_min = bay_center - 0.5 * bay_size
    bay_max = bay_center + 0.5 * bay_size
    
    print(f"  Object position: {obj_pose[:3]}")
    print(f"  Goal position: {goal_pos}")
    print(f"  Bay bounds: x=[{bay_min[0]:.3f}, {bay_max[0]:.3f}], "
          f"y=[{bay_min[1]:.3f}, {bay_max[1]:.3f}], "
          f"z=[{bay_min[2]:.3f}, {bay_max[2]:.3f}]")
    print(f"  Object is inside bay: {bay_min[0] < obj_pose[0] < bay_max[0]}")
    print()

    # 3) Planning loop with phase tracking
    phase = "approach"
    obj_traj: List[np.ndarray] = []
    tcp_traj: List[np.ndarray] = []
    phases_traj: List[str] = []

    for t in range(MAX_STEPS):
        extra = obs["extra"]
        obj_pose = np.asarray(extra["obj_pose"])
        tcp_pose = np.asarray(extra["tcp_pose"])
        goal_pos = np.asarray(extra["goal_pos"])
        
        obj_pos = obj_pose[:3]
        tcp_pos = tcp_pose[:3]
        
        obj_traj.append(obj_pos.copy())
        tcp_traj.append(tcp_pos.copy())
        phases_traj.append(phase)

        # Compute metrics
        obj_to_goal_dist = np.linalg.norm(obj_pos - goal_pos)
        tcp_to_obj_dist = np.linalg.norm(tcp_pos - obj_pos)
        obj_x = obj_pos[0]
        bay_max_x = bay_max[0]
        is_outside = obj_x > bay_max_x + 0.02

        if t % 10 == 0 or phase != phases_traj[-2] if len(phases_traj) > 1 else True:
            print(f"Step {t:3d} [{phase:8s}]: "
                  f"obj=({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}), "
                  f"obj->goal={obj_to_goal_dist:.3f}, "
                  f"tcp->obj={tcp_to_obj_dist:.3f}, "
                  f"outside={is_outside}")

        # Check success
        success = w.unwrapped._compute_success()
        if success:
            print(f"\n✓ SUCCESS! Object retrieved from shelf at step {t}")
            print(f"  Final object position: {obj_pos}")
            print(f"  Final distance to goal: {obj_to_goal_dist:.4f}")
            try:
                w.render()
            except Exception:
                pass
            break

        # Render
        try:
            w.render()
        except Exception:
            pass

        # Compute action
        action, next_phase = two_phase_retrieve_policy(w, obs, phase)
        if next_phase != phase:
            print(f"  → Phase transition: {phase} → {next_phase}")
        phase = next_phase

        # Step
        obs, reward, terminated, truncated, info = w.step(action)
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {t} (terminated={terminated}, truncated={truncated})")
            final_success = w.unwrapped._compute_success()
            if final_success:
                print("  ✓ SUCCESS! Object was retrieved.")
            else:
                print("  ✗ Did not succeed.")
            break

    w.close()
    
    # Summary
    print("\n" + "="*60)
    print("Demo Summary")
    print("="*60)
    print(f"Total steps: {len(obj_traj)}")
    if len(obj_traj) > 0:
        initial_obj = obj_traj[0]
        final_obj = obj_traj[-1]
        print(f"Initial object position: {initial_obj}")
        print(f"Final object position: {final_obj}")
        print(f"Distance traveled: {np.linalg.norm(final_obj - initial_obj):.4f} m")
        print(f"Final distance to goal: {np.linalg.norm(final_obj - goal_pos):.4f} m")
        
        # Phase statistics
        approach_steps = sum(1 for p in phases_traj if p == "approach")
        pull_steps = sum(1 for p in phases_traj if p == "pull")
        print(f"Phase breakdown: approach={approach_steps}, pull={pull_steps}")


if __name__ == "__main__":
    main()

