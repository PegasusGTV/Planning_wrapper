"""
Enhanced visualization demo for ObjectRetrieveFromShelf-v1 with LARGER shelf dimensions.

This demo:
- Uses a larger shelf with more space to avoid hitting walls
- Shows panda robot interacting with multiple objects
- Demonstrates retrieving a specified target object
- Shows objects moving and interacting with each other
- Uses improved camera angle for better visualization
"""

from typing import List, Tuple, Dict, Any

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym

import envs  # noqa: F401  # register shelf retrieval env

from planning_wrapper.wrappers.maniskill_planning import ManiSkillPlanningWrapper
from planning_wrapper.adapters.shelf_retrieve import ShelfRetrieveTaskAdapter


MAX_STEPS = 2000  # Increased time for object manipulation
# Slower, smoother motion
STEP_SIZE = 0.004  # meters per step (reduced for slower, smoother motion)
BACKOFF = 0.02  # position behind object from robot's side (left side)
ABOVE_Z = 0.05  # approach height above object
PUSH_Z_OFFSET = -0.005  # push slightly below center for better leverage
CONTACT_EPS = 0.008  # contact distance - get close to object surface
GOAL_CLOSE_XY = 0.05  # stop if object is close to goal in XY


def make_delta_action(wrapper: ManiSkillPlanningWrapper, delta_xyz: np.ndarray) -> np.ndarray:
    """
    Convert a delta (meters) into the controller's normalized action space.
    Exact copy from pusht_demo3.py to ensure proper scaling.
    """
    low, high = wrapper.get_controller_bounds()
    sample = wrapper.action_space.sample()
    action = np.zeros_like(sample, dtype=np.float32)
    flat = action.reshape(-1)
    
    # Convert delta from meters to normalized action space
    center = 0.5 * (high + low)
    half_range = 0.5 * (high - low)
    if half_range < 1e-6:
        normalized_delta = np.zeros_like(delta_xyz, dtype=np.float32)
    else:
        normalized_delta = (delta_xyz - center) / half_range
    
    # Clip to [-1, 1] to respect action space bounds
    normalized_delta = np.clip(normalized_delta, -1.0, 1.0)
    
    flat[:3] = normalized_delta.astype(np.float32)
    return action


def move_towards(curr: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    """Move towards target with max step size. From pusht_demo3.py"""
    d = target - curr
    dist = float(np.linalg.norm(d))
    if dist < 1e-6:
        return np.zeros(3, dtype=np.float32)
    step = min(max_step, dist)
    return (d / dist * step).astype(np.float32)


def touch_and_push_policy(
    wrapper: ManiSkillPlanningWrapper, 
    obs,
    ctx: dict,
    target_obj_id: int = 0,
) -> np.ndarray:
    """
    Policy following pusht_demo3.py pattern:
      0) compute push direction, pre-contact point
      1) go above pre-contact
      2) descend to push height
      3) push forward toward goal (keeping height) in a straight line
    """
    planning_obs = wrapper.get_planning_obs(obs)
    
    # Extract target object pose
    target_obj_pose = None
    if "target_obj_pose" in planning_obs:
        target_obj_pose = np.asarray(planning_obs["target_obj_pose"]).reshape(-1, 7)[0]
    elif "obj_poses" in planning_obs:
        obj_poses = np.asarray(planning_obs["obj_poses"])
        if len(obj_poses) > target_obj_id:
            target_obj_pose = obj_poses[target_obj_id]
        elif len(obj_poses) > 0:
            target_obj_pose = obj_poses[0]
        else:
            target_obj_pose = np.full((7,), np.nan, dtype=np.float32)
    else:
        target_obj_pose = np.full((7,), np.nan, dtype=np.float32)
    
    goal_pos = np.asarray(planning_obs["goal_pos"]).reshape(-1, 3)[0]
    tcp_pose = np.asarray(planning_obs["tcp_pose"]).reshape(-1, 7)[0]
    
    target_obj_pos = target_obj_pose[:3]
    tcp = tcp_pose[:3]
    obj = target_obj_pos
    goal = goal_pos
    
    # Handle NaN values
    if np.any(np.isnan(tcp)):
        try:
            if wrapper.agent and hasattr(wrapper.agent, 'tcp'):
                tcp_pose_direct = wrapper.agent.tcp.pose
                tcp = np.asarray(tcp_pose_direct.p, dtype=np.float32)
            else:
                tcp = np.array([0.3, 0.0, 0.6], dtype=np.float32)
        except Exception:
            tcp = np.array([0.3, 0.0, 0.6], dtype=np.float32)
    
    if np.any(np.isnan(obj)):
        if "obj_poses" in planning_obs:
            obj_poses = np.asarray(planning_obs["obj_poses"])
            if len(obj_poses) > target_obj_id:
                obj = obj_poses[target_obj_id][:3]
            elif len(obj_poses) > 0:
                obj = obj_poses[0][:3]
            else:
                obj = np.array([0.6, 0.0, 0.8], dtype=np.float32)
        else:
            obj = np.array([0.6, 0.0, 0.8], dtype=np.float32)
    
    # Stop if object already near goal (xy)
    if np.linalg.norm((obj - goal)[:2]) < GOAL_CLOSE_XY:
        ctx["phase"] = 4  # done
    
    # Get shelf bounds and center if available
    shelf_min_x = None
    shelf_max_x = None
    shelf_center_xy = None
    if "bay_center" in planning_obs and "bay_size" in planning_obs:
        bay_center = np.asarray(planning_obs["bay_center"]).reshape(-1, 3)[0]
        bay_size = np.asarray(planning_obs["bay_size"]).reshape(-1, 3)[0]
        shelf_min_x = float(bay_center[0] - bay_size[0] / 2.0)
        shelf_max_x = float(bay_center[0] + bay_size[0] / 2.0)
        shelf_center_xy = bay_center[:2]  # Shelf center in XY
    
    # Compute push direction: move object TOWARDS shelf center instead of towards goal
    # This creates a more controlled motion towards the center
    if "push_dir_xy" not in ctx:
        if shelf_center_xy is not None:
            # Push towards shelf center
            d_xy = (shelf_center_xy - obj[:2])
        else:
            # Fallback: use goal direction
            d_xy = (goal - obj)[:2]
        
        n = float(np.linalg.norm(d_xy))
        if n < 1e-6:
            push_dir_xy = np.array([1.0, 0.0], dtype=np.float32)
        else:
            push_dir_xy = (d_xy / n).astype(np.float32)
        ctx["push_dir_xy"] = push_dir_xy
        
        # For shelf: robot is at x=-0.3 (left), objects at x~0.45 (inside), goal at x>0.45 (outside)
        # Push direction is +x (towards goal/right)
        # Position TCP from robot's side (left/smaller x) to push object right (+x)
        # "Behind" in push direction means opposite to push, so -push_dir = left side (robot's side)
        contact_xy_candidate = obj[:2] - push_dir_xy * CONTACT_EPS
        
        # Ensure contact point is not too far into shelf (avoid back wall)
        # With much larger shelf, we have plenty of space, but still keep safe margin
        if shelf_min_x is not None:
            min_safe_x = shelf_min_x + 0.08  # 8cm from back wall (generous margin with much larger shelf)
            if contact_xy_candidate[0] < min_safe_x:
                contact_xy_candidate[0] = min_safe_x
        
        ctx["pre_xy"] = obj[:2] - push_dir_xy * BACKOFF  # Approach from left (robot side)
        ctx["contact_xy"] = contact_xy_candidate  # Contact position (slightly behind object, but safe from wall)
    else:
        push_dir_xy = ctx["push_dir_xy"]
        pre_xy = ctx["pre_xy"]
        contact_xy = ctx["contact_xy"]
    
    # Pre-contact: approach from robot's side (left, smaller x), then contact slightly behind object
    pre_xy = ctx.get("pre_xy", obj[:2] - push_dir_xy * BACKOFF)
    contact_xy = ctx.get("contact_xy", obj[:2] - push_dir_xy * CONTACT_EPS)
    
    # Heights
    z_above = float(obj[2] + ABOVE_Z)
    z_push = float(obj[2] + PUSH_Z_OFFSET)
    
    phase = ctx.get("phase", 0)
    
    # Phase 0: initialize ctx
    if phase == 0:
        ctx["phase"] = 1
        ctx["push_steps"] = 0
    
    # Phase 1: go above pre-contact
    if ctx["phase"] == 1:
        target = np.array([pre_xy[0], pre_xy[1], z_above], dtype=np.float32)
        delta = move_towards(tcp, target, STEP_SIZE)
        if np.linalg.norm(target - tcp) < 0.01:
            ctx["phase"] = 2
        return make_delta_action(wrapper, delta)
    
    # Phase 2: go down to push height at the *contact* XY
    if ctx["phase"] == 2:
        target = np.array([contact_xy[0], contact_xy[1], z_push], dtype=np.float32)
        delta = move_towards(tcp, target, STEP_SIZE)
        if np.linalg.norm(target - tcp) < 0.008:
            ctx["phase"] = 3
        return make_delta_action(wrapper, delta)
    
    # Phase 3: push forward in XY while holding Z (slower, smoother motion)
    if ctx["phase"] == 3:
        ctx["push_steps"] += 1
        
        # Slower, smoother pushing - no aggressive initial steps
        fwd = STEP_SIZE  # Use consistent step size for smooth motion
        
        # move forward along push direction, keep z near push height
        target = np.array(
            [tcp[0] + push_dir_xy[0] * fwd,
             tcp[1] + push_dir_xy[1] * fwd,
             z_push],
            dtype=np.float32
        )
        delta = move_towards(tcp, target, STEP_SIZE)
        
        # Check if object is close to shelf center (if pushing towards center)
        if shelf_center_xy is not None:
            obj_to_center_dist = np.linalg.norm((obj[:2] - shelf_center_xy))
            # Stop if close to center or after many steps
            if ctx["push_steps"] > 500 or obj_to_center_dist < GOAL_CLOSE_XY:
                ctx["phase"] = 4
        else:
            # Fallback: use goal distance
            if ctx["push_steps"] > 500 or np.linalg.norm((obj - goal)[:2]) < GOAL_CLOSE_XY:
                ctx["phase"] = 4
        return make_delta_action(wrapper, delta)
    
    # Phase 4: done — hold still
    return make_delta_action(wrapper, np.zeros(3, dtype=np.float32))


def main():
    # 1) Build env + wrapper + adapter with MUCH LARGER shelf dimensions
    # Significantly increased shelf size to provide ample space and avoid hitting walls
    env = gym.make(
        "ObjectRetrieveFromShelf-v1",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        render_mode="human",  # Enable visualization
        spawn_mode="fixed",  # Fixed object positions (not random)
        num_objects=16,  # Spawn 16 objects
        target_object_id=0,  # Target the first object (red cube)
        # MUCH LARGER shelf dimensions for maximum workspace
        bay_center=(0.55, 0.0, 0.8),  # Further forward to give more space behind
        bay_size=(0.7, 0.6, 0.3),  # Much larger: depth 0.7m (was 0.3m), width 0.6m (was 0.4m)
        table_center=(0.55, 0.0, 0.6),  # Match shelf position
        table_size=(1.2, 1.2, 0.05),  # Much larger table to support bigger shelf
    )

    adapter = ShelfRetrieveTaskAdapter()
    w = ManiSkillPlanningWrapper(env, adapter=adapter)

    # 2) Reset
    obs, info = w.reset(seed=42)
    print("="*70)
    print("Multi-Object Shelf Retrieval Demo - Panda Robot (LARGER SHELF)")
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
    print(f"  Shelf bounds: x=[{bay_center[0] - bay_size[0]/2:.3f}, {bay_center[0] + bay_size[0]/2:.3f}]")
    
    # Print positions of all objects
    print("\n  All object positions:")
    for i, obj_pose in enumerate(obj_poses):
        obj_pos = obj_pose[:3]
        marker = " <-- TARGET" if i == target_obj_id else ""
        print(f"    Object {i}: {obj_pos}{marker}")
    print()

    # 3) Planning loop with context-based state machine (following pusht_demo3.py)
    ctx: dict = {"phase": 0}
    target_obj_traj: List[np.ndarray] = []
    tcp_traj: List[np.ndarray] = []
    phases_traj: List[int] = []
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
        phases_traj.append(ctx.get("phase", 0))
        
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

        phase_names = {0: "init", 1: "above", 2: "descend", 3: "push", 4: "done"}
        phase_name = phase_names.get(ctx.get("phase", 0), "unknown")
        
        if t % 20 == 0 or (len(phases_traj) > 1 and ctx.get("phase", 0) != phases_traj[-2]):
            print(f"Step {t:4d} [phase={ctx.get('phase', 0)} ({phase_name:8s})]: "
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

        # Compute action using context-based policy
        action = touch_and_push_policy(w, obs, ctx, target_obj_id=target_obj_id)

        # Step
        obs, reward, terminated, truncated, info = w.step(action)
        
        # Handle termination/truncation
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
        phase_counts = {}
        for p in phases_traj:
            phase_counts[p] = phase_counts.get(p, 0) + 1
        print(f"Phase breakdown: {phase_counts}")
        
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

