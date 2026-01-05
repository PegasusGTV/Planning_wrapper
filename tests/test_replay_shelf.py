"""
Deterministic replay test for ObjectRetrieveFromShelf-v1.

Pattern:
  - reset(seed=0)
  - snapshot S via wrapper.clone_state()
  - run N actions (pre-sampled), record reward, obj_pose, success
  - restore S
  - run same N actions, record again
  - assert all signals match within tolerance
"""

import numpy as np
import gymnasium as gym  # or gym

import envs  # noqa: F401
from planning_wrapper.wrappers.maniskill_planning import ManiSkillPlanningWrapper


ENV_ID = "ObjectRetrieveFromShelf-v1"
N_STEPS = 30
ATOL_REWARD = 1e-6
ATOL_POSE = 1e-6


def get_success(env) -> bool:
    u = env.unwrapped
    if hasattr(u, "_compute_success"):
        return bool(u._compute_success())
    try:
        return bool(u.evaluate().get("is_success", False))
    except Exception:
        return False


def run_rollout(wrapped_env, actions):
    rewards = []
    obj_poses = []
    successes = []

    obs = None
    done = False

    for a in actions:
        if done:
            break

        obs, reward, terminated, truncated, info = wrapped_env.step(a)
        extra = obs.get("extra", {})

        rewards.append(float(reward))

        obj_pose = np.asarray(extra.get("obj_pose", np.full(7, np.nan, np.float32)))
        obj_poses.append(obj_pose.copy())

        succ = get_success(wrapped_env)
        successes.append(bool(succ))

        done = bool(terminated or truncated)

    if len(obj_poses) == 0:
        obj_poses = np.zeros((0, 7), dtype=np.float32)
    else:
        obj_poses = np.stack(obj_poses)

    return (
        np.array(rewards, dtype=np.float32),
        obj_poses,
        np.array(successes, dtype=bool),
    )


def test_replay_shelf_deterministic():
    env = gym.make(
        ENV_ID,
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        spawn_mode="random_small",
    )
    wrapped = ManiSkillPlanningWrapper(env)

    # Reset with fixed seed
    obs, info = wrapped.reset(seed=0)

    # Pre-sample actions deterministically
    action_space = wrapped.action_space
    action_space.seed(0)
    actions = [action_space.sample() for _ in range(N_STEPS)]

    # Snapshot state
    snapshot = wrapped.clone_state()  # if your API needs obs, pass obs

    # First rollout
    r1, p1, s1 = run_rollout(wrapped, actions)

    # Restore snapshot and second rollout
    wrapped.restore_state(snapshot)
    r2, p2, s2 = run_rollout(wrapped, actions)

    # Lengths must match
    assert len(r1) == len(r2)
    assert p1.shape == p2.shape
    assert len(s1) == len(s2)

    # Rewards must match (within numerical tolerance)
    assert np.allclose(r1, r2, atol=ATOL_REWARD)

    # Object poses must match (allowing for tiny FP noise + NaNs)
    assert np.allclose(p1, p2, atol=ATOL_POSE, equal_nan=True)

    # Success flags must match exactly
    assert np.array_equal(s1, s2)

    wrapped.close()
