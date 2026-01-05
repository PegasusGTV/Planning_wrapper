"""
Smoke test: deterministic replay for ObjectRetrieveFromShelf-v1.

Pattern:
  - reset(seed=0)
  - snapshot S via wrapper.clone_state()
  - run N actions (pre-sampled), record reward, obj_pose, success
  - restore S via wrapper.restore_state(S)
  - run same N actions, compare signals

Run:
    python examples/replay_shelf_smoke.py
"""

import numpy as np
import gymnasium as gym  # or `import gym` if you're on old gym

import envs  # ensure env registration side-effects run
from planning_wrapper.wrappers.maniskill_planning import ManiSkillPlanningWrapper


ENV_ID = "ObjectRetrieveFromShelf-v1"
N_STEPS = 30


def get_success(env) -> bool:
    """Helper to get scalar success from env."""
    u = env.unwrapped
    if hasattr(u, "_compute_success"):
        return bool(u._compute_success())
    # fallback to evaluate() if you changed naming
    try:
        return bool(u.evaluate().get("is_success", False))
    except Exception:
        return False


def run_rollout(wrapped_env, actions):
    """Run rollout from current state, return recorded signals."""
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

        # reward
        rewards.append(float(reward))

        # obj_pose from obs["extra"] (shape (7,))
        obj_pose = np.asarray(extra.get("obj_pose", np.full(7, np.nan, np.float32)))
        obj_poses.append(obj_pose.copy())

        # success
        succ = get_success(wrapped_env)
        successes.append(bool(succ))

        done = bool(terminated or truncated)

    return np.array(rewards, dtype=np.float32), np.stack(obj_poses), np.array(
        successes, dtype=bool
    )


def main():
    # 1) Make env and wrapper
    env = gym.make(
        ENV_ID,
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        spawn_mode="random_small",  # or "fixed"
    )
    wrapped = ManiSkillPlanningWrapper(env)

    # 2) Reset with fixed seed
    obs, info = wrapped.reset(seed=0)

    # 3) Pre-sample actions so controller/env RNG doesn’t matter
    action_space = wrapped.action_space
    action_space.seed(0)
    actions = [action_space.sample() for _ in range(N_STEPS)]

    # 4) Snapshot state S
    # If your clone_state signature takes obs, pass `obs` here instead.
    snapshot = wrapped.clone_state()

    # 5) First rollout
    r1, p1, s1 = run_rollout(wrapped, actions)

    # 6) Restore snapshot and re-run
    wrapped.restore_state(snapshot)
    r2, p2, s2 = run_rollout(wrapped, actions)

    # 7) Print differences
    print("=== Replay Smoke Test (Shelf) ===")
    print("len(r1), len(r2):", len(r1), len(r2))
    print("len(p1), len(p2):", len(p1), len(p2))
    print("len(s1), len(s2):", len(s1), len(s2))

    min_len = min(len(r1), len(r2), len(p1), len(p2), len(s1), len(s2))

    print("\nFirst few rewards (run 1 vs run 2):")
    for i in range(min(5, min_len)):
        print(f"  step {i}: {r1[i]: .4f} vs {r2[i]: .4f}")

    print("\nMax |reward diff|:", float(np.max(np.abs(r1[:min_len] - r2[:min_len]))))
    print(
        "Max |obj_pose diff|:",
        float(np.nanmax(np.abs(p1[:min_len] - p2[:min_len]))),
    )
    print("Success sequence equal:", np.array_equal(s1[:min_len], s2[:min_len]))

    wrapped.close()


if __name__ == "__main__":
    main()
