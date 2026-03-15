#!/usr/bin/env python3
"""
Batched scripted demo collection for PushBoundary.

On GPU  → uses ManiSkill's native num_envs (fastest).
On CPU  → uses gymnasium AsyncVectorEnv across N worker processes,
          since ManiSkill's CPU backend only supports single-env instances.

No rendering is done — purely low-dimensional state collection.

Usage
-----
    # CPU (multiprocessing, works on Mac / no-GPU Linux)
    python scripts/collect_scripted_demos_batch.py \
        --num_envs 8 --num_demos 200 --shape cube --mode mixed

    # GPU (native ManiSkill vectorization, much faster)
    python scripts/collect_scripted_demos_batch.py \
        --num_envs 32 --num_demos 500 --shape T --num_extra_blocks 5 \
        --backend gpu --record_dir demos/PushBoundary/scripted_batch
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import gymnasium as gym
from mani_skill.utils.wrappers.record import RecordEpisode

from scripts.scripted_push import (
    ScriptedPushPolicy,
    _unwrap,
    EE_Z,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_np(x) -> np.ndarray:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Squeeze wrapper — strips the (1,...) batch dim ManiSkill adds to each field
# so AsyncVectorEnv receives plain scalars / 1-D arrays, not (1,) tensors.
# ─────────────────────────────────────────────────────────────────────────────

class _SqueezeWrapper(gym.Wrapper):
    """
    Strips the leading batch dimension that ManiSkill adds even for num_envs=1.

    AsyncVectorEnv calls each worker's step() and expects:
      reward    → Python float  (or 0-D array)
      terminated → Python bool
      truncated  → Python bool
      obs        → plain numpy arrays (no leading 1-dim)

    Without this wrapper ManiSkill returns e.g. reward=(1,) tensor, which
    causes AsyncVectorEnv's np.array(rewards) to get inhomogeneous shapes.
    """

    def _squeeze_obs(self, obs):
        """Recursively squeeze leading size-1 dims from all arrays in obs."""
        if isinstance(obs, dict):
            return {k: self._squeeze_obs(v) for k, v in obs.items()}
        arr = _to_np(obs)
        if arr.ndim > 0 and arr.shape[0] == 1:
            arr = arr[0]    # (1, ...) → (...)
        return arr

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._squeeze_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._squeeze_obs(obs)
        # Scalars — AsyncVectorEnv stacks these across workers
        reward     = float(_to_np(reward).reshape(-1)[0])
        terminated = bool(_to_np(terminated).reshape(-1)[0])
        truncated  = bool(_to_np(truncated).reshape(-1)[0])
        return obs, reward, terminated, truncated, info


# ─────────────────────────────────────────────────────────────────────────────
# GPU path  — native ManiSkill num_envs
# ─────────────────────────────────────────────────────────────────────────────

def run_gpu(args: argparse.Namespace):
    N = args.num_envs
    rng_master = np.random.default_rng(args.seed)

    env = gym.make(
        "PushBoundary",
        obs_mode="state_dict",   # dict so we can read extra keys cleanly
        control_mode="pd_ee_delta_pose",
        render_mode=None,        # headless — no rendering needed
        sim_backend="gpu",
        num_envs=N,
        shape=args.shape,
        num_extra_blocks=args.num_extra_blocks,
    )
    env = RecordEpisode(
        env,
        output_dir=args.record_dir,
        save_trajectory=True,
        save_video=False,        # never render/save video during batch collection
        trajectory_name="scripted_batch",
    )

    base_env = _unwrap(env)
    block_half_width = max(base_env.block_dims.half_x, base_env.block_dims.half_y)

    rngs     = [np.random.default_rng(rng_master.integers(0, 2**32)) for _ in range(N)]
    policies = [
        ScriptedPushPolicy(rng=rngs[i], mode=args.mode, block_half_width=block_half_width)
        for i in range(N)
    ]

    seeds = [int(rng_master.integers(0, 2**32)) for _ in range(N)]
    obs, _info = env.reset(seed=seeds)
    for p in policies:
        p.reset()

    completed   = 0
    total_steps = 0
    t0          = time.time()

    while completed < args.num_demos:
        # Read directly from base_env — avoids any dict parsing overhead
        ee_poses    = _to_np(base_env.agent.tcp.pose.p)  # (N, 3)
        block_poses = _to_np(base_env.block.pose.p)       # (N, 3)

        # raw_pose is (1, 7) per single-env worker, so AsyncVectorEnv stacks
        # to (N, 1, 7). Flatten each pose to (7,) so scalar indexing works
        # and the policy doesn't get ambiguous boolean comparisons.
        ee_poses    = ee_poses.reshape(N, -1)     # (N, 7)
        block_poses = block_poses.reshape(N, -1)  # (N, 7)

        actions = np.zeros((N, 6), dtype=np.float32)
        for i in range(N):
            ee_xyz    = ee_poses[i, :3]    # (3,) — plain floats now
            block_xyz = block_poses[i, :3] # (3,)
            actions[i]    = policies[i].act(ee_xy=ee_xyz[:2], block_xy=block_xyz[:2])
            actions[i, 2] = -(float(ee_xyz[2]) - EE_Z)

        obs, rewards, terminated, truncated, info = env.step(actions)
        total_steps += N

        dones = _to_np(terminated).reshape(N).astype(bool) | \
                _to_np(truncated).reshape(N).astype(bool)

        for i in np.where(dones)[0]:
            completed += 1
            elapsed = time.time() - t0
            print(f"  [{completed:>5d}/{args.num_demos}]  env={i}  "
                  f"fps={total_steps/elapsed:.0f}  elapsed={elapsed:.1f}s")
            policies[i].reset()
            if completed >= args.num_demos:
                break

    env.close()
    elapsed = time.time() - t0
    print(f"\nDone. {completed} demos in {elapsed:.1f}s  "
          f"({total_steps/elapsed:.0f} steps/s)")


# ─────────────────────────────────────────────────────────────────────────────
# CPU path  — gymnasium AsyncVectorEnv (one process per env)
# ─────────────────────────────────────────────────────────────────────────────

def _make_env_fn(shape: str, num_extra_blocks: int,
                 record_dir: str, worker_seed: int, worker_idx: int):
    """Returns a thunk that creates one wrapped single env (for AsyncVectorEnv)."""
    def _thunk():
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

        import gymnasium as gym
        from mani_skill.utils.wrappers.record import RecordEpisode
        import envs  # noqa: F401 — registers PushBoundary

        env = gym.make(
            "PushBoundary",
            obs_mode="state_dict",   # need dict obs to extract tcp + block pose
            control_mode="pd_ee_delta_pose",
            render_mode=None,        # headless — no rendering
            sim_backend="cpu",
            num_envs=1,
            shape=shape,
            num_extra_blocks=num_extra_blocks,
        )
        # Each worker writes into its own sub-folder to avoid HDF5 write conflicts.
        worker_record_dir = str(Path(record_dir) / f"worker_{worker_idx:03d}")
        env = RecordEpisode(
            env,
            output_dir=worker_record_dir,
            save_trajectory=True,
            save_video=False,        # never render/save video
            trajectory_name="scripted_batch",
        )
        # Strip ManiSkill's (1,...) batch dim so AsyncVectorEnv gets scalars.
        # Defined inline so the subprocess worker doesn't need to re-import
        # this module (which can fail depending on __main__ guard).
        import gymnasium as _gym
        import numpy as _np

        def _to_np_local(x):
            try:
                import torch
                if isinstance(x, torch.Tensor):
                    x = x.cpu().numpy()
            except ImportError:
                pass
            return _np.asarray(x, dtype=_np.float32)

        class _SQ(_gym.Wrapper):
            def _sq(self, obs):
                if isinstance(obs, dict):
                    return {k: self._sq(v) for k, v in obs.items()}
                a = _to_np_local(obs)
                return a[0] if (a.ndim > 0 and a.shape[0] == 1) else a
            def reset(self, **kw):
                o, i = self.env.reset(**kw); return self._sq(o), i
            def step(self, action):
                o, r, te, tr, i = self.env.step(action)
                return (self._sq(o),
                        float(_to_np_local(r).reshape(-1)[0]),
                        bool(_to_np_local(te).reshape(-1)[0]),
                        bool(_to_np_local(tr).reshape(-1)[0]),
                        i)

        env = _SQ(env)
        env.reset(seed=worker_seed)
        return env

    return _thunk


def run_cpu(args: argparse.Namespace):
    N = args.num_envs
    rng_master = np.random.default_rng(args.seed)
    worker_seeds = [int(rng_master.integers(0, 2**32)) for _ in range(N)]

    print(f"  Using AsyncVectorEnv with {N} worker processes …")

    # Probe one single env just to read block_dims, then close it.
    import envs  # noqa: F401
    _probe = gym.make(
        "PushBoundary",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        render_mode=None,
        sim_backend="cpu",
        num_envs=1,
        shape=args.shape,
        num_extra_blocks=args.num_extra_blocks,
    )
    block_half_width = max(_probe.unwrapped.block_dims.half_x,
                           _probe.unwrapped.block_dims.half_y)
    _probe.close()

    vec_env = gym.vector.AsyncVectorEnv([
        _make_env_fn(
            shape            = args.shape,
            num_extra_blocks = args.num_extra_blocks,
            record_dir       = args.record_dir,
            worker_seed      = worker_seeds[i],
            worker_idx       = i,
        )
        for i in range(N)
    ])

    rngs     = [np.random.default_rng(rng_master.integers(0, 2**32)) for _ in range(N)]
    policies = [
        ScriptedPushPolicy(rng=rngs[i], mode=args.mode, block_half_width=block_half_width)
        for i in range(N)
    ]

    obs, _info = vec_env.reset(seed=worker_seeds)
    for p in policies:
        p.reset()

    completed   = 0
    total_steps = 0
    t0          = time.time()

    while completed < args.num_demos:
        # AsyncVectorEnv stacks obs → each field is (N, ...) numpy array.
        # _get_obs_extra puts tcp_pose and block_pose under obs["extra"].
        ee_poses    = np.asarray(obs["extra"]["tcp_pose"],   dtype=np.float32)  # (N, 7)
        block_poses = np.asarray(obs["extra"]["block_pose"], dtype=np.float32)  # (N, 7)

        # raw_pose is (1, 7) per single-env worker so AsyncVectorEnv stacks
        # to (N, 1, 7). Flatten to (N, 7) so scalar indexing gives plain (3,)
        # arrays and the policy's boolean comparisons don't ambiguously fail.
        ee_poses    = ee_poses.reshape(N, -1)     # (N, 7)
        block_poses = block_poses.reshape(N, -1)  # (N, 7)

        actions = np.zeros((N, 6), dtype=np.float32)
        for i in range(N):
            ee_xyz    = ee_poses[i, :3]    # (3,) plain float array
            block_xyz = block_poses[i, :3] # (3,) plain float array
            actions[i]    = policies[i].act(ee_xyz[:2], block_xyz[:2])
            actions[i, 2] = -(float(ee_xyz[2]) - EE_Z)

        obs, rewards, terminated, truncated, info = vec_env.step(actions)
        total_steps += N

        dones = np.asarray(terminated, bool) | np.asarray(truncated, bool)

        for i in np.where(dones)[0]:
            completed += 1
            elapsed = time.time() - t0
            print(f"  [{completed:>5d}/{args.num_demos}]  worker={i}  "
                  f"fps={total_steps/elapsed:.0f}  elapsed={elapsed:.1f}s")
            policies[i].reset()
            if completed >= args.num_demos:
                break

    vec_env.close()
    elapsed = time.time() - t0
    print(f"\nDone. {completed} demos saved under '{args.record_dir}/worker_*/' "
          f"in {elapsed:.1f}s  ({total_steps/elapsed:.0f} steps/s)")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace):
    print(
        f"PushBoundary batched collection\n"
        f"  shape={args.shape}  extra_blocks={args.num_extra_blocks}  "
        f"mode={args.mode}  backend={args.backend}\n"
        f"  num_envs={args.num_envs}  target_demos={args.num_demos}\n"
        f"  record_dir={args.record_dir}"
    )
    if args.backend == "gpu":
        run_gpu(args)
    else:
        # cpu or auto: ManiSkill CPU backend raises RuntimeError for num_envs > 1,
        # so we always use AsyncVectorEnv (multiprocessing) on this path.
        run_cpu(args)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batched scripted PushBoundary demo collection"
    )
    # ── same env args as scripted_push.py ──
    p.add_argument("--shape",            type=str,   default="cube",
                   choices=["cube", "T"])
    p.add_argument("--num_extra_blocks", type=int,   default=3)
    p.add_argument("--mode",             type=str,   default="mixed",
                   choices=["standard", "direct", "mixed"])
    p.add_argument("--record_dir",       type=str,
                   default="demos/PushBoundary/scripted_batch")
    p.add_argument("--seed",             type=int,   default=None)

    # ── batch args ──
    p.add_argument("--num_envs",   type=int,   default=8,
                   help="Parallel environments (default: 8).")
    p.add_argument("--num_demos",  type=int,   default=200,
                   help="Stop after this many complete episodes (default: 200).")
    p.add_argument("--backend",    type=str,   default="auto",
                   choices=["auto", "cpu", "gpu"],
                   help="gpu uses ManiSkill native vectorization; "
                        "cpu/auto uses AsyncVectorEnv multiprocessing (default: auto).")

    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())