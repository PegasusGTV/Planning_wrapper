#!/usr/bin/env python3
"""
Batched face-push demo collection for PushBoundary.

4-face protocol
---------------
Each block configuration (block at centre, random yaw) is used for exactly 4
episodes — one per cube face (indices 0–3, permuted randomly per group). A new
yaw is only drawn after all 4 faces have been visited.  This gives balanced
coverage of push directions while keeping diverse block orientations.

Gripper spawn is re-sampled independently for each of the 4 episodes, so the
approach angle varies even within a group.

Backends
--------
  --backend cpu   gymnasium AsyncVectorEnv (multiprocessing) — works everywhere
  --backend gpu   native ManiSkill num_envs                  — fastest on GPU

Usage
-----
    python collect_face_push_batch.py --num_envs 8 --num_demos 200 --backend cpu
    python collect_face_push_batch.py --num_envs 32 --num_demos 1000 --backend gpu
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))   # so workers can import collect_face_push

import gymnasium as gym
from mani_skill.utils.wrappers.record import RecordEpisode

import envs.floating_gripper  # noqa: F401

from envs.push_boundary import (
    BOUNDARY_CENTER_X as BCX,
    BOUNDARY_CENTER_Y as BCY,
    BOUNDARY_HALF_X   as BHX,
    BOUNDARY_HALF_Y   as BHY,
)

from collect_face_push import (
    FaceApproachPushPolicy,
    _get_xyz, _get_xyzw_quat, _unwrap,
    _set_gripper_xy, _yaw_to_sapien_quat,
    _obb_support, _wall_dist, _truncated_gaussian,
    GRIPPER_RADIUS, SPAWN_CLEARANCE, GRIPPER_SPAWN_STD, WALL_MARGIN,
    SPAWN_MIN_X, SPAWN_MAX_X, SPAWN_MIN_Y, SPAWN_MAX_Y,
)


# ---------------------------------------------------------------------------
# Per-env configuration tracker
# ---------------------------------------------------------------------------

class EnvConfig:
    """
    Tracks the 4-face rotation for one environment.

    Call next(rng) to get the (block_yaw, face_index) for the next episode.
    A fresh yaw is sampled and a new permutation of [0,1,2,3] is queued only
    when the previous group of 4 is exhausted.
    """

    def __init__(self):
        self._face_queue : list[int] = []
        self.block_yaw   : float     = 0.0

    def next(self, rng: np.random.Generator) -> tuple[float, int]:
        if not self._face_queue:
            self.block_yaw   = rng.uniform(-np.pi, np.pi)
            self._face_queue = rng.permutation(4).tolist()
        return self.block_yaw, self._face_queue.pop(0)

    def remaining(self) -> int:
        return len(self._face_queue)


# ---------------------------------------------------------------------------
# Spawn helpers (called from main process for both backends)
# ---------------------------------------------------------------------------

def _sample_gripper_spawn(hx: float, hy: float, block_yaw: float,
                           rng: np.random.Generator) -> tuple[float, float]:
    """
    Sample a valid gripper spawn (gx, gy) using the same logic as custom_init.
    Returns a deterministic position so the main process knows where the
    gripper will be placed without having to read it back from the worker.
    """
    for _ in range(100):
        angle     = rng.uniform(0.0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        support   = _obb_support(hx, hy, block_yaw, direction)
        wall_d    = _wall_dist(direction)
        lo        = GRIPPER_RADIUS + SPAWN_CLEARANCE
        hi        = wall_d - support - WALL_MARGIN - GRIPPER_RADIUS
        if hi <= lo:
            continue
        extra = _truncated_gaussian(rng, mu=0.0, sigma=GRIPPER_SPAWN_STD,
                                    lo=lo, hi=hi)
        gx = BCX + direction[0] * (support + extra)
        gy = BCY + direction[1] * (support + extra)
        return gx, gy
    raise RuntimeError("Failed to sample valid gripper spawn after 100 attempts")


def _to_np(x) -> np.ndarray:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------------------
# CPU worker wrapper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CPU worker wrapper
# ---------------------------------------------------------------------------

def _make_env_fn(record_dir: str, worker_seed: int, worker_idx: int,
                 hx: float, hy: float):
    """
    Returns a thunk for AsyncVectorEnv.

    The wrapper owns all randomization (4-face queue, spawn sampling) so no
    options need to be passed from the main process.  Instead, each reset()
    returns the chosen (block_yaw, face_index, gx, gy) inside info so the
    main process can initialise the corresponding policy.
    """
    def _thunk():
        import sys
        from pathlib import Path
        _root = Path(__file__).resolve().parent
        sys.path.insert(0, str(_root.parent))
        sys.path.insert(0, str(_root))

        import numpy as np
        import sapien
        import gymnasium as gym
        from mani_skill.utils.wrappers.record import RecordEpisode
        import envs.floating_gripper  # noqa: F401
        import envs                   # noqa: F401 — registers PushBoundary

        from collect_face_push import (
            _yaw_to_sapien_quat, _set_gripper_xy, _get_xyz,
            _obb_support, _wall_dist, _truncated_gaussian,
            BCX, BCY,
            GRIPPER_RADIUS, SPAWN_CLEARANCE, GRIPPER_SPAWN_STD, WALL_MARGIN,
        )

        env = gym.make(
            "PushBoundary",
            obs_mode="state_dict",
            control_mode="floating_vel",
            render_mode=None,
            sim_backend="cpu",
            num_envs=1,
            shape="cube",
            num_extra_blocks=0,
            robot_uids="floating_gripper",
        )

        class _InitWrapper(gym.Wrapper):
            def __init__(self, env):
                super().__init__(env)
                self._rng        = np.random.default_rng(worker_seed)
                self._face_queue : list[int] = []
                self._block_yaw  : float     = 0.0
                self._gripper_z  = None
                self._block_z    = None

            def _base(self):
                e = self.env
                while hasattr(e, "env"):
                    e = e.env
                return e

            def _cache_z(self):
                if self._gripper_z is None:
                    base = self._base()
                    self._gripper_z = float(_get_xyz(base.agent.tcp.pose.p)[2])
                    self._block_z   = float(_get_xyz(base.block.pose.p)[2])

            def _next_config(self):
                if not self._face_queue:
                    self._block_yaw  = self._rng.uniform(-np.pi, np.pi)
                    self._face_queue = self._rng.permutation(4).tolist()
                return self._block_yaw, self._face_queue.pop(0)

            def _sample_spawn(self, block_yaw):
                for _ in range(20):
                    angle     = self._rng.uniform(0.0, 2 * np.pi)
                    direction = np.array([np.cos(angle), np.sin(angle)])
                    support   = _obb_support(hx, hy, block_yaw, direction)
                    wall_d    = _wall_dist(direction)
                    lo        = GRIPPER_RADIUS + SPAWN_CLEARANCE
                    hi        = wall_d - support - WALL_MARGIN - GRIPPER_RADIUS
                    if hi <= lo:
                        continue
                    extra = _truncated_gaussian(self._rng, 0.0, GRIPPER_SPAWN_STD, lo, hi)
                    gx = BCX + direction[0] * (support + extra)
                    gy = BCY + direction[1] * (support + extra)
                    return gx, gy
                return BCX + hx + 0.12, BCY

            def _sq(self, obs):
                if isinstance(obs, dict):
                    return {k: self._sq(v) for k, v in obs.items()}
                a = np.asarray(obs, dtype=np.float32)
                return a[0] if (a.ndim > 0 and a.shape[0] == 1) else a

            def reset(self, **kwargs):
                kwargs.pop("options", None)
                obs, info = self.env.reset(**kwargs)
                self._cache_z()

                block_yaw, face_index = self._next_config()
                gx, gy                = self._sample_spawn(block_yaw)

                base = self._base()

                # Set block pose and zero its velocity so it doesn't drift
                # from residual momentum carried over from the previous episode
                base.block.set_pose(
                    sapien.Pose(p=[BCX, BCY, self._block_z],
                                q=_yaw_to_sapien_quat(block_yaw))
                )
                try:
                    base.block.set_velocity([0.0, 0.0, 0.0])
                    base.block.set_angular_velocity([0.0, 0.0, 0.0])
                except Exception:
                    pass  # kinematic actors don't have set_velocity

                _set_gripper_xy(base, gx, gy, self._gripper_z)

                # Note: we intentionally return the obs from self.env.reset()
                # (the pre-set_pose obs) rather than trying to re-read it here.
                # RecordEpisode (which wraps this class) will call get_state()
                # on the env AFTER this reset() returns, at which point the
                # simulator state already reflects our set_pose calls — so
                # env_states[0] in the h5 file will be correct. The returned
                # obs[0] may lag by one frame, which is handled by skipping
                # frame 0 during h5→wds conversion.
                info["init_block_yaw"]  = float(block_yaw)
                info["init_face_index"] = int(face_index)
                info["init_gx"]         = float(gx)
                info["init_gy"]         = float(gy)

                return self._sq(obs), info

            def step(self, action):
                obs, r, te, tr, info = self.env.step(action)
                r  = float(np.asarray(r,  dtype=np.float32).reshape(-1)[0])
                te = bool(np.asarray(te, dtype=bool).reshape(-1)[0])
                tr = bool(np.asarray(tr, dtype=bool).reshape(-1)[0])
                return self._sq(obs), r, te, tr, info

        # _InitWrapper wraps the raw env first — custom positions are set
        # inside its reset(). RecordEpisode then wraps _InitWrapper, so it
        # calls _InitWrapper.reset() and records state AFTER set_pose runs.
        env = _InitWrapper(env)
        worker_record_dir = str(Path(record_dir) / f"worker_{worker_idx:03d}")
        env = RecordEpisode(
            env,
            output_dir=worker_record_dir,
            save_trajectory=True,
            save_video=False,
            trajectory_name="face_push_batch",
        )
        env.reset()
        return env

    return _thunk


# ---------------------------------------------------------------------------
# GPU path
# ---------------------------------------------------------------------------

def run_gpu(args: argparse.Namespace):
    import torch
    from mani_skill.utils.structs import Pose

    N          = args.num_envs
    rng_master = np.random.default_rng(args.seed)

    env = gym.make(
        "PushBoundary",
        obs_mode="state",
        control_mode="floating_vel",
        render_mode=None,
        sim_backend="gpu",
        num_envs=N,
        shape="cube",
        num_extra_blocks=0,
        robot_uids="floating_gripper",
    )
    env = RecordEpisode(
        env,
        output_dir=args.record_dir,
        save_trajectory=True,
        save_video=False,
        trajectory_name="face_push_batch",
    )
    base_env = _unwrap(env)
    hx, hy   = base_env.block_dims.half_x, base_env.block_dims.half_y

    obs, _ = env.reset(seed=int(rng_master.integers(2**31)))
    gripper_z = float(_to_np(base_env.agent.tcp.pose.p)[0, 2])
    block_z   = float(_to_np(base_env.block.pose.p)[0, 2])

    rngs    = [np.random.default_rng(rng_master.integers(2**31)) for _ in range(N)]
    configs = [EnvConfig() for _ in range(N)]
    policies = [FaceApproachPushPolicy() for _ in range(N)]

    # def init_env_gpu(i: int):
    #     """Apply custom init for env i and reset its policy."""
    #     yaw, face     = configs[i].next(rngs[i])
    #     gx, gy        = _sample_gripper_spawn(hx, hy, yaw, rngs[i])

    #     # Set block pose for env i
    #     p_all = _to_np(base_env.block.pose.p)           # (N, 3)
    #     q_all = _to_np(base_env.block.pose.q)           # (N, 4)
    #     q_new = np.array(_yaw_to_sapien_quat(yaw), dtype=np.float32)
    #     p_all[i] = [BCX, BCY, block_z]
    #     q_all[i] = q_new
    #     base_env.block.set_pose(Pose.create_from_pq(
    #         p=torch.tensor(p_all, device=base_env.device),
    #         q=torch.tensor(q_all, device=base_env.device),
    #     ))

    #     # Set gripper qpos for env i
    #     n_dof = int(base_env.agent.robot.dof[0].item())
    #     qpos    = _to_np(base_env.agent.robot.get_qpos())  # (N, n_dof)
    #     qvel    = np.zeros_like(qpos)
    #     qpos[i, 0] = gx - BCX
    #     qpos[i, 1] = gy - BCY
    #     base_env.agent.robot.set_qpos(
    #         torch.tensor(qpos, device=base_env.device))
    #     base_env.agent.robot.set_qvel(
    #         torch.tensor(qvel, device=base_env.device))

    #     ee_xy    = np.array([gx, gy])
    #     block_xy = np.array([BCX, BCY])
    #     policies[i].reset(ee_xy, block_xy, yaw, hx, hy, rngs[i],
    #                       initial_face=face)
    #     return yaw, face
    
    def init_env_gpu(i: int):
        yaw, face = configs[i].next(rngs[i])
        gx, gy    = _sample_gripper_spawn(hx, hy, yaw, rngs[i])

        # --- block pose + velocity for env i only ---
        p_all = _to_np(base_env.block.pose.p).copy()
        q_all = _to_np(base_env.block.pose.q).copy()
        p_all[i] = [BCX, BCY, block_z]
        q_all[i] = np.array(_yaw_to_sapien_quat(yaw), dtype=np.float32)
        base_env.block.set_pose(Pose.create_from_pq(
            p=torch.tensor(p_all, device=base_env.device),
            q=torch.tensor(q_all, device=base_env.device),
        ))

        # zero block velocity for env i only
        lin_vel = _to_np(base_env.block.linear_velocity).copy()
        ang_vel = _to_np(base_env.block.angular_velocity).copy()
        lin_vel[i] = 0.0
        ang_vel[i] = 0.0
        base_env.block.set_linear_velocity(torch.tensor(lin_vel, device=base_env.device))
        base_env.block.set_angular_velocity(torch.tensor(ang_vel, device=base_env.device))

        # --- gripper qpos/qvel for env i only ---
        qpos = _to_np(base_env.agent.robot.get_qpos()).copy()   # (N, n_dof)
        qvel = _to_np(base_env.agent.robot.get_qvel()).copy()   # (N, n_dof)
        qpos[i, 0] = gx - BCX
        qpos[i, 1] = gy - BCY
        qvel[i]    = 0.0   # zero only env i
        base_env.agent.robot.set_qpos(torch.tensor(qpos, device=base_env.device))
        base_env.agent.robot.set_qvel(torch.tensor(qvel, device=base_env.device))

        ee_xy    = np.array([gx, gy])
        block_xy = np.array([BCX, BCY])
        policies[i].reset(ee_xy, block_xy, yaw, hx, hy, rngs[i], initial_face=face)
        return yaw, face

    # Initial per-env setup
    for i in range(N):
        init_env_gpu(i)

    completed = alive_steps_arr = np.zeros(N, dtype=int)
    alive_steps_arr = np.zeros(N, dtype=int)
    total_steps = 0
    t0 = time.time()

    print(f"GPU batch: {N} envs  target={args.num_demos} demos")

    while completed.sum() < args.num_demos:
        ee_poses    = _to_np(base_env.agent.tcp.pose.p)   # (N, 3)
        block_poses = _to_np(base_env.block.pose.p)        # (N, 3)
        block_quats = _to_np(base_env.block.pose.q)        # (N, 4) [w,x,y,z]

        actions = np.zeros((N, 2), dtype=np.float32)
        for i in range(N):
            q    = block_quats[i]
            bq   = np.array([q[1], q[2], q[3], q[0]])  # [x,y,z,w] for scipy
            byaw = float(Rotation.from_quat(bq).as_euler("xyz")[2])
            actions[i] = policies[i].act(ee_poses[i, :2], block_poses[i, :2], byaw)

        obs, _, terminated, truncated, _ = env.step(actions)
        total_steps += N
        alive_steps_arr += 1

        dones = (_to_np(terminated).reshape(N).astype(bool)
               | _to_np(truncated).reshape(N).astype(bool)
               | (alive_steps_arr >= args.max_episode_steps))

        done_envs = np.where(dones)[0]

        # 1. Logging + bookkeeping (no sim writes yet)
        for i in done_envs:
            completed[i] += 1
            elapsed = time.time() - t0
            n_done  = int(completed.sum())
            print(f"  [{n_done:>5d}/{args.num_demos}]  env={i}  "
                  f"steps={alive_steps_arr[i]}  "
                  f"fps={total_steps/elapsed:.0f}")
            alive_steps_arr[i] = 0

        if int(completed.sum()) >= args.num_demos:
            break

        # 2. Per-env resets first (RecordEpisode bookkeeping only)
        for i in done_envs:
            env.reset(options={"env_idx": torch.tensor([i], device=base_env.device)})

        # 3. Single batched read → modify → write for all done envs
        p_all  = _to_np(base_env.block.pose.p).copy()
        q_all  = _to_np(base_env.block.pose.q).copy()
        qpos   = _to_np(base_env.agent.robot.get_qpos()).copy()
        qvel   = _to_np(base_env.agent.robot.get_qvel()).copy()

        init_params = {}
        for i in done_envs:
            yaw, face  = configs[i].next(rngs[i])
            gx, gy     = _sample_gripper_spawn(hx, hy, yaw, rngs[i])
            p_all[i]   = [BCX, BCY, block_z]
            q_all[i]   = _yaw_to_sapien_quat(yaw)
            qpos[i, 0] = gx - BCX
            qpos[i, 1] = gy - BCY
            qvel[i, :] = 0
            init_params[i] = (yaw, face, gx, gy)

        base_env.block.set_pose(Pose.create_from_pq(
            p=torch.tensor(p_all, device=base_env.device),
            q=torch.tensor(q_all, device=base_env.device)))
        base_env.agent.robot.set_qpos(torch.tensor(qpos, device=base_env.device))
        base_env.agent.robot.set_qvel(torch.tensor(qvel, device=base_env.device))

        # 4. Read back actual TCP after the single write to init policies
        for i in done_envs:
            yaw, face, gx, gy = init_params[i]
            policies[i].reset(np.array([gx, gy]), np.array([BCX, BCY]),
                            yaw, hx, hy, rngs[i], initial_face=face)

    env.close()
    elapsed = time.time() - t0
    print(f"\nDone. {int(completed.sum())} demos  {elapsed:.1f}s  "
          f"({total_steps/elapsed:.0f} steps/s)")


# ---------------------------------------------------------------------------
# CPU path
# ---------------------------------------------------------------------------

def run_cpu(args: argparse.Namespace):
    N          = args.num_envs
    rng_master = np.random.default_rng(args.seed)
    w_seeds    = [int(rng_master.integers(2**31)) for _ in range(N)]

    # Probe block dims before forking
    _probe = gym.make(
        "PushBoundary",
        obs_mode="state", control_mode="floating_vel",
        render_mode=None, sim_backend="cpu",
        num_envs=1, shape="cube", num_extra_blocks=0,
        robot_uids="floating_gripper",
    )
    hx = _probe.unwrapped.block_dims.half_x
    hy = _probe.unwrapped.block_dims.half_y
    _probe.close()

    vec_env = gym.vector.AsyncVectorEnv(
        [_make_env_fn(record_dir=args.record_dir,
                      worker_seed=w_seeds[i], worker_idx=i,
                      hx=hx, hy=hy)
         for i in range(N)],
        context="spawn",
    )

    rngs     = [np.random.default_rng(rng_master.integers(2**31)) for _ in range(N)]
    policies = [FaceApproachPushPolicy() for _ in range(N)]
    block_xy = np.array([BCX, BCY])

    def init_policies_from_info(info):
        # AsyncVectorEnv stacks info values into (N,) arrays keyed by field name
        yaws  = np.asarray(info.get("init_block_yaw",  np.zeros(N)),  dtype=float)
        faces = np.asarray(info.get("init_face_index", np.zeros(N)),  dtype=int)
        gxs   = np.asarray(info.get("init_gx", np.full(N, BCX + 0.12)), dtype=float)
        gys   = np.asarray(info.get("init_gy", np.full(N, BCY)),         dtype=float)
        for i in range(N):
            policies[i].reset(np.array([gxs[i], gys[i]]), block_xy,
                               float(yaws[i]), hx, hy, rngs[i],
                               initial_face=int(faces[i]))

    obs, info = vec_env.reset(seed=w_seeds)
    init_policies_from_info(info)

    completed   = 0
    alive_steps = np.zeros(N, dtype=int)
    total_steps = 0
    t0          = time.time()

    print(f"CPU batch: {N} workers  target={args.num_demos} demos")
    
    while completed < args.num_demos:
        tcp_poses   = np.asarray(obs["extra"]["tcp_pose"],   dtype=np.float32).reshape(N, -1)
        block_poses = np.asarray(obs["extra"]["block_pose"], dtype=np.float32).reshape(N, -1)

        actions = np.zeros((N, 2), dtype=np.float32)
        for i in range(N):
            q    = block_poses[i, 3:7]
            bq   = np.array([q[1], q[2], q[3], q[0]])
            byaw = float(Rotation.from_quat(bq).as_euler("xyz")[2])
            actions[i] = policies[i].act(tcp_poses[i, :2], block_poses[i, :2], byaw)

        # 1. Step the environment. 
        # Any worker that returns terminated or truncated will AUTO-RESET internally.
        obs, rewards, terminated, truncated, info = vec_env.step(actions)
        total_steps  += N
        alive_steps  += 1

        # Calculate dones. (Note: Ensure your environment handles its own max_episode_steps 
        # so it naturally returns truncated=True. If not, auto-reset won't trigger!)
        dones = np.asarray(terminated, bool) | np.asarray(truncated, bool)

        if dones.any():
            # Extract the new reset infos. AsyncVectorEnv stacks these.
            yaws  = np.asarray(info.get("init_block_yaw", np.zeros(N)), dtype=float)
            faces = np.asarray(info.get("init_face_index", np.zeros(N)), dtype=int)
            gxs   = np.asarray(info.get("init_gx", np.full(N, BCX + 0.12)), dtype=float)
            gys   = np.asarray(info.get("init_gy", np.full(N, BCY)), dtype=float)

            # 2. Loop ONLY over the workers that actually finished
            for i in np.where(dones)[0]:
                completed += 1
                elapsed    = time.time() - t0
                print(f"  [{completed:>5d}/{args.num_demos}]  worker={i}  "
                      f"steps={alive_steps[i]}  fps={total_steps/elapsed:.0f}")
                
                alive_steps[i] = 0

                # 3. Reset the policy for this specific worker using the new spawn info
                policies[i].reset(
                    np.array([gxs[i], gys[i]]), 
                    block_xy, 
                    float(yaws[i]), 
                    hx, hy, rngs[i],
                    initial_face=int(faces[i])
                )

            if completed >= args.num_demos:
                break

    # while completed < args.num_demos:
    #     tcp_poses   = np.asarray(obs["extra"]["tcp_pose"],   dtype=np.float32).reshape(N, -1)
    #     block_poses = np.asarray(obs["extra"]["block_pose"], dtype=np.float32).reshape(N, -1)

    #     actions = np.zeros((N, 2), dtype=np.float32)
    #     for i in range(N):
    #         q    = block_poses[i, 3:7]
    #         bq   = np.array([q[1], q[2], q[3], q[0]])
    #         byaw = float(Rotation.from_quat(bq).as_euler("xyz")[2])
    #         actions[i] = policies[i].act(tcp_poses[i, :2], block_poses[i, :2], byaw)

    #     obs, _, terminated, truncated, _ = vec_env.step(actions)
    #     total_steps  += N
    #     alive_steps  += 1

    #     dones = (np.asarray(terminated, bool)
    #            | np.asarray(truncated,  bool)
    #            | (alive_steps >= args.max_episode_steps))

    #     if dones.any():
    #         for i in np.where(dones)[0]:
    #             completed += 1
    #             elapsed    = time.time() - t0
    #             print(f"  [{completed:>5d}/{args.num_demos}]  worker={i}  "
    #                   f"steps={alive_steps[i]}  fps={total_steps/elapsed:.0f}")
    #             alive_steps[i] = 0

    #         if completed >= args.num_demos:
    #             break

    #         # Workers generate fresh init internally; main process reads back via info
    #         obs, info = vec_env.reset()
    #         init_policies_from_info(info)

    vec_env.close()
    elapsed = time.time() - t0
    print(f"\nDone. {completed} demos  {elapsed:.1f}s  ({total_steps/elapsed:.0f} steps/s)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace):
    print(
        f"PushBoundary batched face-push collection\n"
        f"  backend={args.backend}  num_envs={args.num_envs}  "
        f"target_demos={args.num_demos}\n"
        f"  max_episode_steps={args.max_episode_steps}  "
        f"record_dir={args.record_dir}"
    )
    if args.backend == "gpu":
        run_gpu(args)
    else:
        run_cpu(args)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batched face-push demo collection for PushBoundary."
    )
    p.add_argument("--num_envs",          type=int, default=8)
    p.add_argument("--num_demos",         type=int, default=1000)
    p.add_argument("--max_episode_steps", type=int, default=400)
    p.add_argument("--seed",              type=int, default=None)
    p.add_argument("--record_dir",        type=str,
                   default="demos/PushBoundary/face_push_batch_v8")
    p.add_argument("--backend",           type=str, default="cpu",
                   choices=["cpu", "gpu"])
    return p.parse_args()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    run(parse_args())