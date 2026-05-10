"""
Closed-loop executor: generate subgoals → execute skill → re-plan → repeat.

Push-O (O-shaped disk) with wall obstacles. Default env: PushO-WallObstacles-v1.

Usage:
    python high_level_planner/executor.py --seed 42 --reach-checkpoint <run_name>
    python high_level_planner/executor.py --seed 42 --skill mpc
    python high_level_planner/executor.py --skill auto --reach-checkpoint <run_name>
        # reach: compares lookahead_reach_mppi_score vs lookahead_rl_score (skills.metrics), then MPC or PPO

Checkpoint args accept either a full path or just a run name under checkpoints/{name}/final_ckpt.pt.
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

_CHECKPOINTS_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def _resolve_checkpoint(ckpt: str | None) -> str | None:
    """Accept a full path or just a run name — expand the latter to checkpoints/{name}/final_ckpt.pt."""
    if ckpt is None:
        return None
    p = Path(ckpt)
    if p.exists():
        return str(p)
    candidate = _CHECKPOINTS_DIR / ckpt / "final_ckpt.pt"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(f"Checkpoint not found: {ckpt!r} (tried {candidate})")

import datetime

import numpy as np
import torch

# Make project root and all skill dirs importable
_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "skills"))
sys.path.insert(0, os.path.join(_ROOT, "skills", "reach"))
sys.path.insert(0, os.path.join(_ROOT, "skills", "pick"))
sys.path.insert(0, os.path.join(_ROOT, "skills", "place"))
# sys.path.insert(0, os.path.join(_ROOT, "skills", "push_cube"))
sys.path.insert(0, os.path.join(_ROOT, "skills", "push_o"))

import envs  # noqa: F401 — registers ManiSkill envs
import gymnasium as gym
from planning_wrapper.adapters import PushOTaskAdapter
from planning_wrapper.wrappers.maniskill_planning import ManiSkillPlanningWrapper
from mpc_base import get_ee_pos
from skills.metrics import (
    lookahead_reach_mppi_score,
    lookahead_rl_score,
    lookahead_rollout_score,
    select_reach_backend,
)
from ppo_base import load_agent

from llm_plan import region_to_xy, place_fallback_candidates
from env_subgoal_runner import subgoals_from_wrapper
from reach_mpc import execute as mpc_execute
from reach_ppo import _build_flat_obs, execute as ppo_execute
from pick_cube_ppo import _build_pick_obs, execute as pick_ppo_execute
from pick_cube_mpc import PickMPCPreviewSession, execute as pick_mpc_execute
from place_cube_ppo import _build_place_obs, execute as place_ppo_execute
from place_cube_mpc import PlaceMPCPreviewSession, execute as place_mpc_execute
# from push_cube_ppo import _build_push_cube_obs, execute as push_cube_ppo_execute
# from push_cube_mpc import PushCubeMPCPreviewSession, execute as push_cube_mpc_execute
from push_o_ppo import execute as push_o_ppo_execute
from push_o_mpc import PushOMPCPreviewSession, execute as push_o_mpc_execute


def _fmt_m(m: dict) -> str:
    return (f"(M:{m.get('manip_norm', 0):.2f} "
            f"P:{m.get('progress_norm', 0):.2f} "
            f"C:{m.get('clearance_norm', 0):.2f})")


def _parse_region(state_str: str) -> str | None:
    """Extract the region name from a PDDL atom, e.g. '(robot-at robot1 r_6_3)' → 'r_6_3'."""
    m = re.search(r"(r_\d+_\d+)", state_str)
    return m.group(1) if m else None


def _parse_block(state_str: str) -> int | None:
    """Extract obstacle index from a PDDL atom, e.g. '(holding robot1 obstacle3)' → 3."""
    m = re.search(r"obstacle(\d+)", state_str)
    return int(m.group(1)) if m else None


def _reach_policy_act(goal_xyz: np.ndarray, agent, env, device: str):
    """Single-step policy callable for lookahead_rl_score (matches reach_ppo action clamping)."""
    action_low = torch.from_numpy(env.action_space.low.reshape(-1)).to(device)
    action_high = torch.from_numpy(env.action_space.high.reshape(-1)).to(device)
    g = goal_xyz.astype(np.float32)

    def policy_act(obs):
        flat = _build_flat_obs(obs, g)
        obs_t = torch.from_numpy(flat).float().unsqueeze(0).to(device)
        with torch.no_grad():
            return torch.clamp(agent.get_action(obs_t, deterministic=True), action_low, action_high)

    return policy_act


def _pick_ppo_policy_act(block_idx: int, agent, env, device: str):
    raw = env.unwrapped
    obstacle = raw.obstacles[block_idx]
    action_low = torch.from_numpy(env.action_space.low.reshape(-1)).to(device)
    action_high = torch.from_numpy(env.action_space.high.reshape(-1)).to(device)

    def policy_act(obs):
        flat = _build_pick_obs(obs, raw, obstacle)
        obs_t = torch.from_numpy(flat).float().unsqueeze(0).to(device)
        with torch.no_grad():
            return torch.clamp(agent.get_action(obs_t, deterministic=True), action_low, action_high)

    return policy_act


def _place_ppo_policy_act(block_idx: int, goal_xyz: np.ndarray, agent, env, device: str):
    raw = env.unwrapped
    obstacle = raw.obstacles[block_idx]
    g = np.asarray(goal_xyz, dtype=np.float32).reshape(3)
    action_low = torch.from_numpy(env.action_space.low.reshape(-1)).to(device)
    action_high = torch.from_numpy(env.action_space.high.reshape(-1)).to(device)

    def policy_act(obs):
        flat = _build_place_obs(obs, raw, obstacle, g)
        obs_t = torch.from_numpy(flat).float().unsqueeze(0).to(device)
        with torch.no_grad():
            return torch.clamp(agent.get_action(obs_t, deterministic=True), action_low, action_high)

    return policy_act


# def _push_cube_ppo_policy_act(block_idx: int, goal_xyz: np.ndarray, agent, env, device: str):
#     raw = env.unwrapped
#     obstacle = raw.obstacles[block_idx]
#     g = np.asarray(goal_xyz, dtype=np.float32).reshape(3)
#     action_low = torch.from_numpy(env.action_space.low.reshape(-1)).to(device)
#     action_high = torch.from_numpy(env.action_space.high.reshape(-1)).to(device)

#     def policy_act(obs):
#         flat = _build_push_cube_obs(obs, raw, obstacle, g)
#         obs_t = torch.from_numpy(flat).float().unsqueeze(0).to(device)
#         with torch.no_grad():
#             return torch.clamp(agent.get_action(obs_t, deterministic=True), action_low, action_high)

#     return policy_act


def _push_o_ppo_policy_act(goal_xyz: np.ndarray, agent, env, device: str):
    from push_o_ppo import _build_flat_obs as _build_push_o_flat

    raw = env.unwrapped
    g = np.asarray(goal_xyz, dtype=np.float32).reshape(3)
    action_low = torch.from_numpy(env.action_space.low.reshape(-1)).to(device)
    action_high = torch.from_numpy(env.action_space.high.reshape(-1)).to(device)

    def policy_act(obs):
        flat = _build_push_o_flat(obs, raw, g)
        obs_t = torch.from_numpy(flat).float().unsqueeze(0).to(device)
        with torch.no_grad():
            return torch.clamp(agent.get_action(obs_t, deterministic=True), action_low, action_high)

    return policy_act


class _VideoRecorder:
    """Wraps a gym env and captures an rgb_array frame after every step/reset."""

    def __init__(self, env, video_path: str):
        self._env = env
        self._video_path = video_path
        self._frames: list = []

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _capture(self):
        frame = self._env.render()
        if frame is None:
            return
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        f = np.asarray(frame)
        if f.ndim == 4:  # ManiSkill batched: (N, H, W, C) → take first
            f = f[0]
        self._frames.append(f.copy())

    def step(self, action):
        result = self._env.step(action)
        self._capture()
        return result

    def reset(self, **kwargs):
        result = self._env.reset(**kwargs)
        self._capture()
        return result

    def close(self):
        self._env.close()

    def save_video(self):
        if not self._frames:
            print("[video] No frames captured — nothing saved.")
            return
        try:
            import imageio
        except ImportError:
            raise ImportError("imageio is required for video capture: pip install imageio[ffmpeg]")
        Path(self._video_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(self._video_path, self._frames, fps=20)
        print(f"[video] Saved {len(self._frames)} frames → {self._video_path}")


def run(
    seed: int = 0,
    max_replans: int = 10,
    offline: bool = False,
    model: str = "gemini-2.5-flash",
    render: bool = False,
    skill: str = "ppo",
    env_id: str = "PushO-WallObstacles-v1",
    checkpoint: str | None = None,
    pick_checkpoint: str | None = None,
    place_checkpoint: str | None = None,
    # push_cube_checkpoint: str | None = None,
    push_o_checkpoint: str | None = None,
    reach_device: str | None = None,
    capture_video: bool = False,
    video_dir: str = "./videos",
):
    control_mode = "pd_ee_delta_pose"
    dev = reach_device or ("cuda" if torch.cuda.is_available() else "cpu")
    skill_agents: dict = {"reach": None, "pick": None, "place": None, "push_cube": None, "push_o": None}

    # render_mode: rgb_array for video capture, human for live viewer, else None
    if capture_video:
        render_mode = "rgb_array"
        effective_render = False  # no separate viewer window when recording
    elif render:
        render_mode = "human"
        effective_render = True
    else:
        render_mode = None
        effective_render = False

    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode="state_dict",
        control_mode=control_mode,
        sim_backend="physx_cpu",
        render_mode=render_mode,
    )
    if effective_render:
        _orig_render = env.render
        env.render = lambda *a, **kw: (_orig_render(*a, **kw), time.sleep(0.05))[0]

    if capture_video:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = str(Path(video_dir) / f"run_seed{seed}_{timestamp}.mp4")
        env = _VideoRecorder(env, video_path)
        print(f"[video] Recording to {video_path}")

    wrapper = ManiSkillPlanningWrapper(env, adapter=PushOTaskAdapter())
    obs, _ = wrapper.reset(seed=seed)

    _init_ee = np.asarray(obs["extra"]["tcp_pose"], dtype=np.float32).reshape(-1)[:3].copy()

    pick_fail_counts: dict[int, int] = {}
    place_fail_counts: dict[int, int] = {}
    stuck_obstacles: set[int] = set()

    task_succeeded = False
    give_up = False
    n_replans_used = 0
    skill_fail_counts: dict[str, int] = {}
    for replan_i in range(max_replans):
        n_replans_used = replan_i + 1
        if give_up:
            break
        print(f"\n--- Plan {replan_i + 1}/{max_replans} ---")
        if stuck_obstacles:
            print(f"  [info] treating as immovable: {sorted(f'obstacle{i}' for i in stuck_obstacles)}")
        temperature = min(replan_i * 0.2, 1.0)
        problem_str, subgoals = subgoals_from_wrapper(
            wrapper, obs, offline=offline, model=model, stuck_obstacles=stuck_obstacles,
            temperature=temperature,
        )

        if not subgoals:
            print("No subgoals returned — task complete or unreachable.")
            break

        skill_failed = False
        for sg in subgoals:
            sg_skill = sg["skill"]
            state    = sg["state"]
            region   = _parse_region(state)
            block_idx = _parse_block(state)
            print(f"  {sg_skill}\t{state}")

            if sg_skill == "reach":
                if region is None:
                    print(f"    [WARN] could not parse region from: {state!r}")
                    skill_failed = True
                    break
                ee_z = float(
                    np.asarray(obs["extra"]["tcp_pose"], dtype=np.float32).reshape(-1)[2]
                )
                x, y = region_to_xy(region)
                goal_xyz = np.array([x, y, max(ee_z, 0.10)], dtype=np.float32)
                print(f"    → reach {np.round(goal_xyz, 3)}")

                if skill == "auto":
                    if checkpoint is None:
                        print("    [ERR] --skill auto requires --reach-checkpoint")
                        skill_failed = True
                        break
                    if skill_agents["reach"] is None:
                        skill_agents["reach"] = load_agent(checkpoint, dev)
                    rag = skill_agents["reach"]
                    ms, mi = lookahead_reach_mppi_score(wrapper, obs, goal_xyz)
                    policy_act = _reach_policy_act(goal_xyz, rag, env, dev)
                    rs, ri = lookahead_rl_score(wrapper, goal_xyz, policy_act, obs)
                    choice = select_reach_backend(ms, rs)
                    method = "reach_mpc" if choice == "planner" else "reach_ppo"
                    print(f"    [metrics] mpc={ms:.3f} {_fmt_m(mi)} | ppo={rs:.3f} {_fmt_m(ri)} → {method}")
                    if choice == "planner":
                        success, obs = mpc_execute(env, obs, goal_xyz, render=effective_render)
                    else:
                        success, obs = ppo_execute(
                            env, obs, goal_xyz, checkpoint=checkpoint, render=effective_render, device=dev, agent=rag
                        )
                elif skill == "ppo":
                    if skill_agents["reach"] is None:
                        skill_agents["reach"] = load_agent(checkpoint, dev)
                    success, obs = ppo_execute(
                        env, obs, goal_xyz, checkpoint=checkpoint, render=effective_render, device=dev, agent=skill_agents["reach"]
                    )
                else:
                    success, obs = mpc_execute(env, obs, goal_xyz, render=effective_render)

                print(f"    → {'OK' if success else 'FAIL'}")
                if not success:
                    skill_failed = True
                    break

            elif sg_skill == "pick":
                if block_idx is None:
                    print(f"    [WARN] could not parse obstacle index from: {state!r}")
                    skill_failed = True
                    break
                print(f"    → pick cube{block_idx}")

                if skill == "auto":
                    if pick_checkpoint is None:
                        print("    [SKIP] pick: no --pick-checkpoint provided — re-planning")
                        skill_failed = True
                        break
                    if skill_agents["pick"] is None:
                        skill_agents["pick"] = load_agent(pick_checkpoint, dev)
                    raw_e = env.unwrapped

                    def _pick_prog(o, inf):
                        ee = get_ee_pos(o)
                        c = raw_e.obstacles[block_idx].pose.p.cpu().numpy().reshape(3)
                        return float(np.linalg.norm(ee - c))

                    mpc_sess = PickMPCPreviewSession(env, block_idx)
                    mpc_sess.reset()
                    ms, mi = lookahead_rollout_score(
                        wrapper, lambda o: mpc_sess.step_action(o), obs, _pick_prog
                    )
                    ppo_act = _pick_ppo_policy_act(block_idx, skill_agents["pick"], env, dev)
                    rs, ri = lookahead_rollout_score(wrapper, ppo_act, obs, _pick_prog)
                    choice = select_reach_backend(ms, rs)
                    method = "pick_cube_mpc" if choice == "planner" else "pick_cube_ppo"
                    print(f"    [metrics] mpc={ms:.3f} {_fmt_m(mi)} | ppo={rs:.3f} {_fmt_m(ri)} → {method}")
                    if choice == "planner":
                        success, obs = pick_mpc_execute(env, obs, block_idx, render=effective_render)
                    else:
                        success, obs = pick_ppo_execute(
                            env, obs, block_idx, checkpoint=pick_checkpoint, render=effective_render, device=dev, agent=skill_agents["pick"]
                        )
                elif skill == "ppo":
                    if pick_checkpoint is None:
                        print("    [SKIP] pick: no --pick-checkpoint provided — re-planning")
                        skill_failed = True
                        break
                    if skill_agents["pick"] is None:
                        skill_agents["pick"] = load_agent(pick_checkpoint, dev)
                    success, obs = pick_ppo_execute(
                        env, obs, block_idx, checkpoint=pick_checkpoint, render=effective_render, device=dev, agent=skill_agents["pick"]
                    )
                else:
                    success, obs = pick_mpc_execute(
                        env, obs, block_idx, render=effective_render
                    )

                print(f"    → {'OK' if success else 'FAIL'}")
                if not success:
                    pick_fail_counts[block_idx] = pick_fail_counts.get(block_idx, 0) + 1
                    if pick_fail_counts[block_idx] >= 2:
                        stuck_obstacles.add(block_idx)
                        print(f"    [WARN] pick obstacle{block_idx} failed {pick_fail_counts[block_idx]}x — marking as immovable")
                    skill_failed = True
                    break

            elif sg_skill == "place":
                if block_idx is None or region is None:
                    print(f"    [WARN] could not parse block/region from: {state!r}")
                    skill_failed = True
                    break

                if skill in ("auto", "ppo"):
                    if place_checkpoint is None:
                        print("    [SKIP] place: no --place-checkpoint provided — re-planning")
                        skill_failed = True
                        break
                    if skill_agents["place"] is None:
                        skill_agents["place"] = load_agent(place_checkpoint, dev)

                px, py = region_to_xy(region)
                goal_xyz = np.array([px, py, 0.02], dtype=np.float32)
                print(f"    → place cube{block_idx} at {np.round(goal_xyz, 3)}")
                success = False
                force_mpc_place = skill == "auto" and place_fail_counts.get(block_idx, 0) >= 1

                if force_mpc_place:
                    print(f"    [retry] place cube{block_idx} failed before — forcing place_cube_mpc")
                    success, obs = place_mpc_execute(env, obs, block_idx, goal_xyz, render=effective_render)
                elif skill == "auto":
                    raw_e = env.unwrapped
                    g = goal_xyz.copy()

                    def _place_prog(o, inf, _bi=block_idx, _g=g, _re=raw_e):
                        cube = _re.obstacles[_bi].pose.p.cpu().numpy().reshape(3)
                        return float(np.linalg.norm(cube[:2] - _g[:2]) + 0.3 * abs(cube[2] - _g[2]))

                    mpc_sess = PlaceMPCPreviewSession(env, block_idx, goal_xyz)
                    mpc_sess.reset()
                    ms, mi = lookahead_rollout_score(
                        wrapper, lambda o, _s=mpc_sess: _s.step_action(o), obs, _place_prog
                    )
                    ppo_act = _place_ppo_policy_act(block_idx, goal_xyz, skill_agents["place"], env, dev)
                    rs, ri = lookahead_rollout_score(wrapper, ppo_act, obs, _place_prog)
                    choice = select_reach_backend(ms, rs)
                    method = "place_cube_mpc" if choice == "planner" else "place_cube_ppo"
                    print(f"    [metrics] mpc={ms:.3f} {_fmt_m(mi)} | ppo={rs:.3f} {_fmt_m(ri)} → {method}")
                    if choice == "planner":
                        success, obs = place_mpc_execute(env, obs, block_idx, goal_xyz, render=effective_render)
                    else:
                        success, obs = place_ppo_execute(
                            env, obs, block_idx, goal_xyz, checkpoint=place_checkpoint, render=effective_render, device=dev, agent=skill_agents["place"]
                        )
                elif skill == "ppo":
                    success, obs = place_ppo_execute(
                        env, obs, block_idx, goal_xyz, checkpoint=place_checkpoint, render=effective_render, device=dev, agent=skill_agents["place"]
                    )
                else:
                    success, obs = place_mpc_execute(
                        env, obs, block_idx, goal_xyz, render=effective_render
                    )

                print(f"    → {'OK' if success else 'FAIL'}")

                if not success:
                    place_fail_counts[block_idx] = place_fail_counts.get(block_idx, 0) + 1
                    print("    [WARN] place failed — opening gripper and replanning")
                    act_dim = env.action_space.shape[0]
                    open_action = np.zeros(act_dim, dtype=np.float32)
                    open_action[-1] = 1.0  # open gripper
                    for _ in range(15):
                        obs, _, _, _, _ = env.step(open_action)
                    skill_failed = True
                    break

            elif sg_skill == "push_cube":
                print("    [SKIP] push_cube skill disabled — re-planning")
                skill_failed = True
                break
                # if block_idx is None or region is None:
                #     print(f"    [WARN] could not parse block/region from: {state!r}")
                #     skill_failed = True
                #     break
                # x, y = region_to_xy(region)
                # goal_xyz = np.array([x, y, 0.0], dtype=np.float32)
                # print(f"    → push_cube obstacle{block_idx} to {np.round(goal_xyz, 3)}")

                # if skill == "auto":
                #     if push_cube_checkpoint is None:
                #         print("    [SKIP] push_cube: no --push-cube-checkpoint provided — re-planning")
                #         skill_failed = True
                #         break
                #     if skill_agents["push_cube"] is None:
                #         skill_agents["push_cube"] = load_agent(push_cube_checkpoint, dev)
                #     g = np.asarray(goal_xyz, dtype=np.float32).reshape(3)
                #     raw_e = env.unwrapped

                #     def _pc_prog(o, inf):
                #         cube = raw_e.obstacles[block_idx].pose.p.cpu().numpy().reshape(3)
                #         return float(np.linalg.norm(cube[:2] - g[:2]))

                #     mpc_sess = PushCubeMPCPreviewSession(env, block_idx, goal_xyz)
                #     mpc_sess.reset()
                #     ms, mi = lookahead_rollout_score(
                #         wrapper, lambda o: mpc_sess.step_action(o), obs, _pc_prog
                #     )
                #     ppo_act = _push_cube_ppo_policy_act(block_idx, goal_xyz, skill_agents["push_cube"], env, dev)
                #     rs, ri = lookahead_rollout_score(wrapper, ppo_act, obs, _pc_prog)
                #     choice = select_reach_backend(ms, rs)
                #     method = "push_cube_mpc" if choice == "planner" else "push_cube_ppo"
                #     print(f"    [metrics] mpc={ms:.3f} {_fmt_m(mi)} | ppo={rs:.3f} {_fmt_m(ri)} → {method}")
                #     if choice == "planner":
                #         success, obs = push_cube_mpc_execute(env, obs, block_idx, goal_xyz, render=effective_render)
                #     else:
                #         success, obs = push_cube_ppo_execute(
                #             env, obs, block_idx, goal_xyz, checkpoint=push_cube_checkpoint, render=effective_render, device=dev, agent=skill_agents["push_cube"]
                #         )
                # elif skill == "ppo":
                #     if push_cube_checkpoint is None:
                #         print("    [SKIP] push_cube: no --push-cube-checkpoint provided — re-planning")
                #         skill_failed = True
                #         break
                #     if skill_agents["push_cube"] is None:
                #         skill_agents["push_cube"] = load_agent(push_cube_checkpoint, dev)
                #     success, obs = push_cube_ppo_execute(
                #         env, obs, block_idx, goal_xyz, checkpoint=push_cube_checkpoint, render=effective_render, device=dev, agent=skill_agents["push_cube"]
                #     )
                # else:
                #     success, obs = push_cube_mpc_execute(
                #         env, obs, block_idx, goal_xyz, render=effective_render
                #     )
                # print(f"    → {'OK' if success else 'FAIL'}")
                # if not success:
                #     skill_failed = True
                #     break

            elif sg_skill == "push_disk":
                if region is None:
                    print(f"    [WARN] could not parse region from: {state!r}")
                    skill_failed = True
                    break
                x, y = region_to_xy(region)
                goal_xyz = np.array([x, y, 0.0], dtype=np.float32)
                print(f"    → push_disk to {np.round(goal_xyz, 3)}")

                if skill == "auto":
                    if push_o_checkpoint is None:
                        print("    [SKIP] push_disk: no --push-o-checkpoint provided — re-planning")
                        skill_failed = True
                        break
                    if skill_agents["push_o"] is None:
                        skill_agents["push_o"] = load_agent(push_o_checkpoint, dev)
                    g = np.asarray(goal_xyz, dtype=np.float32).reshape(3)
                    raw_e = env.unwrapped

                    def _pod_prog(o, inf):
                        d = raw_e.disk.pose.p.cpu().numpy().reshape(3)
                        return float(np.linalg.norm(d[:2] - g[:2]))

                    mpc_sess = PushOMPCPreviewSession(env, goal_xyz)
                    mpc_sess.reset()
                    ms, mi = lookahead_rollout_score(
                        wrapper, lambda o: mpc_sess.step_action(o), obs, _pod_prog
                    )
                    ppo_act = _push_o_ppo_policy_act(goal_xyz, skill_agents["push_o"], env, dev)
                    rs, ri = lookahead_rollout_score(wrapper, ppo_act, obs, _pod_prog)
                    choice = select_reach_backend(ms, rs)
                    method = "push_o_mpc" if choice == "planner" else "push_o_ppo"
                    print(f"    [metrics] mpc={ms:.3f} {_fmt_m(mi)} | ppo={rs:.3f} {_fmt_m(ri)} → {method}")
                    if choice == "planner":
                        success, obs = push_o_mpc_execute(env, obs, goal_xyz, render=effective_render)
                    else:
                        success, obs = push_o_ppo_execute(
                            env, obs, goal_xyz, checkpoint=push_o_checkpoint, render=effective_render, device=dev, agent=skill_agents["push_o"]
                        )
                elif skill == "ppo":
                    if push_o_checkpoint is None:
                        print("    [SKIP] push_disk: no --push-o-checkpoint provided — re-planning")
                        skill_failed = True
                        break
                    if skill_agents["push_o"] is None:
                        skill_agents["push_o"] = load_agent(push_o_checkpoint, dev)
                    success, obs = push_o_ppo_execute(
                        env, obs, goal_xyz, checkpoint=push_o_checkpoint, render=effective_render, device=dev, agent=skill_agents["push_o"]
                    )
                else:
                    success, obs = push_o_mpc_execute(env, obs, goal_xyz, render=effective_render)
                print(f"    → {'OK' if success else 'FAIL'}")
                if not success:
                    skill_failed = True
                    break

            else:
                print(f"    [SKIP] '{sg_skill}' unknown — re-planning")
                skill_failed = True
                break

        if skill_failed:
            skill_fail_counts[sg_skill] = skill_fail_counts.get(sg_skill, 0) + 1
        else:
            print("All subgoals executed successfully.")
            task_succeeded = True
            break

    if capture_video:
        env.save_video()
    wrapper.close()
    return {
        "success": task_succeeded,
        "n_replans": n_replans_used,
        "skill_fail_counts": skill_fail_counts,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",                type=int,   default=0)
    ap.add_argument("--max_replans",         type=int,   default=5)
    ap.add_argument("--offline",             action="store_true")
    ap.add_argument("--render",              action="store_true", help="Open a viewer window")
    ap.add_argument("--model",               default="gemini-2.5-flash")
    ap.add_argument("--skill",               default="ppo", choices=["mpc", "ppo", "auto"],
                    help="auto: MPC vs PPO preview metrics per subgoal")
    ap.add_argument("--env-id",             default="PushO-WallObstacles-v1", choices=["PushO-WallObstacles-v1", "PushO-Scattered", "PushO-TrappedDisk"])
    ap.add_argument("--reach-checkpoint",    default="Reach", dest="reach_checkpoint",
                    help="Reach PPO checkpoint (required for --skill ppo or auto)")
    ap.add_argument("--reach-device",        default=None, dest="reach_device",
                    help="cuda|cpu for reach PPO / auto RL preview (default: auto-detect)")
    ap.add_argument("--pick-checkpoint",     default="PickSkill", dest="pick_checkpoint",
                    help="Pick skill PPO checkpoint")
    ap.add_argument("--place-checkpoint",    default="PlaceSkill", dest="place_checkpoint",
                    help="Place skill PPO checkpoint")
    # ap.add_argument("--push-cube-checkpoint", default="PushCube", dest="push_cube_checkpoint",
    #                 help="Push-cube skill PPO checkpoint")
    ap.add_argument("--push-o-checkpoint", default="PushO", dest="push_o_checkpoint",
                    help="Push-O skill PPO checkpoint")
    ap.add_argument("--capture-video",     action="store_true", dest="capture_video",
                    help="Record the entire run and save as an MP4")
    ap.add_argument("--video-dir",         default="./videos", dest="video_dir",
                    help="Directory to save recorded video (default: ./videos)")
    args = ap.parse_args()
    if args.skill in ("ppo", "auto") and args.reach_checkpoint is None:
        ap.error("--reach-checkpoint is required when using --skill ppo or auto")
    result = run(
        seed=args.seed,
        max_replans=args.max_replans,
        offline=args.offline,
        model=args.model,
        render=args.render,
        skill=args.skill,
        env_id=args.env_id,
        checkpoint=_resolve_checkpoint(args.reach_checkpoint),
        reach_device=args.reach_device,
        pick_checkpoint=_resolve_checkpoint(args.pick_checkpoint),
        place_checkpoint=_resolve_checkpoint(args.place_checkpoint),
        # push_cube_checkpoint=_resolve_checkpoint(args.push_cube_checkpoint),
        push_o_checkpoint=_resolve_checkpoint(args.push_o_checkpoint),
        capture_video=args.capture_video,
        video_dir=args.video_dir,
    )
    print(f"\n[result] success={result['success']}  replans={result['n_replans']}  skill_fails={result['skill_fail_counts']}")

