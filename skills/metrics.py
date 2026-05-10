"""
Skill-agnostic scoring: TCP manipulability, obstacle clearance, task progress along a preview.

Includes MPPI reach preview (lookahead_reach_mppi_score) and generic policy rollouts
(lookahead_rollout_score, lookahead_rl_score).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

_POLARIS_ROOT = Path(__file__).resolve().parent.parent
_POLARIS_MANIP_CACHE_ATTR = "_polaris_tcp_manip_cache"

# Panda hand TCP link index in SAPIEN compact Pinocchio model (matches reach_mpc).
_DEFAULT_TCP_LINK_IDX = 10


@dataclass
class _TcpManipCache:
    """One SAPIEN PinocchioModel per robot — creating a new model every step is unsafe (crash / heap corruption)."""

    pmodel: Any
    tcp_link_idx: int = _DEFAULT_TCP_LINK_IDX


def _get_tcp_manip_cache(robot: Any) -> _TcpManipCache:
    c = getattr(robot, _POLARIS_MANIP_CACHE_ATTR, None)
    if isinstance(c, _TcpManipCache):
        return c
    scene = getattr(robot, "scene", None)
    if scene is not None and getattr(scene, "gpu_sim_enabled", False):
        raise RuntimeError(
            "TCP manipulability cache needs CPU simulation (create_pinocchio_model is disabled on GPU)."
        )
    pmodel = robot.create_pinocchio_model()
    cache = _TcpManipCache(pmodel=pmodel, tcp_link_idx=_DEFAULT_TCP_LINK_IDX)
    setattr(robot, _POLARIS_MANIP_CACHE_ATTR, cache)
    return cache


def _tcp_world_position(pmodel: Any, q_full: np.ndarray, tcp_link_idx: int) -> np.ndarray:
    """TCP linear position in world frame using only SAPIEN PinocchioModel (no separate pinocchio import)."""
    q_full = np.ascontiguousarray(np.asarray(q_full, dtype=np.float64).reshape(-1))
    pmodel.compute_forward_kinematics(q_full)
    pose = pmodel.get_link_pose(tcp_link_idx)
    p = getattr(pose, "p", pose)
    if hasattr(p, "cpu"):
        p = p.cpu().numpy()
    return np.asarray(p, dtype=np.float64).reshape(3)


def _manip_from_linear_jacobian(J: np.ndarray) -> float:
    """Scalar manipulability: w = sqrt(det(J @ J.T)) for J in R^{3 x k} (position part)."""
    gram = J @ J.T
    det = float(np.linalg.det(gram))
    return float(np.sqrt(max(det, 0.0)))


def _numerical_position_jacobian(
    pmodel: Any,
    q_full: np.ndarray,
    tcp_link_idx: int,
    arm_dofs: int,
    eps: float,
) -> np.ndarray:
    """3 x arm_dofs ∂x_tcp / ∂q_arm via central-ish one-sided FD (stable, matches SAPIEN FK only)."""
    q_full = np.asarray(q_full, dtype=np.float64).reshape(-1).copy()
    p0 = _tcp_world_position(pmodel, q_full, tcp_link_idx)
    J = np.zeros((3, arm_dofs), dtype=np.float64)
    for i in range(arm_dofs):
        dq = q_full.copy()
        dq[i] += eps
        p1 = _tcp_world_position(pmodel, dq, tcp_link_idx)
        J[:, i] = (p1 - p0) / eps
    return J


def unwrap_maniskill_root(env: Any) -> Any:
    """Walk wrappers until we find a ManiSkill-style env with agent.robot."""
    seen: set[int] = set()
    e: Any = env
    for _ in range(32):
        if id(e) in seen:
            break
        seen.add(id(e))
        u = getattr(e, "unwrapped", e)
        agent = getattr(u, "agent", None)
        if agent is not None and getattr(agent, "robot", None) is not None:
            return u
        if hasattr(e, "env") and getattr(e, "env", None) is not e:
            e = e.env
        elif hasattr(e, "unwrapped") and e.unwrapped is not e:
            e = e.unwrapped
        else:
            break
    raise RuntimeError("unwrap_maniskill_root: could not find env.agent.robot")


def tcp_manipulability(
    robot: Any,
    tcp_frame_name: str = "panda_hand_tcp",
    *,
    arm_dofs: int = 7,
    fd_eps: float = 1e-4,
) -> np.ndarray:

    del tcp_frame_name  # API compatibility; TCP index fixed for Panda in SAPIEN Pinocchio model

    cache = _get_tcp_manip_cache(robot)
    pmodel, tcp_link_idx = cache.pmodel, cache.tcp_link_idx

    q_t = robot.qpos
    if hasattr(q_t, "detach"):
        q_t = q_t.detach()
    if hasattr(q_t, "cpu"):
        q_t = q_t.cpu().numpy()
    q_all = np.ascontiguousarray(np.asarray(q_t, dtype=np.float64))
    if q_all.ndim == 1:
        q_all = q_all.reshape(1, -1)

    ws: list[float] = []
    for i in range(q_all.shape[0]):
        q_row = q_all[i]
        J = _numerical_position_jacobian(pmodel, q_row, tcp_link_idx, arm_dofs, fd_eps)
        ws.append(_manip_from_linear_jacobian(J))
    return np.asarray(ws, dtype=np.float64)


def _q_full_from_arm(robot: Any, q7: np.ndarray) -> np.ndarray:
    q_t = robot.qpos
    if hasattr(q_t, "detach"):
        q_t = q_t.detach()
    if hasattr(q_t, "cpu"):
        q_t = q_t.cpu().numpy()
    q_full = np.asarray(q_t, dtype=np.float64).reshape(-1).copy()
    q7 = np.asarray(q7, dtype=np.float64).reshape(-1)
    q_full[: min(7, len(q_full))] = q7[: min(7, len(q_full))]
    return q_full


def _tcp_from_q7(robot: Any, q7: np.ndarray) -> np.ndarray:
    cache = _get_tcp_manip_cache(robot)
    q_full = _q_full_from_arm(robot, q7)
    return _tcp_world_position(cache.pmodel, q_full, cache.tcp_link_idx)


def _tcp_obstacle_clearance(root: Any, tcp_pos: np.ndarray) -> float:
    tcp = np.asarray(tcp_pos, dtype=np.float64).reshape(3)
    obstacles = getattr(root, "obstacles", None)
    if not obstacles:
        return 1.0
    specs = getattr(root, "OBSTACLE_SPECS", None)
    if specs is None:
        specs = [(0.02, []), (0.015, []), (0.025, []), (0.018, [])]
    dmin = float("inf")
    for i, o in enumerate(obstacles):
        half = float(specs[i][0]) if i < len(specs) else 0.02
        p = o.pose.p
        if hasattr(p, "cpu"):
            p = p.cpu().numpy()
        c = np.asarray(p, dtype=np.float64).reshape(3)
        dd = np.maximum(np.abs(tcp - c) - half, 0.0)
        dmin = min(dmin, float(np.linalg.norm(dd)))
    return float(dmin) if dmin < float("inf") else 1.0


def _normalize_triplet(
    manip_vals: list[float],
    clear_vals: list[float],
    dist_start: float,
    dist_end: float,
    *,
    manip_scale: float = 0.25,
    clear_scale: float = 0.25,
) -> Tuple[float, float, float]:
    if not manip_vals:
        m_n = 0.0
    else:
        m_n = float(np.clip(np.mean(manip_vals) / manip_scale, 0.0, 1.0))
    if not clear_vals:
        c_n = 0.0
    else:
        c_n = float(np.clip(np.mean(clear_vals) / clear_scale, 0.0, 1.0))
    ds = max(float(dist_start), 1e-9)
    p_n = float(np.clip((dist_start - dist_end) / ds, 0.0, 1.0))
    return m_n, p_n, c_n


def _dynamic_weights(
    manip_vals: list[float],
    clear_vals: list[float],
    dist_start: float,
    dist_end: float,
    *,
    manip_scale: float = 0.25,
    clear_scale: float = 0.25,
) -> Tuple[float, float, float]:
    """
    Compute per-rollout weights from the raw metric values:
      w_prog  ∝ dist_end / dist_start   — high when still far from goal
      w_clear ∝ 1 / (mean_clear + ε)   — high when near an obstacle
      w_manip ∝ 1 / (mean_manip + ε)   — high when near a singularity
    Returns (w_manip, w_prog, w_clear) normalised to sum to 1.
    """
    ds = max(float(dist_start), 1e-9)
    dist_frac = float(np.clip(dist_end / ds, 0.0, 1.0))

    mean_manip = float(np.mean(manip_vals)) if manip_vals else 0.0
    mean_clear = float(np.mean(clear_vals)) if clear_vals else 0.0

    w_p = max(dist_frac, 1e-3)
    w_c = 1.0 / (mean_clear / clear_scale + 0.1)
    w_m = 1.0 / (mean_manip / manip_scale + 0.1)

    total = w_p + w_c + w_m
    return w_m / total, w_p / total, w_c / total


def weighted_reach_score(
    manip_vals: list[float],
    clear_vals: list[float],
    dist_start: float,
    dist_end: float,
) -> Tuple[float, float, float, float]:
    w_manip, w_prog, w_clear = _dynamic_weights(manip_vals, clear_vals, dist_start, dist_end)
    m_n, p_n, c_n = _normalize_triplet(manip_vals, clear_vals, dist_start, dist_end)
    s = w_manip * m_n + w_prog * p_n + w_clear * c_n
    return s, m_n, p_n, c_n


def _dist_tcp_to_goal(obs: Dict[str, Any], goal_pos: np.ndarray) -> float:
    g = np.asarray(goal_pos, dtype=np.float64).reshape(3)
    ee = obs.get("extra", {}) if isinstance(obs, dict) else {}
    if not isinstance(ee, dict):
        return float("nan")
    if "ee_pos" in ee:
        ep = np.asarray(ee["ee_pos"], dtype=np.float64).reshape(-1)[:3]
        return float(np.linalg.norm(ep - g))
    if "tcp_pose" in ee:
        ep = np.asarray(ee["tcp_pose"], dtype=np.float64).reshape(-1)[:3]
        return float(np.linalg.norm(ep - g))
    return float("nan")


def lookahead_rollout_score(
    wrapper: Any,
    policy_act: Callable[[Dict[str, Any]], Any],
    obs: Dict[str, Any],
    progress_fn: Callable[[Dict[str, Any], Optional[Dict[str, Any]]], float],
    *,
    preview_steps: int = 24,
    dist_start_override: Optional[float] = None,
    goal_pos_for_fallback: Optional[np.ndarray] = None,
) -> Tuple[float, dict]:
    """`progress_fn(obs, info)` — smaller is closer to subgoal; `info=None` for initial distance."""
    root = wrapper.root
    robot = wrapper.robot
    if robot is None:
        return float("-inf"), {"reason": "no_robot"}

    if dist_start_override is not None:
        dist_start = float(dist_start_override)
    else:
        dist_start = float(progress_fn(obs, None))

    snapshot = wrapper.clone_state()
    manip_vals: list[float] = []
    clear_vals: list[float] = []

    try:
        o = obs
        info: Dict[str, Any] = {}
        for _ in range(preview_steps):
            act = policy_act(o)
            if hasattr(act, "detach"):
                act = act.detach().cpu().numpy()
            act = np.asarray(act, dtype=np.float32).reshape(-1)
            o, _, term, trunc, info = wrapper.env.step(act)
            try:
                w = tcp_manipulability(robot)
                manip_vals.append(float(np.mean(w)))
            except Exception:
                manip_vals.append(0.0)
            ee = o.get("extra", {}) if isinstance(o, dict) else {}
            if isinstance(ee, dict) and "ee_pos" in ee:
                tcp = np.asarray(ee["ee_pos"], dtype=np.float64).reshape(-1)[:3]
            elif isinstance(ee, dict) and "tcp_pose" in ee:
                tcp = np.asarray(ee["tcp_pose"], dtype=np.float64).reshape(-1)[:3]
            else:
                q_t = robot.qpos
                if hasattr(q_t, "detach"):
                    q_t = q_t.detach().cpu().numpy()
                tcp = _tcp_from_q7(robot, np.asarray(q_t, dtype=np.float64).reshape(-1)[:7])
            clear_vals.append(_tcp_obstacle_clearance(root, tcp))
            if np.asarray(term).any() or np.asarray(trunc).any():
                break

        inf = info if isinstance(info, dict) else {}
        dist_end = float(progress_fn(o, inf))
        if dist_end != dist_end:
            if goal_pos_for_fallback is not None:
                dist_end = _dist_tcp_to_goal(o, goal_pos_for_fallback)
            else:
                dist_end = float(progress_fn(o, {}))
        if dist_start != dist_start:
            dist_start = dist_end if dist_end == dist_end else 1.0
        if dist_end != dist_end:
            dist_end = dist_start if dist_start == dist_start else 1.0
        if dist_start != dist_start:
            dist_start = 1.0
        if dist_end != dist_end:
            dist_end = 1.0
    finally:
        wrapper.restore_state(snapshot)

    s, m_n, p_n, c_n = weighted_reach_score(
        manip_vals, clear_vals, dist_start, dist_end
    )
    return s, {"manip_norm": m_n, "progress_norm": p_n, "clearance_norm": c_n}


def lookahead_rl_score(
    wrapper: Any,
    goal_pos: np.ndarray,
    policy_act: Callable[[Dict[str, Any]], Any],
    obs: Dict[str, Any],
    *,
    preview_steps: int = 24,
    dist_start_override: Optional[float] = None,
) -> Tuple[float, dict]:
    gp = np.asarray(goal_pos, dtype=np.float64).reshape(3)

    def _prog(o: Dict[str, Any], inf: Optional[Dict[str, Any]]) -> float:
        if inf is not None and isinstance(inf, dict):
            d = inf.get("dist_to_goal")
            if d is not None:
                return float(np.asarray(d).reshape(-1)[0])
        return _dist_tcp_to_goal(o, gp)

    return lookahead_rollout_score(
        wrapper,
        policy_act,
        obs,
        _prog,
        preview_steps=preview_steps,
        dist_start_override=dist_start_override,
        goal_pos_for_fallback=gp,
    )


def lookahead_reach_mppi_score(
    wrapper: Any,
    obs: Dict[str, Any],
    goal_pos: np.ndarray,
    *,
    preview_steps: int = 24,
    **mppi_kwargs: Any,
) -> Tuple[float, dict]:
    """MPPI reach preview — uses `reach_mpc.ReachMPPI` (same as `reach_mpc.execute`)."""
    import sys

    for p in (str(_POLARIS_ROOT / "skills"), str(_POLARIS_ROOT / "skills" / "reach")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from mpc_base import get_ee_pos
    from reach_mpc import ReachMPPI

    root = wrapper.root
    robot = wrapper.robot
    if robot is None:
        return float("-inf"), {"reason": "no_robot"}
    gp = np.asarray(goal_pos, dtype=np.float64).reshape(3)
    ctrl = ReachMPPI(goal_xyz=gp, **mppi_kwargs)
    act_dim = wrapper.env.action_space.shape[0]

    def _mppi_act(o: Dict[str, Any]) -> np.ndarray:
        ee = get_ee_pos(o)
        delta = ctrl.get_action({"ee_pos": ee})
        a = np.zeros(act_dim, dtype=np.float32)
        a[: min(3, act_dim)] = delta[: min(3, act_dim)]
        return a

    def _prog(o: Dict[str, Any], inf: Optional[Dict[str, Any]]) -> float:
        if inf is not None and isinstance(inf, dict):
            d = inf.get("dist_to_goal")
            if d is not None:
                return float(np.asarray(d).reshape(-1)[0])
        return _dist_tcp_to_goal(o, gp)

    return lookahead_rollout_score(
        wrapper,
        _mppi_act,
        obs,
        _prog,
        preview_steps=preview_steps,
        goal_pos_for_fallback=gp,
    )


def select_reach_backend(
    mpc_score: float,
    ppo_score: float,
) -> str:
    """Returns 'planner' (MPC / classical-style) or 'rl' (PPO)."""
    if mpc_score != mpc_score and ppo_score == ppo_score:
        return "rl"
    if ppo_score != ppo_score and mpc_score == mpc_score:
        return "planner"
    return "planner" if float(mpc_score) >= float(ppo_score) else "rl"


select_skill_backend = select_reach_backend
