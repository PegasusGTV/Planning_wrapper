# envs/pusht_v2.py
"""
PushT-v2: Continuous multi-goal PushT with multiple extra objects.

Key features vs PushT-v1 / PushT-WithExtraObject-v1:
  - Smaller T block (reduced half-sizes)
  - N randomly spawned extra objects at episode start
  - Goal cycling: when T reaches goal, a NEW random goal is immediately set
    instead of ending the episode.
  - Object respawn: after `respawn_threshold` consecutive goals in a row,
    all extra objects are relocated to new random positions, simulating a
    "new configuration". This also fires the `config_changed` info flag
    that the teleop recorder uses to start a new video segment.
  - Episode only ends on truncation (max_episode_steps). There is no
    terminal success state.
  
Success metric (internal, for reward shaping):
  - Identical to PushT-v1: the IoU overlap between the live T and the ghost
    goal-T exceeds the threshold defined in ManiSkill's PushTEnv.evaluate().
  - We intercept that signal, cycle the goal, and suppress episode termination.

T block size:
  ManiSkill's PushTEnv builds the T from `self.tee_half_size` (float, half
  the side-length of each bar of the T) if that attribute is set before
  `super()._load_scene()` is called.  We set it to a smaller value here.
  If your ManiSkill version does not read that attribute the T geometry
  will remain at its default size; everything else will still work correctly.
"""

from __future__ import annotations

import numpy as np
import sapien
import torch
from typing import Any, Dict, List, Optional, Tuple

import mani_skill.envs  # noqa: F401 – registers built-in envs
from mani_skill.envs.tasks.tabletop.push_t import PushTEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.building import actors
from mani_skill.utils.structs import Pose


# ──────────────────────────────────────────────────────────────────────────────
# Table geometry helpers (approximate; matches PushT default table layout)
# ──────────────────────────────────────────────────────────────────────────────
_TABLE_XY_HALF = 0.18       # usable half-width of the table top in XY
_TABLE_Z = 0.0              # table top z in PushT world frame (approx.)
_GOAL_XY_HALF = 0.13        # smaller range for goal positions to stay central
_EXTRA_OBJ_XY_HALF = 0.16   # spawn range for extra objects


@register_env("PushT-v2", max_episode_steps=100_000)
class PushTv2Env(PushTEnv):
    """
    Continuous multi-goal PushT with multiple extra objects.

    Constructor parameters
    ----------------------
    num_extra_objects : int
        Number of extra cube objects to spawn on the table (default 5).
    respawn_threshold : int
        After this many consecutive goals the extra objects are relocated
        (default 3).  The info dict will have ``config_changed=True`` for
        that one step so recorders can start a new video segment.
    extra_obj_half_size : float
        Half-size of each extra cube in metres (default 0.018 m → 3.6 cm).
    T_scale : float
        Scale factor applied to T block dimensions vs. ManiSkill default.
        Values < 1 give a smaller T (default 0.70 → 30 % smaller).
        Only effective when ManiSkill reads ``tee_half_size`` from the env.
    """

    # ── init ─────────────────────────────────────────────────────────────────
    def __init__(
        self,
        *args,
        num_extra_objects: int = 5,
        respawn_threshold: int = 3,
        extra_obj_half_size: float = 0.018,
        T_scale: float = 0.70,
        **kwargs,
    ) -> None:
        # ---- store params BEFORE super().__init__() which calls _load_scene ----
        self.num_extra_objects_v2: int = num_extra_objects
        self.respawn_threshold: int = respawn_threshold
        self.extra_obj_half_size: float = extra_obj_half_size

        # Reduce T block mass & friction for easier pushing
        self.T_mass: float = 0.15          # default ~0.8
        self.T_dynamic_friction: float = 0.4
        self.T_static_friction: float = 0.4

        # Attempt to make T block physically smaller.
        # ManiSkill ≥ 3.x reads tee_half_size if present.
        _DEFAULT_TEE_HALF = 0.04           # ManiSkill default (metres)
        self.tee_half_size: float = _DEFAULT_TEE_HALF * T_scale

        # Goal / consecutive-goals tracking
        self._consecutive_goals: int = 0
        self._total_goals: int = 0
        self._was_success_last_step: bool = False   # debounce
        self._current_step_success: bool = False    # written by evaluate(), read by step()
        self._config_changed: bool = False          # consumed by step()

        # List of extra SAPIEN actors (populated in _load_scene)
        self.extra_objects_v2: list = []

        super().__init__(*args, **kwargs)

    # ── scene loading ─────────────────────────────────────────────────────────
    def _load_scene(self, options: dict) -> None:
        """Build base PushT scene then append extra objects."""
        super()._load_scene(options)

        # Hide the robot-home-position marker (a grey cylinder that PushT
        # renders as a shadow on the table).  We sink it below the table so
        # it never appears in the viewport.
        try:
            self.ee_goal_pos.set_pose(sapien.Pose(p=[0.0, 0.0, -1.0]))
        except Exception:
            pass

        scene = self.scene.sub_scenes[0]

        import sapien.physx as physx
        import sapien.render

        # Low-friction material so extra objects don't impede the T
        pm = physx.PhysxMaterial(
            static_friction=0.15,
            dynamic_friction=0.15,
            restitution=0.15,
        )

        # Distinct colours for each extra object
        palette = [
            [0.85, 0.25, 0.25, 1.0],  # red
            [0.25, 0.75, 0.25, 1.0],  # green
            [0.25, 0.35, 0.90, 1.0],  # blue
            [0.90, 0.75, 0.15, 1.0],  # yellow
            [0.80, 0.35, 0.80, 1.0],  # purple
            [0.25, 0.80, 0.80, 1.0],  # cyan
            [0.90, 0.55, 0.20, 1.0],  # orange
            [0.55, 0.80, 0.30, 1.0],  # lime
        ]

        hs = self.extra_obj_half_size
        self.extra_objects_v2 = []
        for i in range(self.num_extra_objects_v2):
            color = palette[i % len(palette)]
            vm = sapien.render.RenderMaterial(
                base_color=color,
                metallic=0.0,
                roughness=0.45,
            )
            builder = scene.create_actor_builder()
            builder.add_box_collision(
                half_size=[hs, hs, hs], material=pm, density=60
            )
            builder.add_box_visual(half_size=[hs, hs, hs], material=vm)
            builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -1.0])
            obj = builder.build(name=f"extra_cube_v2_{i}")
            self.extra_objects_v2.append(obj)

    # ── episode init ──────────────────────────────────────────────────────────
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
        """Reset T + goal (via parent) then place extra objects."""
        super()._initialize_episode(env_idx, options)

        # Re-hide the home marker that the parent just moved back into view.
        try:
            self.ee_goal_pos.set_pose(sapien.Pose(p=[0.0, 0.0, -1.0]))
        except Exception:
            pass
        self._consecutive_goals = 0
        self._total_goals = 0
        self._was_success_last_step = False
        self._current_step_success = False
        self._config_changed = False
        self._place_extra_objects_random()

    # ── extra-object placement ────────────────────────────────────────────────
    def _place_extra_objects_random(self) -> None:
        """Scatter extra objects across the table, away from the T block."""
        if not self.extra_objects_v2:
            return

        with torch.device(self.device):
            try:
                tee_xy = self.tee.pose.p[0, :2].cpu().numpy()
            except Exception:
                tee_xy = np.zeros(2)

        hs = self.extra_obj_half_size
        placed: List[np.ndarray] = []

        rng = np.random.default_rng()

        for obj in self.extra_objects_v2:
            for _ in range(50):  # rejection-sample
                xy = rng.uniform(-_EXTRA_OBJ_XY_HALF, _EXTRA_OBJ_XY_HALF, size=2)
                # keep away from T and other placed objects
                too_close = np.linalg.norm(xy - tee_xy) < 0.07
                if not too_close:
                    for prev in placed:
                        if np.linalg.norm(xy - prev) < 0.055:
                            too_close = True
                            break
                if not too_close:
                    placed.append(xy.copy())
                    break
            else:
                placed.append(xy.copy())  # give up on perfect spacing

            z = _TABLE_Z + hs + 1e-3
            pos = np.array([placed[-1][0], placed[-1][1], z], dtype=np.float32)
            obj.set_pose(sapien.Pose(pos, [1.0, 0.0, 0.0, 0.0]))

    # ── goal cycling ─────────────────────────────────────────────────────────
    def _cycle_goal(self) -> None:
        """
        Assign a fresh random goal.

        Three things must stay in sync — the visual ghost T, the scalar
        self.goal_z_rot used by compute_dense_reward, and (critically)
        self.world_to_goal_trans used by pseudo_render_intersection().
        _load_scene() builds world_to_goal_trans once and never touches it
        again, so we must recompute it here whenever the goal changes.
        """
        with torch.device(self.device):
            # ── sample new goal pose ──────────────────────────────────────────
            new_z_rot = float(torch.rand(1).item() * 2 * np.pi)

            new_offset = torch.tensor(
                np.random.uniform(-_GOAL_XY_HALF, _GOAL_XY_HALF, size=2),
                dtype=torch.float32,
            )  # shape (2,) on CPU — mirrors how goal_offset is stored

            # ── move the visual ghost T ───────────────────────────────────────
            goal_tee_z = 1e-3  # same as _initialize_episode
            p_goal = torch.tensor(
                [[new_offset[0].item(), new_offset[1].item(), goal_tee_z]],
                dtype=torch.float32,
                device=self.device,
            ).expand(self.num_envs, -1)

            c = float(np.cos(new_z_rot / 2))
            s = float(np.sin(new_z_rot / 2))
            q_goal = torch.tensor(
                [[c, 0.0, 0.0, s]], dtype=torch.float32, device=self.device
            ).expand(self.num_envs, -1)

            try:
                self.goal_tee.set_pose(Pose.create_from_pq(p=p_goal, q=q_goal))
            except Exception:
                pass

            # ── update goal_z_rot (used by compute_dense_reward) ─────────────
            # Must be shape (num_envs,) — NOT (1,1) — to avoid broadcasting
            # an extra dimension into the reward tensor.
            self.goal_z_rot = torch.full(
                (self.num_envs,),
                new_z_rot,
                dtype=torch.float32,
                device=self.device,
            )

            # ── update goal_offset (used below and by any obs code) ───────────
            self.goal_offset = new_offset.to(self.device)

            # ── RECOMPUTE world_to_goal_trans ─────────────────────────────────
            # pseudo_render_intersection() uses ONLY this matrix — it never
            # reads goal_tee.pose at all.  Copied verbatim from _load_scene().
            goal_fake_quat = torch.tensor(
                [(torch.tensor([new_z_rot]) / 2).cos(), 0, 0, 0.0]
            ).unsqueeze(0)
            zrot = self.quat_to_zrot(goal_fake_quat).squeeze(0)  # 3x3
            goal_trans = torch.eye(3)
            goal_trans[:2, :2] = zrot[:2, :2]
            goal_trans[0:2, 2] = self.goal_offset
            self.world_to_goal_trans = torch.linalg.inv(goal_trans).to(self.device)

    # ── evaluate – ONLY returns standard tensor fields ───────────────────────
    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """
        Ask the parent whether the overlap condition is met and stash the
        result in ``self._current_step_success`` so that ``step()`` can act
        on it.  We always return ``success=False`` here so that ManiSkill
        never raises a termination signal.

        IMPORTANT: only return keys whose values are bool tensors of shape
        (num_envs,).  ManiSkill's RecordEpisode wrapper feeds every key from
        evaluate() into the trajectory buffer; any Python int/bool will cause
        shape-mismatch errors in ``common.append_dict_array``.
        """
        parent_result = super().evaluate(**kwargs)
        raw_success = parent_result.get("success", False)

        if isinstance(raw_success, torch.Tensor):
            self._current_step_success = bool(raw_success.any().item())
        else:
            self._current_step_success = bool(raw_success)

        # Suppress termination – return only standard, consistently-typed keys
        false_t = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return {
            "success": false_t,
            "is_success": false_t,
        }

    # ── step – goal cycling + metadata ───────────────────────────────────────
    def step(self, action):
        """
        1. Run the parent step (which internally calls evaluate(), setting
           self._current_step_success).
        2. If success was detected (debounced), cycle the goal and optionally
           respawn extra objects.
        3. Append goal-tracking metadata to info.  These are plain Python
           scalars – RecordEpisode does not put info keys into the tensor
           trajectory buffer, so there is no shape-mismatch risk here.
        """
        obs, reward, terminated, truncated, info = super().step(action)

        # ── Goal-reached event (debounced) ────────────────────────────────────
        if self._current_step_success and not self._was_success_last_step:
            self._was_success_last_step = True
            self._total_goals += 1
            self._consecutive_goals += 1

            self._cycle_goal()

            if self._consecutive_goals >= self.respawn_threshold:
                self._place_extra_objects_random()
                self._consecutive_goals = 0
                self._config_changed = True
        elif not self._current_step_success:
            self._was_success_last_step = False

        # ── Attach metadata to info ───────────────────────────────────────────
        info["config_changed"] = self._config_changed
        info["goals_reached"] = self._total_goals
        info["consecutive_goals"] = self._consecutive_goals

        # config_changed is a one-shot flag; clear after broadcasting to info
        self._config_changed = False

        return obs, reward, terminated, truncated, info