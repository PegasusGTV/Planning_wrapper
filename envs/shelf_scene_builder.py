# planning_wrapper/envs/shelf_scene_builder.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

import sapien.core as sapien
import sapien.physx as physx
import sapien.render
from sapien.utils import Viewer


@dataclass
class ShelfBayGeometry:
    """Axis-aligned bounding box for a single shelf bay in world frame."""

    bay_min_xyz: np.ndarray  # shape (3,)
    bay_max_xyz: np.ndarray  # shape (3,)

    @property
    def bay_center(self) -> np.ndarray:
        return 0.5 * (self.bay_min_xyz + self.bay_max_xyz)

    @property
    def bay_size(self) -> np.ndarray:
        return self.bay_max_xyz - self.bay_min_xyz


class ShelfSceneBuilder:
    """
    Builder for a simple static scene:
      - A table (box)
      - A single open-front shelf bay on top of the table (built from static boxes)

    All geometry is STATIC (no dynamics), so it doesn't need to be saved/restored
    for planning. We only keep the bay AABB for success checks and spawn regions.
    """

    def __init__(
        self,
        # Bay interior geometry in world frame
        bay_center: Tuple[float, float, float] = (0.6, 0.0, 0.8),
        bay_size: Tuple[float, float, float] = (0.3, 0.4, 0.3),
        wall_thickness: float = 0.02,
        # Table geometry
        table_center: Tuple[float, float, float] = (0.6, 0.0, 0.6),
        table_size: Tuple[float, float, float] = (0.8, 0.8, 0.05),
    ) -> None:
        """
        Args:
            bay_center: (x, y, z) center of the *interior* of the bay (world frame).
            bay_size:   (sx, sy, sz) interior size along x (depth), y (width), z (height).
            wall_thickness: thickness of shelf walls.
            table_center: center of table top box (world frame).
            table_size:   full size of table box (sx, sy, sz).
        """
        self.bay_center = np.asarray(bay_center, dtype=np.float32)
        self.bay_size = np.asarray(bay_size, dtype=np.float32)
        self.wall_thickness = float(wall_thickness)

        self.table_center = np.asarray(table_center, dtype=np.float32)
        self.table_size = np.asarray(table_size, dtype=np.float32)

        # Computed after build()
        self._bay_geom: ShelfBayGeometry | None = None

        self.success_margin_x = 0.02  # 2 cm beyond bay front
        self.lift_thresh = 0.02       # 2 cm above table top
        self.require_lift = True

    # ------------------ public geometry API ------------------ #

    @property
    def table_top_z(self) -> float:
        """Z-height of the table's top surface in world frame."""
        cx, cy, cz = self.table_center
        sx, sy, sz = self.table_size
        return float(cz + sz / 2.0)

    @property
    def bay_geometry(self) -> ShelfBayGeometry:
        if self._bay_geom is None:
            # compute from center + size
            half = 0.5 * self.bay_size
            bay_min = self.bay_center - half
            bay_max = self.bay_center + half
            self._bay_geom = ShelfBayGeometry(bay_min_xyz=bay_min, bay_max_xyz=bay_max)
        return self._bay_geom

    @property
    def bay_min_xyz(self) -> np.ndarray:
        return self.bay_geometry.bay_min_xyz

    @property
    def bay_max_xyz(self) -> np.ndarray:
        return self.bay_geometry.bay_max_xyz

    @property
    def bay_center_xyz(self) -> np.ndarray:
        return self.bay_geometry.bay_center

    @property
    def bay_size_xyz(self) -> np.ndarray:
        return self.bay_geometry.bay_size

    def spawn_region(
        self,
        margin: Tuple[float, float, float] = (0.02, 0.02, 0.02),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns a shrunken AABB (min, max) inside the bay where it's safe to spawn objects.

        Args:
            margin: (mx, my, mz) margin to shrink from each side.

        Returns:
            spawn_min, spawn_max: 3D corners in world frame.
        """
        margin = np.asarray(margin, dtype=np.float32)

        bay_min = self.bay_min_xyz
        bay_max = self.bay_max_xyz

        spawn_min = bay_min + margin
        spawn_max = bay_max - margin

        # Make sure we didn't invert axes by too large margin
        spawn_min = np.minimum(spawn_min, spawn_max - 1e-4)

        return spawn_min, spawn_max
    

    def _compute_success(self) -> bool:
        """Return True if the cube has been pulled out (and optionally lifted)."""
        if self.obj is None:
            return False

        try:
            pose = self.obj.get_pose()
        except Exception:
            return False

        obj_pos = np.asarray(pose.p, dtype=np.float32)
        obj_x, _, obj_z = obj_pos

        # Bay bounds from builder
        bay_max_x = float(self.shelf_builder.bay_max_xyz[0])

        # Pulled out along +x
        pulled_out = obj_x > (bay_max_x + self.success_margin_x)

        if not self.require_lift:
            return bool(pulled_out)

        # Lift condition: above table top by a small margin
        table_top_z = self.shelf_builder.table_top_z
        lifted = obj_z > (table_top_z + self.lift_thresh)

        return bool(pulled_out and lifted)

    # ------------------ building SAPIEN scene ------------------ #

    def build(self, scene: sapien.Scene) -> None:
        """
        Build static table + shelf geometry into the given SAPIEN scene.

        The bay is an open-front box:
          - floor/base
          - left wall
          - right wall
          - back wall
          (top is optional; we omit it for now).
        """
        # Default physical + visual materials
        pm = physx.PhysxMaterial(static_friction=0.4, dynamic_friction=0.4, restitution=0.01)
        vm = sapien.render.RenderMaterial(
            base_color=[0.7, 0.7, 0.7, 1.0],
            metallic=0.0,
            roughness=0.8,
        )

        # 1) Build table as a single static box
        self._build_table(scene, pm, vm)

        # 2) Build shelf bay on top of the table
        self._build_shelf(scene, pm, vm)

    def _build_table(
        self,
        scene: sapien.Scene,
        pm: physx.PhysxMaterial,
        vm: sapien.render.RenderMaterial,
    ) -> None:
        half_size = 0.5 * self.table_size
        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size, material=pm)
        builder.add_box_visual(half_size=half_size, material=vm)

        pose = sapien.Pose(self.table_center, [1, 0, 0, 0])
        builder.initial_pose = pose
        builder.build_static(name="table")

    def _build_shelf(
        self,
        scene: sapien.Scene,
        pm: physx.PhysxMaterial,
        vm: sapien.render.RenderMaterial,
    ) -> None:
        """
        Construct four static boxes around the bay interior:
          - floor/base
          - left wall
          - right wall
          - back wall
        """
        sx, sy, sz = self.bay_size
        t = self.wall_thickness

        # Convenient aliases
        cx, cy, cz = self.bay_center

        # Precompute AABB (for bay_geometry)
        _ = self.bay_geometry  # ensures _bay_geom is populated

        # Helper: build a static box
        def build_static_box(
            half_size: np.ndarray,
            center: np.ndarray,
            name: str,
        ) -> None:
            builder = scene.create_actor_builder()
            builder.add_box_collision(half_size=half_size, material=pm)
            builder.add_box_visual(half_size=half_size, material=vm)
            pose = sapien.Pose(center, [1, 0, 0, 0])
            builder.initial_pose = pose
            builder.build_static(name=name)

        # (a) Floor/base: lies at the bay bottom, same x/y footprint, thin in z
        floor_half = np.array([sx / 2.0, sy / 2.0, t / 2.0], dtype=np.float32)
        floor_center = np.array(
            [cx, cy, cz - sz / 2.0 - t / 2.0], dtype=np.float32
        )
        build_static_box(floor_half, floor_center, "shelf_floor")

        # (b) Left wall: along +y side
        wall_y_half = np.array([sx / 2.0, t / 2.0, sz / 2.0], dtype=np.float32)
        left_center = np.array(
            [cx, cy + sy / 2.0 + t / 2.0, cz], dtype=np.float32
        )
        right_center = np.array(
            [cx, cy - sy / 2.0 - t / 2.0, cz], dtype=np.float32
        )
        build_static_box(wall_y_half, left_center, "shelf_left_wall")
        build_static_box(wall_y_half, right_center, "shelf_right_wall")

        # (c) Back wall: along +x (at "deep" side of bay)
        wall_x_half = np.array([t / 2.0, sy / 2.0, sz / 2.0], dtype=np.float32)
        back_center = np.array(
            [cx + sx / 2.0 + t / 2.0, cy, cz], dtype=np.float32
        )
        build_static_box(wall_x_half, back_center, "shelf_back_wall")

        # (d) Optional top (commented, can be enabled later)
        # top_half = np.array([sx / 2.0, sy / 2.0, t / 2.0], dtype=np.float32)
        # top_center = np.array(
        #     [cx, cy, cz + sz / 2.0 + t / 2.0], dtype=np.float32
        # )
        # build_static_box(top_half, top_center, "shelf_top")


# ------------------ Quick manual test: visualize in SAPIEN ------------------ #

if __name__ == "__main__":
    # Minimal viewer script to visually inspect the shelf bay.
    # Create physics and render systems
    physx_system = physx.PhysxCpuSystem()
    try:
        render_device = sapien.Device("cuda")
    except RuntimeError:
        render_device = sapien.Device("cpu")
    render_system = sapien.render.RenderSystem(render_device)
    
    # Create scene with both systems
    scene = sapien.Scene(systems=[physx_system, render_system])
    scene.set_timestep(1 / 240.0)
    physx.set_scene_config(gravity=[0.0, 0.0, -9.81])

    # Basic lighting
    scene.set_ambient_light([0.4, 0.4, 0.4])
    scene.add_directional_light([1, -1, -1], [0.8, 0.8, 0.8], shadow=True)

    # Build table + shelf
    builder = ShelfSceneBuilder()
    builder.build(scene)

    print("Bay center:", builder.bay_center_xyz)
    print("Bay size:", builder.bay_size_xyz)
    print("Bay min:", builder.bay_min_xyz)
    print("Bay max:", builder.bay_max_xyz)
    print("Spawn region:", builder.spawn_region())

    # Simple viewer
    viewer = Viewer()
    viewer.set_scene(scene)

    # Camera looking at shelf
    cam_pos = [0.3, -1.5, 1.4]
    cam_look_at = builder.bay_center_xyz.tolist()
    viewer.set_camera_pose(sapien.Pose(cam_pos, [1, 0, 0, 0]))
    viewer.scene.set_environment_map(None)

    while not viewer.closed:
        scene.step()
        scene.update_render()
        viewer.render()
