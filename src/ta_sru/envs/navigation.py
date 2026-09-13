"""Isaac Lab DirectRLEnv：六迷宫 Hummingbird 导航任务。

本模块必须在 ``AppLauncher`` 启动 Isaac Sim 后导入。
"""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.assets import (
    Articulation,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCollection,
    RigidObjectCfg,
    RigidObjectCollectionCfg,
)
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, MultiMeshRayCasterCamera, MultiMeshRayCasterCameraCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from ta_sru.config import EnvConfig
from ta_sru.envs.curriculum import select_training_maze_curriculum_stage
from ta_sru.envs.layouts import MAZE_LAYOUTS
from ta_sru.envs.toa import build_toa_bank, save_toa_global_maps
from ta_sru.models.hummingbird import HummingbirdParameters, rotate_inverse, yaw_from_quaternion
from ta_sru.models.hummingbird_asset import HUMMINGBIRD_CFG
from ta_sru.models.lee_controller import LeePositionController


INNER_WALL_LENGTH_SCALES = (1.0, 1.0, 1.0, 0.5, 0.5)
MAX_INNER_WALLS = len(INNER_WALL_LENGTH_SCALES)
MAX_RANDOM_CYLINDERS = 60
CYLINDER_VARIANTS = ((0.30, 2.5), (0.45, 3.2), (0.60, 4.0))
TRAJECTORY_MAX_POINTS = 512
TRAJECTORY_POINT_SPACING = 0.15


def _camera_mesh_targets(
    cylinder_count: int,
) -> list[MultiMeshRayCasterCameraCfg.RaycastTargetCfg]:
    """逐槽位配置相机网格，使克隆环境能够命中共享网格缓存。"""

    # 同一表达式匹配多个同形障碍物时，Isaac Lab 的去重分支可能不缓存别名，
    # 导致后续环境重复解析网格；只对环境编号使用通配符。
    names = (
        "Floor",
        "BoundaryNorth",
        "BoundarySouth",
        "BoundaryEast",
        "BoundaryWest",
        *(f"InnerWall_{index}" for index in range(MAX_INNER_WALLS)),
        *(f"RandomCylinder_{index:02d}" for index in range(cylinder_count)),
    )
    return [
        MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
            prim_expr=f"{{ENV_REGEX_NS}}/{name}",
            is_shared=True,
            track_mesh_transforms=True,
        )
        for name in names
    ]


def _fixed_cuboid(size: tuple[float, float, float]) -> sim_utils.CuboidCfg:
    return sim_utils.CuboidCfg(
        size=size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )


def _fixed_cylinder(radius: float, height: float) -> sim_utils.CylinderCfg:
    return sim_utils.CylinderCfg(
        radius=radius,
        height=height,
        axis="Z",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32, 0.43, 0.55)),
    )


@configclass
class NavigationSceneCfg(InteractiveSceneCfg):
    """每个克隆环境包含一架飞机、一组迷宫墙和随机圆柱。"""

    robot = HUMMINGBIRD_CFG
    floor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Floor",
        spawn=_fixed_cuboid((40.0, 40.0, 0.1)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
    )
    boundary_north = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BoundaryNorth",
        spawn=_fixed_cuboid((40.0, 0.7, 4.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 20.0, 2.0)),
    )
    boundary_south = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BoundarySouth",
        spawn=_fixed_cuboid((40.0, 0.7, 4.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -20.0, 2.0)),
    )
    boundary_east = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BoundaryEast",
        spawn=_fixed_cuboid((0.7, 40.0, 4.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(20.0, 0.0, 2.0)),
    )
    boundary_west = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BoundaryWest",
        spawn=_fixed_cuboid((0.7, 40.0, 4.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-20.0, 0.0, 2.0)),
    )
    inner_wall_0 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InnerWall_0",
        spawn=_fixed_cuboid((12.0, 0.7, 4.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
    )
    inner_wall_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InnerWall_1",
        spawn=_fixed_cuboid((12.0, 0.7, 4.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
    )
    inner_wall_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InnerWall_2",
        spawn=_fixed_cuboid((12.0, 0.7, 4.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
    )
    inner_wall_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InnerWall_3",
        spawn=_fixed_cuboid((6.0, 0.7, 4.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
    )
    inner_wall_4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InnerWall_4",
        spawn=_fixed_cuboid((6.0, 0.7, 4.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
    )
    random_cylinders = RigidObjectCollectionCfg(
        rigid_objects={
            f"cylinder_{index:02d}": RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/RandomCylinder_{index:02d}",
                spawn=_fixed_cylinder(*CYLINDER_VARIANTS[index % len(CYLINDER_VARIANTS)]),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
            )
            for index in range(MAX_RANDOM_CYLINDERS)
        }
    )
    camera = MultiMeshRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.04,
        mesh_prim_paths=_camera_mesh_targets(MAX_RANDOM_CYLINDERS),
        data_types=["distance_to_image_plane"],
        max_distance=12.0,
        depth_clipping_behavior="max",
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=10.4775,
            horizontal_aperture=20.955,
            height=48,
            width=64,
        ),
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, -1.0),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(base_link|rotor_0|rotor_1|rotor_2|rotor_3)",
        update_period=1.0 / 120.0,
        history_length=6,
        debug_vis=False,
    )
    light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 0.75)),
    )


@configclass
class IsaacNavigationEnvCfg(DirectRLEnvCfg):
    debug_vis: bool = False
    decimation = 5
    episode_length_s = 40.0
    action_space = gym.spaces.Box(
        low=np.asarray((-0.1, -0.5, -np.pi / 3), dtype=np.float32),
        high=np.asarray((2.0, 0.5, np.pi / 3), dtype=np.float32),
        dtype=np.float32,
    )
    observation_space = {
        "camera": gym.spaces.Box(-1.0, 1.0, (1, 12, 16), dtype=np.float32),
        "robot_state": gym.spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32),
        "critic_toa": gym.spaces.Box(-1.0, 1.0, (1, 16, 16), dtype=np.float32),
    }
    state_space = 0
    sim = SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)
    sim.physx.enable_external_forces_every_iteration = True
    sim.physx.gpu_found_lost_pairs_capacity = 2**23
    scene = NavigationSceneCfg(
        num_envs=4,
        env_spacing=52.0,
        replicate_physics=True,
        filter_collisions=True,
    )


def make_isaac_env_cfg(task: EnvConfig, sim_device: str) -> IsaacNavigationEnvCfg:
    """把与训练框架无关的配置转换成 Isaac Lab 配置。"""

    if task.random_cylinder_count > MAX_RANDOM_CYLINDERS:
        raise ValueError(f"随机圆柱最多 {MAX_RANDOM_CYLINDERS} 个")
    if task.random_cylinder_count <= 0:
        raise ValueError("Isaac 环境至少需要一个随机圆柱；可将其数量设为 1")
    cfg = IsaacNavigationEnvCfg()
    cfg.seed = task.seed
    cfg.decimation = task.control_decimation
    cfg.episode_length_s = task.episode_seconds
    cfg.sim.dt = task.physics_dt
    cfg.sim.render_interval = task.control_decimation
    cfg.sim.device = sim_device
    cfg.scene.num_envs = task.num_envs
    cfg.debug_vis = task.debug
    cfg.action_space = gym.spaces.Box(
        np.asarray(task.action_low, dtype=np.float32),
        np.asarray(task.action_high, dtype=np.float32),
        dtype=np.float32,
    )
    cfg.observation_space = {
        "camera": gym.spaces.Box(
            -1.0, 1.0, (1, task.depth_height, task.depth_width), dtype=np.float32
        ),
        "robot_state": gym.spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32),
        "critic_toa": gym.spaces.Box(
            -1.0, 1.0, (1, task.toa_crop_size, task.toa_crop_size), dtype=np.float32
        ),
    }
    cfg.scene.camera.pattern_cfg.height = task.depth_height * 4
    cfg.scene.camera.pattern_cfg.width = task.depth_width * 4
    for wall_id, length_scale in enumerate(INNER_WALL_LENGTH_SCALES):
        wall_cfg = getattr(cfg.scene, f"inner_wall_{wall_id}")
        wall_cfg.spawn.size = (
            task.inner_wall_length * length_scale,
            task.wall_thickness,
            task.arena_height,
        )
    cfg.scene.random_cylinders.rigid_objects = dict(
        list(cfg.scene.random_cylinders.rigid_objects.items())[: task.random_cylinder_count]
    )
    # 相机目标与实际生成的圆柱保持一致，避免减少数量后匹配到不存在的路径。
    cfg.scene.camera.mesh_prim_paths = _camera_mesh_targets(task.random_cylinder_count)
    return cfg


class NavigationEnv(DirectRLEnv):
    """由 Isaac Sim/PhysX 驱动的向量化导航环境。"""

    cfg: IsaacNavigationEnvCfg

    def __init__(
        self,
        cfg: IsaacNavigationEnvCfg,
        task_config: EnvConfig,
        render_mode: str | None = None,
    ) -> None:
        self.task = task_config
        super().__init__(cfg, render_mode=render_mode)

        self.parameters = HummingbirdParameters()
        self.controller = LeePositionController(self.parameters).to(self.device)
        self.base_link_ids, body_names = self.robot.find_bodies("base_link")
        if len(self.base_link_ids) != 1:
            raise RuntimeError(f"需要唯一的 base_link，实际找到 {body_names}")

        self.actions = torch.zeros((self.num_envs, 3), device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.goal_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self.position_setpoints = torch.zeros_like(self.goal_positions)
        self.yaw_setpoints = torch.zeros((self.num_envs, 1), device=self.device)
        self.maze_ids = torch.arange(self.num_envs, device=self.device) % len(MAZE_LAYOUTS)
        self.route_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.direction_reversed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.toa_map_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.previous_toa = torch.full((self.num_envs,), torch.nan, device=self.device)
        self.training_step_offset = 0
        self.training_curriculum_stage = 0
        self.training_curriculum_maze_count = 1
        self.training_curriculum_cylinder_count = 0
        self.training_curriculum_contact_scale = 0.0
        self.training_curriculum_contact_force_threshold = (
            self.task.collision_force_threshold
        )
        self._create_layout_tensors()
        self._create_toa_tensors()
        if self.task.debug:
            self._trajectory_history = torch.zeros(
                (self.num_envs, TRAJECTORY_MAX_POINTS, 3), device=self.device
            )
            self._trajectory_counts = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self) -> None:
        self.robot: Articulation = self.scene["robot"]
        self.camera: MultiMeshRayCasterCamera = self.scene["camera"]
        self.contact_sensor: ContactSensor = self.scene["contact_forces"]
        self.inner_walls: list[RigidObject] = [
            self.scene[f"inner_wall_{index}"] for index in range(MAX_INNER_WALLS)
        ]
        self.random_cylinders: RigidObjectCollection = self.scene["random_cylinders"]

    def _create_layout_tensors(self) -> None:
        layout_count = len(MAZE_LAYOUTS)
        self.wall_xy = torch.zeros((layout_count, MAX_INNER_WALLS, 2), device=self.device)
        self.wall_yaw = torch.zeros((layout_count, MAX_INNER_WALLS), device=self.device)
        self.wall_length = torch.tensor(
            INNER_WALL_LENGTH_SCALES, device=self.device
        ).repeat(layout_count, 1)
        self.wall_length *= self.task.inner_wall_length
        self.wall_active = torch.zeros(
            (layout_count, MAX_INNER_WALLS), dtype=torch.bool, device=self.device
        )
        self.route_starts = torch.zeros((layout_count, 2, 2), device=self.device)
        self.route_goals = torch.zeros_like(self.route_starts)
        for layout_id, layout in enumerate(MAZE_LAYOUTS):
            next_slot = {1.0: 0, 0.5: 3}
            for wall in layout.walls:
                if wall.length_scale not in next_slot:
                    raise ValueError(f"不支持的墙体长度比例：{wall.length_scale}")
                wall_id = next_slot[wall.length_scale]
                next_slot[wall.length_scale] += 1
                self.wall_xy[layout_id, wall_id] = torch.tensor(
                    wall.center_xy, device=self.device
                )
                self.wall_yaw[layout_id, wall_id] = np.pi / 2 if wall.vertical else 0.0
                self.wall_active[layout_id, wall_id] = True
            for route_id, (start, goal) in enumerate(layout.routes):
                self.route_starts[layout_id, route_id] = torch.tensor(start, device=self.device)
                self.route_goals[layout_id, route_id] = torch.tensor(goal, device=self.device)

        specs = [
            CYLINDER_VARIANTS[index % len(CYLINDER_VARIANTS)]
            for index in range(self.task.random_cylinder_count)
        ]
        self.cylinder_radii = torch.tensor(
            [radius for radius, _ in specs], device=self.device
        )
        self.cylinder_heights = torch.tensor(
            [height for _, height in specs], device=self.device
        )

    def _create_toa_tensors(self) -> None:
        toa_bank = build_toa_bank(self.task)
        self.toa_maps = torch.from_numpy(toa_bank).to(self.device)
        if self.task.debug:
            output_dir = self.task.debug_output_dir or "debug"
            saved = save_toa_global_maps(self.task, toa_bank, output_dir)
            print(f"[DEBUG] 已保存 {len(saved)} 种迷宫的全局 TOA 图：{output_dir}")
        half_extent = self.task.toa_crop_size * self.task.toa_crop_spacing / 2
        coordinates = torch.linspace(
            -half_extent + self.task.toa_crop_spacing / 2,
            half_extent - self.task.toa_crop_spacing / 2,
            self.task.toa_crop_size,
            device=self.device,
        )
        grid_x, grid_y = torch.meshgrid(coordinates, coordinates, indexing="xy")
        self.toa_crop_body = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """创建或隐藏仅由命令行 ``--debug`` 启用的导航标记。"""

        if not self.task.debug:
            return
        if debug_vis and not hasattr(self, "goal_visualizer"):
            self.goal_visualizer = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/TA_SRU/Goals",
                    markers={
                        "goal": sim_utils.SphereCfg(
                            radius=0.35,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.1, 1.0, 0.15)
                            ),
                        )
                    },
                )
            )
            self.velocity_visualizer = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/TA_SRU/VelocityCones",
                    markers={
                        "velocity": sim_utils.ConeCfg(
                            radius=0.20,
                            height=1.0,
                            axis="X",
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(1.0, 0.35, 0.05)
                            ),
                        )
                    },
                )
            )
            self.trajectory_visualizer = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/TA_SRU/Trajectories",
                    markers={
                        "point": sim_utils.SphereCfg(
                            radius=0.055,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.1, 0.65, 1.0)
                            ),
                        )
                    },
                )
            )

        if hasattr(self, "goal_visualizer"):
            self.goal_visualizer.set_visibility(debug_vis)
            self.velocity_visualizer.set_visibility(debug_vis)
            self.trajectory_visualizer.set_visibility(debug_vis)

    def _debug_vis_callback(self, event) -> None:
        """刷新目标、实际速度圆锥和当前回合轨迹。"""

        if not self.task.debug or not self.robot.is_initialized:
            return

        self.goal_visualizer.visualize(self.goal_positions)

        velocity_xy = self.robot.data.root_lin_vel_w[:, :2]
        speed = torch.linalg.vector_norm(velocity_xy, dim=-1)
        direction = velocity_xy / speed[:, None].clamp_min(1.0e-6)
        cone_length = speed.clamp(0.10, 4.0)
        cone_positions = self.robot.data.root_pos_w.clone()
        cone_positions[:, 2] += 0.75
        cone_positions[:, :2] += direction * (0.5 * cone_length)[:, None]
        cone_orientations = self._yaw_quaternion(torch.atan2(velocity_xy[:, 1], velocity_xy[:, 0]))
        cone_scales = torch.ones((self.num_envs, 3), device=self.device)
        cone_scales[:, 0] = cone_length
        self.velocity_visualizer.visualize(
            cone_positions,
            orientations=cone_orientations,
            scales=cone_scales,
        )

        positions = self.robot.data.root_pos_w
        empty = self._trajectory_counts == 0
        last_indices = (self._trajectory_counts - 1).clamp_min(0)
        last_positions = self._trajectory_history[
            torch.arange(self.num_envs, device=self.device), last_indices
        ]
        moved = torch.linalg.vector_norm(positions - last_positions, dim=-1) >= TRAJECTORY_POINT_SPACING
        append = (empty | moved) & (self._trajectory_counts < TRAJECTORY_MAX_POINTS)
        env_ids = torch.nonzero(append, as_tuple=False).squeeze(-1)
        if len(env_ids):
            slots = self._trajectory_counts[env_ids]
            self._trajectory_history[env_ids, slots] = positions[env_ids]
            self._trajectory_counts[env_ids] += 1

        valid = torch.arange(TRAJECTORY_MAX_POINTS, device=self.device)[None, :] < (
            self._trajectory_counts[:, None]
        )
        self.trajectory_visualizer.visualize(self._trajectory_history[valid])

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.previous_actions.copy_(self.actions)
        low = actions.new_tensor(self.task.action_low)
        high = actions.new_tensor(self.task.action_high)
        self.actions.copy_(actions.clamp(low, high))

    def _apply_action(self) -> None:
        quaternion = self.robot.data.root_quat_w
        w, x, y, z = quaternion.unbind(dim=-1)
        forward = torch.stack((w * w + x * x - y * y - z * z, 2 * (x * y + w * z)), dim=-1)
        forward = forward / torch.linalg.vector_norm(forward, dim=-1, keepdim=True).clamp_min(1.0e-6)
        side = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
        target_velocity_xy = forward * self.actions[:, 0:1] + side * self.actions[:, 1:2]
        target_velocity = torch.cat(
            (target_velocity_xy, torch.zeros((self.num_envs, 1), device=self.device)), dim=-1
        )
        self.position_setpoints[:, :2] += target_velocity_xy * self.physics_dt
        self.position_setpoints[:, 2] = 2.0
        self.yaw_setpoints += self.actions[:, 2:3] * self.physics_dt
        self.yaw_setpoints[:] = (self.yaw_setpoints + torch.pi) % (2 * torch.pi) - torch.pi

        root_state = torch.cat(
            (
                self.robot.data.root_pos_w,
                quaternion,
                self.robot.data.root_lin_vel_w,
                self.robot.data.root_ang_vel_w,
            ),
            dim=-1,
        )
        wrench = self.controller(
            root_state,
            self.position_setpoints,
            target_velocity,
            torch.zeros_like(target_velocity),
            self.yaw_setpoints,
        )
        forces = torch.zeros((self.num_envs, 1, 3), device=self.device)
        forces[:, 0, 2] = wrench[:, 0]
        self.robot.set_external_force_and_torque(
            forces,
            wrench[:, 1:4].unsqueeze(1),
            body_ids=self.base_link_ids,
            is_global=False,
        )

    def _depth_observation(self) -> torch.Tensor:
        depth = self.camera.data.output["distance_to_image_plane"].clone()
        if depth.ndim == 3:
            depth = depth.unsqueeze(-1)
        depth = depth.permute(0, 3, 1, 2)
        depth = -F.max_pool2d(-depth, kernel_size=4, stride=4)
        depth = torch.nan_to_num(
            depth,
            nan=self.task.depth_max_distance,
            posinf=self.task.depth_max_distance,
            neginf=0.0,
        )
        depth = depth.clamp(0.0, self.task.depth_max_distance)
        return depth / self.task.depth_max_distance * 2.0 - 1.0

    def _sample_toa(self, points_xy: torch.Tensor) -> torch.Tensor:
        grid_size = self.toa_maps.shape[-1]
        spacing = 2.0 * self.task.arena_half_extent / (grid_size - 1)
        x = ((points_xy[..., 0] + self.task.arena_half_extent) / spacing).clamp(0, grid_size - 1)
        y = ((points_xy[..., 1] + self.task.arena_half_extent) / spacing).clamp(0, grid_size - 1)
        x0, y0 = torch.floor(x).long(), torch.floor(y).long()
        x1, y1 = (x0 + 1).clamp(max=grid_size - 1), (y0 + 1).clamp(max=grid_size - 1)
        wx, wy = x - x0, y - y0
        offsets = self.toa_map_ids[:, None] * (grid_size * grid_size)
        flat = self.toa_maps.reshape(-1)

        def gather(row: torch.Tensor, column: torch.Tensor) -> torch.Tensor:
            return flat[offsets + row * grid_size + column]

        value_0 = gather(y0, x0) * (1 - wx) + gather(y0, x1) * wx
        value_1 = gather(y1, x0) * (1 - wx) + gather(y1, x1) * wx
        return value_0 * (1 - wy) + value_1 * wy

    def _critic_toa_observation(self) -> torch.Tensor:
        local_position = self.robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        yaw = yaw_from_quaternion(self.robot.data.root_quat_w)
        body_x = self.toa_crop_body[:, 0].unsqueeze(0)
        body_y = self.toa_crop_body[:, 1].unsqueeze(0)
        cos_yaw, sin_yaw = torch.cos(yaw).unsqueeze(1), torch.sin(yaw).unsqueeze(1)
        points = torch.stack(
            (
                body_x * cos_yaw - body_y * sin_yaw + local_position[:, 0:1],
                body_x * sin_yaw + body_y * cos_yaw + local_position[:, 1:2],
            ),
            dim=-1,
        )
        values = self._sample_toa(points)
        values = torch.nan_to_num(
            values,
            nan=self.task.toa_normalization_max,
            posinf=self.task.toa_normalization_max,
            neginf=0.0,
        )
        values = values.clamp(0.0, self.task.toa_normalization_max)
        values = values / self.task.toa_normalization_max * 2.0 - 1.0
        return values.reshape(self.num_envs, 1, self.task.toa_crop_size, self.task.toa_crop_size)

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        quaternion = self.robot.data.root_quat_w
        relative_goal = rotate_inverse(
            quaternion, self.goal_positions - self.robot.data.root_pos_w
        )
        velocity_body = rotate_inverse(quaternion, self.robot.data.root_lin_vel_w)
        angular_velocity_body = rotate_inverse(quaternion, self.robot.data.root_ang_vel_w)
        robot_state = torch.cat(
            (
                relative_goal[:, :2].clamp(-5.0, 5.0),
                self.actions,
                velocity_body[:, :2],
                angular_velocity_body[:, 2:3],
            ),
            dim=-1,
        )
        return {
            "policy": {
                "camera": self._depth_observation(),
                "robot_state": robot_state,
                "critic_toa": self._critic_toa_observation(),
            }
        }

    def _contact_force(self) -> torch.Tensor:
        forces = self.contact_sensor.data.net_forces_w
        return torch.linalg.vector_norm(forces, dim=-1).amax(dim=1)

    @property
    def training_elapsed_steps(self) -> int:
        """返回包含 checkpoint 偏移的累计环境 transition 数。"""

        simulated_steps = int(self.common_step_counter * self.num_envs)
        return max(self.training_step_offset + simulated_steps, 0)

    def set_training_progress(self, elapsed_steps: int) -> None:
        """同步恢复训练后的课程进度，而不修改 Isaac Lab 内部计数器。"""

        if elapsed_steps < 0:
            raise ValueError(f"elapsed_steps 不能为负数，实际为 {elapsed_steps}")
        simulated_steps = int(self.common_step_counter * self.num_envs)
        self.training_step_offset = int(elapsed_steps) - simulated_steps

    def _contact_curriculum(self) -> tuple[float, float]:
        """返回接触惩罚比例及碰撞重置共用的动态接触力阈值。"""

        curriculum = min(
            float(self.training_elapsed_steps)
            / (
                self.task.total_training_steps * self.task.contact_penalty_ramp_fraction
            ),
            1.0,
        )
        threshold = (
            self.task.collision_force_threshold * (1.0 - curriculum)
            + self.task.minimum_contact_force_threshold * curriculum
        )
        self.training_curriculum_contact_scale = curriculum
        self.training_curriculum_contact_force_threshold = threshold
        return curriculum, threshold

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        distance = torch.linalg.vector_norm(
            self.goal_positions[:, :2] - self.robot.data.root_pos_w[:, :2], dim=-1
        )
        success = distance <= self.task.goal_threshold
        _, contact_force_threshold = self._contact_curriculum()
        collided = self._contact_force() >= contact_force_threshold
        local_xy = self.robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        outside = torch.any(torch.abs(local_xy) > 19.5, dim=-1)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = success | collided | outside

        self.extras = {
            "success": success.clone(),
            "collided": collided.clone(),
            "time_out": time_out.clone(),
            "outside": outside.clone(),
            "distance_to_goal": distance.clone(),
        }
        if torch.any(terminated | time_out):
            terminal = self._get_observations()["policy"]
            self.extras["terminal_observation"] = {
                key: value.clone() for key, value in terminal.items()
            }
        else:
            self.extras["terminal_observation"] = None
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        """计算与 manager-based training_mazes 等价的逐策略步奖励。"""

        goal_delta = self.goal_positions[:, :2] - self.robot.data.root_pos_w[:, :2]
        goal_direction = goal_delta / (
            torch.linalg.vector_norm(goal_delta, dim=-1, keepdim=True) + 1.0e-6
        )
        goal_velocity = torch.sum(self.robot.data.root_lin_vel_w[:, :2] * goal_direction, dim=-1)
        smoothness = torch.linalg.vector_norm(self.actions - self.previous_actions, dim=-1)

        local_position = self.robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        current_toa = self._sample_toa(local_position.unsqueeze(1)).squeeze(1)
        valid = torch.isfinite(self.previous_toa) & torch.isfinite(current_toa)
        toa_progress = torch.where(
            valid, self.previous_toa - current_toa, torch.zeros_like(current_toa)
        ).clamp(-0.25, 0.25)
        # 缓存只属于当前回合；_reset_idx 会将重置环境对应的值恢复为 NaN，
        # 从而保证新回合第一次奖励不会与上一个回合做 TOA 差分。
        self.previous_toa.copy_(current_toa.detach())

        curriculum, contact_force_threshold = self._contact_curriculum()
        contact_penalty = (self._contact_force() / contact_force_threshold).clamp(
            max=1.0
        ) * curriculum
        success = self.extras["success"].float()
        return (
            self.task.goal_velocity_weight * goal_velocity
            + self.task.toa_progress_weight * toa_progress
            + self.task.action_smoothness_weight * smoothness
            + self.task.contact_force_weight * contact_penalty
            + self.task.goal_reached_weight * success
        )

    @staticmethod
    def _yaw_quaternion(yaw: torch.Tensor) -> torch.Tensor:
        quaternion = torch.zeros((yaw.shape[0], 4), device=yaw.device)
        quaternion[:, 0] = torch.cos(yaw / 2)
        quaternion[:, 3] = torch.sin(yaw / 2)
        return quaternion

    def _sample_cylinder_positions(
        self,
        env_ids: torch.Tensor,
        start_xy: torch.Tensor,
        goal_xy: torch.Tensor,
        selected_wall_xy: torch.Tensor,
        selected_wall_yaw: torch.Tensor,
        selected_wall_length: torch.Tensor,
        selected_wall_active: torch.Tensor,
        cylinder_count: int,
    ) -> torch.Tensor:
        if not 0 <= cylinder_count <= self.task.random_cylinder_count:
            raise ValueError(
                f"cylinder_count 必须位于 [0, {self.task.random_cylinder_count}]，"
                f"实际为 {cylinder_count}"
            )
        positions = torch.empty((len(env_ids), cylinder_count, 2), device=self.device)
        vertical = torch.isclose(
            selected_wall_yaw.abs(),
            torch.full_like(selected_wall_yaw, torch.pi / 2),
            atol=1.0e-6,
            rtol=0.0,
        )
        half_x = torch.where(
            vertical, self.task.wall_thickness / 2, selected_wall_length / 2
        )
        half_y = torch.where(
            vertical, selected_wall_length / 2, self.task.wall_thickness / 2
        )
        for cylinder_id in range(cylinder_count):
            radius = self.cylinder_radii[cylinder_id]
            limit = self.task.arena_half_extent - self.task.wall_thickness / 2 - radius - 0.35
            unresolved = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
            for _ in range(128):
                if not torch.any(unresolved):
                    break
                candidate = (torch.rand((len(env_ids), 2), device=self.device) * 2 - 1) * limit
                delta = (candidate[:, None] - selected_wall_xy).abs()
                dx = torch.clamp(delta[..., 0] - half_x, min=0.0)
                dy = torch.clamp(delta[..., 1] - half_y, min=0.0)
                valid = torch.all(
                    (~selected_wall_active) | (dx.square() + dy.square() >= (radius + 0.35).square()),
                    dim=1,
                )
                clearance = (radius + 1.0).square()
                valid &= torch.sum((candidate - start_xy).square(), dim=-1) >= clearance
                valid &= torch.sum((candidate - goal_xy).square(), dim=-1) >= clearance
                if cylinder_id:
                    distance = torch.sum(
                        (candidate[:, None] - positions[:, :cylinder_id]).square(), dim=-1
                    )
                    minimum = radius + self.cylinder_radii[:cylinder_id] + 0.35
                    valid &= torch.all(distance >= minimum.square(), dim=1)
                accepted = unresolved & valid
                positions[accepted, cylinder_id] = candidate[accepted]
                unresolved &= ~accepted
            if torch.any(unresolved):
                raise RuntimeError(f"无法放置第 {cylinder_id} 个随机圆柱")
        return positions

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        super()._reset_idx(env_ids)
        if self.task.debug:
            self._trajectory_counts[env_ids] = 0
        count = len(env_ids)
        curriculum_stage = select_training_maze_curriculum_stage(
            elapsed_steps=self.training_elapsed_steps,
            max_steps=self.task.total_training_steps,
            stage_fractions=self.task.training_curriculum_stage_fractions,
            maze_counts=self.task.training_curriculum_maze_counts,
            cylinder_counts=self.task.training_curriculum_cylinder_counts,
        )
        # --cylinders 可以减少实际创建的槽位；课程数量相应封顶，但默认 60
        # 个槽位时与参考任务的 0/0/10/30/60 调度完全一致。
        active_cylinder_count = min(
            curriculum_stage.cylinder_count, self.task.random_cylinder_count
        )
        self.training_curriculum_stage = curriculum_stage.index
        self.training_curriculum_maze_count = curriculum_stage.maze_count
        self.training_curriculum_cylinder_count = active_cylinder_count
        maze_ids = torch.remainder(env_ids, curriculum_stage.maze_count)
        route_ids = torch.randint(0, 2, (count,), device=self.device)
        reversed_direction = torch.rand(count, device=self.device) < 0.5
        starts = self.route_starts[maze_ids, route_ids]
        goals = self.route_goals[maze_ids, route_ids]
        start_xy = torch.where(reversed_direction[:, None], goals, starts)
        goal_xy = torch.where(reversed_direction[:, None], starts, goals)
        origins = self.scene.env_origins[env_ids]

        self.maze_ids[env_ids] = maze_ids
        self.route_ids[env_ids] = route_ids
        self.direction_reversed[env_ids] = reversed_direction
        self.toa_map_ids[env_ids] = maze_ids * 4 + route_ids * 2 + reversed_direction.long()
        self.goal_positions[env_ids, :2] = origins[:, :2] + goal_xy
        self.goal_positions[env_ids, 2] = 2.0
        self.previous_toa[env_ids] = torch.nan

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += origins
        root_state[:, :2] = origins[:, :2] + start_xy
        root_state[:, 2] = 2.0
        direction = goal_xy - start_xy
        yaw = torch.atan2(direction[:, 1], direction[:, 0])
        root_state[:, 3:7] = self._yaw_quaternion(yaw)
        root_state[:, 7:] = 0.0
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos[env_ids],
            self.robot.data.default_joint_vel[env_ids],
            env_ids=env_ids,
        )
        self.position_setpoints[env_ids] = root_state[:, :3]
        self.yaw_setpoints[env_ids, 0] = yaw

        selected_xy = self.wall_xy[maze_ids]
        selected_yaw = self.wall_yaw[maze_ids]
        selected_length = self.wall_length[maze_ids]
        selected_active = self.wall_active[maze_ids]
        for wall_id, wall in enumerate(self.inner_walls):
            pose = wall.data.default_root_state[env_ids, :7].clone()
            pose[:, :2] = origins[:, :2] + selected_xy[:, wall_id]
            pose[:, 2] = torch.where(selected_active[:, wall_id], 2.0, -10.0)
            pose[:, 3:7] = self._yaw_quaternion(selected_yaw[:, wall_id])
            wall.write_root_pose_to_sim(pose, env_ids)
            wall.write_root_velocity_to_sim(torch.zeros((count, 6), device=self.device), env_ids)

        cylinder_xy = self._sample_cylinder_positions(
            env_ids,
            start_xy,
            goal_xy,
            selected_xy,
            selected_yaw,
            selected_length,
            selected_active,
            active_cylinder_count,
        )
        cylinder_pose = self.random_cylinders.data.default_object_state[env_ids, :, :7].clone()
        # 未启用的固定槽位停放到地面以下，避免动态改变场景拓扑。
        cylinder_pose[:, :, :2] = origins[:, None, :2]
        cylinder_pose[:, :, 2] = -10.0
        if active_cylinder_count > 0:
            cylinder_pose[:, :active_cylinder_count, :2] = (
                origins[:, None, :2] + cylinder_xy
            )
            cylinder_pose[:, :active_cylinder_count, 2] = (
                self.cylinder_heights[:active_cylinder_count][None] / 2
            )
        self.random_cylinders.write_object_pose_to_sim(cylinder_pose, env_ids=env_ids)
        self.random_cylinders.write_object_velocity_to_sim(
            torch.zeros((count, self.task.random_cylinder_count, 6), device=self.device),
            env_ids=env_ids,
        )


__all__ = ["IsaacNavigationEnvCfg", "NavigationEnv", "make_isaac_env_cfg"]
