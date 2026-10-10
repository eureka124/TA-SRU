"""Isaac Lab DirectRLEnv：DFS 地图池 Hummingbird 导航任务。

本模块必须在 ``AppLauncher`` 启动 Isaac Sim 后导入。
"""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import (
    ContactSensor,
    ContactSensorCfg,
    MultiMeshRayCasterCamera,
    MultiMeshRayCasterCameraCfg,
)
from isaaclab.sensors.ray_caster import patterns
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass

from ta_sru.config import EnvConfig
from ta_sru.envs.contact import peak_contact_force
from ta_sru.envs.depth import (
    DEPTH_DOWNSAMPLE,
    downsample_depth,
    mask_invalid_depth,
    sanitize_depth,
)
from ta_sru.envs.maze import pool_mesh, pool_origins
from ta_sru.envs.toa import (
    build_toa_bank,
    resolve_toa_normalization_max,
    save_toa_global_maps,
)
from ta_sru.envs.toa_sampling import normalized_progress, sample_toa
from ta_sru.evaluation import MapQuotaScheduler
from ta_sru.models.hummingbird import (
    HummingbirdParameters,
    rotate_inverse,
    yaw_from_quaternion,
)
from ta_sru.models.hummingbird_asset import HUMMINGBIRD_CFG
from ta_sru.models.lee_controller import LeePositionController

TRAJECTORY_MAX_POINTS = 512
TRAJECTORY_POINT_SPACING = 0.15


def _spawn_dfs_mesh(prim_path, cfg, translation=None, orientation=None):
    """在场景解析期间创建全局网格，让首次碰撞过滤包含地图。"""
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = SimulationContext.instance().stage
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    # 显式创建标准变换顺序，兼容新版 Isaac Lab 的 XformPrimView 校验。
    xform = UsdGeom.Xformable(mesh.GetPrim())
    position = (0.0, 0.0, 0.0) if translation is None else translation
    rotation = (1.0, 0.0, 0.0, 0.0) if orientation is None else orientation
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*position))
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(rotation[0], Gf.Vec3d(*rotation[1:]))
    )
    xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 1.0, 1.0))
    mesh.CreatePointsAttr(cfg.vertices.tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(cfg.faces))
    mesh.CreateFaceVertexIndicesAttr(cfg.faces.reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    return mesh.GetPrim()


@configclass
class DfsMeshCfg(sim_utils.SpawnerCfg):
    func = _spawn_dfs_mesh
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None


@configclass
class NavigationSceneCfg(InteractiveSceneCfg):
    """各环境只克隆机器人和传感器，静态地图池全局共享。"""

    robot = HUMMINGBIRD_CFG
    dfs_maps: AssetBaseCfg | None = None
    camera = MultiMeshRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.04,
        mesh_prim_paths=["/World/DfsMaps"],
        data_types=["distance_to_image_plane"],
        max_distance=12.0,
        depth_clipping_behavior="max",
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=10.4775,
            horizontal_aperture=20.955,
            height=192,
            width=256,
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
        history_length=5,
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
    # 占位值；make_isaac_env_cfg 会按 EnvConfig 重新填写，这里的默认值只为让
    # 直接构造的测试配置也能通过校验。
    observation_space = {  # noqa: RUF012
        "camera": gym.spaces.Box(0.0, 10.0, (1, 48, 64), dtype=np.float32),
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
        filter_collisions=False,
    )


def make_isaac_env_cfg(task: EnvConfig, sim_device: str) -> IsaacNavigationEnvCfg:
    """把与训练框架无关的配置转换成 Isaac Lab 配置。"""

    task.validate()
    cfg = IsaacNavigationEnvCfg()
    cfg.seed = task.seed
    cfg.decimation = task.control_decimation
    cfg.episode_length_s = task.episode_seconds
    cfg.sim.dt = task.physics_dt
    cfg.sim.render_interval = task.control_decimation
    cfg.sim.device = sim_device
    cfg.scene.num_envs = task.num_envs
    cfg.debug_vis = task.debug
    # 历史窗口恰好覆盖一个策略步，并随物理步长和抽取倍数同步调整。
    cfg.scene.contact_forces.update_period = task.physics_dt
    cfg.scene.contact_forces.history_length = task.control_decimation
    cfg.action_space = gym.spaces.Box(
        np.asarray(task.action_low, dtype=np.float32),
        np.asarray(task.action_high, dtype=np.float32),
        dtype=np.float32,
    )
    cfg.observation_space = {
        # 编码器输入是米制深度，0 表示无效，因此上界就是量程上限。
        "camera": gym.spaces.Box(
            0.0,
            task.depth_max_distance,
            (1, task.depth_height, task.depth_width),
            dtype=np.float32,
        ),
        "robot_state": gym.spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32),
        "critic_toa": gym.spaces.Box(
            -1.0, 1.0, (1, task.toa_crop_size, task.toa_crop_size), dtype=np.float32
        ),
    }
    # 相机按网络输入的分辨率乘以池化倍数配置，两处必须用同一个常量。
    cfg.scene.camera.pattern_cfg.height = task.depth_height * DEPTH_DOWNSAMPLE
    cfg.scene.camera.pattern_cfg.width = task.depth_width * DEPTH_DOWNSAMPLE
    cfg.scene.camera.update_period = task.policy_dt
    cfg.scene.camera.max_distance = task.depth_max_distance
    # 地图池按场地范围加相机截断间距平铺，环境间距必须与 pool_origins 保持一致。
    cfg.scene.env_spacing = 2 * task.arena_half_extent + 2 * task.depth_max_distance + 2
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
        self.toa_bank = build_toa_bank(self.task)
        self.map_definitions = self.toa_bank.maps
        self.map_names = [maze.name for maze in self.map_definitions]
        self._pool_origins = pool_origins(
            len(self.map_names),
            self.task.arena_half_extent,
            2 * self.task.depth_max_distance + 2,
        )
        vertices, faces = pool_mesh(self.map_definitions, self._pool_origins)
        cfg.scene.dfs_maps = AssetBaseCfg(
            prim_path="/World/DfsMaps",
            collision_group=-1,
            spawn=DfsMeshCfg(vertices=vertices, faces=faces),
        )
        self.scene_manifest = self.toa_bank.manifest
        self.quota_scheduler = (
            MapQuotaScheduler(
                len(self.map_names), self.task.evaluation_episodes_per_map
            )
            if self.task.evaluation_episodes_per_map
            else None
        )
        self.episode_counts = np.full(self.task.num_envs, -1, dtype=np.int64)
        # 仅由评估入口安装记录器，训练时不采集视频或回放数据。
        self.play_debug_recorder = None
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
        self.map_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.goal_ids = torch.zeros_like(self.map_ids)
        self.toa_map_ids = torch.zeros_like(self.map_ids)
        self.map_origins = torch.zeros((self.num_envs, 3), device=self.device)
        self.start_positions = torch.zeros((self.num_envs, 2), device=self.device)
        self.evaluation_active = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.episode_start_toa = torch.ones(self.num_envs, device=self.device)
        self.previous_toa = torch.full((self.num_envs,), torch.nan, device=self.device)
        self.training_step_offset = 0
        self.training_curriculum_contact_scale = 0.0
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
        # 显式设置各机器人碰撞组和全局地图；CPU 的首次过滤也包含该路径。
        self.scene.filter_collisions(global_prim_paths=["/World/DfsMaps"])

    def _create_toa_tensors(self) -> None:
        self.toa_normalization_max = resolve_toa_normalization_max(
            self.toa_bank.values, self.task.toa_normalization_max
        )
        self.task.toa_normalization_max = self.toa_normalization_max
        self.toa_maps = torch.from_numpy(self.toa_bank.values).to(self.device)
        self.pool_origins = torch.from_numpy(self._pool_origins).to(self.device)
        self.target_positions = torch.from_numpy(self.toa_bank.goals).to(self.device)
        if self.task.debug:
            output_dir = self.task.debug_output_dir or "debug"
            saved = save_toa_global_maps(self.task, self.toa_bank, output_dir)
            print(f"[DEBUG] 已保存 {len(saved)} 张目标 TOA 图：{output_dir}")
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
        cone_orientations = self._yaw_quaternion(
            torch.atan2(velocity_xy[:, 1], velocity_xy[:, 0])
        )
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
        moved = (
            torch.linalg.vector_norm(positions - last_positions, dim=-1)
            >= TRAJECTORY_POINT_SPACING
        )
        append = (empty | moved) & (self._trajectory_counts < TRAJECTORY_MAX_POINTS)
        env_ids = torch.nonzero(append, as_tuple=False).squeeze(-1)
        if len(env_ids):
            slots = self._trajectory_counts[env_ids]
            self._trajectory_history[env_ids, slots] = positions[env_ids]
            self._trajectory_counts[env_ids] += 1

        valid = (
            torch.arange(TRAJECTORY_MAX_POINTS, device=self.device)[None, :]
            < (self._trajectory_counts[:, None])
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
        forward = torch.stack(
            (w * w + x * x - y * y - z * z, 2 * (x * y + w * z)), dim=-1
        )
        forward = forward / torch.linalg.vector_norm(
            forward, dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        side = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
        target_velocity_xy = (
            forward * self.actions[:, 0:1] + side * self.actions[:, 1:2]
        )
        target_velocity = torch.cat(
            (target_velocity_xy, torch.zeros((self.num_envs, 1), device=self.device)),
            dim=-1,
        )
        self.position_setpoints[:, :2] += target_velocity_xy * self.physics_dt
        self.position_setpoints[:, 2] = 2.0
        self.yaw_setpoints += self.actions[:, 2:3] * self.physics_dt
        self.yaw_setpoints[:] = (self.yaw_setpoints + torch.pi) % (
            2 * torch.pi
        ) - torch.pi

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
        """相机读数 → 米制深度图，量程外与无效读数置 0。"""

        depth = self.camera.data.output["distance_to_image_plane"].clone()
        if depth.ndim == 3:
            depth = depth.unsqueeze(-1)
        depth = depth.permute(0, 3, 1, 2)
        depth = sanitize_depth(
            depth, invalid_distance=self.task.depth_invalid_distance
        )
        depth = downsample_depth(depth)
        return mask_invalid_depth(
            depth,
            min_distance=self.task.depth_min_distance,
            max_distance=self.task.depth_max_distance,
        )

    def _sample_toa(self, points_xy: torch.Tensor) -> torch.Tensor:
        values, _ = sample_toa(
            self.toa_maps, self.toa_map_ids, points_xy, self.task.arena_half_extent
        )
        return values

    def _critic_toa_observation(self) -> torch.Tensor:
        local_position = self.robot.data.root_pos_w[:, :2] - self.map_origins[:, :2]
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
            nan=self.toa_normalization_max,
            posinf=self.toa_normalization_max,
            neginf=0.0,
        )
        values = values.clamp(0.0, self.toa_normalization_max)
        values = values / self.toa_normalization_max * 2.0 - 1.0
        return values.reshape(
            self.num_envs, 1, self.task.toa_crop_size, self.task.toa_crop_size
        )

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        quaternion = self.robot.data.root_quat_w
        relative_goal = rotate_inverse(
            quaternion, self.goal_positions - self.robot.data.root_pos_w
        )
        velocity_body = rotate_inverse(quaternion, self.robot.data.root_lin_vel_w)
        angular_velocity_body = rotate_inverse(
            quaternion, self.robot.data.root_ang_vel_w
        )
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
        # 历史从新到旧排列；只取本策略步，避免短暂接触漏检或跨步重复惩罚。
        return peak_contact_force(
            self.contact_sensor.data.net_forces_w_history, self.cfg.decimation
        )

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

    def _contact_penalty_scale(self) -> float:
        """返回接触惩罚渐增比例，碰撞判定阈值始终固定。"""

        curriculum = min(
            float(self.training_elapsed_steps)
            / (
                self.task.total_training_steps * self.task.contact_penalty_ramp_fraction
            ),
            1.0,
        )
        self.training_curriculum_contact_scale = curriculum
        return curriculum

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        distance = torch.linalg.vector_norm(
            self.goal_positions[:, :2] - self.robot.data.root_pos_w[:, :2], dim=-1
        )
        success = distance <= self.task.goal_threshold
        collided = self._contact_force() >= self.task.collision_force_threshold
        local_xy = self.robot.data.root_pos_w[:, :2] - self.map_origins[:, :2]
        outside = torch.any(torch.abs(local_xy) > self.task.arena_half_extent, dim=-1)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = success | collided | outside

        self.extras = {
            "success": success.clone(),
            "collided": collided.clone(),
            "time_out": time_out.clone(),
            "outside": outside.clone(),
            "distance_to_goal": distance.clone(),
            "map_id": self.map_ids.clone(),
            "goal_id": self.goal_ids.clone(),
            "evaluation_active": self.evaluation_active.clone(),
            "episode_start_toa": self.episode_start_toa.clone(),
            "start_xy": self.start_positions.clone(),
            "episode_id": torch.as_tensor(
                self.episode_counts.copy(), device=self.device
            ),
        }
        if torch.any(terminated | time_out):
            terminal = self._get_observations()["policy"]
            self.extras["terminal_observation"] = {
                key: value.clone() for key, value in terminal.items()
            }
        else:
            self.extras["terminal_observation"] = None
        if self.play_debug_recorder is not None:
            # DirectRLEnv 随后会自动重置，必须在这里保存终止帧和真实末端位置。
            self.play_debug_recorder.capture(self, terminated | time_out)
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        """计算按起点 TOA 归一化的逐策略步奖励。"""

        goal_delta = self.goal_positions[:, :2] - self.robot.data.root_pos_w[:, :2]
        goal_direction = goal_delta / (
            torch.linalg.vector_norm(goal_delta, dim=-1, keepdim=True) + 1.0e-6
        )
        goal_velocity = torch.sum(
            self.robot.data.root_lin_vel_w[:, :2] * goal_direction, dim=-1
        )
        smoothness = torch.linalg.vector_norm(
            self.actions - self.previous_actions, dim=-1
        )

        local_position = self.robot.data.root_pos_w[:, :2] - self.map_origins[:, :2]
        current_toa = self._sample_toa(local_position.unsqueeze(1)).squeeze(1)
        toa_progress = normalized_progress(
            self.previous_toa, current_toa, self.episode_start_toa
        )
        # 起点为本回合固定分母；无效采样切断差分历史，成功时不补发剩余进度。
        self.previous_toa.copy_(current_toa.detach())

        curriculum = self._contact_penalty_scale()
        contact_penalty = (
            self._contact_force() / self.task.collision_force_threshold
        ).clamp(max=1.0) * curriculum
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

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        super()._reset_idx(env_ids)
        # 动作观测和平滑惩罚的缓存只属于当前回合。
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        if self.task.debug:
            self._trajectory_counts[env_ids] = 0
        if len(env_ids) == 0:
            return
        samples, active = [], []
        for env_id in env_ids.cpu().tolist():
            self.episode_counts[env_id] += 1
            selected = self.quota_scheduler.assign() if self.quota_scheduler else None
            active.append(selected != -1)
            samples.append(
                self.toa_bank.sample(
                    self.task,
                    env_id,
                    int(self.episode_counts[env_id]),
                    0 if selected == -1 else selected,
                )
            )
        map_ids = torch.tensor([item[0] for item in samples], device=self.device)
        goal_ids = torch.tensor([item[1] for item in samples], device=self.device)
        start_xy = torch.as_tensor(
            np.stack([item[2] for item in samples]), device=self.device
        )
        initial_toa = torch.tensor([item[3] for item in samples], device=self.device)
        bank_ids = map_ids * self.task.goals_per_map + goal_ids
        goal_xy = self.target_positions[bank_ids]
        origins = self.pool_origins[map_ids]
        self.map_ids[env_ids] = map_ids
        self.goal_ids[env_ids] = goal_ids
        self.toa_map_ids[env_ids] = bank_ids
        self.map_origins[env_ids] = origins
        self.start_positions[env_ids] = start_xy
        self.evaluation_active[env_ids] = torch.tensor(active, device=self.device)
        self.goal_positions[env_ids, :2] = origins[:, :2] + goal_xy
        self.goal_positions[env_ids, 2] = 2.0
        self.episode_start_toa[env_ids] = initial_toa
        self.previous_toa[env_ids] = initial_toa

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

        # 在机器人传送后使下一次读取重新生成相机数据。
        self.camera.reset(env_ids)


__all__ = ["IsaacNavigationEnvCfg", "NavigationEnv", "make_isaac_env_cfg"]
