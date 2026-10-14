"""独立无障碍控制器测试场景；必须在 AppLauncher 启动后导入。"""

from __future__ import annotations

from dataclasses import asdict

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ta_sru.config import EnvConfig
from ta_sru.models.hummingbird import HummingbirdParameters
from ta_sru.models.hummingbird_asset import HUMMINGBIRD_CFG
from ta_sru.models.lee_controller import LeePositionController
from ta_sru.models.navigation_control import compute_velocity_yaw_wrench


@configclass
class ControllerTestSceneCfg(InteractiveSceneCfg):
    """只包含地面、灯光、无人机与接触传感器。"""

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        collision_group=-1,
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 0.75)),
    )
    robot = HUMMINGBIRD_CFG.copy()
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(base_link|rotor_0|rotor_1|rotor_2|rotor_3)",
        update_period=1.0 / 128.0,
        history_length=4,
    )


class ControllerStepTestRig:
    """按物理步执行共享控制逻辑，不包含回合或自动重置。"""

    def __init__(self, device: str, env_spacing: float) -> None:
        self.task = EnvConfig(num_envs=3, control_decimation=4)
        self.physics_dt = self.task.physics_dt
        self.sample_dt = self.task.policy_dt
        sim_cfg = sim_utils.SimulationCfg(
            device=device,
            dt=self.physics_dt,
            render_interval=self.task.control_decimation,
        )
        sim_cfg.physx.enable_external_forces_every_iteration = True
        sim_cfg.physx.gpu_max_rigid_contact_count = 2**20
        self.sim = sim_utils.SimulationContext(sim_cfg)
        scene_cfg = ControllerTestSceneCfg(num_envs=3, env_spacing=env_spacing)
        scene_cfg.ground.spawn.size = (env_spacing * 6, env_spacing * 6)
        scene_cfg.contact_forces.update_period = self.physics_dt
        self.scene = InteractiveScene(scene_cfg)
        self.sim.set_camera_view((8.0, 8.0, 6.0), (0.0, 0.0, 2.0))
        self.sim.reset()
        self.robot = self.scene["robot"]
        self.contact_sensor = self.scene["contact_forces"]
        self.device = self.robot.device
        self.parameters = HummingbirdParameters()
        self.controller = LeePositionController(self.parameters).to(self.device)
        self.base_link_ids, names = self.robot.find_bodies("base_link")
        if len(self.base_link_ids) != 1:
            raise RuntimeError(f"需要唯一的 base_link，实际找到 {names}")

        root_state = self.robot.data.default_root_state.clone()
        root_state[:, :3] = self.scene.env_origins
        root_state[:, 2] += 2.0
        root_state[:, 3:7] = root_state.new_tensor((1.0, 0.0, 0.0, 0.0))
        root_state[:, 7:] = 0.0
        self.robot.write_root_pose_to_sim(root_state[:, :7])
        self.robot.write_root_velocity_to_sim(root_state[:, 7:])
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos, self.robot.data.default_joint_vel
        )
        self.scene.reset()
        self.initial_root_state = root_state.clone()
        self.position_setpoints = root_state[:, :3].clone()
        self.yaw_setpoints = root_state.new_zeros((3, 1))
        self.actions = root_state.new_zeros((3, 3))
        self.last_wrench = root_state.new_zeros((3, 4))
        self.physics_steps = 0

    @torch.no_grad()
    def step(self, commands: torch.Tensor) -> None:
        """执行一个物理步，保持与训练相同的动作限幅和机体系力矩施加方式。"""

        self.actions.copy_(
            commands.clamp(
                commands.new_tensor(self.task.action_low),
                commands.new_tensor(self.task.action_high),
            )
        )
        wrench = compute_velocity_yaw_wrench(
            self.controller,
            self.robot.data.root_state_w,
            self.actions,
            self.position_setpoints,
            self.yaw_setpoints,
            self.physics_dt,
        )
        if not torch.isfinite(wrench).all():
            raise FloatingPointError("控制器输出了非有限推力或力矩")
        self.last_wrench.copy_(wrench)
        forces = wrench.new_zeros((3, 1, 3))
        forces[:, 0, 2] = wrench[:, 0]
        self.robot.set_external_force_and_torque(
            forces,
            wrench[:, 1:4].unsqueeze(1),
            body_ids=self.base_link_ids,
            is_global=False,
        )
        self.scene.write_data_to_sim()
        # Isaac Sim 4.5 的带渲染 step 可能推进多个物理子步，必须分开调用。
        self.sim.step(render=False)
        self.physics_steps += 1
        if self.physics_steps % self.task.control_decimation == 0:
            self.sim.render()
        self.scene.update(self.physics_dt)

    def contact_force(self) -> torch.Tensor:
        """返回当前物理步各无人机全部机体部件的最大接触力。"""

        return torch.linalg.vector_norm(
            self.contact_sensor.data.net_forces_w, dim=-1
        ).amax(dim=-1)

    def metadata(self) -> dict:
        """记录实际使用的物理参数、原始控制增益与初始状态。"""

        return {
            "parameters": asdict(self.parameters),
            "physics_dt": self.physics_dt,
            "sample_dt": self.sample_dt,
            "action_low": self.task.action_low,
            "action_high": self.task.action_high,
            "contact_threshold_n": self.task.collision_force_threshold,
            "position_gain": self.controller.position_gain.cpu().tolist(),
            "velocity_gain": self.controller.velocity_gain.cpu().tolist(),
            "attitude_gain": (self.controller.attitude_gain * self.controller.inertia)
            .cpu()
            .tolist(),
            "angular_rate_gain": (
                self.controller.angular_rate_gain * self.controller.inertia
            )
            .cpu()
            .tolist(),
            "initial_root_states": self.initial_root_state.cpu().tolist(),
            "body_masses": self.robot.root_physx_view.get_masses().cpu().tolist(),
            "env_spacing": self.scene.cfg.env_spacing,
        }
