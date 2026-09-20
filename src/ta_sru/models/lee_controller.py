"""精简后的 Lee 位置控制器。

算法来自旧项目所用 OmniDrones 控制器，只去掉 TensorDict、TorchRL、YAML 注册器和
未使用的控制模式。输入输出是普通张量，因而可独立测试和替换仿真后端。
"""

from __future__ import annotations

import torch
from torch import nn

from ta_sru.models.hummingbird import (
    HummingbirdParameters,
    normalize,
    quaternion_to_matrix,
    rotate_inverse,
    yaw_from_quaternion,
)


class LeePositionController(nn.Module):
    """根据位置、速度和偏航目标计算总推力与机体系力矩。"""

    def __init__(
        self,
        parameters: HummingbirdParameters | None = None,
        position_gain: tuple[float, float, float] = (4.0, 4.0, 4.0),
        velocity_gain: tuple[float, float, float] = (2.2, 2.2, 2.2),
        attitude_gain: tuple[float, float, float] = (0.7, 0.7, 0.035),
        angular_rate_gain: tuple[float, float, float] = (0.1, 0.1, 0.025),
    ) -> None:
        super().__init__()
        parameters = parameters or HummingbirdParameters()
        inertia = torch.tensor(parameters.inertia, dtype=torch.float32)
        self.mass = parameters.mass
        self.register_buffer("position_gain", torch.tensor(position_gain))
        self.register_buffer("velocity_gain", torch.tensor(velocity_gain))
        self.register_buffer("attitude_gain", torch.tensor(attitude_gain) / inertia)
        self.register_buffer(
            "angular_rate_gain", torch.tensor(angular_rate_gain) / inertia
        )
        self.register_buffer("inertia", inertia)

    def forward(
        self,
        root_state: torch.Tensor,
        target_position: torch.Tensor | None = None,
        target_velocity: torch.Tensor | None = None,
        target_acceleration: torch.Tensor | None = None,
        target_yaw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """返回 ``[..., thrust, torque_x, torque_y, torque_z]``。"""

        position, quaternion, velocity, angular_velocity_world = torch.split(
            root_state, (3, 4, 3, 3), dim=-1
        )
        target_position = position if target_position is None else target_position
        target_velocity = (
            torch.zeros_like(velocity) if target_velocity is None else target_velocity
        )
        target_acceleration = (
            torch.zeros_like(velocity)
            if target_acceleration is None
            else target_acceleration
        )
        if target_yaw is None:
            target_yaw = yaw_from_quaternion(quaternion).unsqueeze(-1)
        elif target_yaw.ndim == root_state.ndim - 1:
            target_yaw = target_yaw.unsqueeze(-1)

        angular_velocity_body = rotate_inverse(quaternion, angular_velocity_world)
        acceleration_command = (
            (position - target_position) * self.position_gain
            + (velocity - target_velocity) * self.velocity_gain
            - root_state.new_tensor((0.0, 0.0, 9.81))
            - target_acceleration
        )

        rotation = quaternion_to_matrix(quaternion)
        desired_body_x = torch.cat(
            (
                torch.cos(target_yaw),
                torch.sin(target_yaw),
                torch.zeros_like(target_yaw),
            ),
            dim=-1,
        )
        desired_body_z = -normalize(acceleration_command)
        desired_body_y = normalize(
            torch.linalg.cross(desired_body_z, desired_body_x, dim=-1)
        )
        desired_rotation = torch.stack(
            (
                torch.linalg.cross(desired_body_y, desired_body_z, dim=-1),
                desired_body_y,
                desired_body_z,
            ),
            dim=-1,
        )

        error_matrix = 0.5 * (
            desired_rotation.transpose(-2, -1) @ rotation
            - rotation.transpose(-2, -1) @ desired_rotation
        )
        attitude_error = torch.stack(
            (error_matrix[..., 2, 1], error_matrix[..., 0, 2], error_matrix[..., 1, 0]),
            dim=-1,
        )
        angular_acceleration = (
            -attitude_error * self.attitude_gain
            - angular_velocity_body * self.angular_rate_gain
        )
        thrust = -self.mass * torch.sum(
            acceleration_command * rotation[..., :, 2], dim=-1
        )
        # 刚体方程为 JΩ_dot + Ω×(JΩ) = τ；补偿项在力矩域相加。
        gyroscopic_torque = torch.linalg.cross(
            angular_velocity_body, angular_velocity_body * self.inertia, dim=-1
        )
        torque = angular_acceleration * self.inertia + gyroscopic_torque
        return torch.cat((thrust.unsqueeze(-1), torque), dim=-1)
