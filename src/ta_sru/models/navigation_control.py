"""训练与阶跃测试共用的速度、偏航动作转换。"""

from __future__ import annotations

import torch

from ta_sru.models.lee_controller import LeePositionController


def compute_velocity_yaw_wrench(
    controller: LeePositionController,
    root_state: torch.Tensor,
    actions: torch.Tensor,
    position_setpoints: torch.Tensor,
    yaw_setpoints: torch.Tensor,
    physics_dt: float,
) -> torch.Tensor:
    """积分已限幅的动作并返回推力、力矩；原地更新位置与偏航设定点。"""

    quaternion = root_state[:, 3:7]
    w, x, y, z = quaternion.unbind(dim=-1)
    forward = torch.stack((w * w + x * x - y * y - z * z, 2 * (x * y + w * z)), dim=-1)
    forward = forward / torch.linalg.vector_norm(
        forward, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    side = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
    target_velocity_xy = forward * actions[:, 0:1] + side * actions[:, 1:2]
    target_velocity = torch.cat(
        (target_velocity_xy, root_state.new_zeros((len(root_state), 1))), dim=-1
    )
    position_setpoints[:, :2] += target_velocity_xy * physics_dt
    position_setpoints[:, 2] = 2.0
    yaw_setpoints += actions[:, 2:3] * physics_dt
    yaw_setpoints[:] = (yaw_setpoints + torch.pi) % (2 * torch.pi) - torch.pi
    return controller(
        root_state,
        position_setpoints,
        target_velocity,
        torch.zeros_like(target_velocity),
        yaw_setpoints,
    )
