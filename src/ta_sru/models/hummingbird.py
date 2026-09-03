"""Hummingbird 四旋翼参数和姿态计算工具。

参数来自旧项目实际使用的 ``hummingbird.yaml``。这里不保留 USD、关节可视化和
OmniDrones 类层次，只留下本地 Lee 控制器所需的数据。刚体动力学由 Isaac Sim
PhysX 负责。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HummingbirdParameters:
    """Hummingbird 的物理参数（SI 单位）。"""

    mass: float = 0.716
    inertia: tuple[float, float, float] = (0.007, 0.007, 0.012)
    drag_coefficient: float = 0.2
    arm_lengths: tuple[float, ...] = (0.17, 0.17, 0.17, 0.17)
    rotor_angles: tuple[float, ...] = (
        0.0,
        1.57079632679,
        3.14159265359,
        -1.57079632679,
    )
    rotor_directions: tuple[float, ...] = (-1.0, 1.0, -1.0, 1.0)
    force_constants: tuple[float, ...] = (8.54858e-6,) * 4
    moment_constants: tuple[float, ...] = (1.3677728816219314e-7,) * 4
    max_rotation_velocities: tuple[float, ...] = (838.0,) * 4


def normalize(vector: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    return vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(eps)


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """把 ``(w, x, y, z)`` 四元数转换成旋转矩阵。"""

    quaternion = normalize(quaternion)
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def rotate_inverse(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """把世界坐标向量旋转到机体坐标系。"""

    rotation = quaternion_to_matrix(quaternion)
    return torch.matmul(rotation.transpose(-2, -1), vector.unsqueeze(-1)).squeeze(-1)


def yaw_from_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_to_quaternion(yaw: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(yaw)
    return torch.stack((torch.cos(yaw / 2), zeros, zeros, torch.sin(yaw / 2)), dim=-1)


