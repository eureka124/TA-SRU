"""障碍物图元与二维几何，不依赖 Isaac Sim。

DFS 迷宫把占据栅格转成轴对齐长方体，随机障碍场地直接给出圆柱与带朝向的长方体。
两类场景都用这里的图元描述水平截面，供碰撞网格、深度相机、TOA 距离场和摆放校验共用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# 圆柱碰撞与深度网格的棱柱边数。多边形取外接圆，保证网格不小于解析圆。
CYLINDER_SIDES = 12


@dataclass(frozen=True)
class Box:
    """绕竖直轴旋转 yaw 弧度的长方体，size_x/size_y 是水平截面边长。"""

    x: float
    y: float
    yaw: float
    size_x: float
    size_y: float
    z0: float
    z1: float

    def corners(self) -> np.ndarray:
        """返回 (4, 2) 水平截面顶点，逆时针排列。"""
        half_x, half_y = self.size_x / 2.0, self.size_y / 2.0
        local = np.asarray(
            ((-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y))
        )
        cosine, sine = math.cos(self.yaw), math.sin(self.yaw)
        rotation = np.asarray(((cosine, -sine), (sine, cosine)))
        return local @ rotation.T + np.asarray((self.x, self.y))


@dataclass(frozen=True)
class Disc:
    """竖直圆柱障碍物，radius 是水平半径。"""

    x: float
    y: float
    radius: float
    z0: float
    z1: float

    def corners(self) -> np.ndarray:
        """用外接正多边形逼近水平截面，保证网格大于解析圆。"""
        angles = np.linspace(0.0, 2.0 * math.pi, CYLINDER_SIDES, endpoint=False)
        reach = self.radius / math.cos(math.pi / CYLINDER_SIDES)
        return np.column_stack(
            (self.x + reach * np.cos(angles), self.y + reach * np.sin(angles))
        )


Solid = Box | Disc


def footprint_sdf(points: np.ndarray, solid: Solid) -> np.ndarray:
    """points（..., 2）到图元水平截面的有符号距离，内部为负。"""

    offset_x = points[..., 0] - solid.x
    offset_y = points[..., 1] - solid.y
    if isinstance(solid, Disc):
        return np.hypot(offset_x, offset_y) - solid.radius
    # 三角函数的操作数保持 Python float，避免向上转型成 float64 距离场。
    cosine, sine = math.cos(solid.yaw), math.sin(solid.yaw)
    local_x = offset_x * cosine + offset_y * sine
    local_y = offset_y * cosine - offset_x * sine
    delta_x = np.abs(local_x) - solid.size_x / 2.0
    delta_y = np.abs(local_y) - solid.size_y / 2.0
    return np.hypot(np.maximum(delta_x, 0.0), np.maximum(delta_y, 0.0)) + np.minimum(
        np.maximum(delta_x, delta_y), 0.0
    )


def footprint_distance(a: Solid, b: Solid) -> float:
    """两个图元水平截面的表面最小距离；相交或包含时返回 0。"""

    if isinstance(a, Disc) and isinstance(b, Disc):
        return max(math.hypot(a.x - b.x, a.y - b.y) - a.radius - b.radius, 0.0)
    if isinstance(a, Disc) or isinstance(b, Disc):
        disc, box = (a, b) if isinstance(a, Disc) else (b, a)
        centre = np.asarray(((disc.x, disc.y),))
        return max(float(footprint_sdf(centre, box)[0]) - disc.radius, 0.0)
    if _boxes_intersect(a, b):
        return 0.0
    # 不相交的凸多边形之间，最近点必是某一方顶点到另一方的垂足或顶点。
    return min(_corner_distance(a, b), _corner_distance(b, a))


def solid_record(solid: Solid) -> list:
    """图元的可序列化描述，用于内容散列与调试输出。"""

    if isinstance(solid, Disc):
        return ["disc", float(solid.x), float(solid.y), float(solid.radius)]
    return [
        "box",
        float(solid.x),
        float(solid.y),
        float(solid.yaw),
        float(solid.size_x),
        float(solid.size_y),
    ]


def _corner_distance(source: Box, target: Box) -> float:
    """source 顶点到 target 的最小距离；有顶点落在 target 内时为 0。"""

    return max(float(footprint_sdf(source.corners(), target).min()), 0.0)


def _boxes_intersect(a: Box, b: Box) -> bool:
    """二维分离轴测试；共边或共点算相交，与距离公式的零值保持一致。"""

    corners_a, corners_b = a.corners(), b.corners()
    for corners in (corners_a, corners_b):
        for index in range(len(corners)):
            axis = corners[(index + 1) % len(corners)] - corners[index]
            projection_a = corners_a @ axis
            projection_b = corners_b @ axis
            if projection_a.max() < projection_b.min() or (
                projection_b.max() < projection_a.min()
            ):
                return False
    return True
