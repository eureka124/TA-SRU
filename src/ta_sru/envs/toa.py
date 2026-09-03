"""六迷宫的静态 TOA（到达时间）地图。"""

from __future__ import annotations

import numpy as np
import skfmm

from ta_sru.config import EnvConfig
from ta_sru.envs.layouts import MAZE_LAYOUTS, wall_rectangle


def layout_rectangles(config: EnvConfig, layout_id: int) -> list[tuple[float, float, float, float]]:
    """返回边界和内部墙体的 ``center_x, center_y, size_x, size_y``。"""

    extent = config.arena_half_extent
    thickness = config.wall_thickness
    rectangles = [
        (0.0, extent, 2 * extent, thickness),
        (0.0, -extent, 2 * extent, thickness),
        (extent, 0.0, thickness, 2 * extent),
        (-extent, 0.0, thickness, 2 * extent),
    ]
    rectangles.extend(
        wall_rectangle(wall, thickness, config.inner_wall_length)
        for wall in MAZE_LAYOUTS[layout_id].walls
    )
    return rectangles


def _rectangle_sdf(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    rectangle: tuple[float, float, float, float],
    inflation: float,
) -> np.ndarray:
    center_x, center_y, size_x, size_y = rectangle
    dx = np.abs(grid_x - center_x) - (size_x / 2 + inflation)
    dy = np.abs(grid_y - center_y) - (size_y / 2 + inflation)
    outside = np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0))
    return outside + np.minimum(np.maximum(dx, dy), 0.0)


def build_toa_map(
    config: EnvConfig,
    layout_id: int,
    goal_xy: tuple[float, float],
) -> np.ndarray:
    """用 Fast Marching Method 构建一张到达时间图。"""

    coordinates = np.linspace(
        -config.arena_half_extent,
        config.arena_half_extent,
        config.toa_grid_size,
        dtype=np.float32,
    )
    spacing = float(coordinates[1] - coordinates[0])
    grid_x, grid_y = np.meshgrid(coordinates, coordinates, indexing="xy")
    clearance = np.full_like(grid_x, 1.0e6, dtype=np.float32)
    for rectangle in layout_rectangles(config, layout_id):
        clearance = np.minimum(
            clearance,
            _rectangle_sdf(grid_x, grid_y, rectangle, config.drone_radius),
        )

    blocked = clearance <= 0.0
    speed = np.ones_like(clearance)
    near_wall = clearance <= config.toa_safe_distance
    speed[near_wall] = config.toa_slow_speed + (1.0 - config.toa_slow_speed) * (
        np.maximum(clearance[near_wall], 0.0) / config.toa_safe_distance
    )
    speed = np.ma.array(np.clip(speed, 1.0e-3, 1.0), mask=blocked)

    goal_radius = max(1.5 * spacing, 1.0e-3)
    level_set = np.hypot(grid_x - goal_xy[0], grid_y - goal_xy[1]) - goal_radius
    level_set = np.ma.array(level_set, mask=blocked)
    arrival_time = skfmm.travel_time(level_set, speed, dx=spacing)
    if np.ma.isMaskedArray(arrival_time):
        arrival_time = arrival_time.filled(1.0e6)
    arrival_time = np.asarray(arrival_time, dtype=np.float32)

    goal_x = int(np.argmin(np.abs(coordinates - goal_xy[0])))
    goal_y = int(np.argmin(np.abs(coordinates - goal_xy[1])))
    goal_time = float(arrival_time[goal_y, goal_x])
    if np.isfinite(goal_time):
        arrival_time = np.maximum(arrival_time - goal_time, 0.0)
    return arrival_time


def build_toa_bank(config: EnvConfig) -> np.ndarray:
    """构建 6 个迷宫 × 2 组路线 × 正反方向，共 24 张共享地图。"""

    maps = []
    for layout_id, layout in enumerate(MAZE_LAYOUTS):
        for start, goal in layout.routes:
            maps.append(build_toa_map(config, layout_id, goal))
            maps.append(build_toa_map(config, layout_id, start))
    return np.stack(maps).astype(np.float32, copy=False)
