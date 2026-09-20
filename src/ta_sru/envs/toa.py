"""六迷宫的静态 TOA（到达时间）地图。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import skfmm

from ta_sru.config import EnvConfig
from ta_sru.envs.layouts import MAZE_LAYOUTS, wall_rectangle


def layout_rectangles(
    config: EnvConfig, layout_id: int
) -> list[tuple[float, float, float, float]]:
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


def _traversable_toa_max(values: np.ndarray) -> float:
    """返回剔除墙体/不可达区域后的最大 TOA。"""

    traversable = np.isfinite(values) & (values < 1.0e5)
    if not np.any(traversable):
        return 1.0
    return max(float(np.max(values[traversable])), 1.0e-6)


def resolve_toa_normalization_max(
    values: np.ndarray, configured: float | None
) -> float:
    """以全部共享地图的有效范围统一缩放，排除墙体和不可达哨兵值。"""
    if configured is not None:
        if not np.isfinite(configured) or configured <= 0.0:
            raise ValueError("TOA 归一化量程必须为有限正数")
        return float(configured)
    return _traversable_toa_max(values)


def _toa_rgb(values: np.ndarray) -> np.ndarray:
    """使用蓝到红的 HSV 色相渐变渲染一张 TOA 图。"""

    blocked = ~np.isfinite(values) | (values >= 1.0e5)
    normalized = np.clip(values / _traversable_toa_max(values), 0.0, 1.0)

    # HSV: 最短到达时间为蓝色，最大到达时间为红色。
    hue = (1.0 - normalized) * (2.0 / 3.0)
    sector = np.floor(hue * 6.0).astype(np.int64)
    fraction = hue * 6.0 - sector
    value = np.ones_like(hue)
    zero = np.zeros_like(hue)
    inverse = 1.0 - fraction
    choices = (
        np.stack((value, fraction, zero), axis=-1),
        np.stack((inverse, value, zero), axis=-1),
        np.stack((zero, value, fraction), axis=-1),
        np.stack((zero, inverse, value), axis=-1),
        np.stack((fraction, zero, value), axis=-1),
        np.stack((value, zero, inverse), axis=-1),
    )
    rgb = np.choose((sector % 6)[..., None], choices) * 255.0
    rgb[blocked] = (20, 20, 20)
    # 数组第 0 行对应最小 y；保存时翻转，使地图上方对应世界坐标 +Y。
    return np.flipud(rgb.astype(np.uint8))


def save_toa_global_maps(
    config: EnvConfig,
    toa_bank: np.ndarray,
    output_dir: str | Path,
) -> list[Path]:
    """为六种迷宫各保存一张 4×1 全局 TOA 汇总图和原始数组。

    每张 PNG 依次包含路线 1 正向/反向、路线 2 正向/反向。原始 ``npz``
    中保留相同的四张浮点 TOA 图，便于后续分析。
    """

    from PIL import Image, ImageDraw

    expected_maps = len(MAZE_LAYOUTS) * 4
    if toa_bank.shape != (expected_maps, config.toa_grid_size, config.toa_grid_size):
        raise ValueError(
            f"TOA bank 形状应为 {(expected_maps, config.toa_grid_size, config.toa_grid_size)}，"
            f"实际为 {toa_bank.shape}"
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    labels = (
        "route 1 forward",
        "route 1 reverse",
        "route 2 forward",
        "route 2 reverse",
    )
    tile_size = config.toa_grid_size
    label_height = 24
    gap = 8
    canvas_size = 4 * tile_size + 3 * gap, tile_size + label_height
    saved: list[Path] = []

    for layout_id, layout in enumerate(MAZE_LAYOUTS):
        maps = toa_bank[layout_id * 4 : layout_id * 4 + 4]
        canvas = Image.new("RGB", canvas_size, (245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        for map_id, (values, label) in enumerate(zip(maps, labels)):
            column, row = map_id, 0
            left = column * (tile_size + gap)
            top = row * (tile_size + label_height + gap)
            maximum = _traversable_toa_max(values)
            draw.text(
                (left + 4, top + 4),
                f"{label} | traversable max: {maximum:.2f}",
                fill=(20, 20, 20),
            )
            tile = Image.fromarray(_toa_rgb(values), mode="RGB")
            canvas.paste(tile, (left, top + label_height))

            route_id = map_id // 2
            reversed_direction = bool(map_id % 2)
            route_start, route_goal = layout.routes[route_id]
            start, goal = (
                (route_goal, route_start)
                if reversed_direction
                else (route_start, route_goal)
            )

            def pixel(
                point: tuple[float, float], left: int = left, top: int = top
            ) -> tuple[int, int]:
                scale = (tile_size - 1) / (2.0 * config.arena_half_extent)
                x = round((point[0] + config.arena_half_extent) * scale) + left
                y = (
                    round((config.arena_half_extent - point[1]) * scale)
                    + top
                    + label_height
                )
                return x, y

            for point, color in ((start, (0, 255, 255)), (goal, (255, 70, 70))):
                x, y = pixel(point)
                radius = max(3, tile_size // 100)
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius), fill=color
                )

        image_path = destination / f"{layout.name}_toa_global.png"
        data_path = destination / f"{layout.name}_toa_global.npz"
        canvas.save(image_path)
        np.savez_compressed(
            data_path,
            toa=maps,
            labels=np.asarray(labels),
            arena_half_extent=np.float32(config.arena_half_extent),
        )
        saved.append(image_path)
    return saved


__all__ = ["build_toa_bank", "build_toa_map", "save_toa_global_maps"]
