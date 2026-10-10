#!/usr/bin/env python3
"""把场景地图渲染成 PNG，用于在不启动 Isaac Sim 的情况下检查随机布局。

默认按每张地图的本地坐标绘制围墙、障碍物、可通行起点池、目标点，以及若干采样回合
的起点→目标连线，并在终端打印障碍物数量、最小表面间距和每个采样回合的起点 TOA。
``--episodes 0`` 跳过 TOA 计算，只画几何，此时不需要 scikit-fmm。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ta_sru.config import SCENE_TYPES, EnvConfig
from ta_sru.envs.maze import build_map_pools
from ta_sru.envs.primitives import Disc, footprint_distance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-type", choices=SCENE_TYPES, default="obstacles")
    parser.add_argument("--map-split", choices=("train", "eval"), default="train")
    parser.add_argument("--maps", type=int, default=4, help="渲染的地图数量")
    parser.add_argument("--pixels", type=int, default=760, help="单张地图的图像边长")
    parser.add_argument(
        "--episodes", type=int, default=3, help="每张地图标注的采样回合数；0 只画几何"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("debug/preview"))
    parser.add_argument("--maze-seed", type=int, default=None)
    parser.add_argument("--maze-size", type=int, default=None)
    parser.add_argument("--maze-cell-size", type=float, default=None)
    parser.add_argument("--arena-size", type=float, default=None)
    parser.add_argument("--obstacle-min-separation", type=float, default=None)
    args = parser.parse_args()
    if args.maps <= 0 or args.pixels <= 0 or args.episodes < 0:
        parser.error("--maps 与 --pixels 必须为正数，--episodes 不能为负数")
    return args


def build_config(args: argparse.Namespace) -> EnvConfig:
    values = {
        "scene_type": args.scene_type,
        "train_map_count": args.maps,
        "eval_map_count": 1,
        "goals_per_map": EnvConfig().goals_per_map,
        "map_split": args.map_split,
        "toa_cache_dir": None,
    }
    if args.maze_seed is not None:
        values["maze_seed"] = args.maze_seed
    if args.maze_size is not None:
        values["maze_size"] = args.maze_size
    if args.maze_cell_size is not None:
        values["maze_cell_size"] = args.maze_cell_size
    if args.arena_size is not None:
        values["arena_size"] = args.arena_size
    if args.obstacle_min_separation is not None:
        values["obstacle_min_separation"] = args.obstacle_min_separation
    return EnvConfig(**values)


def render(maze, pixels: int, overlay: dict | None = None) -> Image.Image:
    """按地图本地坐标绘制俯视图，米到像素等比缩放。"""

    extent = maze.extent
    scale = pixels / (2.0 * extent)

    def point(x: float, y: float) -> tuple[float, float]:
        return (pixels / 2 + x * scale, pixels / 2 - y * scale)

    image = Image.new("RGB", (pixels, pixels), (247, 249, 252))
    draw = ImageDraw.Draw(image)
    # 起点采样环：最外围一圈格子的内缩带，仅作参考。
    ring = extent - 2.0 * maze.cell_size
    draw.rectangle(
        [point(-ring, ring), point(ring, -ring)], outline=(203, 213, 225), width=1
    )
    for step in range(1, int(2 * extent // 5) + 1):
        offset = step * 5.0 - extent
        draw.line([point(offset, -extent), point(offset, extent)], fill=(226, 232, 240))
        draw.line([point(-extent, offset), point(extent, offset)], fill=(226, 232, 240))

    for solid in maze.solids():
        if isinstance(solid, Disc):
            draw.ellipse(
                [
                    point(solid.x - solid.radius, solid.y + solid.radius),
                    point(solid.x + solid.radius, solid.y - solid.radius),
                ],
                fill=(100, 116, 139),
            )
        else:
            draw.polygon([point(*corner) for corner in solid.corners()], fill=(100, 116, 139))
    if overlay is not None:
        _draw_overlay(draw, point, overlay)
    return image


def _draw_overlay(draw, point, overlay: dict) -> None:
    """画出可通行起点池、目标点以及采样的起点→目标连线。"""

    for x, y in overlay["starts"]:
        px, py = point(x, y)
        draw.point((px, py), fill=(147, 197, 253))
    for index, (x, y) in enumerate(overlay["goals"]):
        px, py = point(x, y)
        draw.ellipse([px - 6, py - 6, px + 6, py + 6], outline=(22, 163, 74), width=2)
        draw.text((px + 8, py - 14), f"G{index}", fill=(21, 128, 61))
    for episode in overlay["episodes"]:
        start, goal = point(*episode["start"]), point(*episode["goal"])
        draw.line([start, goal], fill=(56, 189, 248), width=1)
        draw.ellipse([start[0] - 5, start[1] - 5, start[0] + 5, start[1] + 5], fill=(14, 165, 233))


def layout_summary(maze) -> str:
    """统计障碍物数量与最小表面间距，作为“互不重叠”的直接证据。"""

    walls = maze.wall_boxes()
    groups = maze.obstacles
    cylinders = sum(1 for group in groups if isinstance(group[0], Disc))
    solids = [solid for group in groups for solid in group]
    if not solids:
        return f"{maze.name}: 无额外障碍物（围墙 {len(walls)} 块）"
    gaps = [footprint_distance(solid, wall) for solid in solids for wall in walls] + [
        footprint_distance(solid, other)
        for index, group in enumerate(groups)
        for other_group in groups[index + 1 :]
        for solid in group
        for other in other_group
    ]
    return (
        f"{maze.name}: 圆柱 {cylinders} 个、U 形 {len(groups) - cylinders} 个、"
        f"实体 {len(solids)} 块、最小间距 {min(gaps):.3f} m"
    )


def build_overlay(config: EnvConfig, bank, map_id: int, episodes: int) -> dict:
    """组装一张地图的起点池、目标点和采样回合。"""

    from ta_sru.envs.toa import coordinates

    coords = coordinates(config)
    starts = coords[bank.starts[map_id][:, ::-1]]
    goals = bank.goals[
        map_id * config.goals_per_map : (map_id + 1) * config.goals_per_map
    ]
    sampled = []
    for index in range(episodes):
        episode = index + 1
        _, goal_id, start_xy, initial_toa = bank.sample(
            config, env_id=index, episode_id=episode, map_id=map_id
        )
        sampled.append(
            {
                "start": start_xy,
                "goal": goals[goal_id],
                "goal_id": goal_id,
                "distance": float(np.linalg.norm(goals[goal_id] - start_xy)),
                "toa": initial_toa,
            }
        )
    return {"starts": starts, "goals": goals, "episodes": sampled}


def main() -> None:
    args = parse_args()
    config = build_config(args)
    pools = build_map_pools(config)
    bank = None
    if args.episodes:
        # 导入放在这里，让 --episodes 0 的纯几何预览不需要 scikit-fmm。
        from ta_sru.envs.toa import build_toa_bank

        bank = build_toa_bank(config)
    destination = args.output_dir.expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    for map_id, maze in enumerate(pools[args.map_split]):
        overlay = (
            None if bank is None else build_overlay(config, bank, map_id, args.episodes)
        )
        path = destination / f"{args.scene_type}_{maze.name}.png"
        render(maze, args.pixels, overlay).save(path)
        print(layout_summary(maze))
        if overlay is not None:
            print(f"  目标 {len(overlay['goals'])} 个，可通行起点 {len(overlay['starts'])} 个")
            for episode in overlay["episodes"]:
                print(
                    f"  回合 G{episode['goal_id']}: 起点→终点直线 "
                    f"{episode['distance']:.1f} m，起点 TOA {episode['toa']:.1f}"
                )
        print(f"  已保存 {path}")
    print(f"场地边长 {2 * pools[args.map_split][0].extent:g} m，共 {len(pools[args.map_split])} 张地图")


if __name__ == "__main__":
    main()
