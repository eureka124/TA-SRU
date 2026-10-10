"""点目标 FMM、目标池及磁盘缓存，不依赖 Isaac Sim。"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import skfmm

from ta_sru.config import EnvConfig
from ta_sru.envs.maze import (
    GENERATOR_VERSIONS,
    MazeDefinition,
    build_map_pools,
    digest,
    scene_version,
)
from ta_sru.envs.primitives import footprint_sdf

TOA_VERSION = "fmm-point-zero-v1"


def coordinates(config: EnvConfig) -> np.ndarray:
    return np.linspace(
        -config.arena_half_extent,
        config.arena_half_extent,
        config.toa_grid_size,
        dtype=np.float32,
    )


def obstacle_clearance(config: EnvConfig, maze: MazeDefinition) -> np.ndarray:
    grid_x, grid_y = np.meshgrid(coordinates(config), coordinates(config))
    points = np.stack((grid_x, grid_y), axis=-1)
    clearance = np.full(grid_x.shape, np.inf, dtype=np.float32)
    for solid in maze.solids():
        clearance = np.minimum(clearance, footprint_sdf(points, solid))
    return clearance - (config.drone_radius + config.safety_margin)


def components(free: np.ndarray) -> np.ndarray:
    """四连通分量避免对角穿墙，外圈由边界墙封闭。"""
    labels = np.full(free.shape, -1, dtype=np.int32)
    component = 0
    height, width = free.shape
    for y, x in zip(*np.nonzero(free)):
        if labels[y, x] >= 0:
            continue
        labels[y, x] = component
        queue = deque([(y, x)])
        while queue:
            row, col = queue.popleft()
            for nr, nc in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if (
                    0 <= nr < height
                    and 0 <= nc < width
                    and free[nr, nc]
                    and labels[nr, nc] < 0
                ):
                    labels[nr, nc] = component
                    queue.append((nr, nc))
        component += 1
    return labels


def start_candidates(config: EnvConfig, clearance: np.ndarray) -> np.ndarray:
    """只保留边界墙内侧第一圈格子中满足出生余量的 TOA 栅格点。"""
    candidates = np.argwhere(clearance > config.spawn_margin)
    cells = np.floor(
        (coordinates(config)[candidates] + config.arena_half_extent) / config.cell_size
    ).astype(np.int32)
    outer_ring = np.any((cells == 1) | (cells == config.maze_size - 2), axis=1)
    return candidates[outer_ring]


def sample_goals(
    config: EnvConfig, maze: MazeDefinition, clearance: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    candidates = np.argwhere(clearance > config.spawn_margin)
    starts = start_candidates(config, clearance)
    if not len(starts):
        raise ValueError(f"{maze.name} 最外围一圈格子没有满足出生余量的起点")
    rng = np.random.default_rng(np.random.SeedSequence([maze.seed, 1701]))
    order = rng.permutation(len(candidates))
    goals = []
    coords = coordinates(config)
    starts_xy = coords[starts[:, ::-1]]
    for index in order:
        goal = candidates[index]
        same = labels[starts[:, 0], starts[:, 1]] == labels[tuple(goal)]
        distance = np.linalg.norm(starts_xy - coords[goal[::-1]], axis=1)
        if np.any(same & (distance >= config.min_start_goal_distance)):
            goals.append(goal)
            if len(goals) == config.goals_per_map:
                return np.asarray(goals, dtype=np.int32)
    raise ValueError(
        f"{maze.name} 无法采样足够目标：最外围起点到目标的可达直线距离"
        f"需要至少 {config.min_start_goal_distance:g} m"
    )


def cache_key(config: EnvConfig, maze: MazeDefinition, goal: np.ndarray) -> str:
    return digest(
        {
            "version": TOA_VERSION,
            "solver": skfmm.__version__,
            "map": maze.content_hash,
            "goal": goal.tolist(),
            "shape": config.toa_grid_size,
            "spacing": config.toa_spacing,
            "radius": config.drone_radius,
            "margin": config.safety_margin,
            "safe": config.toa_safe_distance,
            "slow": config.toa_slow_speed,
        }
    )


def build_toa_map(
    config: EnvConfig, clearance: np.ndarray, goal: np.ndarray
) -> np.ndarray:
    """目标栅格为唯一零值源点，障碍及不可达区域保存为正无穷。"""
    blocked = clearance <= 0
    if blocked[tuple(goal)]:
        raise ValueError("TOA 目标位于障碍内")
    speed = config.toa_slow_speed + (1 - config.toa_slow_speed) * np.clip(
        clearance / config.toa_safe_distance, 0, 1
    )
    phi = np.ones(clearance.shape, dtype=np.float64)
    phi[tuple(goal)] = 0.0
    arrival = skfmm.travel_time(
        np.ma.array(phi, mask=blocked),
        np.ma.array(speed, mask=blocked),
        dx=config.toa_spacing,
    )
    values = np.asarray(np.ma.filled(arrival, np.inf), dtype=np.float32)
    values[blocked] = np.inf
    values[tuple(goal)] = 0.0
    return values


def cached_toa(
    config: EnvConfig,
    maze: MazeDefinition,
    clearance: np.ndarray,
    labels: np.ndarray,
    goal: np.ndarray,
) -> tuple[np.ndarray, str]:
    key = cache_key(config, maze, goal)
    destination = (
        Path(config.toa_cache_dir).expanduser() / f"{key}.npz"
        if config.toa_cache_dir
        else None
    )
    reachable = labels == labels[tuple(goal)]
    if destination:
        try:
            with np.load(destination, allow_pickle=False) as cached:
                values = cached["toa"]
                checksum = hashlib.sha256(values.tobytes()).hexdigest()
                if (
                    str(cached["key"]) == key
                    and str(cached["checksum"]) == checksum
                    and values.dtype == np.float32
                    and values.shape == clearance.shape
                    and np.array_equal(np.isfinite(values), reachable)
                    and values[tuple(goal)] == 0
                    and np.count_nonzero(values == 0) == 1
                    and np.all(values >= 0)
                ):
                    return values, key
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            pass
    values = build_toa_map(config, clearance, goal)
    if (
        not np.array_equal(np.isfinite(values), reachable)
        or np.count_nonzero(values == 0) != 1
    ):
        raise ValueError(f"{maze.name} FMM 可达性或点目标零值异常")
    if destination:
        temporary = None
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, suffix=".npz", delete=False
            ) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(
                    stream,
                    toa=values,
                    key=key,
                    checksum=hashlib.sha256(values.tobytes()).hexdigest(),
                )
            os.replace(temporary, destination)
        except OSError as exc:
            warnings.warn(f"TOA 磁盘缓存不可写，使用内存缓存：{exc}", stacklevel=2)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
    return values, key


@dataclass
class ToaBank:
    maps: list[MazeDefinition]
    goals: np.ndarray
    values: np.ndarray
    # 每张地图一个外围起点候选池，不随目标重复存储。
    starts: list[np.ndarray]
    manifest: dict

    def sample(
        self, config: EnvConfig, env_id: int, episode_id: int, map_id: int | None = None
    ) -> tuple[int, int, np.ndarray, float]:
        rng = np.random.default_rng(
            np.random.SeedSequence([config.seed, 2309, env_id, episode_id])
        )
        map_id = int(rng.integers(len(self.maps))) if map_id is None else map_id
        choices = self.starts[map_id]
        row, col = choices[int(rng.integers(len(choices)))]
        xy = coordinates(config)[[col, row]]
        # 随机排列等价于不断拒绝不合格目标，且最多检查一遍缓存目标池。
        for goal_id in rng.permutation(config.goals_per_map):
            bank_id = map_id * config.goals_per_map + int(goal_id)
            distance = np.linalg.norm(self.goals[bank_id] - xy)
            initial_toa = float(self.values[bank_id, row, col])
            if (
                distance >= config.min_start_goal_distance
                and np.isfinite(initial_toa)
                and initial_toa > 1e-6
            ):
                return map_id, int(goal_id), xy, initial_toa
        raise ValueError(
            f"{self.maps[map_id].name} 外围起点没有符合距离和可达性要求的目标"
        )


def build_toa_bank(config: EnvConfig) -> ToaBank:
    pools = build_map_pools(config)
    metadata = {}
    values, goals_xy, starts = [], [], []
    coords = coordinates(config)
    for split, maps in pools.items():
        records = []
        for maze in maps:
            clearance = obstacle_clearance(config, maze)
            labels = components(clearance > 0)
            goals = sample_goals(config, maze, clearance, labels)
            records.append(
                {
                    "name": maze.name,
                    "seed": maze.seed,
                    "hash": maze.content_hash,
                    "goals": goals.tolist(),
                }
            )
            if split != config.map_split:
                continue
            candidates = start_candidates(config, clearance)
            eligible_start = np.zeros(len(candidates), dtype=bool)
            for goal in goals:
                toa, _ = cached_toa(config, maze, clearance, labels, goal)
                valid = np.isfinite(toa[candidates[:, 0], candidates[:, 1]])
                valid &= toa[candidates[:, 0], candidates[:, 1]] > 1e-6
                valid &= (
                    np.linalg.norm(
                        coords[candidates[:, ::-1]] - coords[goal[::-1]], axis=1
                    )
                    >= config.min_start_goal_distance
                )
                if not valid.any():
                    raise ValueError(f"{maze.name} 目标没有合法起点")
                eligible_start |= valid
                values.append(toa)
                goals_xy.append(coords[goal[::-1]])
            starts.append(candidates[eligible_start].astype(np.int32))
        metadata[split] = records
    manifest = {
        "scene_version": scene_version(config.scene_type),
        "generator": GENERATOR_VERSIONS[config.scene_type],
        "toa_version": TOA_VERSION,
        "solver": skfmm.__version__,
        "toa_reward_normalization": "episode_start",
        "sampling": {
            "version": "outer-ring-start-first-v1",
            "start_region": "first_inner_cell_ring",
            "goal_selection": "reachable_cached_goal_min_euclidean_distance",
        },
        "pools": metadata,
        "grid_size": config.toa_grid_size,
        "spacing": config.toa_spacing,
        "safety": {
            "drone_radius": config.drone_radius,
            "margin": config.safety_margin,
            "spawn_margin": config.spawn_margin,
            "min_distance": config.min_start_goal_distance,
        },
        "speed": {
            "safe_distance": config.toa_safe_distance,
            "slow": config.toa_slow_speed,
        },
        "success_radius": config.goal_threshold,
    }
    # DFS 清单保持历史结构；只有障碍场地才追加参数，旧 checkpoint 仍能逐字段比对。
    if config.scene_type == "obstacles":
        manifest["obstacle_field"] = {
            "arena_size": config.arena_size,
            "cell_size": config.cell_size,
            "height": config.arena_height,
            "min_separation": config.obstacle_min_separation,
            "cylinders": {
                "count": config.cylinder_count,
                "radius": config.cylinder_radius,
            },
            "u_shapes": {
                "count": config.u_shape_count,
                "arm_length": config.u_shape_arm_length,
                "arm_thickness": config.u_shape_arm_thickness,
                "opening": config.u_shape_opening,
            },
        }
    bank = ToaBank(
        pools[config.map_split],
        np.asarray(goals_xy, dtype=np.float32),
        np.stack(values),
        starts,
        manifest,
    )
    byte_count = bank.values.nbytes + bank.goals.nbytes + sum(v.nbytes for v in starts)
    finite = bank.values[np.isfinite(bank.values)]
    print(
        f"[DFS] {config.map_split}: {len(bank.maps)} 张地图 × {config.goals_per_map} 目标; "
        f"TOA {bank.values.nbytes / 2**20:.2f} MiB; CPU 池 {byte_count / 2**20:.2f} MiB; "
        f"有效 TOA P50/P95/max={np.percentile(finite, [50, 95, 100]).round(2).tolist()}"
    )
    return bank


def resolve_toa_normalization_max(
    values: np.ndarray, configured: float | None
) -> float:
    if configured is not None:
        if not np.isfinite(configured) or configured <= 0:
            raise ValueError("TOA 观测量程必须为有限正数")
        return float(configured)
    finite = values[np.isfinite(values)]
    if not finite.size or np.max(finite) <= 0:
        raise ValueError("TOA 池没有有效正值")
    return float(np.max(finite))


def save_toa_global_maps(
    config: EnvConfig, bank: ToaBank, output_dir: str | Path
) -> list[Path]:
    from PIL import Image

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    saved = []
    for map_id, maze in enumerate(bank.maps):
        for goal_id in range(config.goals_per_map):
            index = map_id * config.goals_per_map + goal_id
            values = bank.values[index]
            maximum = resolve_toa_normalization_max(values, None)
            normalized = np.clip(np.nan_to_num(values / maximum, posinf=1), 0, 1)
            rgb = np.stack(
                (normalized, 1 - normalized, np.zeros_like(normalized)), axis=-1
            )
            rgb[~np.isfinite(values)] = 0.08
            path = destination / f"{maze.name}_goal_{goal_id:02d}.png"
            Image.fromarray(np.flipud((rgb * 255).astype(np.uint8))).save(path)
            np.savez_compressed(
                path.with_suffix(".npz"),
                toa=values,
                goal=bank.goals[index],
                extent=config.arena_half_extent,
            )
            saved.append(path)
    (destination / "map_manifest.json").write_text(json.dumps(bank.manifest, indent=2))
    return saved
