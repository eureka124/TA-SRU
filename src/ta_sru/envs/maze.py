"""独立于仿真的场景地图、共用几何与可复现地图池。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from ta_sru.config import EnvConfig
from ta_sru.envs.primitives import Box, Solid, solid_record

GENERATOR_VERSION = "dfs-walls-v1"
SCENE_VERSION = "dfs-point-toa-start-normalized-v1"
OBSTACLE_GENERATOR_VERSION = "obstacle-field-bridson-v1"
OBSTACLE_SCENE_VERSION = "obstacle-field-point-toa-start-normalized-v1"

SCENE_VERSIONS = {"dfs": SCENE_VERSION, "obstacles": OBSTACLE_SCENE_VERSION}
GENERATOR_VERSIONS = {"dfs": GENERATOR_VERSION, "obstacles": OBSTACLE_GENERATOR_VERSION}


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def scene_version(scene_type: str) -> str:
    """返回场景类型的清单版本，用于校验 checkpoint 属于同一种场景。"""

    if scene_type not in SCENE_VERSIONS:
        raise ValueError(f"未知场景类型：{scene_type}")
    return SCENE_VERSIONS[scene_type]


@dataclass(frozen=True)
class MazeDefinition:
    name: str
    seed: int
    occupied: np.ndarray
    cell_size: float
    wall_height: float
    # 每个元素是一个障碍物，由若干实体图元组成：圆柱一个，U 形障碍三块墙。
    obstacles: tuple[tuple[Solid, ...], ...] = ()

    @property
    def extent(self) -> float:
        return len(self.occupied) * self.cell_size / 2

    @property
    def content_hash(self) -> str:
        payload = [self.occupied.astype(int).tolist(), self.cell_size, self.wall_height]
        # 只有障碍场地才扩展散列，让 DFS 的内容散列保持历史取值，
        # 已有 TOA 磁盘缓存与 checkpoint 继续有效。
        if self.obstacles:
            payload.append(
                [solid_record(solid) for group in self.obstacles for solid in group]
            )
        return digest(payload)

    def rectangles(self) -> list[tuple[float, float, float, float]]:
        """按行合并相邻墙格，严格保持物理占用区域的并集。"""
        result = []
        for row, cells in enumerate(self.occupied):
            edges = np.diff(np.r_[False, cells, False].astype(np.int8))
            for left, right in zip(
                np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            ):
                result.append(
                    (
                        -self.extent + (left + right) * self.cell_size / 2,
                        -self.extent + (row + 0.5) * self.cell_size,
                        (right - left) * self.cell_size,
                        self.cell_size,
                    )
                )
        return result

    def wall_boxes(self) -> list[Box]:
        """围墙的图元形式，与 rectangles 的并集一致。"""

        return [
            Box(x, y, 0.0, size_x, size_y, 0.0, self.wall_height)
            for x, y, size_x, size_y in self.rectangles()
        ]

    def solids(self) -> list[Solid]:
        """围墙与障碍物的全部实体，供距离场、网格和回放共用。"""

        return self.wall_boxes() + [
            solid for group in self.obstacles for solid in group
        ]


def generate_maze(config: EnvConfig, seed: int, name: str = "maze") -> MazeDefinition:
    """奇数索引为逻辑节点，只拆除连接相邻节点的内部隔墙。"""
    rng = np.random.default_rng(seed)
    size = config.maze_size
    occupied = np.ones((size, size), dtype=bool)
    start = (
        int(rng.integers((size - 1) // 2)) * 2 + 1,
        int(rng.integers((size - 1) // 2)) * 2 + 1,
    )
    occupied[start] = False
    stack = [start]
    while stack:
        y, x = stack[-1]
        neighbors = [
            (y + dy, x + dx)
            for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2))
            if 0 < y + dy < size - 1
            and 0 < x + dx < size - 1
            and occupied[y + dy, x + dx]
        ]
        if not neighbors:
            stack.pop()
            continue
        ny, nx = neighbors[int(rng.integers(len(neighbors)))]
        occupied[ny, nx] = occupied[(y + ny) // 2, (x + nx) // 2] = False
        stack.append((ny, nx))
    for y in range(1, size - 1):
        for x in range(1, size - 1):
            if (
                (y % 2 != x % 2)
                and occupied[y, x]
                and rng.random() < config.maze_wall_removal_probability
            ):
                occupied[y, x] = False
    occupied.flags.writeable = False
    return MazeDefinition(
        name, seed, occupied, config.maze_cell_size, config.arena_height
    )


def generate_map(
    config: EnvConfig, seed: int, name: str = "map"
) -> MazeDefinition:
    """按场景类型生成一张地图。"""

    if config.scene_type == "obstacles":
        # 延迟导入避免 maze 与 obstacles 两个模块互相引用。
        from ta_sru.envs.obstacles import generate_obstacle_field

        return generate_obstacle_field(config, seed, name)
    return generate_maze(config, seed, name)


def build_map_pools(config: EnvConfig) -> dict[str, list[MazeDefinition]]:
    """两个池的内容去重，不依赖环境数、网络随机流或当前选择的池。"""
    config.validate()
    result = {}
    seen = set()
    for namespace, (split, count) in enumerate(
        (("train", config.train_map_count), ("eval", config.eval_map_count))
    ):
        maps = []
        for index in range(count):
            for attempt in range(1000):
                seed = int(
                    np.random.SeedSequence(
                        [config.maze_seed, namespace, index, attempt]
                    ).generate_state(1)[0]
                )
                maze = generate_map(config, seed, f"{split}_{index:03d}")
                if maze.content_hash not in seen:
                    seen.add(maze.content_hash)
                    maps.append(maze)
                    break
            else:
                raise ValueError(
                    "无法生成足够的不重复地图，请增大地图或减少地图数/拆墙概率"
                )
        result[split] = maps
    return result


def pool_origins(count: int, extent: float, gap: float) -> np.ndarray:
    width = int(np.ceil(np.sqrt(count)))
    ids = np.arange(count)
    return np.column_stack((ids % width, ids // width, np.zeros(count))).astype(
        np.float32
    ) * (2 * extent + gap)


def prism(
    solid: Solid, origin: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """把图元水平截面拉伸成棱柱，返回顶点与逆时针朝外的三角面。"""

    corners = solid.corners()
    count = len(corners)
    bottom = np.column_stack((corners, np.full(count, solid.z0)))
    top = np.column_stack((corners, np.full(count, solid.z1)))
    points = np.vstack((bottom, top)) + origin
    faces = []
    for index in range(count):
        following = (index + 1) % count
        faces.append((index, following, count + following))
        faces.append((index, count + following, count + index))
    for index in range(1, count - 1):
        faces.append((0, index + 1, index))
        faces.append((count, count + index, count + index + 1))
    return points, np.asarray(faces, dtype=np.int32)


def pool_mesh(
    maps: list[MazeDefinition], origins: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """墙体、障碍物和地面合为静态三角网格，供碰撞与相机共用。"""

    vertices, triangles = [], []
    offset = 0
    for maze, origin in zip(maps, origins):
        floor = Box(0.0, 0.0, 0.0, 2 * maze.extent, 2 * maze.extent, -0.1, 0.0)
        for solid in [floor, *maze.solids()]:
            points, faces = prism(solid, origin)
            vertices.append(points)
            triangles.append(faces + offset)
            offset += len(points)
    return np.concatenate(vertices).astype(np.float32), np.concatenate(
        triangles
    ).astype(np.int32)
