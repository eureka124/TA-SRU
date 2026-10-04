"""独立于仿真的 DFS 地图、几何与可复现地图池。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from ta_sru.config import EnvConfig

GENERATOR_VERSION = "dfs-walls-v1"
SCENE_VERSION = "dfs-point-toa-start-normalized-v1"


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class MazeDefinition:
    name: str
    seed: int
    occupied: np.ndarray
    cell_size: float
    wall_height: float

    @property
    def extent(self) -> float:
        return len(self.occupied) * self.cell_size / 2

    @property
    def content_hash(self) -> str:
        return digest(
            [self.occupied.astype(int).tolist(), self.cell_size, self.wall_height]
        )

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
                maze = generate_maze(config, seed, f"{split}_{index:03d}")
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


def pool_mesh(
    maps: list[MazeDefinition], origins: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """墙体和地面合为静态三角网格，供碰撞与相机共用。"""
    vertices, triangles = [], []
    faces = np.asarray(
        (
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        )
    )
    for maze, origin in zip(maps, origins):
        boxes = [(0, 0, 2 * maze.extent, 2 * maze.extent, -0.1, 0.0)]
        boxes += [(*rect, 0.0, maze.wall_height) for rect in maze.rectangles()]
        for x, y, sx, sy, z0, z1 in boxes:
            points = np.asarray(
                [
                    (x - sx / 2, y - sy / 2, z0),
                    (x + sx / 2, y - sy / 2, z0),
                    (x + sx / 2, y + sy / 2, z0),
                    (x - sx / 2, y + sy / 2, z0),
                    (x - sx / 2, y - sy / 2, z1),
                    (x + sx / 2, y - sy / 2, z1),
                    (x + sx / 2, y + sy / 2, z1),
                    (x - sx / 2, y + sy / 2, z1),
                ]
            )
            triangles.append(faces + len(vertices) * 8)
            vertices.append(points + origin)
    return np.concatenate(vertices).astype(np.float32), np.concatenate(
        triangles
    ).astype(np.int32)
