"""随机圆柱与 U 形障碍场地生成，不依赖 Isaac Sim。

场地是四面围墙围出的空地：先摆 U 形障碍，再用泊松盘采样摆圆柱。摆放阶段要求障碍物
之间、障碍物与围墙之间至少留出 ``obstacle_min_separation`` 的表面间距，因此场地内每条
缝隙都宽于无人机直径加安全余量，起点与目标之间始终存在可行路径。
"""

from __future__ import annotations

import math

import numpy as np

from ta_sru.config import EnvConfig
from ta_sru.envs.maze import MazeDefinition
from ta_sru.envs.primitives import Box, Disc, Solid, footprint_distance

# 一张地图摆不下时换子种子重试的总次数，以及单个障碍物的采样次数上限。
MAP_ATTEMPTS = 32
PLACEMENT_ATTEMPTS = 256
# 泊松盘采样中每个活跃点尝试新候选点的次数。
POISSON_CANDIDATES = 30
# 摆放随机子流编号，与地图种子共同决定障碍物布局。
PLACEMENT_SALT = 5209


def generate_obstacle_field(
    config: EnvConfig, seed: int, name: str = "obstacles"
) -> MazeDefinition:
    """生成一张随机障碍场地；单个子种子摆不下时自动换种子重试。"""

    for attempt in range(MAP_ATTEMPTS):
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, PLACEMENT_SALT, attempt])
        )
        obstacles = _place_obstacles(config, rng)
        if obstacles is not None:
            return MazeDefinition(
                name,
                seed,
                _boundary_occupancy(config),
                config.cell_size,
                config.arena_height,
                tuple(obstacles),
            )
    raise ValueError(
        f"{name} 在 {config.arena_size:g} m 见方的场地内摆不下 "
        f"{config.cylinder_count} 个半径 {config.cylinder_radius:g} m 的圆柱与 "
        f"{config.u_shape_count} 个 U 形障碍（最小表面间距 "
        f"{config.obstacle_min_separation:g} m），请缩小障碍物或放大场地"
    )


def _place_obstacles(
    config: EnvConfig, rng: np.random.Generator
) -> list[tuple[Solid, ...]] | None:
    """返回障碍物分组；任一障碍物摆不下时返回 None 触发整体重采样。"""

    inner = config.arena_half_extent - config.cell_size
    separation = config.obstacle_min_separation
    blocking: list[tuple[Solid, ...]] = [_boundary_solids(config)]
    obstacles: list[tuple[Solid, ...]] = []
    for _ in range(config.u_shape_count):
        group = _sample_u_shape(config, rng, inner, separation, blocking)
        if group is None:
            return None
        obstacles.append(group)
        blocking.append(group)
    cylinders = _poisson_cylinders(config, rng, inner, separation, blocking)
    if cylinders is None:
        return None
    return obstacles + [(cylinder,) for cylinder in cylinders]


def _sample_u_shape(
    config: EnvConfig,
    rng: np.random.Generator,
    inner: float,
    separation: float,
    blocking: list[tuple[Solid, ...]],
) -> tuple[Solid, ...] | None:
    """拒绝采样一个与其他障碍物保持间距的 U 形障碍。"""

    reach = _u_shape_reach(config)
    limit = inner - separation - reach
    if limit <= 0.0:
        return None
    for _ in range(PLACEMENT_ATTEMPTS):
        group = u_shape_solids(
            config,
            rng.uniform(-limit, limit),
            rng.uniform(-limit, limit),
            rng.uniform(-math.pi, math.pi),
        )
        if _keeps_distance(group, blocking, separation):
            return group
    return None


def _poisson_cylinders(
    config: EnvConfig,
    rng: np.random.Generator,
    inner: float,
    separation: float,
    blocking: list[tuple[Solid, ...]],
) -> list[Disc] | None:
    """泊松盘采样圆柱中心：邻域网格保证圆柱互不重叠，其余障碍物逐个精确测距。"""

    if config.cylinder_count == 0:
        return []
    radius = config.cylinder_radius
    spacing = 2.0 * radius + separation
    limit = inner - separation - radius
    if limit <= 0.0:
        return None
    cell = spacing / math.sqrt(2.0)
    grid: dict[tuple[int, int], list[tuple[float, float]]] = {}

    def neighbours(point: tuple[float, float]):
        column, row = _cell_key(point, limit, cell)
        for x in range(column - 2, column + 3):
            for y in range(row - 2, row + 3):
                yield from grid.get((x, y), ())

    def accepts(point: tuple[float, float]) -> bool:
        if abs(point[0]) > limit or abs(point[1]) > limit:
            return False
        if any(math.dist(point, other) < spacing for other in neighbours(point)):
            return False
        disc = Disc(point[0], point[1], radius, 0.0, config.arena_height)
        return _keeps_distance((disc,), blocking, separation)

    def remember(point: tuple[float, float]) -> None:
        grid.setdefault(_cell_key(point, limit, cell), []).append(point)

    start = _first_cylinder(config, rng, limit, accepts)
    if start is None:
        return None
    remember(start)
    sites = [start]
    active = [start]
    while active:
        chosen = int(rng.integers(len(active)))
        anchor = active[chosen]
        for _ in range(POISSON_CANDIDATES):
            angle = rng.uniform(0.0, 2.0 * math.pi)
            distance = rng.uniform(spacing, 2.0 * spacing)
            candidate = (
                anchor[0] + distance * math.cos(angle),
                anchor[1] + distance * math.sin(angle),
            )
            if not accepts(candidate):
                continue
            remember(candidate)
            sites.append(candidate)
            active.append(candidate)
            if len(sites) == config.cylinder_count:
                return [
                    Disc(x, y, radius, 0.0, config.arena_height) for x, y in sites
                ]
            break
        else:
            active[chosen] = active[-1]
            active.pop()
    return None


def _first_cylinder(
    config: EnvConfig,
    rng: np.random.Generator,
    limit: float,
    accepts,
) -> tuple[float, float] | None:
    """采出第一个可用圆柱中心，作为泊松盘生长的种子点。"""

    for _ in range(PLACEMENT_ATTEMPTS):
        point = (rng.uniform(-limit, limit), rng.uniform(-limit, limit))
        if accepts(point):
            return point
    return None


def u_shape_solids(
    config: EnvConfig, x: float, y: float, yaw: float
) -> tuple[Solid, ...]:
    """U 形障碍的三块墙：背墙加两条平行臂，开口朝局部 +x 方向。"""

    length = config.u_shape_arm_length
    thickness = config.u_shape_arm_thickness
    opening = config.u_shape_opening
    height = config.arena_height
    return (
        _rotated_box(
            x, y, yaw, -length / 2.0, 0.0, thickness, opening + 2.0 * thickness, height
        ),
        _rotated_box(
            x,
            y,
            yaw,
            thickness / 2.0,
            (opening + thickness) / 2.0,
            length,
            thickness,
            height,
        ),
        _rotated_box(
            x,
            y,
            yaw,
            thickness / 2.0,
            -(opening + thickness) / 2.0,
            length,
            thickness,
            height,
        ),
    )


def _rotated_box(
    x: float,
    y: float,
    yaw: float,
    local_x: float,
    local_y: float,
    size_x: float,
    size_y: float,
    height: float,
) -> Box:
    """把局部坐标下的长方体按 yaw 旋转后平移到 (x, y)。"""

    cosine, sine = math.cos(yaw), math.sin(yaw)
    return Box(
        x + local_x * cosine - local_y * sine,
        y + local_x * sine + local_y * cosine,
        yaw,
        size_x,
        size_y,
        0.0,
        height,
    )


def _u_shape_reach(config: EnvConfig) -> float:
    """U 形障碍水平截面的外接圆半径。"""

    half_x = (config.u_shape_arm_length + config.u_shape_arm_thickness) / 2.0
    half_y = (config.u_shape_opening + 2.0 * config.u_shape_arm_thickness) / 2.0
    return math.hypot(half_x, half_y)


def _keeps_distance(
    group: tuple[Solid, ...],
    blocking: list[tuple[Solid, ...]],
    separation: float,
) -> bool:
    """待放障碍物的每个实体都要与已摆放实体保持最小表面间距。"""

    for solid in group:
        for existing in blocking:
            for other in existing:
                if _far_apart(solid, other, separation):
                    continue
                if footprint_distance(solid, other) < separation:
                    return False
    return True


def _far_apart(a: Solid, b: Solid, separation: float) -> bool:
    """外接圆相距超过间距时无需再做精确几何测试。"""

    reach = separation + _bounding_radius(a) + _bounding_radius(b)
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2 >= reach * reach


def _bounding_radius(solid: Solid) -> float:
    """图元水平截面的外接圆半径。"""

    if isinstance(solid, Disc):
        return solid.radius
    return math.hypot(solid.size_x, solid.size_y) / 2.0


def _cell_key(
    point: tuple[float, float], limit: float, cell: float
) -> tuple[int, int]:
    return (int((point[0] + limit) / cell), int((point[1] + limit) / cell))


def _boundary_solids(config: EnvConfig) -> list[Box]:
    """围墙的四个长方体，与占据栅格导出的墙体并集一致。"""

    half = config.arena_half_extent
    cell = config.cell_size
    height = config.arena_height
    span = 2.0 * half
    offset = half - cell / 2.0
    return [
        Box(0.0, offset, 0.0, span, cell, 0.0, height),
        Box(0.0, -offset, 0.0, span, cell, 0.0, height),
        Box(offset, 0.0, 0.0, cell, span - 2.0 * cell, 0.0, height),
        Box(-offset, 0.0, 0.0, cell, span - 2.0 * cell, 0.0, height),
    ]


def _boundary_occupancy(config: EnvConfig) -> np.ndarray:
    """障碍场地只有最外一圈格子是围墙，内部完全空出来给障碍物。"""

    occupied = np.zeros((config.maze_size, config.maze_size), dtype=bool)
    occupied[[0, -1], :] = True
    occupied[:, [0, -1]] = True
    occupied.flags.writeable = False
    return occupied
