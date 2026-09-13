"""六种训练迷宫的纯数据定义。"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi


@dataclass(frozen=True)
class Wall:
    center_xy: tuple[float, float]
    vertical: bool = False
    length_scale: float = 1.0


@dataclass(frozen=True)
class MazeLayout:
    name: str
    walls: tuple[Wall, ...]
    routes: tuple[tuple[tuple[float, float], tuple[float, float]], ...]


_SCALE = 2.0


def _point(point: tuple[float, float]) -> tuple[float, float]:
    return point[0] * _SCALE, point[1] * _SCALE


def _wall(x: float, y: float, yaw: float = 0.0, length_scale: float = 1.0) -> Wall:
    return Wall(
        _point((x, y)),
        vertical=abs(abs(yaw) - pi / 2) < 1.0e-6,
        length_scale=length_scale,
    )


def _routes(*routes):
    return tuple((_point(start), _point(goal)) for start, goal in routes)


MAZE_LAYOUTS = (
    MazeLayout(
        "maze_01",
        (_wall(0.0, 0.5),),
        _routes(((-1.2, -5.5), (-1.2, 5.5)), ((1.2, -5.5), (1.2, 5.5))),
    ),
    MazeLayout(
        "maze_02",
        (_wall(3.2, 3.0), _wall(-3.2, -2.0)),
        _routes(((-0.8, -6.0), (-1.8, 6.0)), ((1.2, -5.5), (0.8, 6.2))),
    ),
    MazeLayout(
        "maze_03",
        (_wall(-3.2, 3.6), _wall(3.4, 0.4), _wall(0.8, -3.5)),
        _routes(((-1.3, -6.5), (-1.2, 6.2)), ((1.8, -6.3), (1.0, 6.0))),
    ),
    MazeLayout(
        "maze_04",
        (
            _wall(0.0, 1.8),
            _wall(-3.0, 0.3, pi / 2, length_scale=0.5),
            _wall(3.0, 0.3, pi / 2, length_scale=0.5),
        ),
        _routes(((-1.2, -3.0), (-1.2, 6.0)), ((1.2, -3.0), (1.2, 6.0))),
    ),
    MazeLayout(
        "maze_05",
        (_wall(-3.2, 3.2), _wall(3.2, -3.2)),
        _routes(((1.0, -6.3), (-5.0, 6.0)), ((4.5, -6.0), (-2.5, 6.2))),
    ),
    MazeLayout(
        "maze_06",
        (_wall(0.0, 1.8), _wall(0.0, -1.2, pi / 2)),
        _routes(((-3.0, -4.5), (-2.0, 5.5)), ((3.0, -4.5), (2.0, 5.5))),
    ),
)


def wall_rectangle(wall: Wall, wall_thickness: float, wall_length: float) -> tuple[float, float, float, float]:
    """返回 ``center_x, center_y, size_x, size_y``。"""

    scaled_length = wall_length * wall.length_scale
    size_x, size_y = (wall_thickness, scaled_length) if wall.vertical else (scaled_length, wall_thickness)
    return wall.center_xy[0], wall.center_xy[1], size_x, size_y


def validate_layouts(
    arena_half_extent: float = 20.0,
    wall_thickness: float = 0.7,
    wall_length: float = 12.0,
) -> None:
    if len(MAZE_LAYOUTS) != 6:
        raise ValueError("必须定义六种训练迷宫")
    for layout in MAZE_LAYOUTS:
        if len(layout.walls) > 3 or len(layout.routes) != 2:
            raise ValueError(f"{layout.name} 的墙体或路线数量不正确")
        for wall in layout.walls:
            if wall.length_scale <= 0.0:
                raise ValueError(f"{layout.name} 的墙体长度比例必须为正数")
            x, y, size_x, size_y = wall_rectangle(wall, wall_thickness, wall_length)
            if abs(x) + size_x / 2 >= arena_half_extent:
                raise ValueError(f"{layout.name} 的墙体越过 X 边界")
            if abs(y) + size_y / 2 >= arena_half_extent:
                raise ValueError(f"{layout.name} 的墙体越过 Y 边界")


validate_layouts()
