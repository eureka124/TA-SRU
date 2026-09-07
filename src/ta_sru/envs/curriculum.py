"""训练迷宫难度课程的纯 Python 调度逻辑。"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TrainingMazeCurriculumStage:
    """某个训练阶段启用的环境复杂度。"""

    index: int
    start_fraction: float
    maze_count: int
    cylinder_count: int


def select_training_maze_curriculum_stage(
    elapsed_steps: int,
    max_steps: int,
    stage_fractions: Sequence[float],
    maze_counts: Sequence[int],
    cylinder_counts: Sequence[int],
) -> TrainingMazeCurriculumStage:
    """根据累计环境 transition 数选择当前课程阶段。"""

    if max_steps <= 0:
        raise ValueError(f"max_steps 必须为正数，实际为 {max_steps}")
    if not (len(stage_fractions) == len(maze_counts) == len(cylinder_counts)):
        raise ValueError("课程阶段比例、迷宫数量和圆柱数量必须具有相同长度")
    if not stage_fractions:
        raise ValueError("至少需要一个课程阶段")
    if stage_fractions[0] != 0.0:
        raise ValueError("第一个课程阶段必须从 0.0 开始")
    if any(current <= previous for previous, current in zip(stage_fractions, stage_fractions[1:])):
        raise ValueError("课程阶段比例必须严格递增")
    if stage_fractions[-1] > 1.0:
        raise ValueError("课程阶段比例不能超过 1.0")
    if any(count <= 0 for count in maze_counts):
        raise ValueError("每个课程阶段必须至少启用一个迷宫")
    if any(count < 0 for count in cylinder_counts):
        raise ValueError("课程阶段的圆柱数量不能为负数")

    progress = min(max(float(elapsed_steps) / float(max_steps), 0.0), 1.0)
    stage_index = bisect_right(stage_fractions, progress) - 1
    return TrainingMazeCurriculumStage(
        index=stage_index,
        start_fraction=float(stage_fractions[stage_index]),
        maze_count=int(maze_counts[stage_index]),
        cylinder_count=int(cylinder_counts[stage_index]),
    )


__all__ = ["TrainingMazeCurriculumStage", "select_training_maze_curriculum_stage"]
