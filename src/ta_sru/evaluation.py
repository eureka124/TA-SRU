"""独立于仿真的评估配置、终局分类和按布局计数。"""

from __future__ import annotations

from ta_sru.envs.layouts import MAZE_LAYOUTS


def configure_evaluation(values: dict) -> dict:
    """只覆盖本次评估配置，使训练进度不再影响场景或碰撞阈值。"""
    values = dict(values)
    cylinders = values.get("training_curriculum_cylinder_counts", (60,))[-1]
    values.update(
        training_curriculum_stage_fractions=(0.0,),
        training_curriculum_maze_counts=(len(MAZE_LAYOUTS),),
        training_curriculum_maze_start_indices=(0,),
        training_curriculum_cylinder_counts=(cylinders,),
        collision_force_threshold=0.1,
        minimum_contact_force_threshold=0.1,
    )
    return values


def evaluation_outcome(info: dict) -> str:
    """评估终局互斥，碰撞和越界优先于成功及超时。"""
    if info.get("collided", False) or info.get("outside", False):
        return "collision"
    if info.get("success", False):
        return "success"
    if info.get("time_out", False):
        return "timeout"
    raise ValueError("已结束的评估回合缺少终止原因")


class EvaluationStats:
    def __init__(self, episodes_per_layout: int) -> None:
        if episodes_per_layout <= 0:
            raise ValueError("每种布局的评估回合数必须为正数")
        self.target = episodes_per_layout
        self.counts = {
            layout.name: {"success": 0, "collision": 0, "timeout": 0}
            for layout in MAZE_LAYOUTS
        }

    def record(self, maze: str, info: dict) -> str | None:
        counts = self.counts[maze]
        if sum(counts.values()) >= self.target:
            return None
        outcome = evaluation_outcome(info)
        counts[outcome] += 1
        return outcome

    @property
    def complete(self) -> bool:
        return all(
            sum(counts.values()) == self.target for counts in self.counts.values()
        )

    def summary(self) -> dict:
        def rates(counts: dict) -> dict:
            total = sum(counts.values())
            return {
                "episodes": total,
                **counts,
                **{
                    f"{key}_rate": value / total if total else None
                    for key, value in counts.items()
                },
            }

        total = {
            key: sum(counts[key] for counts in self.counts.values())
            for key in ("success", "collision", "timeout")
        }
        return {
            "episodes_per_layout": self.target,
            "complete": self.complete,
            "layouts": {maze: rates(counts) for maze, counts in self.counts.items()},
            "overall": rates(total),
        }
