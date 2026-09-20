"""独立于仿真的评估配置、终局分类和按布局计数。"""

from __future__ import annotations

from unicodedata import east_asian_width

from ta_sru.envs.layouts import MAZE_LAYOUTS


def configure_evaluation(values: dict) -> dict:
    """只覆盖本次评估配置，使训练进度不再影响场景或碰撞阈值。"""
    values = dict(values)
    # 缺少该字段的历史 checkpoint 使用旧版固定归一化量程。
    values.setdefault("toa_normalization_max", 255.0)
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


def load_evaluation_policy(checkpoint: dict, device: str):
    """只加载推理网络，不创建优化器、训练缓冲或训练日志。"""
    import torch

    from ta_sru.config import NetworkConfig
    from ta_sru.models.actor_critic import AsymmetricRecurrentActorCritic

    values = checkpoint["config"]
    network = NetworkConfig(**values["network"])
    algorithm = values.get("algorithm", "recurrent_ppo")
    if algorithm not in ("ppo", "recurrent_ppo") or (
        (algorithm == "ppo") != (network.recurrent_type == "none")
    ):
        raise ValueError("checkpoint 算法与网络结构不一致")
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[提示] CUDA 不可用，网络设备回退到 CPU")
        device = "cpu"
    policy = AsymmetricRecurrentActorCritic(network)
    policy.load_state_dict(checkpoint["policy"], strict=True)
    return policy.to(device).eval()


def evaluation_outcome(info: dict) -> str:
    """评估终局互斥，碰撞和越界优先于成功及超时。"""
    if info.get("collided", False) or info.get("outside", False):
        return "collision"
    if info.get("success", False):
        return "success"
    if info.get("time_out", False):
        return "timeout"
    raise ValueError("已结束的评估回合缺少终止原因")


def format_evaluation_table(summary: dict) -> str:
    """将评估汇总排成终端表格，兼顾中文列宽及未完成评估。"""
    rows = [
        [
            "布局",
            "回合数",
            "成功（数量/比例）",
            "碰撞（数量/比例）",
            "超时（数量/比例）",
        ]
    ]
    for name, counts in [*summary["layouts"].items(), ("整体", summary["overall"])]:
        row = [name, str(counts["episodes"])]
        for outcome in ("success", "collision", "timeout"):
            rate = counts[f"{outcome}_rate"]
            percentage = "—" if rate is None else f"{rate:.2%}"
            row.append(f"{counts[outcome]} / {percentage}")
        rows.append(row)

    def display_width(value: str) -> int:
        return sum(2 if east_asian_width(char) in ("W", "F") else 1 for char in value)

    widths = [
        max(display_width(row[index]) for row in rows) for index in range(len(rows[0]))
    ]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    lines = [separator]
    for index, row in enumerate(rows):
        cells = []
        for column, (value, width) in enumerate(zip(row, widths)):
            padding = " " * (width - display_width(value))
            cells.append(value + padding if column == 0 else padding + value)
        lines.append("| " + " | ".join(cells) + " |")
        if index == 0 or index == len(rows) - 2:
            lines.append(separator)
    lines.append(separator)
    return "\n".join(lines)


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
