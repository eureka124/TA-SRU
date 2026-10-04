"""独立于仿真的评估配置、终局分类和按地图计数。"""

from __future__ import annotations

from unicodedata import east_asian_width

from ta_sru.envs.maze import SCENE_VERSION


def configure_evaluation(values: dict) -> dict:
    """只覆盖本次评估配置，使训练进度不再影响场景或碰撞阈值。"""
    values = dict(values)
    if values.get("toa_normalization_max") is None:
        raise ValueError("DFS 评估需要 checkpoint 中的训练 TOA 观测量程")
    values.update(
        map_split="eval",
        collision_force_threshold=0.1,
        minimum_contact_force_threshold=0.1,
    )
    return values


def load_evaluation_policy(checkpoint: dict, device: str):
    """只加载推理网络，不创建优化器、训练缓冲或训练日志。"""
    import torch

    from ta_sru.config import NetworkConfig
    from ta_sru.models.actor_critic import (
        AsymmetricRecurrentActorCritic,
        require_supported_action_transform,
    )

    # 评估不经过 RecurrentPPO.load，必须在这里单独校验动作参数化方式。
    require_supported_action_transform(checkpoint.get("action_transform"))
    require_dfs_checkpoint(checkpoint)
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
            "地图",
            "回合数",
            "成功（数量/比例）",
            "碰撞（数量/比例）",
            "超时（数量/比例）",
        ]
    ]
    for name, counts in [*summary["maps"].items(), ("整体", summary["overall"])]:
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
    def __init__(self, episodes_per_map: int, map_names: list[str]) -> None:
        if episodes_per_map <= 0:
            raise ValueError("每张地图的评估回合数必须为正数")
        if not map_names or len(set(map_names)) != len(map_names):
            raise ValueError("评估地图清单必须非空且不重复")
        self.target = episodes_per_map
        self.counts = {
            name: {"success": 0, "collision": 0, "timeout": 0} for name in map_names
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
            "episodes_per_map": self.target,
            "complete": self.complete,
            "maps": {maze: rates(counts) for maze, counts in self.counts.items()},
            "overall": rates(total),
        }


def require_dfs_checkpoint(checkpoint: dict) -> None:
    manifest = checkpoint.get("scene_manifest")
    if not isinstance(manifest, dict) or manifest.get("scene_version") != SCENE_VERSION:
        raise ValueError(
            "checkpoint 不属于当前 DFS/起点 TOA 归一化任务，不能恢复或评估旧场景"
        )


class MapQuotaScheduler:
    """回合开始即预留配额，使任意环境数均可遍历完整地图清单。"""

    def __init__(self, map_count: int, episodes_per_map: int) -> None:
        if map_count <= 0 or episodes_per_map <= 0:
            raise ValueError("地图数和评估配额必须为正数")
        self.map_count = map_count
        self.total = map_count * episodes_per_map
        self.assigned = 0

    def assign(self) -> int:
        if self.assigned >= self.total:
            return -1
        selected = self.assigned % self.map_count
        self.assigned += 1
        return selected
