#!/usr/bin/env python3
"""将现有 progress.csv 回填为 TensorBoard 事件。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


LEGACY_TAGS = {
    "policy_loss": "train/policy_gradient_loss",
    "value_loss": "train/value_loss",
    "approx_kl": "train/approx_kl",
    "clip_fraction": "train/clip_fraction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 TA-SRU progress.csv 导入同一运行目录的 TensorBoard"
    )
    parser.add_argument("progress_csv", type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="允许再次导入（可能在 TensorBoard 中产生重复点）",
    )
    return parser.parse_args()


def optional_float(row: dict[str, str], name: str) -> float | None:
    value = row.get(name)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def main() -> None:
    args = parse_args()
    progress_path = args.progress_csv.expanduser().resolve()
    if not progress_path.is_file():
        raise SystemExit(f"错误：找不到 CSV：{progress_path}")
    log_dir = progress_path.parent
    previous_imports = list(log_dir.glob("events.out.tfevents.*.progress-import"))
    if previous_imports and not args.force:
        raise SystemExit(
            "错误：该目录已导入过 progress.csv；"
            "如需重复导入请传 --force"
        )

    with progress_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise SystemExit(f"错误：CSV 没有数据行：{progress_path}")

    ppo_config: dict[str, object] = {}
    config_path = log_dir / "config.json"
    if config_path.is_file():
        values = json.loads(config_path.read_text(encoding="utf-8"))
        ppo_config = values.get("ppo", {})

    writer = SummaryWriter(log_dir=log_dir, filename_suffix=".progress-import")
    try:
        for row in rows:
            step_value = optional_float(row, "timesteps")
            if step_value is None:
                continue
            step = int(step_value)
            for name in row:
                if name == "recurrent_type":
                    continue
                value = optional_float(row, name)
                if value is not None:
                    writer.add_scalar(f"progress/{name}", value, step)

            recurrent_type = row.get("recurrent_type")
            if recurrent_type:
                writer.add_text("progress/recurrent_type", recurrent_type, step)
            for name, tag in LEGACY_TAGS.items():
                value = optional_float(row, name)
                if value is not None:
                    writer.add_scalar(tag, value, step)
            entropy = optional_float(row, "entropy")
            if entropy is not None:
                writer.add_scalar("train/entropy_loss", -entropy, step)

            counts = {
                outcome: optional_float(row, outcome) or 0.0
                for outcome in ("collision", "success", "timeout")
            }
            outcome_count = sum(counts.values())
            if outcome_count > 0:
                for outcome, tag in (
                    ("collision", "Metrics/Collision_Rate"),
                    ("success", "Metrics/Success_Rate"),
                    ("timeout", "Metrics/Timeout_Rate"),
                ):
                    writer.add_scalar(tag, counts[outcome] / outcome_count, step)

            for config_name, tag in (
                ("clip_range", "train/clip_range"),
                ("value_clip_range", "train/clip_range_vf"),
                ("learning_rate", "train/learning_rate"),
            ):
                value = ppo_config.get(config_name)
                if isinstance(value, (int, float)):
                    writer.add_scalar(tag, value, step)
    finally:
        writer.close()
    print(f"已导入 {len(rows)} 条记录到：{log_dir}")


if __name__ == "__main__":
    main()
