"""用 Isaac Sim 测量 Lee 控制器三个动作通道的闭环阶跃响应。"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

from ta_sru.models.hummingbird import rotate_inverse

CHANNELS = (
    ("forward_velocity", 0, "body_vx", "机体前向速度", "m/s"),
    ("lateral_velocity", 1, "body_vy", "机体侧向速度", "m/s"),
    ("yaw_rate", 2, "body_wz", "机体偏航角速度", "rad/s"),
)


def write_response_page(
    path: Path, samples: list[dict], status: str | None = None
) -> None:
    """生成无需额外绘图库的 SVG 响应曲线页面。"""

    width, height = 960, 300
    left, right, top, bottom = 72, 24, 38, 48
    panels = []
    for env_id, (name, _, response_key, title, unit) in enumerate(CHANNELS):
        series = [
            sample
            for sample in samples
            if sample["env_id"] == env_id
            and all(
                math.isfinite(float(sample[key]))
                for key in ("time_s", "command", response_key)
            )
        ]
        if not series:
            continue
        duration = max(float(item["time_s"]) for item in series)
        values = [float(item[response_key]) for item in series]
        target_values = [float(item["command"]) for item in series]
        low = min(0.0, *values, *target_values)
        high = max(0.0, *values, *target_values)
        padding = max((high - low) * 0.12, 0.05)
        low -= padding
        high += padding
        plot_width = width - left - right
        plot_height = height - top - bottom

        response_coords, target_coords = [], []
        for item in series:
            x = left + plot_width * float(item["time_s"]) / max(duration, 1e-6)
            y = top + plot_height * (high - float(item[response_key])) / (high - low)
            target_y = top + plot_height * (high - float(item["command"])) / (
                high - low
            )
            response_coords.append(f"{x:.1f},{y:.1f}")
            target_coords.append(f"{x:.1f},{target_y:.1f}")
        response_points = " ".join(response_coords)
        target_points = " ".join(target_coords)
        grid = []
        for tick in range(5):
            value = high - (high - low) * tick / 4
            y = top + plot_height * tick / 4
            grid.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
                'stroke="#d7dde5"/>'
                f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" '
                'stroke="#4c5663"/><text '
                f'x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{value:.2f}</text>'
            )
        axis_y = top + plot_height
        for tick in range(8):
            time_s = duration * tick / 7
            x = left + plot_width * tick / 7
            grid.append(
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{axis_y}" '
                'stroke="#d7dde5"/>'
                f'<line x1="{x:.1f}" y1="{axis_y}" x2="{x:.1f}" y2="{axis_y + 5}" '
                'stroke="#4c5663"/>'
                f'<text x="{x:.1f}" y="{axis_y + 20}" '
                f'text-anchor="middle">{time_s:.2f}</text>'
            )
        axes = (
            f'<path d="M {left} {top} V {axis_y} H {width - right}" '
            'fill="none" stroke="#4c5663" stroke-width="1.5"/>'
        )
        panels.append(
            f"<section><h2>{html.escape(title)}阶跃响应 ({html.escape(name)})</h2>"
            f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
            f"{''.join(grid)}"
            f"{axes}"
            f'<polyline points="{target_points}" fill="none" stroke="#e58b24" '
            'stroke-width="2" stroke-dasharray="7 5"/>'
            f'<polyline points="{response_points}" fill="none" stroke="#1769aa" '
            'stroke-width="2.5"/>'
            f'<text x="{left}" y="24">{html.escape(unit)}</text>'
            f'<text x="{width / 2}" y="{height - 8}" text-anchor="middle">时间 (s)</text>'
            f'<text x="{width - right - 150}" y="24" fill="#e58b24">虚线：命令</text>'
            f'<text x="{width - right - 65}" y="24" fill="#1769aa">实线：响应</text>'
            "</svg></section>"
        )
    page = (
        """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Lee 控制器阶跃响应</title><style>
body{font:16px sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#222}
section{margin:28px 0}h2{font-size:18px}svg{width:100%;border:1px solid #d7dde5}
svg text{font-size:12px;fill:#4c5663}
</style><h1>Lee 控制器闭环阶跃响应</h1>
<p>三种动作通道分别在独立环境中测试；响应使用机体系线速度和角速度。</p>
"""
        + (f"<p>{html.escape(status)}</p>" if status else "")
        + "\n".join(panels)
        + "</html>"
    )
    path.write_text(page, encoding="utf-8")


CSV_FIELDS = (
    "time_s",
    "env_id",
    "channel",
    "command",
    "response",
    "body_vx",
    "body_vy",
    "body_wz",
    "terminated_or_truncated",
    "phase",
    "contact_force_n",
    "stop_reason",
    "height_m",
    "vertical_velocity_m_s",
    "thrust_n",
)


def record_response(
    samples: list[dict],
    rig,
    phase: str,
    contacts: torch.Tensor,
    stop_reason: str = "",
    stop_env_ids: tuple[int, ...] = (),
) -> None:
    """保存当前状态；异常环境保留无效值，绘图时过滤，不触发任何重置。"""
    root_state = rig.robot.data.root_state_w
    finite = torch.isfinite(root_state).all(dim=-1)
    for env_id, (name, action_index, response_key, _, _) in enumerate(CHANNELS):
        if bool(finite[env_id]):
            velocity = rotate_inverse(root_state[env_id, 3:7], root_state[env_id, 7:10])
            angular_velocity = rotate_inverse(
                root_state[env_id, 3:7], root_state[env_id, 10:13]
            )
            measured = {
                "body_vx": float(velocity[0]),
                "body_vy": float(velocity[1]),
                "body_wz": float(angular_velocity[2]),
            }
        else:
            measured = dict.fromkeys(("body_vx", "body_vy", "body_wz"), math.nan)
        samples.append(
            {
                "time_s": rig.physics_steps * rig.physics_dt,
                "env_id": env_id,
                "channel": name,
                "command": float(rig.actions[env_id, action_index]),
                "response": measured[response_key],
                **measured,
                "terminated_or_truncated": bool(stop_reason),
                "phase": phase,
                "contact_force_n": float(contacts[env_id]),
                "stop_reason": (
                    stop_reason
                    if env_id in stop_env_ids or not stop_env_ids
                    else "peer_stopped"
                    if stop_reason
                    else ""
                ),
                "height_m": float(root_state[env_id, 2]),
                "vertical_velocity_m_s": float(root_state[env_id, 9]),
                "thrust_n": float(rig.last_wrench[env_id, 0]),
            }
        )


def save_results(output_dir: Path, samples: list[dict], summary: dict) -> None:
    """在关闭 Kit 前落盘，使失败或中断后的数据仍可检查。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "response.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(samples)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    status = (
        f"完整采集：{summary['actual_duration_s']:.3f} 秒"
        if summary["completed"]
        else f"采集未完成：{summary['stop_reason']}，"
        f"实际 {summary['actual_duration_s']:.3f} 秒 / 计划 {summary['planned_duration_s']:.3f} 秒"
    )
    write_response_page(output_dir / "index.html", samples, status)
    print(f"[STEP_RESPONSE] {status}", flush=True)
    print(f"CSV: {output_dir / 'response.csv'}", flush=True)
    print(f"曲线: {output_dir / 'index.html'}", flush=True)
    print(f"摘要: {output_dir / 'summary.json'}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-seconds", type=float, default=6.0, help="阶跃保持时间")
    parser.add_argument(
        "--baseline-seconds", type=float, default=1.0, help="阶跃前悬停时间"
    )
    parser.add_argument(
        "--recovery-seconds", type=float, default=6.0, help="阶跃后回零时间"
    )
    parser.add_argument(
        "--step-fraction",
        type=float,
        default=0.5,
        help="阶跃幅度占对应动作上界的比例",
    )
    parser.add_argument(
        "--plot-csv",
        type=Path,
        default=None,
        help="只从已有 response.csv 重新生成曲线，不启动仿真",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录，默认写入 debug/controller_step_<时间戳>",
    )
    plot_args, _ = parser.parse_known_args()
    if plot_args.plot_csv is not None:
        args = parser.parse_args()
        with args.plot_csv.open(newline="", encoding="utf-8") as stream:
            samples = [
                {**row, "env_id": int(row["env_id"])} for row in csv.DictReader(stream)
            ]
        if not samples:
            parser.error("CSV 中没有响应数据")
        output_dir = args.output_dir or args.plot_csv.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        status = None
        summary_path = args.plot_csv.parent / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            status = (
                "完整采集"
                if summary["completed"]
                else f"采集未完成：{summary['stop_reason']}"
            )
        write_response_page(output_dir / "index.html", samples, status)
        print(f"曲线: {output_dir / 'index.html'}", flush=True)
        return 0
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    durations = (args.baseline_seconds, args.step_seconds, args.recovery_seconds)
    if not all(math.isfinite(value) for value in durations):
        parser.error("时长必须为有限数值")
    if args.step_seconds <= 0 or args.baseline_seconds < 0 or args.recovery_seconds < 0:
        parser.error("阶跃时长必须为正，基线和恢复时长不能为负")
    if not 0.0 < args.step_fraction <= 1.0:
        parser.error("--step-fraction 必须位于 (0, 1]")

    output_dir = args.output_dir or Path("debug") / (
        "controller_step_"
        + datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    )
    summary = {
        "completed": False,
        "stop_reason": "initialization_failed",
        "stop_env_ids": [],
        "actual_duration_s": 0.0,
        "planned_duration_s": sum(durations),
        "requested_durations_s": dict(
            zip(("baseline", "step", "recovery"), durations, strict=True)
        ),
        "step_fraction": args.step_fraction,
        "scene": "obstacle_free_controller_test",
    }
    app = None
    rig = None
    exit_code = 0
    samples: list[dict] = []
    phase = "baseline"
    contacts = None
    stage = "启动 Isaac Sim"
    try:
        print(f"[STEP_RESPONSE] {stage}", flush=True)
        app_launcher = AppLauncher(args, fast_shutdown=True)
        app = app_launcher.app
        stage = "构建独立无障碍场景"
        print(f"[STEP_RESPONSE] {stage}", flush=True)
        from ta_sru.envs.controller_test import ControllerStepTestRig

        # 间距随测试时长增加，给超调及回零漂移预留空间。
        spacing = max(20.0, 8.0 * sum(durations) + 10.0)
        rig = ControllerStepTestRig(args.device, spacing)
        summary.update(rig.metadata())
        counts = [round(value / rig.sample_dt) for value in durations]
        counts[1] = max(1, counts[1])
        summary["effective_durations_s"] = dict(
            zip(
                ("baseline", "step", "recovery"),
                (count * rig.sample_dt for count in counts),
                strict=True,
            )
        )
        summary["planned_duration_s"] = sum(counts) * rig.sample_dt
        amplitudes = [
            rig.task.action_high[index] * args.step_fraction
            for _, index, *_ in CHANNELS
        ]
        summary["step_amplitudes"] = amplitudes
        commands = torch.zeros((3, 3), dtype=torch.float32, device=rig.device)
        contacts = torch.zeros(3, device=rig.device)
        record_response(samples, rig, "baseline", contacts)
        stage = "采集阶跃与回零响应"
        print(
            f"[STEP_RESPONSE] {stage}：计划 {summary['planned_duration_s']:.3f}s，"
            f"采样间隔 {rig.sample_dt:.5f}s，阶跃幅度 {amplitudes}",
            flush=True,
        )
        stop_reason = ""
        stop_env_ids: tuple[int, ...] = ()
        for sample_index in range(sum(counts)):
            phase = (
                "baseline"
                if sample_index < counts[0]
                else "step"
                if sample_index < counts[0] + counts[1]
                else "recovery"
            )
            commands.zero_()
            if phase == "step":
                for env_id, (_, action_index, *_rest) in enumerate(CHANNELS):
                    commands[env_id, action_index] = amplitudes[env_id]
            contacts.zero_()
            for _ in range(rig.task.control_decimation):
                if not app.is_running():
                    stop_reason = "application_closed"
                    exit_code = 130
                    break
                rig.step(commands)
                state_valid = torch.isfinite(rig.robot.data.root_state_w).all(dim=-1)
                forces = rig.contact_force()
                valid = state_valid & torch.isfinite(forces)
                contacts = torch.maximum(contacts, forces)
                if not bool(valid.all()):
                    stop_reason = "nonfinite_state"
                    stop_env_ids = tuple(torch.nonzero(~valid).flatten().cpu().tolist())
                    exit_code = 1
                    break
                collided = forces >= rig.task.collision_force_threshold
                if bool(collided.any()):
                    stop_reason = "contact"
                    stop_env_ids = tuple(
                        torch.nonzero(collided).flatten().cpu().tolist()
                    )
                    exit_code = 1
                    break
            record_response(samples, rig, phase, contacts, stop_reason, stop_env_ids)
            if stop_reason:
                break
        summary["completed"] = not bool(stop_reason)
        summary["stop_reason"] = stop_reason or "completed"
        summary["stop_env_ids"] = stop_env_ids
    except KeyboardInterrupt:
        summary["stop_reason"] = "interrupted"
        exit_code = 130
        print("[STEP_RESPONSE] 用户中断，保存已有数据", flush=True)
    except Exception as error:  # noqa: BLE001
        print(f"[STEP_RESPONSE] 失败阶段：{stage}", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        summary["stop_reason"] = (
            "nonfinite_control"
            if isinstance(error, FloatingPointError)
            else "exception"
        )
        summary["failure_stage"] = stage
        summary["error"] = str(error)
        exit_code = 1
    finally:
        if rig is not None:
            summary["actual_duration_s"] = rig.physics_steps * rig.physics_dt
        if not summary["completed"]:
            # 中断或异常可能发生在采样间隔内，给最后已保存状态补上停止原因。
            for sample in samples[-3:]:
                if not sample["stop_reason"]:
                    sample["stop_reason"] = summary["stop_reason"]
                    sample["terminated_or_truncated"] = True
        try:
            save_results(output_dir, samples, summary)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            sys.stderr.flush()
            exit_code = 1
        if exit_code != 0 and app is not None:
            # 数据与报错已落盘，避免 Kit 快速关闭把失败或中断变成成功退出。
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)
        if app is not None:
            # 数据均已落盘；部分 Kit 版本关闭时会挂起，限制清理等待时间。
            def finish_process() -> None:
                print("[STEP_RESPONSE] 数据已保存，结束仿真进程", flush=True)
                sys.stderr.flush()
                os._exit(exit_code)

            shutdown_timer = threading.Timer(5.0, finish_process)
            shutdown_timer.daemon = True
            shutdown_timer.start()
            try:
                app.close(wait_for_replicator=False)
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                exit_code = 1
            finally:
                shutdown_timer.cancel()
            # 仿真资源由本进程独占，避免解释器析构插件时再次触发段错误。
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
