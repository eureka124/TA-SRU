#!/usr/bin/env python3
"""启动 Isaac Sim 并训练非对称 SRU-PPO。"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from training_logging import (
    RUN_DIRECTORY_ENV,
    record_failure,
    run_child,
    supervise,
    write_status,
)

RECURRENT_TYPES = ("sru-lstm", "sru-gru", "sru-lstm-gate", "lstm")


def parse_args() -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="训练 Hummingbird 迷宫导航策略")
    parser.add_argument("--algorithm", choices=("ppo", "recurrent_ppo"), default=None)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--total-timesteps", type=int, default=70_000_000)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument(
        "--recurrent-type",
        type=lambda value: {"nn.lstm": "lstm"}.get(value, value.replace("_", "-")),
        choices=RECURRENT_TYPES,
        default=None,
        help="循环单元；lstm 表示 torch.nn.LSTM",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--network-device", default=None, help="默认与 Isaac 仿真设备相同"
    )
    parser.add_argument("--log-dir", default="runs", help="训练日志根目录")
    parser.add_argument(
        "--run-name", default=None, help="可选实验名，循环单元名称会自动作为前缀"
    )
    parser.add_argument("--checkpoint-dir", default=None, help="可选 checkpoint 根目录")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="每隔多少次 PPO 更新保存 checkpoint（默认：5）",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--commit-id",
        default=None,
        help="patch 对应的基准提交；默认从 diff 文件名识别或使用当前 HEAD",
    )
    parser.add_argument(
        "--diff-file",
        "--diff_file",
        dest="diff_file",
        default=None,
        help="相对基准提交生成的 patch；默认自动执行 git diff",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="每隔多少次 PPO 更新写入 CSV 和 TensorBoard（默认：1）",
    )
    for option in (
        "maze-seed",
        "maze-size",
        "train-map-count",
        "eval-map-count",
        "goals-per-map",
    ):
        parser.add_argument(f"--{option}", type=int, default=None)
    for option in (
        "maze-cell-size",
        "maze-wall-removal-probability",
        "toa-resolution",
        "episode-seconds",
        "safety-margin",
        "spawn-margin",
        "min-start-goal-distance",
    ):
        parser.add_argument(f"--{option}", type=float, default=None)
    parser.add_argument("--toa-cache-dir", default=None)
    parser.add_argument(
        "--toa-normalization-max",
        type=float,
        default=None,
        help="固定 TOA 量程；新训练默认自动计算，恢复时默认沿用 checkpoint",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="显示目标、速度和轨迹标记，并保存地图池的各目标 TOA 图",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _git_stdout(repository_root: Path, *arguments: str) -> bytes:
    """在项目仓库中执行只读 Git 命令并返回标准输出。"""

    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("无法记录代码版本：系统中找不到 git") from error
    except subprocess.CalledProcessError as error:
        message = error.stderr.decode(errors="replace").strip()
        raise RuntimeError(
            f"无法记录代码版本：git {' '.join(arguments)}：{message}"
        ) from error
    return result.stdout


def _capture_source_state(
    repository_root: Path, commit_argument: str | None, diff_argument: str | None
) -> tuple[str, bytes]:
    """确定基准提交，并读取或生成相对该提交的改动。"""

    diff_path = None
    commit_reference = commit_argument
    if diff_argument is not None:
        diff_path = Path(diff_argument).expanduser().resolve()
        if not diff_path.is_file():
            raise FileNotFoundError(f"找不到 diff 文件：{diff_path}")
        if commit_reference is None:
            match = re.fullmatch(r"diff_([0-9a-fA-F]{7,64})\.patch", diff_path.name)
            if match is not None:
                commit_reference = match.group(1)

    commit_reference = commit_reference or "HEAD"
    commit_id = (
        _git_stdout(
            repository_root, "rev-parse", "--verify", f"{commit_reference}^{{commit}}"
        )
        .decode()
        .strip()
    )
    diff_content = (
        diff_path.read_bytes()
        if diff_path is not None
        else _git_stdout(repository_root, "diff", commit_id)
    )
    if diff_path is None:
        # 普通 git diff 不包含新增未跟踪模块；将可见源码一并纳入运行快照。
        untracked = _git_stdout(
            repository_root, "ls-files", "--others", "--exclude-standard", "-z"
        )
        for name in untracked.decode().split("\0"):
            if (
                not name
                or Path(name).parts[0] not in {"src", "scripts", "tests"}
                or Path(name).suffix not in {".py", ".sh", ".md"}
            ):
                continue
            added = subprocess.run(
                ["git", "diff", "--no-index", "--", "/dev/null", name],
                cwd=repository_root,
                capture_output=True,
                check=False,
            )
            if added.returncode not in (0, 1):
                raise RuntimeError(f"无法记录新增源码：{name}")
            diff_content += added.stdout
    return commit_id, diff_content


def _raise_keyboard_interrupt(_signal_number: int, _frame: object) -> None:
    """确保终端 SIGINT 能进入训练脚本的中断保存流程。"""

    raise KeyboardInterrupt


def main() -> None:
    # 日志监控已启动，依赖导入失败也会留下完整异常。
    import torch
    import warp  # noqa: F401
    from isaaclab.app import AppLauncher

    args = parse_args()
    from ta_sru.evaluation import require_dfs_checkpoint
    from ta_sru.scene_config import training_env_config

    checkpoint = (
        torch.load(args.resume, map_location="cpu", weights_only=True)
        if args.resume
        else None
    )
    if checkpoint is not None:
        require_dfs_checkpoint(checkpoint)
    saved = checkpoint["config"] if checkpoint else None
    algorithm = args.algorithm or (
        saved.get("algorithm", "recurrent_ppo") if saved else "recurrent_ppo"
    )
    recurrent_type = args.recurrent_type or (
        saved["network"]["recurrent_type"] if saved else "sru-lstm"
    )
    if algorithm == "ppo":
        if args.recurrent_type is not None:
            raise ValueError("普通 PPO 不接受 --recurrent-type")
        recurrent_type = "none"
    if saved and (
        algorithm != saved.get("algorithm", "recurrent_ppo")
        or recurrent_type != saved["network"]["recurrent_type"]
    ):
        raise ValueError("恢复训练时不能更换算法或循环单元")
    args.recurrent_type = recurrent_type
    repository_root = Path(__file__).resolve().parents[1]
    commit_id, diff_content = _capture_source_state(
        repository_root, args.commit_id, args.diff_file
    )
    simulation_app = None
    env = None
    agent = None
    log_dir = Path(os.environ[RUN_DIRECTORY_ENV])
    run_name = log_dir.name
    try:
        simulation_app = AppLauncher(args).app
        # 依赖 omni/PhysX 的模块只能在 AppLauncher 之后导入。
        from ta_sru.algorithms import RecurrentPPO
        from ta_sru.config import NetworkConfig, PPOConfig, TrainConfig
        from ta_sru.envs import IsaacLabWrapper, NavigationEnv, make_isaac_env_cfg

        fields = (
            "seed",
            "maze_seed",
            "maze_size",
            "train_map_count",
            "eval_map_count",
            "goals_per_map",
            "maze_cell_size",
            "maze_wall_removal_probability",
            "toa_resolution",
            "episode_seconds",
            "safety_margin",
            "spawn_margin",
            "min_start_goal_distance",
            "toa_cache_dir",
            "toa_normalization_max",
        )
        overrides = {field: getattr(args, field) for field in fields}
        overrides.update(
            num_envs=args.num_envs,
            debug=args.debug,
            debug_output_dir=str(log_dir / "debug") if args.debug else None,
        )
        task_config = training_env_config(saved["env"] if saved else None, overrides)
        checkpoint_dir = (
            log_dir / "checkpoints"
            if args.checkpoint_dir is None
            else Path(args.checkpoint_dir) / run_name
        )
        config = TrainConfig(
            env=task_config,
            network=NetworkConfig(**saved["network"])
            if saved
            else NetworkConfig(recurrent_type=args.recurrent_type),
            algorithm=algorithm,
            ppo=PPOConfig(
                rollout_steps=args.rollout_steps,
                batch_size=args.batch_size,
                recurrent_sequence_length=args.sequence_length,
            ),
            total_timesteps=args.total_timesteps,
            device=args.network_device or args.device,
            log_interval=args.log_interval,
            checkpoint_interval=args.checkpoint_interval,
            log_dir=str(log_dir),
            checkpoint_dir=str(checkpoint_dir),
        )
        config.validate()
        (log_dir / "config.json").write_text(
            json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (log_dir / "command.txt").write_text(
            shlex.join(sys.argv) + "\n", encoding="utf-8"
        )
        (log_dir / "commit_id.txt").write_text(commit_id + "\n", encoding="utf-8")
        (log_dir / "diff.patch").write_bytes(diff_content)
        print(f"循环单元：{args.recurrent_type}")
        print(f"训练日志：{log_dir}")
        print(f"模型目录：{checkpoint_dir}")
        print(f"基准提交：{commit_id}")
        if args.debug:
            print(f"调试输出：{task_config.debug_output_dir}")
        isaac_config = make_isaac_env_cfg(task_config, sim_device=args.device)
        env = IsaacLabWrapper(NavigationEnv(isaac_config, task_config))
        (log_dir / "scene_manifest.json").write_text(
            json.dumps(env.scene_manifest, indent=2), encoding="utf-8"
        )
        (log_dir / "config.json").write_text(
            json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        agent = RecurrentPPO(env, config)
        if args.resume:
            agent.load(args.resume)
        previous_sigint_handler = signal.signal(
            signal.SIGINT, _raise_keyboard_interrupt
        )
        try:
            agent.learn()
        except KeyboardInterrupt as error:
            record_failure(log_dir, error)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            interrupted_checkpoint = (
                checkpoint_dir / f"model_interrupted_{agent.timesteps}.pt"
            )
            print(f"\n收到中断，正在保存 checkpoint：{interrupted_checkpoint}")
            try:
                agent.save(interrupted_checkpoint)
                print("中断 checkpoint 已保存，可以使用 --resume 继续训练。")
            except Exception:  # noqa: BLE001
                # 保存或记录失败时保留原始训练异常。
                traceback.print_exc()
            raise
        else:
            agent.save(checkpoint_dir / "model_final.pt")
        finally:
            signal.signal(signal.SIGINT, previous_sigint_handler)
    except BaseException as error:
        # Isaac Sim 的关闭流程可能结束进程，必须提前保存原始异常。
        record_failure(log_dir, error)
        traceback.print_exc()
        raise
    finally:
        active_error = sys.exc_info()[0] is not None
        if agent is not None:
            try:
                write_status(log_dir, last_training_step=agent.timesteps)
            except Exception:  # noqa: BLE001
                # 保存或记录失败时保留原始训练异常。
                traceback.print_exc()
        cleanup_error = None
        for resource in (env, simulation_app):
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as error:  # noqa: BLE001
                # 关闭阶段的异常不能覆盖训练异常，也不能阻止关闭其他资源。
                traceback.print_exc()
                if not active_error:
                    record_failure(log_dir, error)
                cleanup_error = cleanup_error or error
        if cleanup_error is not None and not active_error:
            raise cleanup_error


if __name__ == "__main__":
    if RUN_DIRECTORY_ENV in os.environ:
        # 帮助路径直接退出，不生成训练状态。
        if not os.environ[RUN_DIRECTORY_ENV]:
            main()
        else:
            sys.exit(run_child(main))
    else:
        sys.exit(supervise(Path(__file__).resolve()))
