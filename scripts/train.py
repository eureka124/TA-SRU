#!/usr/bin/env python3
"""启动 Isaac Sim 并训练非对称 SRU-PPO。"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# Isaac Lab 2.3.2 需要先加载 Conda 环境中的新版 Warp。
import warp  # noqa: F401
from isaaclab.app import AppLauncher

RECURRENT_TYPES = ("sru-lstm", "sru-gru", "sru-lstm-gate", "lstm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 Hummingbird 迷宫导航策略")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--total-timesteps", type=int, default=70_000_000)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument(
        "--recurrent-type",
        choices=RECURRENT_TYPES,
        default="sru-lstm",
        help="循环单元；lstm 表示 torch.nn.LSTM",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--network-device", default=None, help="默认与 Isaac 仿真设备相同"
    )
    parser.add_argument("--log-dir", default="runs", help="训练日志根目录")
    parser.add_argument(
        "--run-name", default=None, help="可选实验名，循环单元名称会自动作为前缀"
    )
    parser.add_argument("--checkpoint-dir", default=None, help="可选 checkpoint 根目录")
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
    parser.add_argument("--toa-grid-size", type=int, default=401)
    parser.add_argument("--cylinders", type=int, default=60)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="显示目标、速度和轨迹标记，并保存六种迷宫的全局 TOA 图",
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
    return commit_id, diff_content


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    commit_id, diff_content = _capture_source_state(
        repository_root, args.commit_id, args.diff_file
    )
    simulation_app = AppLauncher(args).app

    # 依赖 omni/PhysX 的模块只能在 AppLauncher 之后导入。
    from ta_sru.algorithms import RecurrentPPO
    from ta_sru.config import EnvConfig, NetworkConfig, PPOConfig, TrainConfig
    from ta_sru.envs import IsaacLabWrapper, NavigationEnv, make_isaac_env_cfg

    env = None
    try:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{args.recurrent_type}_{args.run_name or timestamp}"
        log_dir = Path(args.log_dir) / run_name
        task_config = EnvConfig(
            num_envs=args.num_envs,
            seed=args.seed,
            debug=args.debug,
            debug_output_dir=str(log_dir / "debug") if args.debug else None,
            toa_grid_size=args.toa_grid_size,
            random_cylinder_count=args.cylinders,
        )
        checkpoint_dir = (
            log_dir / "checkpoints"
            if args.checkpoint_dir is None
            else Path(args.checkpoint_dir) / run_name
        )
        config = TrainConfig(
            env=task_config,
            network=NetworkConfig(recurrent_type=args.recurrent_type),
            ppo=PPOConfig(
                rollout_steps=args.rollout_steps,
                batch_size=args.batch_size,
                recurrent_sequence_length=args.sequence_length,
            ),
            total_timesteps=args.total_timesteps,
            device=args.network_device or args.device,
            log_interval=args.log_interval,
            log_dir=str(log_dir),
            checkpoint_dir=str(checkpoint_dir),
        )
        config.validate()
        log_dir.mkdir(parents=True, exist_ok=False)
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
        agent = RecurrentPPO(env, config)
        if args.resume:
            agent.load(args.resume)
        try:
            agent.learn()
        except KeyboardInterrupt:
            print("\n收到中断，保存当前模型。")
        agent.save(checkpoint_dir / "model_final.pt")
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
