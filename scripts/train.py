#!/usr/bin/env python3
"""启动 Isaac Sim 并训练非对称 SRU-PPO。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import shlex
import sys

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
    parser.add_argument("--network-device", default=None, help="默认与 Isaac 仿真设备相同")
    parser.add_argument("--log-dir", default="runs", help="训练日志根目录")
    parser.add_argument("--run-name", default=None, help="可选实验名，循环单元名称会自动作为前缀")
    parser.add_argument("--checkpoint-dir", default=None, help="可选 checkpoint 根目录")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--toa-grid-size", type=int, default=401)
    parser.add_argument("--cylinders", type=int, default=60)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    simulation_app = AppLauncher(args).app

    # 依赖 omni/PhysX 的模块只能在 AppLauncher 之后导入。
    from ta_sru.algorithms import RecurrentPPO
    from ta_sru.config import EnvConfig, NetworkConfig, PPOConfig, TrainConfig
    from ta_sru.envs import IsaacLabWrapper, NavigationEnv, make_isaac_env_cfg

    env = None
    try:
        task_config = EnvConfig(
            num_envs=args.num_envs,
            seed=args.seed,
            toa_grid_size=args.toa_grid_size,
            random_cylinder_count=args.cylinders,
        )
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{args.recurrent_type}_{args.run_name or timestamp}"
        log_dir = Path(args.log_dir) / run_name
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
        print(f"循环单元：{args.recurrent_type}")
        print(f"训练日志：{log_dir}")
        print(f"模型目录：{checkpoint_dir}")
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
