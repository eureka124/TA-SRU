#!/usr/bin/env python3
"""在 Isaac Sim 中运行已训练的策略。"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import warp  # noqa: F401
from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 Hummingbird 导航策略")
    parser.add_argument("checkpoint")
    parser.add_argument("--num-envs", type=int, default=6)
    parser.add_argument("--steps", type=int, default=20000000)
    parser.add_argument("--network-device", default=None)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="显示目标、速度和轨迹标记，并保存六种迷宫的全局 TOA 图",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _print_error(stage: str, error: BaseException) -> None:
    """立即输出异常阶段、异常类型和完整调用栈。"""

    print(
        f"\n[错误] 评估失败（阶段：{stage}）：{type(error).__name__}: {error}",
        file=sys.stderr,
        flush=True,
    )
    traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
    sys.stderr.flush()


def main() -> int:
    args = parse_args()
    stage = "启动 Isaac Sim"
    simulation_app = None
    env = None
    exit_code = 0
    try:
        simulation_app = AppLauncher(args).app

        stage = "导入项目模块"
        from ta_sru.algorithms import RecurrentPPO
        from ta_sru.config import EnvConfig, NetworkConfig, PPOConfig, TrainConfig
        from ta_sru.envs import IsaacLabWrapper, NavigationEnv, make_isaac_env_cfg

        stage = "读取 checkpoint"
        checkpoint_path = Path(args.checkpoint).resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        values = checkpoint["config"]
        env_values = dict(values["env"])
        env_values["num_envs"] = args.num_envs
        ppo_values = dict(values["ppo"])
        rollout_sample_count = int(ppo_values["rollout_steps"]) * args.num_envs
        checkpoint_batch_size = int(ppo_values["batch_size"])
        if checkpoint_batch_size > rollout_sample_count:
            # 评估不会更新 PPO，仅缩小训练批次参数以通过配置一致性校验。
            ppo_values["batch_size"] = rollout_sample_count
            print(
                "评估环境数较少："
                f"batch_size 从 {checkpoint_batch_size} 调整为 {rollout_sample_count}"
            )
        debug_root = (
            checkpoint_path.parent.parent
            if checkpoint_path.parent.name == "checkpoints"
            else checkpoint_path.parent
        )
        # 不继承 checkpoint 中的调试状态；只有本次显式传入 --debug 才启用。
        env_values["debug"] = args.debug
        env_values["debug_output_dir"] = (
            str(debug_root / "debug") if args.debug else None
        )
        config = TrainConfig(
            env=EnvConfig(**env_values),
            network=NetworkConfig(**values["network"]),
            ppo=PPOConfig(**ppo_values),
            total_timesteps=values["total_timesteps"],
            device=args.network_device or args.device,
        )
        print(f"循环单元：{config.network.recurrent_type}")
        if args.debug:
            print(f"调试输出：{config.env.debug_output_dir}")

        stage = "创建仿真环境"
        isaac_config = make_isaac_env_cfg(config.env, sim_device=args.device)
        env = IsaacLabWrapper(NavigationEnv(isaac_config, config.env))

        stage = "创建策略并加载 checkpoint"
        agent = RecurrentPPO(env, config)
        agent.load(args.checkpoint, load_optimizer=False)
        observation, _ = env.reset()
        state = agent.policy.initial_state(env.num_envs)
        episode_starts = np.ones(env.num_envs, dtype=bool)
        outcomes = {"success": 0, "collision": 0, "timeout": 0}

        stage = "执行策略推理"
        agent.policy.eval()
        for _ in range(args.steps):
            tensor_observation = {
                key: torch.as_tensor(value, dtype=torch.float32, device=agent.device)
                for key, value in observation.items()
            }
            action, _, _, state = agent.policy.act(
                tensor_observation,
                state,
                torch.as_tensor(episode_starts, device=agent.device),
                deterministic=True,
            )
            observation, _, terminated, truncated, infos = env.step(action)
            episode_starts = terminated | truncated
            for env_id in np.flatnonzero(episode_starts):
                if infos[env_id]["success"]:
                    outcomes["success"] += 1
                elif infos[env_id]["collided"]:
                    outcomes["collision"] += 1
                else:
                    outcomes["timeout"] += 1
        print(f"运行结束：{outcomes}")
    # 评估入口必须截获所有常规异常，确保 Isaac Sim 关闭前输出完整调用栈。
    except Exception as error:  # noqa: BLE001
        _print_error(stage, error)
        exit_code = 1
    finally:
        if env is not None:
            try:
                env.close()
            # 清理失败也需要明确输出，避免被 Isaac Sim 的关闭日志掩盖。
            except Exception as error:  # noqa: BLE001
                _print_error("关闭仿真环境", error)
                exit_code = 1
        if simulation_app is not None:
            try:
                simulation_app.close()
            # 清理失败也需要明确输出，避免被 Isaac Sim 的关闭日志掩盖。
            except Exception as error:  # noqa: BLE001
                _print_error("关闭 Isaac Sim", error)
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
