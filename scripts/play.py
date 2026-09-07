#!/usr/bin/env python3
"""在 Isaac Sim 中运行已训练的策略。"""

from __future__ import annotations

import argparse
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


def main() -> None:
    args = parse_args()
    simulation_app = AppLauncher(args).app

    from ta_sru.algorithms import RecurrentPPO
    from ta_sru.config import EnvConfig, NetworkConfig, PPOConfig, TrainConfig
    from ta_sru.envs import IsaacLabWrapper, NavigationEnv, make_isaac_env_cfg

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    values = checkpoint["config"]
    env_values = dict(values["env"])
    env_values["num_envs"] = args.num_envs
    debug_root = (
        checkpoint_path.parent.parent
        if checkpoint_path.parent.name == "checkpoints"
        else checkpoint_path.parent
    )
    # 不继承 checkpoint 中的调试状态；只有本次显式传入 --debug 才启用。
    env_values["debug"] = args.debug
    env_values["debug_output_dir"] = str(debug_root / "debug") if args.debug else None
    config = TrainConfig(
        env=EnvConfig(**env_values),
        network=NetworkConfig(**values["network"]),
        ppo=PPOConfig(**values["ppo"]),
        total_timesteps=values["total_timesteps"],
        device=args.network_device or args.device,
    )
    print(f"循环单元：{config.network.recurrent_type}")
    if args.debug:
        print(f"调试输出：{config.env.debug_output_dir}")

    env = None
    try:
        isaac_config = make_isaac_env_cfg(config.env, sim_device=args.device)
        env = IsaacLabWrapper(NavigationEnv(isaac_config, config.env))
        agent = RecurrentPPO(env, config)
        agent.load(args.checkpoint, load_optimizer=False)
        observation, _ = env.reset()
        state = agent.policy.initial_state(env.num_envs)
        episode_starts = np.ones(env.num_envs, dtype=bool)
        outcomes = {"success": 0, "collision": 0, "timeout": 0}

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
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
