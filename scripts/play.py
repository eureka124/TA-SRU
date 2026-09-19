#!/usr/bin/env python3
"""在 Isaac Sim 中运行已训练的策略。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import warp  # noqa: F401
from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 Hummingbird 导航策略")
    parser.add_argument("checkpoint")
    parser.add_argument("--num-envs", type=int, default=6)
    parser.add_argument(
        "--steps", type=int, default=None, help="可选的提前停止步数上限"
    )
    parser.add_argument("--episodes-per-layout", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--network-device", default=None)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="显示导航标记，保存 TOA 图、原始深度视频和动作/状态/俯视轨迹 HTML 回放",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 6:
        parser.error("--num-envs 至少为 6，确保覆盖所有布局")
    if args.episodes_per_layout <= 0 or (args.steps is not None and args.steps <= 0):
        parser.error("回合数量和步数上限必须为正数")
    return args


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
    debug_recorder = None
    stats = None
    results_dir = None
    episode_file = None
    report_metadata = {}
    exit_code = 0
    try:
        simulation_app = AppLauncher(args).app

        stage = "导入项目模块"
        from ta_sru.algorithms import RecurrentPPO
        from ta_sru.config import EnvConfig, NetworkConfig, PPOConfig, TrainConfig
        from ta_sru.envs import IsaacLabWrapper, NavigationEnv, make_isaac_env_cfg
        from ta_sru.envs.layouts import MAZE_LAYOUTS
        from ta_sru.evaluation import (
            EvaluationStats,
            configure_evaluation,
            format_evaluation_table,
        )

        stage = "读取 checkpoint"
        checkpoint_path = Path(args.checkpoint).resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        values = checkpoint["config"]
        env_values = configure_evaluation(values["env"])
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
        run_dir = (
            checkpoint_path.parent.parent
            if checkpoint_path.parent.name == "checkpoints"
            else checkpoint_path.parent
        )
        # 不继承 checkpoint 中的调试状态；只有本次显式传入 --debug 才启用。
        env_values["debug"] = args.debug
        env_values["debug_output_dir"] = str(run_dir / "debug") if args.debug else None
        config = TrainConfig(
            env=EnvConfig(**env_values),
            network=NetworkConfig(**values["network"]),
            algorithm=values.get("algorithm", "recurrent_ppo"),
            ppo=PPOConfig(**ppo_values),
            total_timesteps=values["total_timesteps"],
            device=args.network_device or args.device,
        )
        results_dir = args.output_dir or (
            run_dir
            / "evaluation"
            / datetime.now(timezone.utc).astimezone().strftime("eval_%Y%m%d_%H%M%S_%f")
        )
        results_dir.mkdir(parents=True, exist_ok=False)
        stats = EvaluationStats(args.episodes_per_layout)
        report_metadata = {
            "checkpoint": str(checkpoint_path),
            "num_envs": args.num_envs,
            "collision_force_threshold": config.env.collision_force_threshold,
            "cylinders": min(
                config.env.training_curriculum_cylinder_counts[0],
                config.env.random_cylinder_count,
            ),
        }
        episode_file = (results_dir / "episodes.csv").open(
            "w", encoding="utf-8", newline=""
        )
        episode_writer = csv.DictWriter(
            episode_file,
            fieldnames=("maze", "env_id", "episode", "outcome", "steps", "return"),
        )
        episode_writer.writeheader()
        print(f"评估结果：{results_dir}；每种布局 {stats.target} 回合", flush=True)
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
        if args.debug:
            from ta_sru.debug.playback import PlayDebugRecorder

            debug_recorder = PlayDebugRecorder(
                config.env.debug_output_dir,
                config.env.policy_dt,
                max_episodes_per_layout=min(4, args.episodes_per_layout),
                episodes_per_layout=args.episodes_per_layout,
            )
            env.env.play_debug_recorder = debug_recorder
            print(
                f"[DEBUG] 回放索引：{debug_recorder.output_dir / 'index.html'}",
                flush=True,
            )
            print(
                f"[DEBUG] 每种布局按开始顺序录制第 {debug_recorder.selection_range[0]}–{debug_recorder.selection_range[1]} 个回合",
                flush=True,
            )
        state = agent.policy.initial_state(env.num_envs)
        episode_starts = np.ones(env.num_envs, dtype=bool)
        episode_lengths = np.zeros(env.num_envs, dtype=np.int64)
        episode_returns = np.zeros(env.num_envs, dtype=np.float64)
        step = 0

        stage = "执行策略推理"
        agent.policy.eval()
        # 统计满额后，允许已选中的回放完成；额外回合不再计入统计。
        while (
            not stats.complete
            or (debug_recorder is not None and debug_recorder.episodes)
        ) and (args.steps is None or step < args.steps):
            if not simulation_app.is_running():
                break
            tensor_observation = {
                key: torch.as_tensor(value, dtype=torch.float32, device=agent.device)
                for key, value in observation.items()
            }
            with torch.inference_mode():
                action, _, _, state = agent.policy.act(
                    tensor_observation,
                    state,
                    torch.as_tensor(episode_starts, device=agent.device),
                    deterministic=True,
                )
            if debug_recorder is not None:
                debug_recorder.begin_step(env.env, action)
            # 在自动重置前保留本回合布局，避免终局归入下一回合。
            maze_ids = env.env.maze_ids.detach().cpu().tolist()
            observation, reward, terminated, truncated, infos = env.step(action)
            step += 1
            episode_lengths += 1
            episode_returns += reward
            episode_starts = terminated | truncated
            for env_id in np.flatnonzero(episode_starts):
                maze = MAZE_LAYOUTS[maze_ids[env_id]].name
                outcome = stats.record(maze, infos[env_id])
                if outcome is not None:
                    episode_writer.writerow(
                        {
                            "maze": maze,
                            "env_id": int(env_id),
                            "episode": sum(stats.counts[maze].values()),
                            "outcome": outcome,
                            "steps": int(episode_lengths[env_id]),
                            "return": float(episode_returns[env_id]),
                        }
                    )
                episode_lengths[env_id] = 0
                episode_returns[env_id] = 0
            if np.any(episode_starts):
                episode_file.flush()
                report = {**report_metadata, **stats.summary()}
                temporary = results_dir / "summary.json.tmp"
                temporary.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                temporary.replace(results_dir / "summary.json")
            if step % 100 == 0:
                print(
                    "评估进度："
                    + "，".join(
                        f"{maze}={sum(counts.values())}/{stats.target}"
                        for maze, counts in stats.counts.items()
                    ),
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\n评估已中断，保存已完成回合的统计。", flush=True)
        exit_code = 130
    # 评估入口必须截获所有常规异常，确保 Isaac Sim 关闭前输出完整调用栈。
    except Exception as error:  # noqa: BLE001
        _print_error(stage, error)
        exit_code = 1
    finally:
        if episode_file is not None:
            try:
                episode_file.close()
            except Exception as error:  # noqa: BLE001
                _print_error("关闭评估明细文件", error)
                exit_code = 1
        if stats is not None and results_dir is not None:
            try:
                report = {**report_metadata, **stats.summary()}
                (results_dir / "summary.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(
                    f"\n评估{'完成' if stats.complete else '提前结束'}"
                    f"（每种布局目标：{stats.target} 回合）\n"
                    f"{format_evaluation_table(report)}\n"
                    f"评估结果：{results_dir}",
                    flush=True,
                )
            except Exception as error:  # noqa: BLE001
                _print_error("保存评估统计", error)
                exit_code = 1
        if debug_recorder is not None:
            try:
                debug_recorder.close()
            except Exception as error:  # noqa: BLE001
                _print_error("保存调试回放", error)
                exit_code = 1
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
