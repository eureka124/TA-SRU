"""面向本项目环境接口的简洁 Recurrent PPO 实现。"""

from __future__ import annotations

from collections import deque
import csv
from dataclasses import asdict
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from ta_sru.algorithms.buffer import CpuRolloutBuffer
from ta_sru.config import TrainConfig
from ta_sru.models.actor_critic import AsymmetricRecurrentActorCritic, RecurrentState

ObservationArray = dict[str, np.ndarray]


def _resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[提示] CUDA 不可用，训练设备回退到 CPU")
        return torch.device("cpu")
    return torch.device(requested)


def _observation_tensor(
    observation: ObservationArray, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in observation.items()
    }


class RecurrentPPO:
    """采样、GAE 与 PPO 更新均显式写在一个类中。"""

    def __init__(self, env: Any, config: TrainConfig) -> None:
        config.validate()
        self.env = env
        self.config = config
        self.device = _resolve_device(config.device)
        torch.manual_seed(config.env.seed)
        np.random.seed(config.env.seed)
        self.rng = np.random.default_rng(config.env.seed)
        self.policy = AsymmetricRecurrentActorCritic(config.network).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.ppo.learning_rate, eps=1.0e-5
        )

        self.observation, _ = env.reset()
        self.episode_starts = np.ones(env.num_envs, dtype=bool)
        self.recurrent_state = self.policy.initial_state(env.num_envs)
        self.buffer = CpuRolloutBuffer(
            config.ppo.rollout_steps,
            env.num_envs,
            self.observation,
            config.network.recurrent_layers,
            config.network.recurrent_hidden_size,
            config.ppo.gamma,
            config.ppo.gae_lambda,
        )
        self.timesteps = 0
        self.updates = 0
        self.completed_episodes = 0
        self.recent_outcomes = {"success": 0, "collision": 0, "timeout": 0}
        self.episode_returns = np.zeros(env.num_envs, dtype=np.float64)
        self.episode_lengths = np.zeros(env.num_envs, dtype=np.int64)
        self.recent_episode_returns: deque[float] = deque(maxlen=100)
        self.recent_episode_lengths: deque[int] = deque(maxlen=100)
        self.recent_episode_outcomes: deque[str] = deque(maxlen=100)
        self.progress_path = (
            Path(config.log_dir) / "progress.csv" if config.log_dir is not None else None
        )
        self.tensorboard_writer: SummaryWriter | None = None

    def _progress_values(
        self, metrics: dict[str, float], memory_mb: float
    ) -> dict[str, int | float]:
        return {
            "update": self.updates,
            "timesteps": self.timesteps,
            "policy_loss": metrics.get("policy_loss", float("nan")),
            "value_loss": metrics.get("value_loss", float("nan")),
            "entropy": metrics.get("entropy", float("nan")),
            "approx_kl": metrics.get("approx_kl", float("nan")),
            "clip_fraction": metrics.get("clip_fraction", float("nan")),
            "episodes": self.completed_episodes,
            "success": self.recent_outcomes["success"],
            "collision": self.recent_outcomes["collision"],
            "timeout": self.recent_outcomes["timeout"],
            "curriculum_stage": getattr(self.env, "training_curriculum_stage", 0),
            "active_mazes": getattr(self.env, "training_curriculum_maze_count", 6),
            "active_cylinders": getattr(
                self.env, "training_curriculum_cylinder_count", 0
            ),
            "contact_penalty_scale": getattr(
                self.env, "training_curriculum_contact_scale", 1.0
            ),
            "cpu_buffer_mib": memory_mb,
        }

    def _write_progress(self, metrics: dict[str, float], memory_mb: float) -> None:
        """把与终端输出相同的指标追加到简单、可移植的 CSV 日志。"""

        if self.progress_path is None:
            return
        row: dict[str, str | int | float] = {
            "recurrent_type": self.config.network.recurrent_type
        }
        row.update(self._progress_values(metrics, memory_mb))
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.progress_path.exists()
        with self.progress_path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _recent_outcome_rates(self) -> dict[str, float]:
        outcome_count = len(self.recent_episode_outcomes)
        if outcome_count == 0:
            return {}
        return {
            outcome: sum(item == outcome for item in self.recent_episode_outcomes)
            / outcome_count
            for outcome in ("collision", "success", "timeout")
        }

    def _write_tensorboard(
        self, metrics: dict[str, float], memory_mb: float, fps: float
    ) -> None:
        writer = self.tensorboard_writer
        if writer is None:
            return
        step = self.timesteps

        # progress/* 与 progress.csv 的数值列一一对应。
        for name, value in self._progress_values(metrics, memory_mb).items():
            writer.add_scalar(f"progress/{name}", value, step)
        writer.add_text("progress/recurrent_type", self.config.network.recurrent_type, step)

        # 保留旧 SB3 运行的 tag，方便在同一 TensorBoard 中对比。
        legacy_metrics = {
            "time/fps": fps,
            "train/approx_kl": metrics.get("approx_kl", float("nan")),
            "train/clip_fraction": metrics.get("clip_fraction", float("nan")),
            "train/clip_range": self.config.ppo.clip_range,
            "train/clip_range_vf": (
                self.config.ppo.value_clip_range
                if self.config.ppo.value_clip_range is not None
                else 0.0
            ),
            "train/entropy_loss": -metrics.get("entropy", float("nan")),
            "train/explained_variance": metrics.get(
                "explained_variance", float("nan")
            ),
            "train/learning_rate": metrics.get("learning_rate", float("nan")),
            "train/loss": metrics.get("loss", float("nan")),
            "train/policy_gradient_loss": metrics.get(
                "policy_loss", float("nan")
            ),
            "train/std": metrics.get("std", float("nan")),
            "train/value_loss": metrics.get("value_loss", float("nan")),
        }
        for tag, value in legacy_metrics.items():
            writer.add_scalar(tag, value, step)

        if self.recent_episode_returns:
            writer.add_scalar(
                "rollout/ep_rew_mean", np.mean(self.recent_episode_returns), step
            )
            writer.add_scalar(
                "rollout/ep_len_mean", np.mean(self.recent_episode_lengths), step
            )
        outcome_rates = self._recent_outcome_rates()
        if outcome_rates:
            for outcome, tag in (
                ("collision", "Metrics/Collision_Rate"),
                ("success", "Metrics/Success_Rate"),
                ("timeout", "Metrics/Timeout_Rate"),
            ):
                writer.add_scalar(tag, outcome_rates[outcome], step)
        writer.flush()

    @torch.no_grad()
    def _terminal_value(
        self, terminal_observation: dict[str, np.ndarray], env_id: int, next_state: RecurrentState
    ) -> float:
        observation = {
            key: torch.as_tensor(value[None], dtype=torch.float32, device=self.device)
            for key, value in terminal_observation.items()
        }
        critic_state = (
            next_state.critic[0][:, env_id : env_id + 1].contiguous(),
            next_state.critic[1][:, env_id : env_id + 1].contiguous(),
        )
        value = self.policy.predict_value(
            observation,
            critic_state,
            torch.zeros(1, dtype=torch.bool, device=self.device),
        )
        return float(value.item())

    def collect_rollout(self) -> None:
        """采满一轮数据；buffer 始终是 NumPy/CPU。"""

        self.policy.eval()
        self.buffer.reset()
        final_dones = np.zeros(self.env.num_envs, dtype=bool)
        for _ in range(self.config.ppo.rollout_steps):
            state_before_action = self.recurrent_state
            with torch.no_grad():
                action, value, log_probability, next_state = self.policy.act(
                    _observation_tensor(self.observation, self.device),
                    state_before_action,
                    torch.as_tensor(self.episode_starts, device=self.device),
                )
            action_array = action.cpu().numpy()
            (
                next_observation,
                reward,
                terminated,
                truncated,
                infos,
            ) = self.env.step(action_array)
            done = terminated | truncated
            self.episode_returns += reward
            self.episode_lengths += 1
            adjusted_reward = reward.copy()
            for env_id in np.flatnonzero(truncated):
                adjusted_reward[env_id] += self.config.ppo.gamma * self._terminal_value(
                    infos[env_id]["terminal_observation"], env_id, next_state
                )
            for env_id in np.flatnonzero(done):
                self.completed_episodes += 1
                if infos[env_id].get("success", False):
                    outcome = "success"
                elif infos[env_id].get("collided", False):
                    outcome = "collision"
                else:
                    outcome = "timeout"
                self.recent_outcomes[outcome] += 1
                self.recent_episode_returns.append(float(self.episode_returns[env_id]))
                self.recent_episode_lengths.append(int(self.episode_lengths[env_id]))
                self.recent_episode_outcomes.append(outcome)
                self.episode_returns[env_id] = 0.0
                self.episode_lengths[env_id] = 0

            self.buffer.add(
                self.observation,
                action_array,
                adjusted_reward,
                self.episode_starts,
                value.cpu().numpy(),
                log_probability.cpu().numpy(),
                state_before_action,
            )
            self.observation = next_observation
            self.episode_starts = done
            self.recurrent_state = next_state.detach()
            final_dones = done
            self.timesteps += self.env.num_envs

        with torch.no_grad():
            last_values = self.policy.predict_value(
                _observation_tensor(self.observation, self.device),
                self.recurrent_state.critic,
                torch.as_tensor(self.episode_starts, device=self.device),
            ).cpu().numpy()
        self.buffer.compute_returns_and_advantages(last_values, final_dones)

    def update(self) -> dict[str, float]:
        """对刚采集的 rollout 执行多轮 PPO 更新。"""

        self.policy.train()
        metrics: dict[str, list[float]] = {
            "loss": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
        }
        stop_early = False
        for _ in range(self.config.ppo.epochs):
            for batch in self.buffer.iterate_batches(
                self.config.ppo.batch_size,
                self.config.ppo.recurrent_sequence_length,
                self.device,
                self.rng,
            ):
                mask = batch.mask
                values, log_probability, entropy = self.policy.evaluate_sequences(
                    batch.observations,
                    batch.actions,
                    batch.recurrent_state,
                    batch.episode_starts,
                )
                advantages = batch.advantages
                if self.config.ppo.normalize_advantage:
                    valid_advantage = advantages[mask]
                    advantages = (advantages - valid_advantage.mean()) / (
                        valid_advantage.std(unbiased=False) + 1.0e-8
                    )

                ratio = torch.exp(log_probability - batch.old_log_probabilities)
                unclipped = advantages * ratio
                clipped = advantages * ratio.clamp(
                    1.0 - self.config.ppo.clip_range,
                    1.0 + self.config.ppo.clip_range,
                )
                policy_loss = -torch.minimum(unclipped, clipped)[mask].mean()

                if self.config.ppo.value_clip_range is None:
                    predicted_values = values
                else:
                    predicted_values = batch.old_values + (
                        values - batch.old_values
                    ).clamp(
                        -self.config.ppo.value_clip_range,
                        self.config.ppo.value_clip_range,
                    )
                value_loss = ((batch.returns - predicted_values)[mask] ** 2).mean()
                mean_entropy = entropy[mask].mean()
                loss = (
                    policy_loss
                    + self.config.ppo.value_coefficient * value_loss
                    - self.config.ppo.entropy_coefficient * mean_entropy
                )

                with torch.no_grad():
                    log_ratio = log_probability - batch.old_log_probabilities
                    approximate_kl = (
                        (torch.exp(log_ratio) - 1.0 - log_ratio)[mask].mean()
                    )
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.config.ppo.clip_range)[mask]
                        .float()
                        .mean()
                    )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.ppo.max_grad_norm
                )
                self.optimizer.step()

                metrics["loss"].append(float(loss.detach()))
                metrics["policy_loss"].append(float(policy_loss.detach()))
                metrics["value_loss"].append(float(value_loss.detach()))
                metrics["entropy"].append(float(mean_entropy.detach()))
                metrics["approx_kl"].append(float(approximate_kl))
                metrics["clip_fraction"].append(float(clip_fraction))
                target_kl = self.config.ppo.target_kl
                if target_kl is not None and approximate_kl > 1.5 * target_kl:
                    stop_early = True
                    break
            if stop_early:
                break
        self.updates += 1
        summary = {
            key: float(np.mean(values)) for key, values in metrics.items() if values
        }
        return_variance = float(np.var(self.buffer.returns))
        summary["explained_variance"] = (
            float("nan")
            if return_variance == 0.0
            else 1.0
            - float(np.var(self.buffer.returns - self.buffer.values)) / return_variance
        )
        summary["learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
        summary["std"] = float(self.policy.log_std.detach().exp().mean())
        return summary

    def learn(self, total_timesteps: int | None = None) -> None:
        target = total_timesteps or self.config.total_timesteps
        started_at = time.perf_counter()
        starting_timesteps = self.timesteps
        if self.config.log_dir is not None:
            self.tensorboard_writer = SummaryWriter(log_dir=self.config.log_dir)
        try:
            while self.timesteps < target:
                self.collect_rollout()
                metrics = self.update()
                if self.updates % self.config.log_interval == 0 or self.updates == 1:
                    memory_mb = self.buffer.memory_bytes / 1024**2
                    elapsed = max(time.perf_counter() - started_at, 1.0e-9)
                    fps = (self.timesteps - starting_timesteps) / elapsed
                    self._write_progress(metrics, memory_mb)
                    self._write_tensorboard(metrics, memory_mb, fps)
                    outcome_rates = self._recent_outcome_rates()
                    success_rate = (
                        f"{outcome_rates['success']:.2%}" if outcome_rates else "N/A"
                    )
                    print(
                        f"recurrent={self.config.network.recurrent_type} "
                        f"update={self.updates} steps={self.timesteps} "
                        f"policy_loss={metrics.get('policy_loss', float('nan')):.4f} "
                        f"value_loss={metrics.get('value_loss', float('nan')):.4f} "
                        f"episodes={self.completed_episodes} outcomes={self.recent_outcomes} "
                        f"success_rate={success_rate} "
                        f"curriculum={getattr(self.env, 'training_curriculum_stage', 0)} "
                        f"mazes={getattr(self.env, 'training_curriculum_maze_count', 6)} "
                        f"cylinders={getattr(self.env, 'training_curriculum_cylinder_count', 0)} "
                        f"cpu_buffer={memory_mb:.1f}MiB"
                    )
                    self.recent_outcomes = {
                        "success": 0,
                        "collision": 0,
                        "timeout": 0,
                    }
                if self.updates % self.config.checkpoint_interval == 0:
                    checkpoint = (
                        Path(self.config.checkpoint_dir) / f"model_{self.timesteps}.pt"
                    )
                    self.save(checkpoint)
        finally:
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.close()
                self.tensorboard_writer = None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "timesteps": self.timesteps,
                "updates": self.updates,
                "config": asdict(self.config),
            },
            path,
        )

    def load(self, path: str | Path, *, load_optimizer: bool = True) -> None:
        checkpoint: dict[str, Any] = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.timesteps = int(checkpoint.get("timesteps", 0))
        self.updates = int(checkpoint.get("updates", 0))
        if hasattr(self.env, "set_training_progress"):
            self.env.set_training_progress(self.timesteps)
            # 构造训练器时环境已经按第 0 阶段 reset；恢复 checkpoint 后立即
            # 再 reset 一次，使当前墙体和圆柱数量与恢复的训练步数一致。
            self.observation, _ = self.env.reset()
            self.episode_starts = np.ones(self.env.num_envs, dtype=bool)
            self.recurrent_state = self.policy.initial_state(self.env.num_envs)
