"""面向本项目环境接口的简洁 Recurrent PPO 实现。"""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

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
        self.progress_path = (
            Path(config.log_dir) / "progress.csv" if config.log_dir is not None else None
        )

    def _write_progress(self, metrics: dict[str, float], memory_mb: float) -> None:
        """把与终端输出相同的指标追加到简单、可移植的 CSV 日志。"""

        if self.progress_path is None:
            return
        row: dict[str, str | int | float] = {
            "recurrent_type": self.config.network.recurrent_type,
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
            "cpu_buffer_mib": memory_mb,
        }
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.progress_path.exists()
        with self.progress_path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

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
            adjusted_reward = reward.copy()
            for env_id in np.flatnonzero(truncated):
                adjusted_reward[env_id] += self.config.ppo.gamma * self._terminal_value(
                    infos[env_id]["terminal_observation"], env_id, next_state
                )
            for env_id in np.flatnonzero(done):
                self.completed_episodes += 1
                if infos[env_id].get("success", False):
                    self.recent_outcomes["success"] += 1
                elif infos[env_id].get("collided", False):
                    self.recent_outcomes["collision"] += 1
                else:
                    self.recent_outcomes["timeout"] += 1

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
        return {key: float(np.mean(values)) for key, values in metrics.items() if values}

    def learn(self, total_timesteps: int | None = None) -> None:
        target = total_timesteps or self.config.total_timesteps
        while self.timesteps < target:
            self.collect_rollout()
            metrics = self.update()
            if self.updates % self.config.log_interval == 0 or self.updates == 1:
                memory_mb = self.buffer.memory_bytes / 1024**2
                self._write_progress(metrics, memory_mb)
                print(
                    f"recurrent={self.config.network.recurrent_type} "
                    f"update={self.updates} steps={self.timesteps} "
                    f"policy_loss={metrics.get('policy_loss', float('nan')):.4f} "
                    f"value_loss={metrics.get('value_loss', float('nan')):.4f} "
                    f"episodes={self.completed_episodes} outcomes={self.recent_outcomes} "
                    f"cpu_buffer={memory_mb:.1f}MiB"
                )
                self.recent_outcomes = {"success": 0, "collision": 0, "timeout": 0}
            if self.updates % self.config.checkpoint_interval == 0:
                checkpoint = Path(self.config.checkpoint_dir) / f"model_{self.timesteps}.pt"
                self.save(checkpoint)

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
