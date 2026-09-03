"""只驻留在 CPU 内存的循环 PPO rollout buffer。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch

from ta_sru.models.actor_critic import RecurrentState


@dataclass
class SequenceBatch:
    observations: dict[str, torch.Tensor]
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_probabilities: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    episode_starts: torch.Tensor
    mask: torch.Tensor
    recurrent_state: RecurrentState


@dataclass(frozen=True)
class _Sequence:
    env_id: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


class CpuRolloutBuffer:
    """以 NumPy 数组保存采样数据，仅在取 minibatch 时拷贝到训练设备。"""

    storage_device = "cpu"

    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        observation_example: dict[str, np.ndarray],
        recurrent_layers: int,
        recurrent_hidden_size: int,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.observations = {
            key: np.empty(
                (rollout_steps, num_envs, *value.shape[1:]), dtype=np.float32
            )
            for key, value in observation_example.items()
        }
        self.actions = np.empty((rollout_steps, num_envs, 3), dtype=np.float32)
        scalar_shape = (rollout_steps, num_envs)
        self.rewards = np.empty(scalar_shape, dtype=np.float32)
        self.episode_starts = np.empty(scalar_shape, dtype=bool)
        self.values = np.empty(scalar_shape, dtype=np.float32)
        self.log_probabilities = np.empty(scalar_shape, dtype=np.float32)
        self.advantages = np.empty(scalar_shape, dtype=np.float32)
        self.returns = np.empty(scalar_shape, dtype=np.float32)
        state_shape = (
            rollout_steps,
            recurrent_layers,
            num_envs,
            recurrent_hidden_size,
        )
        self.actor_hidden = np.empty(state_shape, dtype=np.float32)
        self.actor_cell = np.empty(state_shape, dtype=np.float32)
        self.critic_hidden = np.empty(state_shape, dtype=np.float32)
        self.critic_cell = np.empty(state_shape, dtype=np.float32)
        self.position = 0

    def reset(self) -> None:
        self.position = 0

    @property
    def full(self) -> bool:
        return self.position == self.rollout_steps

    def add(
        self,
        observation: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: np.ndarray,
        log_probability: np.ndarray,
        recurrent_state: RecurrentState,
    ) -> None:
        if self.full:
            raise RuntimeError("rollout buffer 已满")
        index = self.position
        for key, array in observation.items():
            self.observations[key][index] = np.asarray(array, dtype=np.float32)
        self.actions[index] = action
        self.rewards[index] = reward
        self.episode_starts[index] = episode_start
        self.values[index] = value
        self.log_probabilities[index] = log_probability
        actor_hidden, actor_cell = recurrent_state.actor
        critic_hidden, critic_cell = recurrent_state.critic
        self.actor_hidden[index] = actor_hidden.detach().cpu().numpy()
        self.actor_cell[index] = actor_cell.detach().cpu().numpy()
        self.critic_hidden[index] = critic_hidden.detach().cpu().numpy()
        self.critic_cell[index] = critic_cell.detach().cpu().numpy()
        self.position += 1

    def compute_returns_and_advantages(
        self, last_values: np.ndarray, last_dones: np.ndarray
    ) -> None:
        if not self.full:
            raise RuntimeError("必须先填满 rollout buffer")
        last_advantage = np.zeros(self.num_envs, dtype=np.float32)
        for step in reversed(range(self.rollout_steps)):
            if step == self.rollout_steps - 1:
                next_non_terminal = 1.0 - last_dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1].astype(np.float32)
                next_values = self.values[step + 1]
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_advantage = (
                delta
                + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            )
            self.advantages[step] = last_advantage
        self.returns[:] = self.advantages + self.values

    def _sequences(self, maximum_length: int) -> list[_Sequence]:
        sequences: list[_Sequence] = []
        for env_id in range(self.num_envs):
            boundaries = [0]
            boundaries.extend(
                int(index)
                for index in np.flatnonzero(self.episode_starts[1:, env_id]) + 1
            )
            boundaries.append(self.rollout_steps)
            for episode_start, episode_end in zip(boundaries[:-1], boundaries[1:]):
                for start in range(episode_start, episode_end, maximum_length):
                    sequences.append(
                        _Sequence(env_id, start, min(start + maximum_length, episode_end))
                    )
        return sequences

    def iterate_batches(
        self,
        batch_size: int,
        sequence_length: int,
        device: torch.device | str,
        rng: np.random.Generator,
    ) -> Iterator[SequenceBatch]:
        """按完整短序列分组、补零，并把当前批次传到训练设备。"""

        if not self.full:
            raise RuntimeError("必须先填满 rollout buffer")
        sequences = self._sequences(sequence_length)
        rng.shuffle(sequences)
        groups: list[list[_Sequence]] = []
        current: list[_Sequence] = []
        current_size = 0
        for sequence in sequences:
            if current and current_size + sequence.length > batch_size:
                groups.append(current)
                current = []
                current_size = 0
            current.append(sequence)
            current_size += sequence.length
        if current:
            groups.append(current)

        for group in groups:
            yield self._make_batch(group, device)

    def _make_batch(
        self, sequences: list[_Sequence], device: torch.device | str
    ) -> SequenceBatch:
        time_size = max(sequence.length for sequence in sequences)
        batch_size = len(sequences)

        def padded(source: np.ndarray) -> np.ndarray:
            output = np.zeros(
                (time_size, batch_size, *source.shape[2:]), dtype=source.dtype
            )
            for sequence_id, sequence in enumerate(sequences):
                output[: sequence.length, sequence_id] = source[
                    sequence.start : sequence.end, sequence.env_id
                ]
            return output

        observations = {
            key: torch.as_tensor(padded(value), device=device)
            for key, value in self.observations.items()
        }
        mask = np.zeros((time_size, batch_size), dtype=bool)
        for sequence_id, sequence in enumerate(sequences):
            mask[: sequence.length, sequence_id] = True

        def initial_state(source: np.ndarray) -> torch.Tensor:
            array = np.stack(
                [source[item.start, :, item.env_id] for item in sequences], axis=1
            )
            return torch.as_tensor(array, device=device)

        state = RecurrentState(
            actor=(initial_state(self.actor_hidden), initial_state(self.actor_cell)),
            critic=(initial_state(self.critic_hidden), initial_state(self.critic_cell)),
        )
        return SequenceBatch(
            observations=observations,
            actions=torch.as_tensor(padded(self.actions), device=device),
            old_values=torch.as_tensor(padded(self.values), device=device),
            old_log_probabilities=torch.as_tensor(
                padded(self.log_probabilities), device=device
            ),
            advantages=torch.as_tensor(padded(self.advantages), device=device),
            returns=torch.as_tensor(padded(self.returns), device=device),
            episode_starts=torch.as_tensor(padded(self.episode_starts), device=device),
            mask=torch.as_tensor(mask, device=device),
            recurrent_state=state,
        )

    @property
    def memory_bytes(self) -> int:
        arrays = [
            *self.observations.values(),
            self.actions,
            self.rewards,
            self.episode_starts,
            self.values,
            self.log_probabilities,
            self.advantages,
            self.returns,
            self.actor_hidden,
            self.actor_cell,
            self.critic_hidden,
            self.critic_cell,
        ]
        return sum(array.nbytes for array in arrays)

