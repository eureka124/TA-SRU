"""非对称循环 Actor–Critic。

数据边界在这个文件中非常明确：Actor 只能访问 ``camera`` 和 ``robot_state``；
Critic 还能访问训练时特权信息 ``critic_toa``。这比依靠特征提取器继承关系隐式
区分输入更容易审计。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.distributions import Normal

from ta_sru.config import NetworkConfig
from ta_sru.models.recurrent import RecurrentStateTuple, build_recurrent

Observation = Mapping[str, torch.Tensor]


@dataclass
class RecurrentState:
    actor: RecurrentStateTuple
    critic: RecurrentStateTuple

    def detach(self) -> "RecurrentState":
        return RecurrentState(
            actor=tuple(tensor.detach() for tensor in self.actor),  # type: ignore[arg-type]
            critic=tuple(tensor.detach() for tensor in self.critic),  # type: ignore[arg-type]
        )


def _mlp(input_size: int, hidden_sizes: tuple[int, ...]) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    previous_size = input_size
    for hidden_size in hidden_sizes:
        layers.extend((nn.Linear(previous_size, hidden_size), nn.LeakyReLU()))
        previous_size = hidden_size
    return nn.Sequential(*layers), previous_size


class ImageEncoder(nn.Module):
    """小型 CNN；自适应池化让输入分辨率可调整。"""

    def __init__(self, channels: tuple[int, ...], output_size: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_channels = 1
        kernels = (2,) + (3,) * (len(channels) - 1)
        for output_channels, kernel_size in zip(channels, kernels):
            layers.extend(
                (
                    nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size),
                    nn.LeakyReLU(),
                )
            )
            input_channels = output_channels
        layers.extend((nn.AdaptiveAvgPool2d((2, 3)), nn.Flatten()))
        self.convolution = nn.Sequential(*layers)
        self.projection = nn.Linear(input_channels * 2 * 3, output_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        prefix = image.shape[:-3]
        flat_image = image.reshape(-1, *image.shape[-3:]).float()
        encoded = self.projection(self.convolution(flat_image))
        return encoded.reshape(*prefix, -1)


class ActorEncoder(nn.Module):
    """只编码部署时可获得的观测。"""

    def __init__(self, robot_state_size: int, feature_size: int) -> None:
        super().__init__()
        self.depth = ImageEncoder((8, 16, 32), feature_size)
        self.robot_state = nn.Linear(robot_state_size, feature_size)

    def forward(self, observation: Observation) -> torch.Tensor:
        return self.depth(observation["camera"]) + self.robot_state(observation["robot_state"].float())


class CriticEncoder(nn.Module):
    """编码普通观测和仅训练时使用的 TOA 地图。"""

    def __init__(self, robot_state_size: int, feature_size: int) -> None:
        super().__init__()
        self.depth = ImageEncoder((8, 16, 32), feature_size)
        self.robot_state = nn.Linear(robot_state_size, feature_size)
        self.toa = ImageEncoder((8, 16, 32), feature_size)

    def forward(self, observation: Observation) -> torch.Tensor:
        return (
            self.depth(observation["camera"])
            + self.robot_state(observation["robot_state"].float())
            + self.toa(observation["critic_toa"])
        )


class AsymmetricRecurrentActorCritic(nn.Module):
    """Actor/Critic 各自拥有编码器、可切换循环单元和 MLP。"""

    action_size = 3
    robot_state_size = 8

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        if config.share_depth_encoder:
            raise NotImplementedError(
                "share_depth_encoder 是未来实验入口；当前版本按要求不实现特征共享"
            )
        self.config = config
        self.actor_encoder = ActorEncoder(self.robot_state_size, config.feature_dim)
        self.critic_encoder = CriticEncoder(self.robot_state_size, config.feature_dim)
        self.actor_recurrent = build_recurrent(
            config.recurrent_type,
            config.feature_dim,
            config.recurrent_hidden_size,
            config.recurrent_layers,
        )
        self.critic_recurrent = build_recurrent(
            config.recurrent_type,
            config.feature_dim,
            config.recurrent_hidden_size,
            config.recurrent_layers,
        )
        self.actor_mlp, actor_output_size = _mlp(
            config.recurrent_hidden_size, config.actor_hidden_sizes
        )
        self.critic_mlp, critic_output_size = _mlp(
            config.recurrent_hidden_size, config.critic_hidden_sizes
        )
        self.action_mean = nn.Linear(actor_output_size, self.action_size)
        self.value_head = nn.Linear(critic_output_size, 1)
        self.log_std = nn.Parameter(torch.full((self.action_size,), config.initial_log_std))
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """使用 PPO 常见的正交初始化；循环单元保留自身初始化。"""

        recurrent_modules = {id(module) for module in self.actor_recurrent.modules()}
        recurrent_modules.update(id(module) for module in self.critic_recurrent.modules())
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and id(module) not in recurrent_modules:
                nn.init.orthogonal_(module.weight, gain=2.0**0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    @property
    def device(self) -> torch.device:
        return self.log_std.device

    def initial_state(self, batch_size: int) -> RecurrentState:
        return RecurrentState(
            actor=self.actor_recurrent.initial_state(batch_size, device=self.device),
            critic=self.critic_recurrent.initial_state(batch_size, device=self.device),
        )

    @staticmethod
    def _with_time_dimension(observation: Observation) -> dict[str, torch.Tensor]:
        return {key: value.unsqueeze(0) for key, value in observation.items()}

    def _distribution(self, latent: torch.Tensor) -> Normal:
        mean = self.action_mean(self.actor_mlp(latent))
        return Normal(mean, self.log_std.exp().expand_as(mean))

    @torch.no_grad()
    def act(
        self,
        observation: Observation,
        state: RecurrentState,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RecurrentState]:
        """处理一个向量环境时间步。"""

        sequence = self._with_time_dimension(observation)
        starts = episode_starts.unsqueeze(0)
        actor_features = self.actor_encoder(sequence)
        critic_features = self.critic_encoder(sequence)
        actor_output, actor_state = self.actor_recurrent(
            actor_features, state.actor, starts
        )
        critic_output, critic_state = self.critic_recurrent(
            critic_features, state.critic, starts
        )
        distribution = self._distribution(actor_output[0])
        action = distribution.mean if deterministic else distribution.sample()
        log_probability = distribution.log_prob(action).sum(dim=-1)
        value = self.value_head(self.critic_mlp(critic_output[0])).squeeze(-1)
        return action, value, log_probability, RecurrentState(actor_state, critic_state)

    def evaluate_sequences(
        self,
        observation: Observation,
        action: torch.Tensor,
        state: RecurrentState,
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """评估 ``[时间, 序列, ...]`` 形式的训练批次。"""

        actor_features = self.actor_encoder(observation)
        critic_features = self.critic_encoder(observation)
        actor_output, _ = self.actor_recurrent(actor_features, state.actor, episode_starts)
        critic_output, _ = self.critic_recurrent(critic_features, state.critic, episode_starts)
        distribution = self._distribution(actor_output)
        log_probability = distribution.log_prob(action).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        value = self.value_head(self.critic_mlp(critic_output)).squeeze(-1)
        return value, log_probability, entropy

    @torch.no_grad()
    def predict_value(
        self,
        observation: Observation,
        critic_state: RecurrentStateTuple,
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        sequence = self._with_time_dimension(observation)
        features = self.critic_encoder(sequence)
        output, _ = self.critic_recurrent(
            features, critic_state, episode_starts.unsqueeze(0)
        )
        return self.value_head(self.critic_mlp(output[0])).squeeze(-1)
