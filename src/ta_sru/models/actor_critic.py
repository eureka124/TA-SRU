"""非对称循环 Actor–Critic。

数据边界在这个文件中非常明确：Actor 只能访问 ``camera`` 和 ``robot_state``；
Critic 还能访问训练时特权信息 ``critic_toa``。这比依靠特征提取器继承关系隐式
区分输入更容易审计。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal

from ta_sru.config import NetworkConfig
from ta_sru.models.recurrent import RecurrentStateTuple, build_recurrent

Observation = Mapping[str, torch.Tensor]

# 动作参数化方式的标识，写入 checkpoint 并在加载时校验。旧 checkpoint 的动作是
# 无界高斯采样经环境 clamp，权重含义与 tanh 版本不同，不能直接复用。
ACTION_TRANSFORM = "tanh"


def require_supported_action_transform(value: str | None) -> None:
    """拒绝加载动作参数化方式不一致的 checkpoint，避免静默产生错误动作。"""

    if value == ACTION_TRANSFORM:
        return
    if value is None:
        raise ValueError(
            "checkpoint 没有动作参数化标识，属于 clamp 版本；其 action_mean 输出的是"
            "物理量纲的无界均值，与当前 tanh 版本含义不同，不能直接加载。请使用"
            "重新训练得到的 checkpoint。"
        )
    raise ValueError(
        f"checkpoint 的动作参数化为 {value!r}，与当前 {ACTION_TRANSFORM!r} 不一致"
    )


@dataclass
class RecurrentState:
    actor: RecurrentStateTuple
    critic: RecurrentStateTuple

    def detach(self) -> RecurrentState:
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
        return self.depth(observation["camera"]) + self.robot_state(
            observation["robot_state"].float()
        )


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
    """Actor/Critic 各自拥有编码器、可切换循环单元和 MLP。

    Actor 输出的是 **归一化动作**：对角高斯采样经 tanh 压缩到 [-1, 1]，环境再线性
    映射到 ``action_low``/``action_high``。缓冲区里存的就是这个 [-1, 1] 的动作，
    因此 ``evaluate_sequences`` 能通过 atanh 精确还原采样值，训练和采样看到的是
    同一个分布。
    """

    action_size = 3
    robot_state_size = 8
    # tanh 压缩的数值余量，保证 atanh 与 log(1 - a²) 在边界处仍然有限。
    squash_epsilon = 1.0e-6

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        if config.share_depth_encoder:
            raise NotImplementedError(
                "share_depth_encoder 是未来实验入口；当前版本按要求不实现特征共享"
            )
        config.validate()
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
        latent_size = (
            config.feature_dim
            if config.recurrent_type == "none"
            else config.recurrent_hidden_size
        )
        self.actor_mlp, actor_output_size = _mlp(latent_size, config.actor_hidden_sizes)
        self.critic_mlp, critic_output_size = _mlp(
            latent_size, config.critic_hidden_sizes
        )
        self.action_mean = nn.Linear(actor_output_size, self.action_size)
        self.value_head = nn.Linear(critic_output_size, 1)
        self.log_std = nn.Parameter(
            torch.full((self.action_size,), config.initial_log_std)
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """使用 PPO 常见的正交初始化；循环单元保留自身初始化。"""

        recurrent_modules = {id(module) for module in self.actor_recurrent.modules()}
        recurrent_modules.update(
            id(module) for module in self.critic_recurrent.modules()
        )
        for module in self.modules():
            if (
                isinstance(module, (nn.Linear, nn.Conv2d))
                and id(module) not in recurrent_modules
            ):
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

    def bounded_log_std(self) -> torch.Tensor:
        """返回裁剪到 [log_std_min, log_std_max] 的可训练对数标准差。

        熵奖励对 log_std 是单向推力，没有上限时动作噪声会一直增长；裁剪也让日志里
        报告的 `train/std` 与实际生效的噪声一致。
        """

        return self.log_std.clamp(self.config.log_std_min, self.config.log_std_max)

    def _distribution(self, latent: torch.Tensor) -> Normal:
        mean = self.action_mean(self.actor_mlp(latent))
        return Normal(mean, self.bounded_log_std().exp().expand_as(mean))

    def _squash(self, raw_action: torch.Tensor) -> torch.Tensor:
        """把无界高斯采样压缩到 [-1, 1]，两端留出数值余量。

        环境负责把 [-1, 1] 线性映射到 action_low/action_high。这里用 tanh 而不是
        直接 clamp：clamp 在边界外梯度恒为 1，策略把均值推出去以后仍会收到等效的
        更新信号，动作噪声只会单向增大；tanh 的梯度随 |raw_action| 增大而衰减，
        饱和区自然停止接受噪声。
        """

        limit = 1.0 - self.squash_epsilon
        return torch.tanh(raw_action).clamp(-limit, limit)

    def _squashed_log_probability(
        self, distribution: Normal, squashed_action: torch.Tensor
    ) -> torch.Tensor:
        """对压缩后的动作求对数概率，并补上 tanh 的 Jacobian 修正。

        采样和训练都必须走同一个入口，否则 log_probability 与实际执行的动作不一致。
        """

        raw_action = torch.atanh(squashed_action)
        correction = torch.log1p(-squashed_action.square())
        return (distribution.log_prob(raw_action) - correction).sum(dim=-1)

    @torch.no_grad()
    def act_actor(
        self,
        observation: Observation,
        state: RecurrentState,
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, RecurrentState]:
        """确定性评估仅计算 Actor，不读取 Critic 的特权观测。"""
        sequence = self._with_time_dimension(observation)
        features = self.actor_encoder(sequence)
        output, actor_state = self.actor_recurrent(
            features, state.actor, episode_starts.unsqueeze(0)
        )
        action = self.action_mean(self.actor_mlp(output[0]))
        return self._squash(action), RecurrentState(actor_state, state.critic)

    @torch.no_grad()
    def act(
        self,
        observation: Observation,
        state: RecurrentState,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RecurrentState]:
        """处理一个向量环境时间步。返回的 action 已压缩到 [-1, 1]。"""

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
        raw_action = distribution.mean if deterministic else distribution.sample()
        action = self._squash(raw_action)
        log_probability = self._squashed_log_probability(distribution, action)
        value = self.value_head(self.critic_mlp(critic_output[0])).squeeze(-1)
        return action, value, log_probability, RecurrentState(actor_state, critic_state)

    def evaluate_sequences(
        self,
        observation: Observation,
        action: torch.Tensor,
        state: RecurrentState,
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """评估 ``[时间, 序列, ...]`` 形式的训练批次。

        ``action`` 是采样时存下来的、已经 tanh 压缩到 [-1, 1] 的动作；这里通过
        atanh 还原成高斯采样值再求对数概率。
        """

        actor_features = self.actor_encoder(observation)
        critic_features = self.critic_encoder(observation)
        actor_output, _ = self.actor_recurrent(
            actor_features, state.actor, episode_starts
        )
        critic_output, _ = self.critic_recurrent(
            critic_features, state.critic, episode_starts
        )
        distribution = self._distribution(actor_output)
        log_probability = self._squashed_log_probability(distribution, action)
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
