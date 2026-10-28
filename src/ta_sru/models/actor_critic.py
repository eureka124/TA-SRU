"""非对称循环 Actor–Critic。

数据边界在这个文件中非常明确：Actor 只能访问深度图特征和 ``robot_state``；
Critic 还能访问训练时特权信息 ``critic_toa``。这比依靠特征提取器继承关系隐式
区分输入更容易审计。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal
from torch.utils.checkpoint import checkpoint

from ta_sru.config import NetworkConfig
from ta_sru.models.recurrent import RecurrentStateTuple, build_recurrent

Observation = Mapping[str, torch.Tensor]

# 动作参数化方式的标识，写入 checkpoint 并在加载时校验。
# 本分支的策略直接输出物理量纲的无界高斯采样，边界由环境的 clamp 施加；main 分支
# 输出的是经 tanh 压缩的归一化动作。两者的 action_mean 权重含义不同，不能互相加载。
ACTION_TRANSFORM = "clamp"


def require_supported_action_transform(value: str | None) -> None:
    """拒绝加载动作参数化方式不一致的 checkpoint，避免静默产生错误动作。"""

    if value == ACTION_TRANSFORM or value is None:
        # 缺少标识的 checkpoint 来自引入该标识之前，属于 clamp 版本，可以加载。
        return
    if value == "tanh":
        raise ValueError(
            "checkpoint 使用 tanh 压缩动作（main 分支）：其 action_mean 输出的是"
            "归一化动作，环境再线性映射到 action_low/action_high。本分支直接输出"
            "物理量纲的动作，两者权重含义不同，不能加载。请改用 main 分支评估，"
            "或在当前分支重新训练。"
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


class DeterministicAdaptiveAvgPool2d(nn.Module):
    """按自适应池化的分区求均值，支持 CUDA 严格确定性反向传播。"""

    def __init__(self, output_size: tuple[int, int]) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        rows, columns = self.output_size
        pooled = []
        for row in range(rows):
            # 起点向下取整、终点向上取整；不能整除时相邻分区可以重叠。
            top = row * height // rows
            bottom = ((row + 1) * height + rows - 1) // rows
            for column in range(columns):
                left = column * width // columns
                right = ((column + 1) * width + columns - 1) // columns
                pooled.append(image[..., top:bottom, left:right].mean(dim=(-2, -1)))
        # 切片与均值的反向传播避免原生 CUDA 自适应池化的非确定性累加。
        return torch.stack(pooled, dim=-1).reshape(*image.shape[:-2], rows, columns)


class ImageEncoder(nn.Module):
    """小型 CNN；自适应池化让输入分辨率可调整。"""

    def __init__(
        self,
        channels: tuple[int, ...],
        output_size: int,
        chunk_size: int = 128,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须为正数")
        self.chunk_size = chunk_size
        self.gradient_checkpointing = gradient_checkpointing
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
        layers.extend((DeterministicAdaptiveAvgPool2d((2, 3)), nn.Flatten()))
        self.convolution = nn.Sequential(*layers)
        self.projection = nn.Linear(input_channels * 2 * 3, output_size)

    def _encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(self.convolution(image))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        prefix = image.shape[:-3]
        flat_image = image.reshape(-1, *image.shape[-3:]).float()
        chunks = []
        for chunk in flat_image.split(self.chunk_size):
            if (
                self.training
                and torch.is_grad_enabled()
                and self.gradient_checkpointing
            ):
                # 原始观测不需要梯度；非重入模式仍能计算 CNN 参数的梯度。
                # 仅分块仍会保留所有块的激活，因此更新时还必须重算激活。
                encoded = checkpoint(
                    self._encode, chunk, use_reentrant=False, preserve_rng_state=False
                )
            else:
                encoded = self._encode(chunk)
            chunks.append(encoded)
        encoded = torch.cat(chunks, dim=0)
        return encoded.reshape(*prefix, -1)


class ActorEncoder(nn.Module):
    """独立可训练的深度图 CNN 与机体状态投影，结构对齐 tutorial。"""

    def __init__(self, robot_state_size: int, config: NetworkConfig) -> None:
        super().__init__()
        self.camera = ImageEncoder(
            (8, 16, 32),
            config.feature_dim,
            config.cnn_chunk_size,
            config.cnn_gradient_checkpointing,
        )
        self.robot_state = nn.Linear(robot_state_size, config.feature_dim)

    def forward(self, observation: Observation) -> torch.Tensor:
        return self.camera(observation["camera"]) + self.robot_state(
            observation["robot_state"].float()
        )


class CrossAttentionActorEncoder(ActorEncoder):
    """状态查询视觉空间 token，保留全局视觉与状态相加的残差路径。"""

    def __init__(self, robot_state_size: int, config: NetworkConfig) -> None:
        super().__init__(robot_state_size, config)
        width = config.feature_dim
        rows, columns = config.attention_grid
        self.spatial_pool = DeterministicAdaptiveAvgPool2d((rows, columns))
        self.token_projection = nn.Linear(32, width)
        # 行列编码组合成二维位置，使注意力能区分图像的上下与左右。
        self.row_position = nn.Parameter(torch.empty(rows, width))
        self.column_position = nn.Parameter(torch.empty(columns, width))
        nn.init.normal_(self.row_position, std=0.02)
        nn.init.normal_(self.column_position, std=0.02)
        self.query_norm = nn.LayerNorm(width)
        self.token_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, config.attention_heads, dropout=0.0, batch_first=True
        )
        self.residual_scale = nn.Parameter(
            torch.tensor(float(config.attention_residual_init))
        )

    def _encode_chunk(
        self, image: torch.Tensor, robot_state: torch.Tensor
    ) -> torch.Tensor:
        # 复用原 CNN 的一次卷积结果；保留其参数键名和 2×3 全局投影路径。
        spatial = self.camera.convolution[:-2](image)
        global_feature = self.camera.projection(self.camera.convolution[-2:](spatial))
        state_feature = self.robot_state(robot_state)
        tokens = self.spatial_pool(spatial).flatten(2).transpose(1, 2)
        position = (
            self.row_position[:, None, :] + self.column_position[None, :, :]
        ).flatten(0, 1)
        tokens = self.token_norm(self.token_projection(tokens) + position)
        query = self.query_norm(state_feature).unsqueeze(1)
        attended, _ = self.attention(query, tokens, tokens, need_weights=False)
        return (
            global_feature + state_feature + self.residual_scale * attended.squeeze(1)
        )

    def forward(self, observation: Observation) -> torch.Tensor:
        image = observation["camera"]
        prefix = image.shape[:-3]
        if observation["robot_state"].shape != (*prefix, self.robot_state.in_features):
            raise ValueError("camera 与 robot_state 的时间和批量维度必须一致")
        flat_image = image.reshape(-1, *image.shape[-3:]).float()
        flat_state = (
            observation["robot_state"].reshape(-1, self.robot_state.in_features).float()
        )
        chunks = []
        # 时间与环境合并到批量轴；注意力仅在每张图的空间轴上计算。
        for image_chunk, state_chunk in zip(
            flat_image.split(self.camera.chunk_size),
            flat_state.split(self.camera.chunk_size),
        ):
            if (
                self.training
                and torch.is_grad_enabled()
                and self.camera.gradient_checkpointing
            ):
                # 连同注意力一起重算，避免保存整个 PPO minibatch 的视觉激活。
                encoded = checkpoint(
                    self._encode_chunk,
                    image_chunk,
                    state_chunk,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                encoded = self._encode_chunk(image_chunk, state_chunk)
            chunks.append(encoded)
        return torch.cat(chunks, dim=0).reshape(*prefix, -1)


class CriticEncoder(nn.Module):
    """独立的两层深度图、TOA CNN 与机体状态投影。"""

    def __init__(self, robot_state_size: int, config: NetworkConfig) -> None:
        super().__init__()
        self.camera = ImageEncoder(
            (8, 16),
            config.feature_dim,
            config.cnn_chunk_size,
            config.cnn_gradient_checkpointing,
        )
        self.robot_state = nn.Linear(robot_state_size, config.feature_dim)
        self.toa = ImageEncoder(
            (8, 16),
            config.feature_dim,
            config.cnn_chunk_size,
            config.cnn_gradient_checkpointing,
        )

    def forward(self, observation: Observation) -> torch.Tensor:
        return (
            self.camera(observation["camera"])
            + self.robot_state(observation["robot_state"].float())
            + self.toa(observation["critic_toa"])
        )


class AsymmetricRecurrentActorCritic(nn.Module):
    """使用 tutorial 独立可训练 CNN 的非对称 Actor–Critic。

    Actor 与 Critic 各自持有深度图 CNN、状态投影、循环单元和 MLP；
    Critic 另有 TOA CNN；Actor 可在相加融合上叠加状态查询视觉的交叉注意力。
    """

    action_size = 3
    robot_state_size = 8

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        actor_encoder = (
            CrossAttentionActorEncoder
            if config.architecture == "cross_attention_cnn_v1"
            else ActorEncoder
        )
        self.actor_encoder = actor_encoder(self.robot_state_size, config)
        self.critic_encoder = CriticEncoder(self.robot_state_size, config)
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

        preserved = set()
        preserved.update(id(module) for module in self.actor_recurrent.modules())
        preserved.update(id(module) for module in self.critic_recurrent.modules())
        for module in self.modules():
            if (
                isinstance(module, (nn.Linear, nn.Conv2d))
                and id(module) not in preserved
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

    def _distribution(self, latent: torch.Tensor) -> Normal:
        mean = self.action_mean(self.actor_mlp(latent))
        self.action_distribution = Normal(mean, self.log_std.exp().expand_as(mean))
        return self.action_distribution

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
            features,
            state.actor,
            episode_starts.unsqueeze(0),
        )
        action = self.action_mean(self.actor_mlp(output[0]))
        return action, RecurrentState(actor_state, state.critic)

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
        actor_output, _ = self.actor_recurrent(
            actor_features,
            state.actor,
            episode_starts,
        )
        critic_output, _ = self.critic_recurrent(
            critic_features,
            state.critic,
            episode_starts,
        )
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
            features,
            critic_state,
            episode_starts.unsqueeze(0),
        )
        return self.value_head(self.critic_mlp(output[0])).squeeze(-1)
