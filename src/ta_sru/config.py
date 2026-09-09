"""项目配置。

全部配置都是普通 dataclass，便于 IDE 跳转、类型检查和命令行覆盖。这里不引入
Hydra 等配置框架，因为 demo 的配置规模不值得增加额外抽象。
"""

from __future__ import annotations

from dataclasses import dataclass, field

RECURRENT_TYPES = ("sru-lstm", "sru-gru", "sru-lstm-gate", "lstm")


@dataclass
class EnvConfig:
    """无人机导航环境配置。"""

    num_envs: int = 4
    seed: int = 123
    debug: bool = False
    debug_output_dir: str | None = None
    physics_dt: float = 1.0 / 120.0
    control_decimation: int = 5
    episode_seconds: float = 40.0

    arena_half_extent: float = 20.0
    arena_height: float = 4.0
    wall_thickness: float = 0.7
    inner_wall_length: float = 12.0
    random_cylinder_count: int = 60
    drone_radius: float = 0.4

    # 难度课程与参考 training_mazes 一致：先学习简单单墙迷宫，再启用全部
    # 六种墙体布局，之后逐步增加随机圆柱。数量表示已生成圆柱槽位的活动前缀。
    training_curriculum_stage_fractions: tuple[float, ...] = (
        0.0,
        0.10,
        0.20,
        0.40,
        0.50,
    )
    training_curriculum_maze_counts: tuple[int, ...] = (3, 4, 5, 5, 6)
    training_curriculum_cylinder_counts: tuple[int, ...] = (10, 10, 10, 30, 60)

    depth_height: int = 12
    depth_width: int = 16
    depth_max_distance: float = 10.0
    camera_horizontal_fov: float = 1.5707963267948966

    toa_grid_size: int = 401
    toa_crop_size: int = 16
    toa_crop_spacing: float = 0.15
    toa_safe_distance: float = 2.0
    toa_slow_speed: float = 0.2
    toa_normalization_max: float = 255.0

    action_low: tuple[float, float, float] = (-0.1, -0.5, -1.0471975511965976)
    action_high: tuple[float, float, float] = (2.0, 0.5, 1.0471975511965976)

    # 与 manager_based/training_mazes 的逐策略步奖励系数保持一致。
    # DirectRLEnv 不会像 RewardManager 一样再乘 policy_dt，因此这里直接保存最终系数。
    goal_velocity_weight: float = 0.1
    toa_progress_weight: float = 5.0
    action_smoothness_weight: float = -0.1
    contact_force_weight: float = -100.0
    goal_reached_weight: float = 300.0
    goal_threshold: float = 0.4
    collision_force_threshold: float = 50.0
    contact_penalty_ramp_fraction: float = 0.4
    total_training_steps: int = 70_000_000

    @property
    def policy_dt(self) -> float:
        return self.physics_dt * self.control_decimation

    @property
    def max_episode_steps(self) -> int:
        return round(self.episode_seconds / self.policy_dt)

    def validate(self) -> None:
        if self.num_envs <= 0:
            raise ValueError("num_envs 必须为正数")
        if self.physics_dt <= 0.0 or self.control_decimation <= 0:
            raise ValueError("physics_dt 和 control_decimation 必须为正数")
        if self.toa_grid_size < 3 or self.toa_grid_size % 2 == 0:
            raise ValueError("toa_grid_size 应为不小于 3 的奇数")
        if self.toa_safe_distance <= 0.0 or not 0.0 < self.toa_slow_speed <= 1.0:
            raise ValueError("TOA 安全距离必须为正，慢速比例必须位于 (0, 1]")
        if self.random_cylinder_count <= 0:
            raise ValueError("random_cylinder_count 必须为正数")
        if self.random_cylinder_count > 60:
            raise ValueError("random_cylinder_count 不能超过 60")
        from ta_sru.envs.curriculum import select_training_maze_curriculum_stage

        select_training_maze_curriculum_stage(
            elapsed_steps=0,
            max_steps=self.total_training_steps,
            stage_fractions=self.training_curriculum_stage_fractions,
            maze_counts=self.training_curriculum_maze_counts,
            cylinder_counts=self.training_curriculum_cylinder_counts,
        )
        if any(count > 6 for count in self.training_curriculum_maze_counts):
            raise ValueError("课程阶段启用的迷宫数量不能超过 6")
        if any(count > 60 for count in self.training_curriculum_cylinder_counts):
            raise ValueError("课程阶段启用的圆柱数量不能超过 60")
        if any(low >= high for low, high in zip(self.action_low, self.action_high)):
            raise ValueError("每个动作下界都必须小于上界")
        if not 0.0 < self.contact_penalty_ramp_fraction <= 1.0:
            raise ValueError("contact_penalty_ramp_fraction 必须位于 (0, 1]")


@dataclass
class NetworkConfig:
    """非对称 Actor–Critic 网络配置。"""

    feature_dim: int = 192
    recurrent_type: str = "sru-lstm"
    recurrent_hidden_size: int = 256
    recurrent_layers: int = 1
    actor_hidden_sizes: tuple[int, ...] = (512, 512)
    critic_hidden_sizes: tuple[int, ...] = (512, 512)
    initial_log_std: float = 0.0
    # 仅作接口预留：当前实现明确保持 Actor/Critic 深度编码器相互独立。
    share_depth_encoder: bool = False

    def validate(self) -> None:
        if self.recurrent_type not in RECURRENT_TYPES:
            choices = ", ".join(RECURRENT_TYPES)
            raise ValueError(f"recurrent_type 必须是以下值之一：{choices}")
        if self.feature_dim <= 0 or self.recurrent_hidden_size <= 0:
            raise ValueError("feature_dim 和 recurrent_hidden_size 必须为正数")
        if self.recurrent_layers <= 0:
            raise ValueError("recurrent_layers 必须为正数")


@dataclass
class PPOConfig:
    """Recurrent PPO 超参数，与旧工程当前训练配置对齐。"""

    rollout_steps: int = 128
    batch_size: int = 128
    recurrent_sequence_length: int = 64
    epochs: int = 5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3.0e-4
    clip_range: float = 0.2
    value_clip_range: float | None = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 2.0
    max_grad_norm: float = 1.0
    normalize_advantage: bool = True
    target_kl: float | None = None

    def validate(self, num_envs: int) -> None:
        if self.rollout_steps <= 0 or self.batch_size <= 0:
            raise ValueError("rollout_steps 和 batch_size 必须为正数")
        if self.recurrent_sequence_length <= 0:
            raise ValueError("recurrent_sequence_length 必须为正数")
        if self.batch_size > self.rollout_steps * num_envs:
            raise ValueError("batch_size 不能大于一次 rollout 的总样本数")


@dataclass
class TrainConfig:
    """训练任务总配置。"""

    env: EnvConfig = field(default_factory=EnvConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    total_timesteps: int = 70_000_000
    device: str = "cuda"
    log_interval: int = 1
    checkpoint_interval: int = 5
    log_dir: str | None = None
    checkpoint_dir: str = "checkpoints"

    def validate(self) -> None:
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps 必须为正数")
        if self.log_interval <= 0 or self.checkpoint_interval <= 0:
            raise ValueError("日志和 checkpoint 间隔必须为正数")
        # 环境的两套课程均以训练器实际使用的总 transition 数为时间轴。
        self.env.total_training_steps = self.total_timesteps
        self.env.validate()
        self.network.validate()
        self.ppo.validate(self.env.num_envs)
