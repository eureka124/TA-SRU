"""项目配置。

全部配置都是普通 dataclass，便于 IDE 跳转、类型检查和命令行覆盖。这里不引入
Hydra 等配置框架，因为 demo 的配置规模不值得增加额外抽象。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, isfinite

RECURRENT_TYPES = ("sru-lstm", "sru-gru", "sru-lstm-gate", "lstm", "none")


def normalize_recurrent_type(value: str) -> str:
    return {"nn.lstm": "lstm"}.get(value.lower(), value.lower().replace("_", "-"))


@dataclass
class EnvConfig:
    """无人机导航环境配置。"""

    # 并行运行的向量化环境数量。会被play.sh中的 --num-envs 参数覆盖。
    num_envs: int = 4
    # 环境、NumPy 与 PyTorch 共用的随机种子。
    seed: int = 123
    # 是否启用目标、速度、轨迹及 TOA 地图等调试输出。
    debug: bool = False
    # 调试文件的保存目录；为 None 时使用默认的 debug 目录。
    debug_output_dir: str | None = None
    # 单次物理仿真的时间步长，单位为秒。
    physics_dt: float = 1.0 / 128.0
    # 每次策略决策之间执行的物理仿真步数。
    control_decimation: int = 32
    # 单个回合允许持续的仿真时长，单位为秒。
    episode_seconds: float = 80.0

    # 地图、目标和回合采样使用独立随机流。
    maze_seed: int = 123
    train_map_count: int = 64
    eval_map_count: int = 16
    goals_per_map: int = 8
    maze_size: int = 15
    maze_cell_size: float = 2.0
    maze_wall_removal_probability: float = 0.25
    arena_height: float = 4.0
    drone_radius: float = 0.4
    safety_margin: float = 0.1
    spawn_margin: float = 0.1
    # 先采样最外围可通行格子中的起点，再排除水平距离不足此值的目标。
    min_start_goal_distance: float = 8.0
    # 评估只切换地图池，沿用训练的 Critic 归一化尺度。
    map_split: str = "train"
    evaluation_episodes_per_map: int = 0
    toa_cache_dir: str | None = ".cache/dfs_toa"

    # 输入网络的下采样深度图高度，单位为像素。
    depth_height: int = 12
    # 输入网络的下采样深度图宽度，单位为像素。
    depth_width: int = 16
    # 深度观测的截断和归一化上限，单位为米。
    depth_max_distance: float = 10.0
    # 深度相机的水平视场角，单位为弧度。
    camera_horizontal_fov: float = 1.5707963267948966

    # 期望 TOA 栅格间距，实际间距由地图范围推导。
    toa_resolution: float = 0.1
    # Critic 使用的局部 TOA 方形裁剪边长。
    toa_crop_size: int = 16
    # 局部 TOA 裁剪中相邻采样点的间距，单位为米。
    toa_crop_spacing: float = 0.15
    # TOA 速度场中开始对近墙区域降速的安全距离，单位为米。
    toa_safe_distance: float = 2.0
    # TOA 速度场贴近障碍物时采用的最低速度比例。
    toa_slow_speed: float = 0.2
    # None 使用全体静态地图的可通行最大值；评估与恢复沿用训练量程。
    toa_normalization_max: float | None = None

    # 动作三个分量的下界，依次为前向速度、侧向速度和偏航角速度。
    action_low: tuple[float, float, float] = (-0.1, -0.5, -1.0471975511965976)
    # 动作三个分量的上界，依次为前向速度、侧向速度和偏航角速度。
    action_high: tuple[float, float, float] = (2.0, 0.5, 1.0471975511965976)

    # 奖励系数按每个策略步计算。
    # DirectRLEnv 不会像 RewardManager 一样再乘 policy_dt，因此这里直接保存最终系数。
    # 朝目标方向速度奖励的权重。
    goal_velocity_weight: float = 0.0
    # 按本回合起点 TOA 归一化后的进度奖励权重。
    toa_progress_weight: float = 500.0
    # 相邻策略步动作变化量惩罚的权重。
    action_smoothness_weight: float = -0.1
    # 接触力归一化惩罚的权重。
    contact_force_weight: float = -100.0
    # 到达目标时一次性奖励的权重。
    goal_reached_weight: float = 300.0
    # 判定无人机到达目标的水平距离阈值，单位为米。
    goal_threshold: float = 0.4
    # 训练全程固定的碰撞判定力阈值，单位为牛顿。
    collision_force_threshold: float = 0.1
    # 接触惩罚完成渐增所占的总训练步数比例。
    contact_penalty_ramp_fraction: float = 0.4
    # 环境课程计算进度时采用的总 transition 数，由 TrainConfig 同步。
    total_training_steps: int = 70_000_000

    @property
    def arena_half_extent(self) -> float:
        return self.maze_size * self.maze_cell_size / 2

    @property
    def toa_grid_size(self) -> int:
        return ceil(2 * self.arena_half_extent / self.toa_resolution) + 1

    @property
    def toa_spacing(self) -> float:
        return 2 * self.arena_half_extent / (self.toa_grid_size - 1)

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
        for name in (
            "maze_cell_size",
            "arena_height",
            "drone_radius",
            "toa_resolution",
            "min_start_goal_distance",
            "episode_seconds",
            "toa_crop_spacing",
            "toa_safe_distance",
            "physics_dt",
            "depth_max_distance",
            "goal_threshold",
            "collision_force_threshold",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为有限正数")
        for name in ("safety_margin", "spawn_margin"):
            if not isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} 必须为有限非负数")
        if self.maze_size < 5 or self.maze_size % 2 == 0:
            raise ValueError("maze_size 必须为不小于 5 的奇数")
        if min(self.train_map_count, self.eval_map_count, self.goals_per_map) <= 0:
            raise ValueError("地图数和目标数必须为正数")
        if min(self.seed, self.maze_seed) < 0:
            raise ValueError("随机种子不能为负数")
        if not 0 <= self.maze_wall_removal_probability <= 1:
            raise ValueError("拆墙概率必须位于 [0, 1]")
        if self.maze_cell_size <= 2 * (self.drone_radius + self.safety_margin + self.spawn_margin):
            raise ValueError("通道宽度不足以容纳无人机及出生余量")
        if self.map_split not in ("train", "eval") or self.evaluation_episodes_per_map < 0:
            raise ValueError("地图池或评估配额不合法")
        if self.toa_crop_size != 16:
            raise ValueError("Critic TOA 输入必须保持 16×16")
        if self.toa_safe_distance <= 0.0 or not 0.0 < self.toa_slow_speed <= 1.0:
            raise ValueError("TOA 安全距离必须为正，慢速比例必须位于 (0, 1]")
        if self.toa_normalization_max is not None and not (0.0 < self.toa_normalization_max < float("inf")):
            raise ValueError("toa_normalization_max 必须为有限正数或 None")
        if any(low >= high for low, high in zip(self.action_low, self.action_high)):
            raise ValueError("每个动作下界都必须小于上界")
        if not 0.0 < self.contact_penalty_ramp_fraction <= 1.0:
            raise ValueError("contact_penalty_ramp_fraction 必须位于 (0, 1]")


@dataclass
class NetworkConfig:
    """非对称 Actor–Critic 网络配置。"""

    # Actor 与 Critic 各观测编码器输出的特征维度。
    feature_dim: int = 192
    # Actor 与 Critic 使用的循环单元类型。
    recurrent_type: str = "sru-lstm"
    # 每层循环单元的隐藏状态维度。
    recurrent_hidden_size: int = 256
    # 循环网络的堆叠层数。
    recurrent_layers: int = 1
    # Actor 在循环网络之后各全连接隐藏层的宽度。
    actor_hidden_sizes: tuple[int, ...] = (512, 512)
    # Critic 在循环网络之后各全连接隐藏层的宽度。
    critic_hidden_sizes: tuple[int, ...] = (512, 512)
    # 高斯动作分布可训练对数标准差的初始值。
    initial_log_std: float = 0.0
    # 仅作接口预留：当前实现明确保持 Actor/Critic 深度编码器相互独立。
    share_depth_encoder: bool = False

    def validate(self) -> None:
        self.recurrent_type = normalize_recurrent_type(self.recurrent_type)
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

    # 每个并行环境在一次参数更新前采集的策略步数。
    rollout_steps: int = 128
    # 每次梯度更新使用的有效时间步样本数。
    batch_size: int = 128
    # 循环网络训练时每段序列包含的最大时间步数。
    recurrent_sequence_length: int = 64
    # 对同一批 rollout 数据重复优化的轮数。
    epochs: int = 5
    # 未来回报的折扣因子。
    gamma: float = 0.95
    # 广义优势估计的偏差与方差权衡系数。
    gae_lambda: float = 0.95
    # Adam 优化器的学习率。
    learning_rate: float = 3.0e-4
    # PPO 策略概率比的裁剪范围。
    clip_range: float = 0.2
    # 价值函数更新的裁剪范围；为 None 时不裁剪。
    value_clip_range: float | None = 0.2
    # 策略熵奖励在总损失中的系数。
    entropy_coefficient: float = 0.01
    # 价值函数损失在总损失中的系数。
    value_coefficient: float = 2.0
    # 梯度裁剪允许的最大整体范数。
    max_grad_norm: float = 1.0
    # 是否在每个训练批次内标准化优势值。
    normalize_advantage: bool = True
    # 提前停止当前 PPO 更新的目标 KL 散度；为 None 时禁用。
    target_kl: float | None = 0.02

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

    # 无人机导航环境及奖励配置。
    env: EnvConfig = field(default_factory=EnvConfig)
    # 非对称循环 Actor–Critic 网络配置。
    network: NetworkConfig = field(default_factory=NetworkConfig)
    # Recurrent PPO 算法超参数配置。
    ppo: PPOConfig = field(default_factory=PPOConfig)
    # 算法名称与保存的网络结构共同校验。
    algorithm: str = "recurrent_ppo"
    # 整个训练任务计划采集的 transition 总数。
    total_timesteps: int = 70_000_000
    # 模型与训练张量所在的计算设备。
    device: str = "cuda"
    # 向终端、CSV 和 TensorBoard 写入指标的更新轮次间隔。
    log_interval: int = 1
    # 定期保存模型 checkpoint 的更新轮次间隔。
    checkpoint_interval: int = 25
    # CSV 和 TensorBoard 日志目录；为 None 时禁用文件日志。
    log_dir: str | None = None
    # 定期 checkpoint 和最佳模型的保存目录。
    checkpoint_dir: str = "checkpoints"

    def validate(self) -> None:
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps 必须为正数")
        if self.log_interval <= 0 or self.checkpoint_interval <= 0:
            raise ValueError("日志和 checkpoint 间隔必须为正数")
        # 环境的接触课程均以训练器实际使用的总 transition 数为时间轴。
        self.env.total_training_steps = self.total_timesteps
        self.env.validate()
        self.network.validate()
        if self.algorithm not in ("ppo", "recurrent_ppo"):
            raise ValueError("algorithm 必须是 ppo 或 recurrent_ppo")
        if (self.algorithm == "ppo") != (self.network.recurrent_type == "none"):
            raise ValueError("普通 PPO 必须使用 recurrent_type=none，循环 PPO 必须指定循环单元")
        self.ppo.validate(self.env.num_envs)
