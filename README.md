# TA-SRU

这是对旧项目 `/home/user/tutorial` 导航训练链路的一次独立重写。旧项目保持不变；
新项目继续使用 **Isaac Sim + Isaac Lab + PhysX**，但不再依赖 Stable-Baselines3、
sb3-contrib 或 OmniDrones。

项目目标是让环境、控制器、非对称网络和 PPO 的数据流都能直接阅读和修改，避免原有
库中多层继承、注册器以及与当前实验无关的抽象。

## 保留的算法逻辑

- 六种迷宫、两组起终点和随机正反方向；每个环境按 `env_id % 6` 选择迷宫。
- Hummingbird USD 飞机模型、Lee 位置控制器、120 Hz 物理频率和 `decimation=5`。
- 动作为机体系 `[前向速度, 侧向速度, 偏航角速度]`，范围分别是
  `[-0.1, 2.0]`、`[-0.5, 0.5]`、`[-π/3, π/3]`。
- Actor 只观察深度图和 8 维机体状态；Critic 额外观察机体对齐的 `16×16` TOA 图。
- Actor/Critic 各自使用深度编码器、可切换循环单元和 MLP。
- Recurrent PPO、GAE、策略/价值裁剪、熵奖励、梯度裁剪和超时 value bootstrap。
- 奖励包含目标方向速度、TOA 进度、动作平滑、接触力惩罚和到达奖励。

深度图由 Isaac Lab 的 `MultiMeshRayCasterCamera` 生成，碰撞力来自
`ContactSensor`，Hummingbird 动力学由 Isaac Sim/PhysX 计算。项目内只保留实际用到的
飞机资产、参数和 Lee 控制器，没有引入 OmniDrones 的类层次。

## 结构与数据流

```text
Isaac Sim / PhysX
        |
NavigationEnv (Isaac Lab DirectRLEnv)
        |
IsaacLabWrapper（自定义，不继承 SB3 wrapper）
        |
RecurrentPPO
   |                    |
   |                    +-- GPU：网络与当前 minibatch
   +-- CPU：完整 rollout buffer（NumPy）

camera + robot_state -------------> Actor encoder -> recurrent core -> policy
camera + robot_state + critic_toa -> Critic encoder -> recurrent core -> value
```

完整 rollout 的观测、动作、价值、GAE 和循环状态都由 `CpuRolloutBuffer` 以 NumPy
数组保存在内存中。只有当前 minibatch 会在更新时复制到训练设备，不会把整个 buffer
常驻显存。

主要模块：

```text
src/ta_sru/
├── config.py                 # 集中的 dataclass 配置
├── envs/
│   ├── layouts.py            # 六种迷宫布局
│   ├── toa.py                # FMM TOA 地图、采样和局部裁剪
│   ├── navigation.py         # Isaac Lab 环境、观测、奖励和终止
│   └── wrappers.py           # 自定义 Isaac Lab wrapper
├── models/
│   ├── hummingbird.py        # 飞机参数与四元数工具
│   ├── hummingbird_asset.py  # 本地 Hummingbird USD 配置
│   ├── lee_controller.py     # 精简的 Lee 控制器
│   ├── recurrent.py          # 三种 SRU 和 torch.nn.LSTM
│   └── actor_critic.py       # 非对称 Actor-Critic
└── algorithms/
    ├── buffer.py             # CPU rollout buffer
    └── ppo.py                # Recurrent PPO
```

## 安装与运行

使用已经安装 Isaac Sim/Isaac Lab 的环境：

```bash
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/user/TA-SRU
python -m pip install -e .
```

短训练检查：

```bash
python scripts/train.py \
  --headless \
  --device cuda:0 \
  --num-envs 2 \
  --total-timesteps 1024 \
  --rollout-steps 32 \
  --batch-size 32 \
  --sequence-length 8 \
  --recurrent-type sru-lstm \
  --toa-grid-size 51 \
  --cylinders 8
```

正式训练、恢复训练与评估：

```bash
python scripts/train.py --headless --device cuda:0 --num-envs 4 --recurrent-type sru-gru
python scripts/train.py --headless --device cuda:0 --recurrent-type lstm
python scripts/play.py runs/sru-gru_时间戳/checkpoints/model_final.pt --device cuda:0
```

需要检查导航过程时，在训练或评估命令后添加 `--debug`。该开关会显示目标点、位于
无人机上方且长度随实际水平速度变化的速度圆锥，以及每个环境当前回合的飞行轨迹；
同时会在运行目录的 `debug/` 下保存六种迷宫各自的 4×1 全局 TOA 汇总 PNG 和原始 NPZ。
未传入 `--debug` 时不会创建这些标记或文件。

`--device` 是 Isaac 仿真设备，`--network-device` 可单独指定网络设备。显存紧张时优先
减小 `--num-envs`、`--batch-size` 和 `--sequence-length`；rollout buffer 始终留在 CPU。

训练前可通过 `--recurrent-type` 选择：

| 参数 | 循环单元 |
|---|---|
| `sru-lstm` | SRU-LSTM（默认） |
| `sru-gru` | SRU-GRU |
| `sru-lstm-gate` | 带 refine gate 的 SRU-LSTM |
| `lstm` | 原生 `torch.nn.LSTM` |

每次运行会创建 `runs/<循环单元>_<时间戳>/`，其中包含 `config.json`、`command.txt`、
`progress.csv` 和 `checkpoints/`。例如：

```text
runs/sru-lstm-gate_20260903-153000/
```

可用 `--run-name 消融实验1` 增加实验名，或用 `--log-dir` 修改日志根目录。

## 自定义网络

在 `NetworkConfig` 中可直接调整特征维度、循环类型、循环层和 Actor/Critic MLP。两个深度编码器
当前明确分开。`share_depth_encoder` 是未来合并特征提取器的接口预留；现在将它设为
`True` 会抛出 `NotImplementedError`，避免无意改变实验结构。

## 测试

无需启动 Isaac Sim 的算法单元测试：

```bash
python -m unittest discover -s tests -v
```

此外，当前版本已在本机 Isaac Sim 4.5 / Isaac Lab 环境中完成真实 PhysX 场景的重置、
步进、rollout 采样和 PPO 更新冒烟测试。第三方来源与许可证见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
