# TA-SRU

TA-SRU 使用 Isaac Sim、Isaac Lab 和 PyTorch 训练 Hummingbird 无人机导航策略。无人机在六种墙体布局及随机圆柱障碍中，从起点飞到目标点。支持普通 PPO，以及使用 SRU-LSTM、SRU-GRU、SRU-LSTM-Gate 或 LSTM 的循环 PPO。

本文中的命令均在**项目根目录**执行。路径中的 `<运行名>`、`<时间戳>` 等占位符需要替换为实际值。

## 1. 环境准备

需要先准备能运行 Isaac Lab 的 Python 环境及其配套 Isaac Sim、PyTorch 和 GPU 驱动。本项目的 `pip install` 只安装项目 Python 依赖，不负责安装 Isaac Sim / Isaac Lab。

已有 Conda 环境时，可按实际安装位置调整以下命令：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /path/to/ta-sru
python -m pip install -e .
```

Python 最低版本由 `pyproject.toml` 声明为 3.10；实际还需满足所用 Isaac Lab 版本的要求。训练和评估都需要启动仿真。`--headless` 表示不显示仿真窗口，仍会计算物理和深度观测。

USD 资产配置了 Git LFS。通过 Git 克隆后，如果资产仍是 LFS 指针文件，需要先安装 Git LFS 并执行 `git lfs pull`，取得真实模型文件。

## 2. 文件与目录用途

### 运行入口

| 文件 | 用途 |
| --- | --- |
| `scripts/train.py` | 新建或恢复训练，启动仿真，创建运行目录，保存配置、日志和 checkpoint。 |
| `scripts/play.py` | 加载命令行指定的 checkpoint，运行六种布局的评估，保存统计及可选回放。 |
| `scripts/evaluation.py` | 评估配置覆盖、终局分类、每种布局的样本配额及汇总计算；由 `play.py` 导入，无需单独运行。 |
| `scripts/progress_to_tensorboard.py` | 将已有训练 `progress.csv` 回填为 TensorBoard 事件。 |
| `scripts/export_actor.py` | 从训练 checkpoint 导出只包含 Actor 权重及相关配置的 PyTorch 文件，无需启动仿真。 |

### 核心代码

| 文件 | 用途 |
| --- | --- |
| `src/ta_sru/config.py` | `EnvConfig`、`NetworkConfig`、`PPOConfig`、`TrainConfig` 默认值和合法性检查。命令行未开放的设置在这里调整。 |
| `src/ta_sru/envs/layouts.py` | `maze_01` 至 `maze_06` 的墙体位置、方向和两组起终点路线。 |
| `src/ta_sru/envs/curriculum.py` | 根据累计训练 transition 数选择布局范围和活动圆柱数量。 |
| `src/ta_sru/envs/navigation.py` | Isaac Lab 场景、传感器、动作执行、观测、奖励、终止和重置逻辑。 |
| `src/ta_sru/envs/wrappers.py` | 将仿真接口转换为 PPO 使用的 NumPy 观测、奖励、终止标记和 episode 信息。 |
| `src/ta_sru/envs/toa.py` | 用 FMM 构建 Time of Arrival（TOA）地图，进行采样、局部裁剪和调试图输出。 |
| `src/ta_sru/models/actor_critic.py` | 非对称 Actor-Critic：Actor 使用深度图和机体状态，Critic 额外使用局部 TOA。 |
| `src/ta_sru/models/recurrent.py` | SRU 系列、PyTorch LSTM 及普通 PPO 的前馈模块，包含循环状态重置。 |
| `src/ta_sru/models/hummingbird.py` | 飞机参数定义、四元数运算及动力学辅助函数。 |
| `src/ta_sru/models/hummingbird_asset.py` | 将本地 Hummingbird USD 资产配置为 Isaac Lab 飞行器。 |
| `src/ta_sru/models/lee_controller.py` | Lee 控制器，将目标运动转换为飞行器控制量。 |
| `src/ta_sru/algorithms/buffer.py` | 存放 rollout、计算 GAE、生成前馈或循环训练批次；完整 buffer 存放在 CPU 内存。 |
| `src/ta_sru/algorithms/ppo.py` | 采样、PPO 更新、训练指标、checkpoint 保存和恢复，同时支持普通及循环 PPO。 |
| `src/ta_sru/debug/playback.py` | 按回合采集深度、动作、状态和场景，生成 NPZ、MP4、JSON、HTML 及回放索引。 |
| `src/ta_sru/debug/playback.html` | 单个 episode 的浏览器回放模板。 |
| `src/ta_sru/export.py` | Actor 导出的具体实现。 |
| 各目录的 `__init__.py` | Python 包入口；`envs/__init__.py` 延迟导入仿真相关类，避免纯 Python 工具启动 Isaac Sim。 |

### 资产、测试及辅助文件

| 文件或目录 | 用途 |
| --- | --- |
| `assets/hummingbird/hummingbird.usd` | 无人机仿真模型资产。 |
| `assets/hummingbird/hummingbird.yaml` | 无人机物理及控制参数。 |
| `tests/test_evaluation.py` | 检查固定评估课程、结局优先级和每种布局配额。 |
| `tests/test_playback.py` | 检查原始深度、视频编码、并行回合隔离及回放保存上限。 |
| `tests/test_core.py`（本地存在时） | 网络、控制器、TOA、buffer、PPO 更新和日志测试；当前忽略规则未将该文件纳入版本管理。 |
| `pyproject.toml` | Python 包信息、依赖及打包配置。 |
| `.gitignore` | 排除运行产物、缓存等；部分本地 shell 脚本也被忽略。 |
| `.gitattributes` | Git 文件属性配置。 |
| `THIRD_PARTY_NOTICES.md` | 控制器及 SRU 实现的第三方来源和许可证。 |
| `runs/` | 训练日志、checkpoint、评估统计和 debug 回放，通常不纳入 Git。 |

工作区还可能包含以下本地便捷脚本。它们含固定环境名、参数或机器路径，使用前应检查内容；本文以 `scripts/*.py` 的命令为准。

| 本地脚本 | 用途及当前注意事项 |
| --- | --- |
| `train.sh` | 在 tmux 中启动循环 PPO。当前预设 `batch-size=10240`，大于 `512 × 6 = 3072` 个 rollout 样本，需先修正才能通过配置检查。 |
| `ppo_train.sh` | 在 tmux 中启动普通 PPO。 |
| `play.sh` | 启动评估；当前本地版本写死 checkpoint 路径，需修改路径后使用。 |
| `tensorboard.sh` | 在 screen 会话中启动 TensorBoard。 |
| `rsync.sh` | 从预设远端同步部分模型和 TensorBoard 文件；地址和筛选规则需要按使用场景调整。 |

## 3. 如何训练

### 先进行短训练检查

以下配置用于确认环境能创建、采样和更新，不用于比较模型性能：

```bash
python scripts/train.py \
  --headless --device cuda:0 --network-device cuda:0 \
  --num-envs 2 --total-timesteps 1024 \
  --rollout-steps 32 --batch-size 32 --sequence-length 8 \
  --recurrent-type sru-lstm --toa-grid-size 51 --cylinders 8
```

### 正式训练

循环 PPO 示例：

```bash
python scripts/train.py \
  --headless --device cuda:0 --network-device cuda:0 \
  --algorithm recurrent_ppo --recurrent-type sru-lstm \
  --num-envs 6 --total-timesteps 70000000 \
  --rollout-steps 512 --batch-size 512 --sequence-length 64 \
  --toa-grid-size 401 --cylinders 60 --log-dir runs
```

`--recurrent-type` 可选 `sru-lstm`、`sru-gru`、`sru-lstm-gate`、`lstm`。未指定算法时默认使用循环 PPO，未指定循环单元时默认使用 `sru-lstm`。

普通 PPO 使用以下命令，**不传 `--recurrent-type`**：

```bash
python scripts/train.py \
  --headless --device cuda:0 --network-device cuda:0 \
  --algorithm ppo --num-envs 6 --total-timesteps 70000000 \
  --rollout-steps 512 --batch-size 512 --toa-grid-size 401 --cylinders 60
```

### 常用训练参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--num-envs` | `4` | 并行仿真实例数。 |
| `--total-timesteps` | `70000000` | 所有环境累计的 transition 目标数，不是 episode 数；一次向量步产生 `num_envs` 个 transition。 |
| `--rollout-steps` | `512` | 每次更新前，每个环境采集的策略步数。 |
| `--batch-size` | `512` | 训练批次大小，不能超过 `rollout_steps × num_envs`。 |
| `--sequence-length` | `64` | 循环 PPO 的训练序列长度。 |
| `--seed` | `123` | 随机种子。 |
| `--cylinders` | `60` | 随机圆柱槽位上限，实际活动数量还受课程控制。 |
| `--toa-grid-size` | `401` | 全局 TOA 网格边长。 |
| `--network-device` | 跟随 `--device` | 网络设备；`--device` 指定仿真设备。 |
| `--log-dir` | `runs` | 日志根目录。 |
| `--run-name` | 时间戳 | 自定义运行名后缀；最终目录带算法或循环单元前缀，同名目录已存在会报错。 |
| `--checkpoint-interval` | `5` | 每隔多少次 PPO 更新保存周期 checkpoint。 |
| `--log-interval` | `1` | 每隔多少次 PPO 更新写 CSV 和 TensorBoard。 |
| `--debug` | 关闭 | 显示导航标记并保存全局 TOA 图；训练时不生成评估 episode 回放。 |

显存不足时先降低并行环境数和 batch size，同时检查上述批次约束。完整 rollout 保存在 CPU，但内存需求仍随环境数和 rollout 长度增加。

### 中断与恢复

训练期间按 Ctrl+C 会保存 `checkpoints/model_interrupted_<步数>.pt`。恢复示例：

```bash
python scripts/train.py \
  --resume runs/<原运行名>/checkpoints/model_interrupted_<步数>.pt \
  --headless --device cuda:0 --network-device cuda:0 \
  --num-envs 6 --total-timesteps 70000000 \
  --rollout-steps 512 --batch-size 512 --sequence-length 64 \
  --toa-grid-size 401 --cylinders 60
```

恢复时加载模型、优化器、训练步数、网络结构和历史最佳回报。算法及循环单元从 checkpoint 推断，不能切换。

**环境和 PPO 配置仍由本次命令及默认值构建**，不会全部沿用 checkpoint。请参照原运行的 `config.json` 和 `command.txt` 传入原参数。`--total-timesteps` 是包含已训练步数的总目标；恢复会创建新的运行目录，不会继续向旧目录追加日志。

### 训练课程与指标

默认课程按总训练进度启用不同布局和圆柱数量：

| 训练进度 | 布局 | 活动圆柱数上限 |
| --- | --- | --- |
| 0%–10% | maze_01–03 | 10 |
| 10%–20% | maze_01–04 | 10 |
| 20%–40% | maze_01–05 | 10 |
| 40%–50% | maze_01–05 | 30 |
| 50%–60% | 全部六种 | 60 |
| 60%–95% | maze_04 | 60 |
| 95% 以后 | 全部六种 | 60 |

布局在 episode 重置时切换；圆柱数量不超过 `--cylinders`。碰撞接触力阈值在前 40% 训练进度从 50 N 线性减至 0.1 N，随后保持不变。

当前训练达到碰撞阈值即终止。训练日志按“成功优先、其次碰撞、其余超时”分类，越界也可能被归为超时。TensorBoard 的 `Metrics/*` 使用最近 100 个已完成 episodes；这些训练指标与后文独立评估的固定场景及分类口径不同。

## 4. 训练产物在哪里、如何查看

默认目录结构如下，未启用 debug 或未运行评估时，相应目录可能尚不存在：

```text
runs/<算法或循环单元>_<时间戳或运行名>/
├── config.json                 # 本次训练配置
├── command.txt                 # 启动命令
├── commit_id.txt               # 基准 Git 提交
├── diff.patch                  # 相对基准提交的代码差异
├── progress.csv                # 训练过程指标
├── events.out.tfevents.*        # TensorBoard 事件
├── checkpoints/
│   ├── model_<步数>.pt         # 周期保存
│   ├── model_best.pt           # 最近最多 100 回合平均回报的历史最佳模型
│   ├── model_final.pt          # 正常训练完成后的模型
│   └── model_interrupted_*.pt  # Ctrl+C 中断时保存
├── evaluation/
│   └── eval_<时间戳>/
│       ├── summary.json        # 分布局与整体统计
│       └── episodes.csv        # 逐回合评估明细
└── debug/
    ├── ...                     # 全局 TOA 的 PNG / NPZ
    └── play_<时间戳>/
        ├── index.html          # 回放索引
        └── env_000_episode_00000/
            ├── index.html
            ├── depth.mp4
            ├── depth_raw.npz
            └── telemetry.json
```

`model_best.pt` 按**训练回报**选择，不代表评估成功率最高。比较模型应分别运行评估。

查看训练曲线：

```bash
tensorboard --logdir runs --port 6006
```

在运行 TensorBoard 的机器上，用浏览器打开 `http://localhost:6006`。常用标签：

| 标签 | 含义 |
| --- | --- |
| `rollout/ep_rew_mean`、`rollout/ep_len_mean` | 最近最多 100 个完整回合的平均回报和长度。 |
| `Metrics/Success_Rate`、`Metrics/Collision_Rate`、`Metrics/Timeout_Rate` | 最近最多 100 个完整训练回合的结局比例。 |
| `train/*` | PPO 损失、KL、裁剪比例等。 |
| `progress/*` | CSV 对应指标，包括训练步数、课程阶段、活动布局/圆柱数和接触力阈值。 |

`progress.csv` 的 `episodes` 是累计完成数量，`success/collision/timeout` 是本次日志区间的计数。已有 CSV 可回填 TensorBoard：

```bash
python scripts/progress_to_tensorboard.py runs/<运行名>/progress.csv
```

回填的三项结局比例根据 CSV 区间计数计算，不等同于在线 TensorBoard 的最近 100 回合窗口。已有事件时通常无需回填；`--force` 允许重复导入，可能产生重复点。

默认 `commit_id.txt` 和 `diff.patch` 来自当前 HEAD 及 `git diff HEAD`。也可用 `--commit-id`、`--diff-file` 指定基准提交和预先生成的 patch。Git diff 不包含未跟踪文件，复现实验所需的新代码应纳入版本管理。

## 5. 如何评估

使用完整训练 checkpoint，命令行第一个位置参数就是模型路径：

```bash
python scripts/play.py runs/<运行名>/checkpoints/model_best.pt \
  --headless --device cuda:0 --network-device cuda:0 \
  --num-envs 6
```

普通 PPO 和各类循环 PPO 使用相同入口，结构从 checkpoint 读取。评估采用确定性动作，不更新网络。

评估固定使用 `layouts.py` 的全部六种布局，不沿用训练课程阶段。圆柱数量采用 checkpoint 最终课程的数量，并受其圆柱槽位上限约束；碰撞阈值固定为 **0.1 N**。

默认**每种布局统计 1000 个完整 episodes，共 6000 个，达到配额后自动退出**。`--num-envs` 至少为 6，建议为 6 的倍数；增加实例数不会增加每种布局的统计配额。某布局满额后，该布局额外完成的回合不计入统计。

先用小样本检查整个评估流程：

```bash
python scripts/play.py runs/<运行名>/checkpoints/model_best.pt \
  --headless --device cuda:0 --num-envs 6 --episodes-per-layout 2
```

以上只统计 12 个回合，不适合用于正式性能比较。

| 参数 | 含义 |
| --- | --- |
| `checkpoint` | 必填的位置参数，指定完整训练 checkpoint。 |
| `--episodes-per-layout` | 每种布局的完整回合目标数，默认 1000。 |
| `--num-envs` | 并行实例数，默认 6，至少 6。 |
| `--debug` | 额外保存每种布局前 4 个完整回合的回放。 |
| `--steps` | 可选的向量步数上限；达到后提前退出，不保证回合配额完成。 |
| `--output-dir` | 自定义统计输出目录，必须是尚不存在的新目录；不改变 debug 保存路径。 |
| `--device`、`--network-device` | 仿真与网络设备。 |
| `--headless` | 关闭仿真窗口；省略可观察实时仿真。 |

Ctrl+C、关闭仿真窗口或达到 `--steps` 上限都会结束评估并保存已有完整回合统计。未完成配额时 `complete=false`；未结束的回合不计入分母，也不保存其 debug 回放。

## 6. 如何查看评估结果

### 找到本次输出

程序启动时打印评估结果目录。标准 checkpoint 路径 `runs/<运行名>/checkpoints/model.pt` 对应：

```text
runs/<运行名>/evaluation/eval_<时间戳>/
```

`evaluation/` 与 `checkpoints/`、`debug/` 并列，每次评估创建新的时间戳子目录。如果 checkpoint 的父目录不叫 `checkpoints`，则以 checkpoint 所在目录作为输出根目录；移动了模型或使用自定义 checkpoint 目录时，可通过 `--output-dir` 明确指定统计位置。

### 查看成功率、碰撞率、超时率

```bash
python -m json.tool runs/<运行名>/evaluation/eval_<时间戳>/summary.json
```

`summary.json` 的主要字段如下：

| 字段 | 含义 |
| --- | --- |
| `checkpoint` | 本次评估模型的绝对路径。 |
| `num_envs`、`cylinders`、`collision_force_threshold` | 实际评估实例数、活动圆柱数和碰撞阈值。 |
| `episodes_per_layout` | 每种布局目标回合数。 |
| `complete` | 六种布局是否全部完成目标配额。 |
| `layouts.maze_01` 至 `layouts.maze_06` | 各布局的计数和比例。 |
| `overall` | 已计入回合的整体计数和比例。 |

每个布局及 `overall` 都包含 `episodes`、`success`、`collision`、`timeout` 和对应的 `success_rate`、`collision_rate`、`timeout_rate`。比例取值为 0–1，例如 `0.85` 表示 85%；没有已完成回合时比例为 `null`。

评估结局互斥，按以下优先级判定：

1. 接触力达到 0.1 N 或触发越界：碰撞。
2. 未碰撞且进入目标范围：成功。
3. 以上均未发生且达到回合时限：超时。

默认目标水平距离阈值为 0.4 m，回合时限为 40 s，实际由 checkpoint 的环境配置决定。场地四周有墙；代码仍保留越界判断，并在评估中归入碰撞。

三项率的分母都是**该范围内已计入的完整回合数**。非空统计的三项率之和为 1，碰撞率就是发生碰撞的回合数除以完整回合数。完整默认评估中，每个布局应为 1000，整体为 6000；提前退出时整体按实际样本数加权，不能把它当作六种布局等量评估结果。

### 查看逐回合记录

用表格软件或文本编辑器打开同目录的 `episodes.csv`：

| 列 | 含义 |
| --- | --- |
| `maze` | 布局名称。 |
| `env_id` | 并行仿真实例编号，不是布局编号。 |
| `episode` | 此布局内已计入回合的序号，从 1 开始。 |
| `outcome` | `success`、`collision` 或 `timeout`。 |
| `steps` | 本回合的策略步数，不是物理子步数。 |
| `return` | 本回合累计环境奖励。 |

程序在有回合完成时刷新统计文件，可以在评估运行中查看。正式比较时，先核对 `complete`、checkpoint、障碍物数量和每种布局样本数，再比较比例。

### 查看 debug 视频与轨迹

评估时启用：

```bash
python scripts/play.py runs/<运行名>/checkpoints/model_best.pt \
  --headless --device cuda:0 --num-envs 6 --debug
```

打开启动日志打印的 `debug/play_<时间戳>/index.html`，点击某个回合进入回放。网页包含深度视频、请求/执行动作、实际运动状态和俯视轨迹，支持播放、调速、逐帧及拖动定位。

保存规则是**每种布局最先完成的 4 个完整回合，共最多 24 个**，不是每个并行实例 4 个。同一步结束的回合按环境编号排序；每种布局评估目标小于 4 时，保存上限随之减小。达到保存上限后仍继续完成全部统计。

| 文件 | 如何使用 |
| --- | --- |
| `index.html` | 浏览器直接打开，查看同步回放。 |
| `depth.mp4` | 灰度深度视频；默认原始分辨率为 64×48，保留相机分辨率，按策略频率采样。 |
| `depth_raw.npz` | 原始米制浮点深度（保留 NaN/Inf）、时间、动作和状态，可用 NumPy 读取。 |
| `telemetry.json` | 本回合布局、结局、障碍物几何及逐步动作/状态/位置数据。 |

灰度视频使用固定深度量程；数值分析应使用 NPZ 原始深度。状态中的 `vx/vy` 为机体系速度（m/s），`w` 为机体系 Z 轴角速度（rad/s）。每帧对应动作执行后的状态，包含自动重置前的终止帧。

在远端运行时，把整个 `play_<时间戳>` 文件夹复制到本地再打开 HTML，保持 HTML 与 MP4 的相对路径。全局 TOA PNG/NPZ 另存于 `debug/`，不占这 24 个 episode 的配额。

## 7. 修改算法与导出模型

主要数据流为：

```text
Isaac Sim / PhysX → NavigationEnv → IsaacLabWrapper → PPO
深度图 + 8 维机体状态 → Actor → 目标前向速度、侧向速度、偏航角速度
深度图 + 机体状态 + 局部 TOA → Critic → 状态价值
策略动作 → Lee 控制器 → 无人机物理运动
```

默认物理频率为 120 Hz，每 5 个物理步执行一次策略决策，即 24 Hz。Actor 与 Critic 使用独立编码器。修改布局看 `envs/layouts.py`，修改奖励/终止看 `envs/navigation.py`，修改网络看 `models/actor_critic.py`、`models/recurrent.py` 和 `NetworkConfig`；`share_depth_encoder=True` 当前尚未实现。

导出 Actor：

```bash
python scripts/export_actor.py \
  runs/<运行名>/checkpoints/model_best.pt \
  runs/<运行名>/actor.pt
```

输出为项目定义的 PyTorch 权重与配置字典，不是 TorchScript 或 ONNX，也不是 `play.py` 接受的完整训练 checkpoint。

## 8. 验证与第三方来源

不启动 Isaac Sim 的单元测试：

```bash
python -m unittest discover -s tests -v
```

这些测试覆盖纯 Python / PyTorch 逻辑和回放编码，不能替代真实仿真验证。安装后可先执行短训练，再用生成的 checkpoint 做每种布局 2 个回合的评估检查。

第三方实现来源及许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
