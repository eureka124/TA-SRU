# TA-SRU

TA-SRU 使用 Isaac Sim、Isaac Lab 和 PyTorch 训练 Hummingbird 无人机导航策略。无人机在启动时生成的地图池中，从起点飞到目标点；地图默认是 DFS＋随机拆墙迷宫，也可以切换成随机圆柱与 U 形障碍场地。支持普通 PPO，以及使用 SRU-LSTM、SRU-GRU、SRU-LSTM-Gate 或 LSTM 的循环 PPO。

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

当前默认网络恢复为 `~/tutorial` 的可训练小 CNN，不加载深度编码器预训练权重。仓库保留的 RegNetX 模块与 `assets/depth_encoder/vae_pretrain_new.pth` 仅用于历史实现对照，当前训练、评估和导出均不使用它们。

## 2. 文件与目录用途

### 运行入口

| 文件 | 用途 |
| --- | --- |
| `scripts/check_dfs_scene.py` | 用两个小型环境检查共享场景、传感器及回合重置；`--scene-type obstacles` 检查障碍场地。 |
| `scripts/controller_step_response.py` | 在独立无障碍场景中测试前向、侧向和偏航角速度阶跃与回零响应；默认悬停 1 秒、阶跃 6 秒、回零 6 秒，输出 CSV、运行摘要和曲线页面。`--plot-csv` 可从旧数据重新生成曲线，无需启动仿真。 |
| `scripts/preview_scene.py` | 不启动仿真，把地图池渲染成 PNG 并打印障碍物数量与最小表面间距。 |
| `scripts/train.py` | 新建或恢复训练，启动仿真，创建运行目录，保存配置、日志和 checkpoint。 |
| `scripts/play.py` | 加载命令行指定的 checkpoint，运行独立 DFS 地图池的评估，保存统计及可选回放。 |
| `scripts/progress_to_tensorboard.py` | 将已有训练 `progress.csv` 回填为 TensorBoard 事件。 |
| `scripts/export_actor.py` | 从训练 checkpoint 导出只包含 Actor 权重及相关配置的 PyTorch 文件，无需启动仿真。 |

### 核心代码

| 文件 | 用途 |
| --- | --- |
| `src/ta_sru/config.py` | `EnvConfig`、`NetworkConfig`、`PPOConfig`、`TrainConfig` 默认值和合法性检查。命令行未开放的设置在这里调整。 |
| `src/ta_sru/evaluation.py` | 独立于仿真的评估配置、终局分类、每张地图的样本配额及汇总计算；作为包内模块供 `scripts/play.py` 和测试导入，无需单独运行。 |
| `src/ta_sru/envs/maze.py` | 场景地图数据结构、DFS 生成、随机拆墙、地图内容去重、共用物理/射线网格及地图池分发。 |
| `src/ta_sru/envs/primitives.py` | 障碍物图元（圆柱、带朝向长方体）、共享水平截面距离场与成对间距计算。 |
| `src/ta_sru/envs/obstacles.py` | 随机圆柱与 U 形障碍场地生成：泊松盘采样摆位并保证最小表面间距。 |
| `src/ta_sru/scene_config.py` | 恢复环境配置及兼容性检查。 |
| `src/ta_sru/envs/toa_sampling.py` | 有效性检查、共享 TOA 采样与起点归一化奖励。 |
| `src/ta_sru/envs/contact.py` | 计算当前策略步所有物理子步、所有机体部件的接触力峰值。 |
| `src/ta_sru/envs/navigation.py` | Isaac Lab 场景、传感器、动作执行、观测、奖励、终止和重置逻辑。 |
| `src/ta_sru/envs/wrappers.py` | 将仿真接口转换为 PPO 使用的 NumPy 观测、奖励、终止标记和 episode 信息。 |
| `src/ta_sru/envs/toa.py` | 用 FMM 构建 Time of Arrival（TOA）地图，进行采样、局部裁剪和调试图输出。 |
| `src/ta_sru/envs/depth.py` | 深度观测预处理：非有限值替换、4×4 最小池化下采样、量程外置 0。 |
| `src/ta_sru/models/actor_critic.py` | 非对称 Actor-Critic：Actor 使用深度图和机体状态，Critic 额外使用局部 TOA。 |
| `src/ta_sru/models/depth_encoder.py` | 历史 RegNetX 冻结编码器，保留作对照，当前默认网络不调用。 |
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
| `assets/depth_encoder/vae_pretrain_new.pth` | 深度编码器的预训练权重（22.7 MB，普通 Git 文件）。 |
| `tests/test_depth_encoder.py` | tutorial CNN 结构与梯度、历史编码器和深度预处理检查。 |
| `tests/test_cnn_memory.py` | CNN 分块与激活重算的输出、梯度等价性及反向传播张量占用检查。 |
| `tests/test_evaluation.py` | 检查独立评估池配置、结局优先级和每张地图配额。 |
| `tests/test_playback.py` | 检查原始深度、视频编码、并行回合隔离及回放保存上限。 |
| `tests/test_dfs.py` | DFS 树结构、几何、缓存、点目标 TOA、归一化奖励及恢复校验。 |
| `tests/test_obstacles.py` | 障碍场地数量与间距、独立边界点复核、自由空间连通性、网格范围及场景版本校验。 |
| `tests/test_core.py` | 网络、控制器惯性补偿、接触历史、TOA 归一化、buffer、PPO 更新和日志测试。 |
| `pyproject.toml` | Python 包信息、依赖及打包配置。 |
| `.gitignore` | 排除运行产物、缓存等；部分本地 shell 脚本也被忽略。 |
| `.gitattributes` | Git 文件属性配置。 |
| `THIRD_PARTY_NOTICES.md` | 控制器及 SRU 实现的第三方来源和许可证。 |
| `runs/` | 训练日志、checkpoint、评估统计和 debug 回放，通常不纳入 Git。 |

HXY 的训练参数可直接写在本地 shell 脚本中。脚本需先激活 Isaac Lab 环境；本地 shell 脚本可能包含机器专用设置，使用前检查其内容。

## 3. 如何训练

### 先进行短训练检查

激活 Isaac Lab 环境后，可先运行独立的场景检查。障碍场地保留真实场地尺寸与障碍物数量，只缩小地图池，用于确认网格生成、碰撞和深度相机都正常：

```bash
python scripts/check_dfs_scene.py --headless --device cuda:0
python scripts/check_dfs_scene.py --headless --device cuda:0 --scene-type obstacles
```


以下配置用于确认环境能创建、采样和更新，不用于比较模型性能：

```bash
python scripts/train.py \
  --headless --device cuda:0 --network-device cuda:0 \
  --num-envs 2 --total-timesteps 1024 \
  --rollout-steps 32 --batch-size 32 --sequence-length 8 \
  --recurrent-type sru-lstm --maze-size 7 --train-map-count 2 --eval-map-count 2 --goals-per-map 2 --toa-resolution 0.2
```

### 场景类型

`--scene-type` 选择地图生成器，两种场景共用同一套 TOA、评估、回放和 checkpoint 校验流程。

| 场景 | 场地 | 内容 |
| --- | --- | --- |
| `dfs`（默认） | `maze_size × maze_cell_size` | DFS 迷宫加随机拆墙。 |
| `obstacles` | `arena_size` | 空地围墙内随机摆放 60 个半径 0.3 m 的圆柱与 10 个图内开口同向、图间朝向随机的 U 形障碍。 |

障碍场地默认 30×30 米（`arena_size=30`，均分给 `maze_size=15` 个格子，每格 2 m 决定围墙厚度与起点采样环）。U 形障碍是背墙加两条平行臂，开口 2.4 m 宽、臂长 1.5 m（即开口深度，外形总深 1.8 m），每张地图独立在 360° 范围内随机采样一个开口方向，图内所有 U 形障碍使用同一朝向（同种子可复现）；开口够宽可以进入，所以它既是绕行障碍，也是需要退出的死胡同。圆柱先在允许区域内生长到饱和，再随机抽取所需数量，避免布局围着单个种子点聚成一团、对侧留出空区。摆放用泊松盘采样加精确成对测距，保证任意两个障碍物之间、障碍物与围墙之间至少留出 `obstacle_min_separation`（默认 1.2 m）的表面间距，因此场地内每条缝隙都宽于无人机直径加安全余量，自由空间保持单连通，起点到任意目标都存在可行路径。障碍物高度等于 `arena_height`，高于 2 m 的飞行高度，无人机无法越过。

训练与评估障碍场地（评估沿用 checkpoint 内保存的同一套场景参数）：

```bash
python scripts/train.py \
  --headless --device cuda:0 --network-device cuda:0 \
  --scene-type obstacles --arena-size 30 \
  --algorithm recurrent_ppo --recurrent-type sru-lstm \
  --num-envs 6 --total-timesteps 70000000 \
  --rollout-steps 512 --batch-size 512 --sequence-length 64 \
  --toa-resolution 0.1 --log-dir runs
```

摆放前可以先看随机布局，该脚本不需要 Isaac Sim，只依赖 NumPy 和 Pillow：

```bash
python scripts/preview_scene.py --scene-type obstacles --maps 4
```

每个子种子摆不下时自动换种子重试，全部失败才报错。场地太小或障碍物太多时，报错会给出实际数量、尺寸和间距，按提示缩小障碍物或放大场地即可。障碍物数量与尺寸的可行上限受最小间距约束：默认保留 60 个半径 0.3 m 的圆柱，实际容量随 U 形布局变化，改动前建议用 `scripts/preview_scene.py` 先确认。

### 正式训练

循环 PPO 示例：

```bash
python scripts/train.py \
  --headless --device cuda:0 --network-device cuda:0 \
  --algorithm recurrent_ppo --recurrent-type sru-lstm \
  --num-envs 6 --total-timesteps 70000000 \
  --rollout-steps 512 --batch-size 512 --sequence-length 64 \
  --toa-resolution 0.1 --log-dir runs
```

`--recurrent-type` 可选 `sru-lstm`、`sru-gru`、`sru-lstm-gate`、`lstm`。未指定算法时默认使用循环 PPO，未指定循环单元时默认使用 `sru-lstm`。

普通 PPO 使用以下命令，**不传 `--recurrent-type`**：

```bash
python scripts/train.py \
  --headless --device cuda:0 --network-device cuda:0 \
  --algorithm ppo --num-envs 6 --total-timesteps 70000000 \
  --rollout-steps 512 --batch-size 512 --toa-resolution 0.1
```

### 常用训练参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--num-envs` | `4` | 并行仿真实例数。 |
| `--total-timesteps` | `70000000` | 所有环境累计的 transition 目标数，不是 episode 数；一次向量步产生 `num_envs` 个 transition。 |
| `--rollout-steps` | `512` | 每次更新前，每个环境采集的策略步数。 |
| `--batch-size` | `512` | 训练批次大小，不能超过 `rollout_steps × num_envs`。 |
| `--sequence-length` | `64` | 循环 PPO 的训练序列长度。 |
| `--seed` | `123` | 统一控制地图生成、回合采样和网络随机种子；各部分仍使用独立随机流。原 `--maze-seed` 参数已移除。 |
| `--train-map-count` / `--eval-map-count` | `64` / `16` | 训练与独立评估的地图数量，内容不重叠。 |
| `--goals-per-map` | `8` | 每张地图预采样并缓存 TOA 的目标数量。 |
| `--maze-size` / `--maze-cell-size` | `15` / `4.0` | DFS 场景：包含外墙的奇数网格尺寸及每格米数；默认地图为 60×60 米。 |
| `--maze-wall-removal-probability` | `0.01` | DFS 后每条剩余内部隔墙的拆除概率。 |
| `--scene-type` | `dfs` | 场景类型：`dfs` 为 DFS 迷宫，`obstacles` 为随机圆柱与 U 形障碍场地。 |
| `--arena-size` | `30.0` | 障碍场地边长，米；同时决定 TOA 栅格范围（30 米对应 301×301）。 |
| `--cylinder-count` / `--cylinder-radius` | `60` / `0.3` | 障碍场地的圆柱数量与半径，米。 |
| `--u-shape-count` | `10` | 障碍场地的 U 形障碍数量。 |
| `--u-shape-arm-length` / `--u-shape-arm-thickness` / `--u-shape-opening` | `1.5` / `0.3` / `2.4` | U 形障碍臂长（开口深度）、臂厚与开口宽度，米。 |
| `--obstacle-min-separation` | `1.2` | 障碍物之间、障碍物与围墙的最小表面间距，米；必须宽于无人机直径加安全余量。 |
| `--toa-resolution` | `0.1` | 期望 TOA 米制栅格间距；默认 DFS 地图生成 601×601 图，30 米障碍场地生成 301×301 图。 |
| `--toa-cache-dir` | `.cache/dfs_toa` | 可复用磁盘缓存，损坏项自动重建；无法写入时仍用内存缓存。 |
| `--min-start-goal-distance` | `8.0` | 起终点最小水平直线距离，米。 |
| `--safety-margin` / `--spawn-margin` | `0.1` / `0.1` | 障碍膨胀及额外出生余量，米。 |
| `--episode-seconds` | `40` | 回合时长，秒；长绕行路径是否适配需通过短检查评估。 |
| `--toa-normalization-max` | 新训练自动计算 | Critic 的固定 TOA 量程；省略时新训练使用静态地图的可通行最大值，恢复训练沿用 checkpoint。 |
| `--network-device` | 跟随 `--device` | 网络设备；`--device` 指定仿真设备。 |
| `--log-dir` | `runs` | 日志根目录。 |
| `--run-name` | 时间戳 | 自定义运行名；最终目录为 `<时间戳或运行名>_<算法或循环单元>`，同名目录已存在会报错。 |
| `--checkpoint-interval` | `5` | 每隔多少次 PPO 更新保存周期 checkpoint。 |
| `--log-interval` | `1` | 每隔多少次 PPO 更新写 CSV 和 TensorBoard。 |
| `--debug` | 关闭 | 显示导航标记并保存全局 TOA 图；训练时不生成评估 episode 回放。 |

显存不足时先降低并行环境数和 batch size，同时检查上述批次约束。完整 rollout 保存在 CPU，但内存需求仍随环境数和 rollout 长度增加。

每回合先在边界墙内侧最外围一圈可通行格子中采样起点，再从缓存目标池中随机选择可达目标；水平直线距离不足 8 m 的目标会被拒绝。障碍场地额外要求起点位于 U 形开口朝向的一侧、目标位于对侧：按开口方向投影，分别选取方形内边界正负两端最外一格宽的区域，斜向开口对应对角附近的边界带。地图池内各图独立随机朝向，共用同一张地图的并行实例共享该方向。起点满足障碍膨胀和出生余量要求，候选池仅包含至少有一个合格目标的起点。若地图或目标池无法满足条件，启动时报错。

### 错误日志与退出状态

直接运行 `python scripts/train.py ...` 即自动启用日志监控，无需额外使用 `tee`。运行目录在导入 torch、Warp 和 Isaac Lab 前创建，完整 stdout/stderr（包括底层库输出）实时写入 `console.log` 并显示在终端。

| 文件 | 内容 |
|---|---|
| `console.log` | 完整终端输出、Python traceback 和底层崩溃调用栈。 |
| `status.json` | `state`（`running`、`completed`、`interrupted`、`failed`）、开始/结束时间、退出码、进程号、最后记录的训练步数。 |
| `error.json` | 可捕获异常的时间、类型、消息和完整 traceback；后续清理错误不会覆盖原始异常。 |

正常完成返回 0；Ctrl+C 返回 130，SIGTERM 返回 143，并在训练循环中尝试保存中断 checkpoint。启动、训练或关闭失败返回非零退出码。`status.json` 的 `raw_returncode` 保留子进程原始返回值，负数表示被信号终止，`signal` 记录相应信号编号。突然终止时的最后步数从 `progress.csv` 恢复，可能落后于实际执行进度。

独立监控进程可以记录训练子进程被 SIGKILL 终止的状态，但 SIGKILL 无法生成 Python traceback，也无法保存 checkpoint。若监控进程也被系统杀死，状态可能停留在 `running`；结合进程是否仍存活和系统日志判断。日志监控不会自动判定 OOM。

本地 `train_hxy.sh` 会显示每次训练的退出码，失败后继续下一种模型；任一次失败则整个批次返回非零。所有 `.sh` 文件仍只在本地维护，不纳入 Git。

### 中断与恢复

训练期间按 Ctrl+C 会保存 `checkpoints/model_interrupted_<步数>.pt`。恢复示例：

```bash
python scripts/train.py \
  --resume runs/<原运行名>/checkpoints/model_interrupted_<步数>.pt \
  --headless --device cuda:0 --network-device cuda:0 \
  --num-envs 6 --total-timesteps 70000000 \
  --rollout-steps 512 --batch-size 512 --sequence-length 64 \
  --toa-resolution 0.1
```

恢复时加载模型、优化器、训练步数、网络结构和历史最佳回报。算法及循环单元从 checkpoint 推断，不能切换。

随机种子统一为 `seed`。旧 checkpoint 的 `maze_seed` 与 `seed` 相同时自动兼容；两者不同时无法同时保持原地图和回合采样，恢复训练或评估会明确报错，需要使用旧版代码加载，或以统一的 `--seed` 重新训练。

**环境配置完整继承 checkpoint**。CLI 缺省值不会覆盖已保存地图参数；显式更改地图、目标、TOA 或奖励定义会报错。运行设备、环境数、日志位置及缓存路径可以调整。PPO 批次参数仍由本次命令配置；`--total-timesteps` 是包含已训练步数的总目标。恢复会创建新的运行目录。

`--scene-type` 及障碍场地参数同属场景定义，因此**不能**在恢复训练或评估时切换：DFS checkpoint 只能按 `dfs` 恢复，障碍场地 checkpoint 只能按 `obstacles` 恢复，否则报 `checkpoint 环境参数不兼容` 或场景版本不匹配。新增场景参数在旧 checkpoint 中缺失时按默认值处理，旧 DFS 运行仍可正常恢复。障碍场地已升级至 v3（图内开口同向、图间随机，起点在开口侧、目标在对侧；默认 10 个 U 形、开口 2.4 m），v1/v2 障碍场地 checkpoint 不能用于恢复或评估新场景。

checkpoint 保存场景/生成器/TOA 版本、两个地图池及目标摘要和训练 TOA 观测量程。恢复时重新生成并核验，拒绝旧布局 checkpoint；恢复会开启新回合，不保证从中断的物理状态逐帧续演。

场景摘要包含起终点采样规则；采用全图起点采样的旧 checkpoint 无法直接恢复到外围起点采样规则下。

### 训练课程与指标

PPO 使用 SRU 参考导航项目的自适应学习率，不再因 KL 提前停止。默认初始学习率 `1e-3`、`schedule=adaptive`、`target_kl=0.01`（对应参考项目的 `desired_kl`）。每个 minibatch 更新前，用新旧高斯分布参数计算解析 KL：超过 `0.02` 时学习率除以 `1.5`（最低 `1e-5`），处于 `(0, 0.005)` 时乘以 `1.5`（最高 `1e-3`），其他情况不变；填充位置不参与 KL 计算。`schedule=fixed` 或 `target_kl=None` 可关闭调节。每轮均执行配置的全部 epochs；终端、CSV 和 TensorBoard 记录 `optimizer_steps`、`analytic_kl` 和轮末 `learning_rate`，原 `approx_kl` 仍保留作对照。恢复自适应 checkpoint 时沿用优化器保存的学习率；旧固定学习率 checkpoint 切换调度时使用本次配置的初始学习率。

地图池在启动时按种子生成，默认训练 64 张、评估 16 张，每图 8 个目标。重置时随机选择训练地图与目标，并从同一可达分量采样合法起点，初始朝向目标、速度清零。目标坐标不抖动。场景只包含平地和墙体，碰撞与相机使用同一静态网格。

训练和评估的碰撞力阈值全程固定为 0.1 N，接触惩罚从第一步起使用完整强度，不随训练进度变化。接触力归一化使用固定阈值。各奖励项按权重直接加总，不再统一缩放：默认成功奖励为 +3，达到碰撞阈值时的接触惩罚为 −1，TOA 进度项的权重为 5。日志中的 `contact_penalty_scale` 保留为常数 1。恢复旧 checkpoint 时将碰撞阈值统一迁移到 0.1 N，并忽略已废弃的最小碰撞阈值和接触惩罚渐增字段；旧模型也使用新的奖励定义，其 Critic 和优化器状态需要适应新尺度，历史回报和最佳回报不可直接与新回报比较。

接触检测和接触惩罚都取当前策略步全部物理子步、全部机体部件的最大接触力，不再只看最后一个子步。历史窗口随 `control_decimation` 调整，避免漏掉碰撞后弹开的短暂接触，也避免重复计入上一策略步。回合重置会清零当前及历史动作缓存。

Critic 的 16×16 TOA 输入使用训练池的统一有效最大值归一化到 [-1,1]，不可达和越界像素为 +1；独立评估沿用训练尺度。Actor 不接收 TOA。

TOA 进度奖励为 `weight × (T_previous − T_current) / T_start`。每回合分母固定为起点到目标点的 FMM 到达时间，包含近墙降速；不使用旧差分裁剪。首步以起点 TOA 建立历史，无效采样切断差分。**目标点 TOA 为零，成功使用 0.4 m 水平半径**；成功时不强制清零剩余 TOA、不补发进度，成功奖励独立计算。

TOA 数组按地图/目标共享，不随环境数复制。默认训练池 float32 TOA 约 176.96 MiB，另有 CPU 起点候选、几何、临时数组及 GPU 副本；启动打印实际池大小和 TOA 分布。HXY 的启动时间、相机/碰撞和 2048 环境规模需要单独验收。

当前训练达到碰撞阈值即终止。训练日志按“成功优先、其次碰撞、其余超时”分类，越界也可能被归为超时。TensorBoard 的 `Metrics/*` 使用最近 100 个已完成 episodes；这些训练指标与后文独立评估的独立地图池及分类口径不同。

## 4. 训练产物在哪里、如何查看

默认目录结构如下，未启用 debug 或未运行评估时，相应目录可能尚不存在：

```text
runs/<时间戳或运行名>_<算法或循环单元>/
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
│       ├── summary.json        # 分地图与整体统计
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
| `progress/*` | CSV 对应指标，包括训练步数、接触惩罚比例和接触力阈值。 |

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

评估直接加载网络，不创建 PPO 训练器、优化器或 rollout buffer；每步仅执行 Actor 前向，不计算 Critic。环境仍生成 TOA 以计算回报。

评估使用 checkpoint 定义的独立地图池，默认 16 张，碰撞阈值固定为 **0.1 N**。地图和目标池的生成与训练使用相同版本及参数，内容摘要必须匹配 checkpoint。

默认每张地图统计 100 个完整回合，共 1600 个。回合开始即预留配额，单个环境也能轮转覆盖全部地图；已分配完配额后的空闲回合不计入统计。

先用小样本检查整个评估流程：

```bash
python scripts/play.py runs/<运行名>/checkpoints/model_best.pt \
  --headless --device cuda:0 --num-envs 6 --episodes-per-map 2
```

默认评估池下以上只统计 32 个回合，不适合用于正式性能比较。

| 参数 | 含义 |
| --- | --- |
| `checkpoint` | 必填的位置参数，指定完整训练 checkpoint。 |
| `--episodes-per-map` | 每张地图的完整回合目标数，默认 100。 |
| `--episode-seconds` | 单回合仿真时限（秒），必须为有限正数；默认沿用 checkpoint，传入 `160` 可覆盖为 160 秒。 |
| `--num-envs` | 并行实例数，默认 6，至少 1。 |
| `--debug` | 按开始顺序预选每张地图第 50 回合附近的 4 个完整回合保存回放；默认第 49–52 个。 |
| `--steps` | 可选的向量步数上限；达到后提前退出，不保证回合配额完成。 |
| `--output-dir` | 自定义统计输出目录，必须是尚不存在的新目录；不改变 debug 保存路径。 |
| `--device`、`--network-device` | 仿真与网络设备。 |
| `--headless` | 关闭仿真窗口；省略可观察实时仿真。 |

Ctrl+C、关闭仿真窗口或达到 `--steps` 上限都会结束评估并保存已有完整回合统计。未完成配额时 `complete=false`；未结束的回合不计入分母，也不保存其 debug 回放。

评估先读取 checkpoint 保存的环境配置，因此修改 `EnvConfig.episode_seconds` 的默认值不会改变已有模型的评估时限。需要延长时，在评估命令中显式添加 `--episode-seconds 160`；启动日志会显示实际时限，`summary.json` 也会记录该值。新参数仅影响重新启动的评估。

## 6. 如何查看评估结果

### 找到本次输出

程序启动时打印评估结果目录。标准 checkpoint 路径 `runs/<运行名>/checkpoints/model.pt` 对应：

```text
runs/<运行名>/evaluation/eval_<时间戳>/
```

`evaluation/` 与 `checkpoints/`、`debug/` 并列，每次评估创建新的时间戳子目录。如果 checkpoint 的父目录不叫 `checkpoints`，则以 checkpoint 所在目录作为输出根目录；移动了模型或使用自定义 checkpoint 目录时，可通过 `--output-dir` 明确指定统计位置。

### 查看成功率、碰撞率、超时率

评估完成或提前结束时，终端会以表格展示所有评估地图及整体的回合数、三类结局的数量和比例。
比例显示为保留两位小数的百分比；尚无完整回合时显示 `—`。JSON 中仍保存原始精度的数据。

```bash
python -m json.tool runs/<运行名>/evaluation/eval_<时间戳>/summary.json
```

`summary.json` 的主要字段如下：

| 字段 | 含义 |
| --- | --- |
| `checkpoint` | 本次评估模型的绝对路径。 |
| `num_envs`、`collision_force_threshold` | 实际评估实例数和碰撞阈值。 |
| `episode_seconds`、`policy_dt` | 本次评估实际采用的单回合仿真时限和策略步长，单位为秒。 |
| `scene_manifest` | 地图种子、摘要、目标栅格和场景版本。 |
| `episodes_per_map` | 每张地图目标回合数。 |
| `complete` | 所有评估地图是否全部完成目标配额。 |
| `maps.eval_000` 等 | 各地图的计数和比例。 |
| `overall` | 已计入回合的整体计数和比例。 |

每张地图及 `overall` 都包含 `episodes`、`success`、`collision`、`timeout` 和对应的 `success_rate`、`collision_rate`、`timeout_rate`。比例取值为 0–1，例如 `0.85` 表示 85%；没有已完成回合时比例为 `null`。

评估结局互斥，按以下优先级判定：

1. 接触力达到 0.1 N 或触发越界：碰撞。
2. 未碰撞且进入目标范围：成功。
3. 以上均未发生且达到回合时限：超时。

默认目标水平距离阈值为 0.4 m，回合时限由 checkpoint 的环境配置决定，可用 `--episode-seconds` 覆盖。场地四周有墙；代码仍保留越界判断，并在评估中归入碰撞。

三项率的分母都是**该范围内已计入的完整回合数**。非空统计的三项率之和为 1，碰撞率就是发生碰撞的回合数除以完整回合数。完整默认评估中，每张地图应为 100，整体为 1600；提前退出时整体按实际样本数加权，不能把它当作所有评估地图等量评估结果。

### 查看逐回合记录

用表格软件或文本编辑器打开同目录的 `episodes.csv`：

| 列 | 含义 |
| --- | --- |
| `maze` | 布局名称。 |
| `env_id` | 并行仿真实例编号，不是布局编号。 |
| `episode` | 该环境的回合编号，从 0 开始。 |
| `map_hash`、`goal_id`、`start_xy` | 地图内容摘要、目标索引和局部出生坐标，用于复查。 |
| `episode_start_toa` | 本回合进度奖励的固定分母。 |
| `outcome` | `success`、`collision` 或 `timeout`。 |
| `steps` | 本回合的策略步数，不是物理子步数。 |
| `return` | 本回合累计环境奖励。 |

程序在有回合完成时刷新统计文件，可以在评估运行中查看。正式比较时，先核对 `complete`、checkpoint、地图摘要和每张地图样本数，再比较比例。

### 查看 debug 视频与轨迹

评估时启用：

```bash
python scripts/play.py runs/<运行名>/checkpoints/model_best.pt \
  --headless --device cuda:0 --num-envs 6 --debug
```

打开启动日志打印的 `debug/play_<时间戳>/index.html`，点击某个回合进入回放。网页包含深度视频、请求/执行动作、实际运动状态和俯视轨迹，支持播放、调速、逐帧及拖动定位。

保存规则是**按开始顺序预选每张地图第 50 回合附近的 4 个完整回合，默认评估池共最多 64 个**，不是每个并行实例 4 个。每张地图评估至少 100 回合时，选择第 **49–52** 个；不足 100 回合时，选择评估范围中间的回合。同一步开始的回合按环境编号排序。目标数为 N 时，保存数量 K=min(4,N)，窗口 W=min(N,100)，起始序号为 floor((W-K)/2)+1，连续选择 K 个。

选择在回合开始时确定，不根据成功、碰撞或结束快慢筛选；未选中回合不采集回放数据。回放索引和 `telemetry.json` 的 `map_start_episode` 标明地图内从 1 开始的序号，它与 CSV 中环境内的回合编号 `episode` 不同。4 个回合仍只是小样本，不保证覆盖三种结局。

若统计配额先满而选中的回放尚未结束，程序会继续仿真至这些回放完整结束，额外完成的回合不计入统计。Ctrl+C、关闭窗口或 `--steps` 上限仍会提前停止；中断时未完成的回放不保存。

| 文件 | 如何使用 |
| --- | --- |
| `index.html` | 浏览器直接打开，查看同步回放。 |
| `depth.mp4` | 灰度深度视频；默认原始分辨率为 256×192，保留相机分辨率，按策略频率采样。 |
| `depth_raw.npz` | 原始米制浮点深度（保留 NaN/Inf）、时间、动作和状态，可用 NumPy 读取。 |
| `telemetry.json` | 本回合布局、结局、障碍物几何及逐步动作/状态/位置数据。 |

灰度视频使用固定深度量程；数值分析应使用 NPZ 原始深度。状态中的 `vx/vy` 为机体系速度（m/s），`w` 为机体系 Z 轴角速度（rad/s）。每帧对应动作执行后的状态，包含自动重置前的终止帧。

在远端运行时，把整个 `play_<时间戳>` 文件夹复制到本地再打开 HTML，保持 HTML 与 MP4 的相对路径。全局 TOA PNG/NPZ 另存于 `debug/`，不占各地图的 episode 的配额。

## 7. 修改算法与导出模型

主要数据流为：

```text
Isaac Sim / PhysX → NavigationEnv → IsaacLabWrapper → PPO
深度图 + 8 维机体状态 → Actor → 目标前向速度、侧向速度、偏航角速度
深度图 + 机体状态 + 局部 TOA → Critic → 状态价值
策略动作 → Lee 控制器 → 无人机物理运动
```

默认物理频率为 120 Hz，每 5 个物理步执行一次策略决策，即 24 Hz。修改地图生成看 `envs/maze.py`，修改奖励/终止看 `envs/navigation.py`，修改网络看 `models/actor_critic.py`、`models/recurrent.py`、`models/depth_encoder.py` 和 `NetworkConfig`。

### 网络结构（对齐 tutorial）

默认结构对齐 `~/tutorial` 的 `ActorFeaturesExtractor`、`CriticFeaturesExtractor` 和 `sb3_ppo_cfg.yaml`：

| 分支 | 卷积结构 | 池化与投影 |
| --- | --- | --- |
| Actor 深度图 | `1→8`（2×2）、`8→16`（3×3）、`16→32`（3×3） | 池化到 2×3，展平 192，线性层输出 192 |
| Critic 深度图 | `1→8`（2×2）、`8→16`（3×3） | 池化到 2×3，展平 96，线性层输出 192 |
| Critic TOA | `1→8`（2×2）、`8→16`（3×3） | 池化到 2×3，展平 96，线性层输出 192 |

各卷积步长为 1、无填充，后接 LeakyReLU。Actor 与 Critic 分别持有 `8→192` 的机体状态线性层：Actor 将深度图与状态特征相加，Critic 将深度图、状态与 TOA 特征相加。融合输出为 **192 维**，直接送入各自的循环网络，默认 SRU-LSTM、隐藏维度 **256**、一层；没有额外循环前投影。循环后的 Actor MLP 为 **[512,512]**，Critic MLP 为 **[256,256]**，激活函数均为 LeakyReLU，分别输出三维高斯动作均值和状态价值。

CNN、状态投影、循环网络与策略头均参与 PPO 更新。rollout buffer 保存原始 `camera`、`robot_state` 和 `critic_toa`；训练 minibatch 重新运行可训练 CNN，不缓存其特征。Actor 不读取 TOA，部署仅需深度图和机体状态。`feature_dim` 默认 192，可独立于相机分辨率调整；Actor 的三次卷积要求深度图高宽至少为 6。

为限制大 minibatch 的显存峰值，三个 CNN 均按 `network.cnn_chunk_size` 分块处理图像（默认 128 张），更新时默认启用 `network.cnn_gradient_checkpointing`：反向传播重算各块的卷积激活，以增加计算换取显存。此设置保持网络结构、PPO batch size 和每次更新的梯度语义；仅分块而不重算，仍会保留整个 minibatch 的激活。采样和评估无需梯度时只分块，不重算。旧 `tutorial_cnn_v1` checkpoint 缺少这两个字段时采用上述默认值，权重仍可直接加载。

此次只恢复网络结构，深度观测仍保留当前米制语义：相机原图先做非有限值清理及 4×4 最小池化，再将量程外读数置 0，默认得到 48×64 深度图。奖励与 PPO 的 KL 自适应学习率设置保留。

新 checkpoint 的 `network.architecture` 为 `tutorial_cnn_v1`，导出版本为 **4**。旧 RegNet、带循环前投影或早期 CNN checkpoint 缺少当前结构标记，不能直接恢复、评估或导出；需要使用对应历史代码或重新训练，避免混用网络与输入语义。

Lee 控制器在机体系力矩中加入 `Ω × (JΩ)`，补偿刚体方程的陀螺耦合。接触历史检测及控制器修复会改变旧 checkpoint 的仿真轨迹和结局；与修复前的评估结果比较时，请同时记录代码版本。

导出 Actor：

```bash
python scripts/export_actor.py \
  runs/<运行名>/checkpoints/model_best.pt \
  runs/<运行名>/actor.pt
```

输出为项目定义的 PyTorch 权重与配置字典，不是 TorchScript 或 ONNX，也不是 `play.py` 接受的完整训练 checkpoint。

导出产物的 `version` 为 4，`network` 段记录当前网络配置，`observation` 段记录深度量程、分辨率和动作边界。部署侧输入 `camera` 仍为**米制深度**（0 表示无效）；`actor` 段包含 `actor_encoder.camera.*`、状态投影、循环网络、MLP 与动作头权重，不包含 Critic 或历史 RegNet 编码器。

## 8. 验证与第三方来源

不启动 Isaac Sim 的单元测试：

```bash
python -m unittest discover -s tests -v
```

这些测试覆盖纯 Python / PyTorch 逻辑和回放编码，不能替代真实仿真验证。安装后可先执行短训练，再用生成的 checkpoint 做每张地图 2 个回合的评估检查。

第三方实现来源及许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
