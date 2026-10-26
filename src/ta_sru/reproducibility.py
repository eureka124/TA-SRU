"""统一训练中的随机种子与 PyTorch 确定性计算设置。"""

from __future__ import annotations

import os
import random


def seed_torch(seed: int) -> None:
    """在仿真、模型初始化前调用；独立 NumPy Generator 仍需显式传入种子。"""
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed 必须位于 [0, 4294967295]")

    # 当前解释器的哈希种子只能在启动时设置；训练入口负责必要的进程重启。
    os.environ["PYTHONHASHSEED"] = str(seed)
    # 必须在首次 CUDA 运算前设置，保证多流 cuBLAS 运算可复现。
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # 没有确定性实现的算子直接报错，避免静默产生不同训练轨迹。
    torch.use_deterministic_algorithms(True)
