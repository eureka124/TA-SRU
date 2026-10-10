"""深度观测预处理：相机读数整理成深度编码器的输入。

编码器是在**米制深度**上预训练的 VAE，因此这里不做 [-1, 1] 归一化：量程外与
非有限的读数统一置 0（0 表示"无效"），其余保留米数，与参考工程的推理路径一致。

三个步骤的先后顺序不能调换：非有限值必须在池化**之前**替换掉，否则一次 NaN
就会沿着最小池化污染掉整个池化块。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# 相机分辨率相对网络输入的倍数，同时也是下采样池化的核大小。
DEPTH_DOWNSAMPLE = 4


def sanitize_depth(
    depth: torch.Tensor, *, invalid_distance: float
) -> torch.Tensor:
    """把非有限读数替换成必然越界的常数。

    ``invalid_distance`` 取得比量程上限大，替换后的像素会在下一步被当作无效值
    置 0；放在池化之前做是因为最小池化遇到 NaN 会把整个块变成 NaN。
    """

    return torch.nan_to_num(
        depth, nan=invalid_distance, posinf=invalid_distance, neginf=0.0
    )


def downsample_depth(
    depth: torch.Tensor, factor: int = DEPTH_DOWNSAMPLE
) -> torch.Tensor:
    """按 ``factor × factor`` 取块内最小深度。

    ``-max_pool2d(-x)`` 即最小池化：块内取最近表面，对细杆、墙沿这类会在一部分
    像素上穿透的障碍是保守取样。
    """

    return -F.max_pool2d(-depth, kernel_size=factor, stride=factor)


def mask_invalid_depth(
    depth: torch.Tensor, *, min_distance: float, max_distance: float
) -> torch.Tensor:
    """把量程外的读数置 0，其余保留米数。

    相机的截断距离就是 ``max_distance``，无返回的射线恰好等于该值，因此这里用
    ``>=`` 而不是 ``>``。
    """

    return torch.where(
        (depth >= max_distance) | (depth < min_distance),
        torch.zeros_like(depth),
        depth,
    )
