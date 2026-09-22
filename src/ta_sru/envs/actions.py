"""归一化动作与物理动作之间的映射。

放在独立模块里，便于在不启动 Isaac Sim 的情况下验证这段纯张量逻辑。策略输出的
是 [-1, 1] 的归一化动作（tanh 压缩后的高斯采样），环境在这里把它映射到
``action_low``/``action_high`` 的物理取值范围。
"""

from __future__ import annotations

import torch


def unit_to_action(
    unit_actions: torch.Tensor, low: torch.Tensor, high: torch.Tensor
) -> torch.Tensor:
    """把 [-1, 1] 的归一化动作线性映射到 [low, high]。

    -1 对应 low，0 对应区间中点，+1 对应 high。越界输入按边界饱和，仅用于防御
    异常调用；策略的输出经 tanh 压缩，正常路径不会越界。
    """

    unit = unit_actions.clamp(-1.0, 1.0)
    return low + (high - low) * (unit + 1.0) * 0.5


__all__ = ["unit_to_action"]
