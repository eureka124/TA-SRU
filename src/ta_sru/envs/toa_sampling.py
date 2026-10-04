"""共享 TOA 的安全插值和按回合起点归一化的奖励。"""

from __future__ import annotations

import torch


def sample_toa(
    maps: torch.Tensor, map_ids: torch.Tensor, points: torch.Tensor, extent: float
) -> tuple[torch.Tensor, torch.Tensor]:
    size = maps.shape[-1]
    grid = (points + extent) * ((size - 1) / (2 * extent))
    valid = (
        torch.isfinite(grid).all(dim=-1)
        & (grid >= 0).all(dim=-1)
        & (grid <= size - 1).all(dim=-1)
    )
    grid = torch.nan_to_num(grid).clamp(0, size - 1)
    low = grid.floor().long()
    high = (low + 1).clamp(max=size - 1)
    weight = grid - low
    values = torch.zeros_like(grid[..., 0])
    flat = maps.reshape(-1)
    offsets = map_ids[:, None] * size * size
    for xi, yi, w in (
        (low[..., 0], low[..., 1], (1 - weight[..., 0]) * (1 - weight[..., 1])),
        (high[..., 0], low[..., 1], weight[..., 0] * (1 - weight[..., 1])),
        (low[..., 0], high[..., 1], (1 - weight[..., 0]) * weight[..., 1]),
        (high[..., 0], high[..., 1], weight[..., 0] * weight[..., 1]),
    ):
        sample = flat[offsets + yi * size + xi]
        finite = torch.isfinite(sample)
        valid &= finite | (w == 0)
        values += torch.where(finite, sample, 0) * w
    return torch.where(valid, values, torch.inf), valid


def normalized_progress(
    previous: torch.Tensor, current: torch.Tensor, start: torch.Tensor
) -> torch.Tensor:
    valid = (
        torch.isfinite(previous)
        & torch.isfinite(current)
        & torch.isfinite(start)
        & (start > 1e-6)
    )
    denominator = torch.where(valid, start, 1)
    delta = torch.where(valid, previous, 0) - torch.where(valid, current, 0)
    return delta / denominator
