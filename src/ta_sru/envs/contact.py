"""接触历史的纯张量计算，可在不启动仿真时验证。"""

import torch


def peak_contact_force(history: torch.Tensor, substeps: int) -> torch.Tensor:
    """返回当前策略步内各机体部件的最大接触力，历史按新到旧排列。"""
    if substeps <= 0 or history.ndim != 4 or history.shape[1] < substeps:
        raise ValueError("接触历史必须为 (N,T,B,3)，且覆盖全部物理子步")
    if history.shape[-1] != 3:
        raise ValueError("接触力必须包含三个空间分量")
    return torch.linalg.vector_norm(history[:, :substeps], dim=-1).amax(dim=(1, 2))
