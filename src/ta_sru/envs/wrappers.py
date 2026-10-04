"""Isaac Lab 与本项目训练器之间的轻量 wrapper。

这里没有继承或导入 SB3 wrapper。Isaac Lab 环境本身会在 ``step`` 内自动重置，本类
只负责动作边界、设备转换、终止信息和 NumPy 接口。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from ta_sru.envs.navigation import NavigationEnv


class IsaacLabWrapper:
    def __init__(self, env: NavigationEnv) -> None:
        self.env = env
        self.num_envs = env.num_envs
        self.sim_device = torch.device(env.device)
        self.action_low = np.asarray(env.task.action_low, dtype=np.float32)
        self.action_high = np.asarray(env.task.action_high, dtype=np.float32)

    @staticmethod
    def _host_observation(
        observation: dict[str, torch.Tensor],
    ) -> dict[str, np.ndarray]:
        return {
            key: value.detach().cpu().numpy().astype(np.float32, copy=False)
            for key, value in observation.items()
        }

    def reset(self) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
        observation, _ = self.env.reset()
        return self._host_observation(observation["policy"]), [
            {} for _ in range(self.num_envs)
        ]

    def set_training_progress(self, elapsed_steps: int) -> None:
        """将训练器的累计 transition 数同步给环境课程。"""

        self.env.set_training_progress(elapsed_steps)

    @property
    def scene_manifest(self) -> dict:
        return self.env.scene_manifest

    @property
    def training_curriculum_contact_scale(self) -> float:
        return self.env.training_curriculum_contact_scale

    @property
    def training_curriculum_contact_force_threshold(self) -> float:
        return self.env.training_curriculum_contact_force_threshold

    def step(
        self, actions: np.ndarray | torch.Tensor
    ) -> tuple[
        dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
    ]:
        if isinstance(actions, torch.Tensor):
            action_tensor = actions.to(device=self.sim_device, dtype=torch.float32)
        else:
            bounded = np.clip(
                np.asarray(actions, dtype=np.float32), self.action_low, self.action_high
            )
            action_tensor = torch.from_numpy(bounded).to(self.sim_device)
        observation, reward, terminated, truncated, extras = self.env.step(
            action_tensor
        )
        host_observation = self._host_observation(observation["policy"])
        terminated_host = terminated.detach().cpu().numpy()
        truncated_host = truncated.detach().cpu().numpy()
        done_ids = np.flatnonzero(terminated_host | truncated_host)

        terminal_batch = extras.get("terminal_observation")
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        scalar_keys = (
            "success",
            "collided",
            "time_out",
            "outside",
            "distance_to_goal",
            "map_id",
            "goal_id",
            "evaluation_active",
            "episode_start_toa",
            "episode_id",
        )
        # 每个批量字段只传回 CPU 一次，避免按环境逐项触发 GPU 同步。
        host_extras = {
            key: extras[key].detach().cpu().numpy()
            for key in scalar_keys
            if isinstance(extras.get(key), torch.Tensor)
        }
        for env_id in range(self.num_envs):
            for key, value in host_extras.items():
                infos[env_id][key] = value[env_id].item()
        for env_id in done_ids:
            infos[env_id]["start_xy"] = (
                extras["start_xy"][env_id].detach().cpu().tolist()
            )
            if isinstance(terminal_batch, dict):
                infos[env_id]["terminal_observation"] = {
                    key: value[env_id]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                    for key, value in terminal_batch.items()
                }
            else:
                # 防御性回退；正常路径始终使用环境在 reset 前保存的观测。
                infos[env_id]["terminal_observation"] = {
                    key: value[env_id].copy() for key, value in host_observation.items()
                }
            infos[env_id]["TimeLimit.truncated"] = bool(
                truncated_host[env_id] and not terminated_host[env_id]
            )
            infos[env_id]["is_success"] = bool(infos[env_id].get("success", False))
        return (
            host_observation,
            reward.detach().cpu().numpy(),
            terminated_host,
            truncated_host,
            infos,
        )

    def close(self) -> None:
        self.env.close()


# 保留短名称，让算法层无需知道具体仿真后端。
AutoResetWrapper = IsaacLabWrapper

__all__ = ["AutoResetWrapper", "IsaacLabWrapper"]
