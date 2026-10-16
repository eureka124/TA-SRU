"""无需启动 Isaac Sim 的 Actor 导出。"""

from dataclasses import asdict
from pathlib import Path

import torch

from ta_sru.config import NetworkConfig
from ta_sru.models.actor_critic import (
    AsymmetricRecurrentActorCritic,
    require_supported_action_transform,
)

# Actor 的独立 CNN 已包含在 actor_encoder 中，不导出 Critic 分支。
ACTOR_PREFIXES = (
    "actor_encoder.",
    "actor_recurrent.",
    "actor_mlp.",
    "action_mean.",
)
ENV_KEYS = (
    "depth_height",
    "depth_width",
    "depth_min_distance",
    "depth_max_distance",
    "depth_invalid_distance",
    "physics_dt",
    "control_decimation",
    "action_low",
    "action_high",
)
# 版本 3：恢复 tutorial 可训练小 CNN 和 192 维融合；深度输入仍为米制深度。
# 版本 4：observation 段移除从未被相机读取的 camera_horizontal_fov。
ARTIFACT_VERSION = 4


def export_actor(checkpoint_path, output_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require_supported_action_transform(checkpoint.get("action_transform"))
    config = checkpoint["config"]
    network = NetworkConfig.from_checkpoint(config["network"])
    network.validate()
    algorithm = config.get("algorithm", "recurrent_ppo")
    if (algorithm == "ppo") != (network.recurrent_type == "none"):
        raise ValueError("算法与网络结构不一致")
    model = AsymmetricRecurrentActorCritic(network)
    model.load_state_dict(checkpoint["policy"], strict=True)
    artifact = {
        "format": "ta_sru_actor",
        "version": ARTIFACT_VERSION,
        "algorithm": algorithm,
        "network": asdict(network),
        "observation": {key: config["env"][key] for key in ENV_KEYS},
        "actor": {
            key: value.cpu()
            for key, value in model.state_dict().items()
            if key.startswith(ACTOR_PREFIXES)
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    return output_path
