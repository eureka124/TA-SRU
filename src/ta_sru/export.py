"""无需启动 Isaac Sim 的 Actor 导出。"""

from dataclasses import asdict
from pathlib import Path
import torch
from ta_sru.config import NetworkConfig
from ta_sru.models.actor_critic import AsymmetricRecurrentActorCritic

ACTOR_PREFIXES = ("actor_encoder.", "actor_recurrent.", "actor_mlp.", "action_mean.")
ENV_KEYS = (
    "depth_height",
    "depth_width",
    "depth_max_distance",
    "camera_horizontal_fov",
    "physics_dt",
    "control_decimation",
    "action_low",
    "action_high",
)


def export_actor(checkpoint_path, output_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    network = NetworkConfig(**config["network"])
    network.validate()
    algorithm = config.get("algorithm", "recurrent_ppo")
    if (algorithm == "ppo") != (network.recurrent_type == "none"):
        raise ValueError("算法与网络结构不一致")
    model = AsymmetricRecurrentActorCritic(network)
    model.load_state_dict(checkpoint["policy"], strict=True)
    artifact = {
        "format": "ta_sru_actor",
        "version": 1,
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
