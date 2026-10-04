"""DFS 场景的配置恢复与兼容性校验。"""

from __future__ import annotations

from dataclasses import asdict

from ta_sru.config import EnvConfig

RUNTIME_FIELDS = {
    "num_envs",
    "debug",
    "debug_output_dir",
    "toa_cache_dir",
    "total_training_steps",
}


def training_env_config(saved: dict | None, overrides: dict) -> EnvConfig:
    """缺省 CLI 不覆盖 checkpoint；仅运行参数允许显式改变。"""
    values = dict(saved) if saved is not None else asdict(EnvConfig())
    for key, value in overrides.items():
        if value is None:
            continue
        if saved is not None and key not in RUNTIME_FIELDS and value != values.get(key):
            raise ValueError(f"恢复训练不能更改环境参数 {key}")
        values[key] = value
    config = EnvConfig(**values)
    config.validate()
    return config


def validate_resume_config(saved: dict, current: EnvConfig) -> None:
    for key, value in asdict(current).items():
        if key not in RUNTIME_FIELDS and saved.get(key) != value:
            raise ValueError(f"checkpoint 环境参数不兼容：{key}")
