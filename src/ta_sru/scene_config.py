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
    # 旧 checkpoint 的阈值课程字段已废弃，恢复时只保留固定阈值。
    values.pop("minimum_contact_force_threshold", None)
    for key, value in overrides.items():
        if value is None:
            continue
        if saved is not None and key not in RUNTIME_FIELDS and value != values.get(key):
            raise ValueError(f"恢复训练不能更改环境参数 {key}")
        values[key] = value
    # 新训练和恢复训练统一使用固定的 0.1 N 碰撞阈值。
    values["collision_force_threshold"] = 0.1
    config = EnvConfig(**values)
    config.validate()
    return config


def validate_resume_config(saved: dict, current: EnvConfig) -> None:
    saved = dict(saved)
    # 旧训练阈值统一迁移到当前固定阈值，其他参数仍严格校验。
    saved["collision_force_threshold"] = 0.1
    for key, value in asdict(current).items():
        if key not in RUNTIME_FIELDS and saved.get(key) != value:
            raise ValueError(f"checkpoint 环境参数不兼容：{key}")
