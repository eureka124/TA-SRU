"""Isaac 环境的延迟导入入口。

延迟导入保证配置、网络和 CPU buffer 的单元测试不需要先启动 Isaac Sim。
"""

from __future__ import annotations


def __getattr__(name: str):
    if name in {"AutoResetWrapper", "IsaacLabWrapper"}:
        from ta_sru.envs.wrappers import IsaacLabWrapper

        return IsaacLabWrapper
    if name in {"IsaacNavigationEnvCfg", "NavigationEnv", "make_isaac_env_cfg"}:
        from ta_sru.envs.navigation import (
            IsaacNavigationEnvCfg,
            NavigationEnv,
            make_isaac_env_cfg,
        )

        return {
            "IsaacNavigationEnvCfg": IsaacNavigationEnvCfg,
            "NavigationEnv": NavigationEnv,
            "make_isaac_env_cfg": make_isaac_env_cfg,
        }[name]
    raise AttributeError(name)


__all__ = [
    "AutoResetWrapper",
    "IsaacLabWrapper",
    "IsaacNavigationEnvCfg",
    "NavigationEnv",
    "make_isaac_env_cfg",
]
