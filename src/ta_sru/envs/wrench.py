"""跨 Isaac Lab 版本的机体外力施加。

项目需要在两类 Isaac Lab 上运行：

- 1.4 以及 2.0–2.3.1：只有 ``Articulation.set_external_force_and_torque``；
- 2.3.2 起：推荐 ``permanent_wrench_composer.set_forces_and_torques``，旧方法已被标记废弃。

两者写入的都是「每个物理步持续生效、``reset()`` 时清零」的常驻 wrench 缓冲，
并由 ``write_data_to_sim()`` 在仿真步前提交给 PhysX，语义一致。这里按运行时能力
选择接口，不比较版本号，也不依赖私有属性，因此训练脚本不必再在启动前用 sed
改写源码。
"""

from __future__ import annotations

from typing import Any, Sequence


def set_body_wrench(
    asset: Any,
    forces: Any,
    torques: Any,
    body_ids: Sequence[int] | None = None,
) -> None:
    """把机体系外力/力矩写入 asset 的常驻缓冲。

    Args:
        asset: Isaac Lab 的 ``Articulation`` / ``RigidObject``。
        forces: 机体系外力，形状 ``(num_envs, len(body_ids), 3)``。
        torques: 机体系力矩，形状 ``(num_envs, len(body_ids), 3)``。
        body_ids: 受力刚体索引；``None`` 表示全部刚体。

    Raises:
        AttributeError: 新旧两套接口都不存在。
    """
    composer = getattr(asset, "permanent_wrench_composer", None)
    setter = getattr(composer, "set_forces_and_torques", None)
    if callable(setter):
        # is_global 在两套接口里都默认 False（机体系），1.x 根本没有该参数，
        # 所以这里显式省略，只传三方共有的参数。
        setter(forces=forces, torques=torques, body_ids=body_ids)
        return

    legacy = getattr(asset, "set_external_force_and_torque", None)
    if callable(legacy):
        legacy(forces, torques, body_ids=body_ids)
        return

    raise AttributeError(
        "asset 既没有 permanent_wrench_composer.set_forces_and_torques，"
        f"也没有 set_external_force_and_torque：{type(asset).__name__}"
    )
