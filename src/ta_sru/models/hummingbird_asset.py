"""本地 Hummingbird USD 的 Isaac Lab 资产配置。

此模块必须在 ``AppLauncher`` 启动 Isaac Sim 后导入。
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


HUMMINGBIRD_USD_PATH = (
    Path(__file__).resolve().parents[3] / "assets" / "hummingbird" / "hummingbird.usd"
)

HUMMINGBIRD_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(HUMMINGBIRD_USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0),
        joint_pos={".*": 0.0},
        joint_vel={
            "rotor_0_joint": 200.0,
            "rotor_1_joint": -200.0,
            "rotor_2_joint": 200.0,
            "rotor_3_joint": -200.0,
        },
    ),
    actuators={
        "rotor_visualization": ImplicitActuatorCfg(
            joint_names_expr=[".*"], stiffness=0.0, damping=0.0
        )
    },
)

__all__ = ["HUMMINGBIRD_CFG", "HUMMINGBIRD_USD_PATH"]
