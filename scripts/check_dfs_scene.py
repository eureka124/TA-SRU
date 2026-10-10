"""用两个小型环境检查 DFS 共享地图、重置、TOA 和传感器。"""

from __future__ import annotations

import argparse
import traceback
from types import SimpleNamespace

import torch
import warp  # noqa: F401
from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-type",
        choices=("dfs", "obstacles"),
        default="dfs",
        help="要检查的场景类型：dfs 用小型迷宫，obstacles 用真实障碍场地尺寸",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    from pxr import Gf, UsdGeom

    from ta_sru.config import EnvConfig
    from ta_sru.envs import NavigationEnv, make_isaac_env_cfg

    env = None
    try:
        # 两种场景都只缩小地图池和 TOA 分辨率，障碍场地保留真实尺寸与障碍物数量。
        config = EnvConfig(
            num_envs=2,
            train_map_count=2,
            eval_map_count=1,
            goals_per_map=2,
            scene_type=args.scene_type,
            maze_size=7 if args.scene_type == "dfs" else 15,
            toa_resolution=0.2,
            toa_cache_dir=None,
        )
        env = NavigationEnv(make_isaac_env_cfg(config, args.device), config)
        # 老版允许空变换栈；显式覆盖新版场景视图要求，避免漏掉兼容问题。
        map_xform = UsdGeom.Xformable(env.sim.stage.GetPrimAtPath("/World/DfsMaps"))
        assert [str(op.GetOpName()) for op in map_xform.GetOrderedXformOps()] == [
            "xformOp:translate",
            "xformOp:orient",
            "xformOp:scale",
        ]
        assert map_xform.GetLocalTransformation() == Gf.Matrix4d(1.0)
        obs, _ = env.reset()
        assert torch.isfinite(obs["policy"]["critic_toa"]).all()
        assert torch.all(env.episode_start_toa > 0)
        assert torch.allclose(env.previous_toa, env.episode_start_toa)
        for _ in range(3):
            obs, reward, _, _, _ = env.step(torch.zeros((2, 3), device=env.device))
            assert torch.isfinite(reward).all()
            assert torch.isfinite(obs["policy"]["camera"]).all()
        other_start = env.episode_start_toa[1].clone()
        env._reset_idx([0])
        assert torch.equal(env.episode_start_toa[1], other_start)
        assert torch.equal(env.previous_toa[0], env.episode_start_toa[0])
        # 将两个不同环境的机器人放在同一地图的同一位姿，验证碰撞和射线隔离。
        env.quota_scheduler = SimpleNamespace(assign=lambda: 0)
        env._reset_idx([0, 1])
        pose = env.robot.data.root_state_w[0:1, :7].repeat(2, 1)
        env.robot.write_root_pose_to_sim(pose)
        env.robot.write_root_velocity_to_sim(torch.zeros((2, 6), device=env.device))
        env.position_setpoints[:] = pose[:, :3]
        env.yaw_setpoints[1] = env.yaw_setpoints[0]
        env.goal_positions[1] = env.goal_positions[0]
        env.toa_map_ids[1] = env.toa_map_ids[0]
        env.camera.reset()
        depth = env.camera.data.output["distance_to_image_plane"]
        torch.testing.assert_close(depth[0], depth[1])
        assert (depth < config.depth_max_distance).any(), "相机未命中共享地图"
        for _ in range(3):
            _, _, _, _, extras = env.step(torch.zeros((2, 3), device=env.device))
            assert torch.all(env._contact_force() < 1e-5), "不同环境的机器人发生接触"

        # 轻微穿入外墙内表面，确认过滤后仍能与共享地图接触。
        env.task.collision_force_threshold = 0.1
        wall_pose = env.robot.data.root_state_w[0:1, :7].clone()
        wall_pose[:, :2] = env.map_origins[0:1, :2]
        wall_pose[:, 1] -= config.arena_half_extent - config.cell_size - 0.05
        env.robot.write_root_pose_to_sim(
            wall_pose, env_ids=torch.tensor([0], device=env.device)
        )
        env.position_setpoints[0] = wall_pose[0, :3]
        wall_velocity = torch.zeros((1, 6), device=env.device)
        wall_velocity[:, 1] = -2.0
        env.robot.write_root_velocity_to_sim(
            wall_velocity, env_ids=torch.tensor([0], device=env.device)
        )
        wall_contact = False
        for _ in range(3):
            _, _, _, _, extras = env.step(torch.zeros((2, 3), device=env.device))
            wall_contact |= bool(extras["collided"][0])
        assert wall_contact, "机器人未与共享墙体碰撞"
        print(
            "DFS_SIM_CHECK_OK",
            env.map_ids.tolist(),
            env.episode_start_toa.tolist(),
            flush=True,
        )
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
