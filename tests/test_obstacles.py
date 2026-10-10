"""随机障碍场地的摆放间距、连通性、网格与场景版本验证。"""

import unittest
from dataclasses import asdict

import numpy as np

from ta_sru.config import EnvConfig
from ta_sru.envs.maze import (
    OBSTACLE_SCENE_VERSION,
    SCENE_VERSION,
    build_map_pools,
    generate_maze,
    pool_mesh,
    pool_origins,
    scene_version,
)
from ta_sru.envs.obstacles import generate_obstacle_field, u_shape_solids
from ta_sru.envs.primitives import Disc, footprint_distance, footprint_sdf
from ta_sru.envs.toa import (
    build_toa_bank,
    components,
    coordinates,
    obstacle_clearance,
)
from ta_sru.evaluation import require_scene_checkpoint
from ta_sru.scene_config import training_env_config, validate_resume_config


class ObstacleFieldTests(unittest.TestCase):
    def config(self, **kwargs):
        values = {
            "scene_type": "obstacles",
            "train_map_count": 2,
            "eval_map_count": 1,
            "goals_per_map": 2,
            "toa_resolution": 0.2,
            "toa_cache_dir": None,
        }
        values.update(kwargs)
        return EnvConfig(**values)

    def test_counts_sizes_and_minimum_separation(self):
        config = self.config()
        maze = generate_obstacle_field(config, 20261010)
        groups = maze.obstacles
        self.assertEqual(len(groups), config.cylinder_count + config.u_shape_count)
        cylinders = [group[0] for group in groups if isinstance(group[0], Disc)]
        u_shapes = [group for group in groups if len(group) == 3]
        self.assertEqual(len(cylinders), config.cylinder_count)
        self.assertEqual(len(u_shapes), config.u_shape_count)
        for cylinder in cylinders:
            self.assertEqual(cylinder.radius, config.cylinder_radius)
            self.assertEqual((cylinder.z0, cylinder.z1), (0.0, config.arena_height))
        expected = u_shape_solids(config, 0.0, 0.0, 0.0)
        for group in u_shapes:
            for solid, reference in zip(group, expected):
                self.assertEqual(
                    (solid.size_x, solid.size_y, solid.z1),
                    (reference.size_x, reference.size_y, reference.z1),
                )
        # 不同障碍物之间、障碍物与围墙之间都保持最小表面间距。
        walls = maze.wall_boxes()
        for index, group in enumerate(groups):
            for solid in group:
                for wall in walls:
                    self.assertGreaterEqual(
                        footprint_distance(solid, wall),
                        config.obstacle_min_separation - 1e-9,
                    )
                for other_group in groups[index + 1 :]:
                    for other in other_group:
                        self.assertGreaterEqual(
                            footprint_distance(solid, other),
                            config.obstacle_min_separation - 1e-9,
                        )

    def test_separation_is_confirmed_by_boundary_sampling(self):
        """沿水平截面边界采样复核间距，不依赖成对测距公式本身。"""

        config = self.config()
        maze = generate_obstacle_field(config, 5)
        groups = maze.obstacles
        for index, group in enumerate(groups):
            others = [
                solid
                for other in groups[index + 1 :]
                for solid in other
            ] + maze.wall_boxes()
            for solid in group:
                outline = _outline_points(solid)
                for other in others:
                    self.assertGreaterEqual(
                        float(footprint_sdf(outline, other).min()),
                        config.obstacle_min_separation - 1e-9,
                    )

    def test_free_space_is_single_connected_component(self):
        config = self.config(toa_resolution=0.1)
        for seed in (7, 20261010):
            maze = generate_obstacle_field(config, seed)
            clearance = obstacle_clearance(config, maze)
            free = clearance > 0
            labels = components(free)
            self.assertEqual(set(labels[free].tolist()), {0})
            self.assertGreater(free.mean(), 0.3)

    def test_obstacle_centres_are_blocked_in_the_free_mask(self):
        config = self.config(toa_resolution=0.1)
        maze = generate_obstacle_field(config, 11)
        clearance = obstacle_clearance(config, maze)
        coords = coordinates(config)
        for group in maze.obstacles:
            for solid in group:
                centre = np.asarray(((solid.x, solid.y),))
                self.assertLess(float(footprint_sdf(centre, solid)[0]), 0.0)
            column = int(np.argmin(np.abs(coords - group[0].x)))
            row = int(np.argmin(np.abs(coords - group[0].y)))
            self.assertLessEqual(float(clearance[row, column]), 0.0)
        # 起点采样环位于围墙内侧，必须留下足够多的可通行格子。
        ring = np.abs(np.abs(coords) - (config.arena_half_extent - 1.5 * config.cell_size))
        band = ring < 0.5
        frame = np.zeros(clearance.shape, dtype=bool)
        frame[band, :] = True
        frame[:, band] = True
        self.assertGreater(float((clearance[frame] > 0).mean()), 0.5)

    def test_dfs_clearance_keeps_bit_identical_values(self):
        """围墙是轴对齐长方体；新 SDF 必须与历史公式逐位一致，TOA 缓存才继续有效。"""

        config = self.config(scene_type="dfs", maze_size=7)
        maze = generate_maze(config, 42)
        grid_x, grid_y = np.meshgrid(coordinates(config), coordinates(config))
        reference = np.full(grid_x.shape, np.inf, dtype=np.float32)
        for x, y, size_x, size_y in maze.rectangles():
            delta_x = np.abs(grid_x - x) - size_x / 2
            delta_y = np.abs(grid_y - y) - size_y / 2
            sdf = np.hypot(
                np.maximum(delta_x, 0), np.maximum(delta_y, 0)
            ) + np.minimum(np.maximum(delta_x, delta_y), 0)
            reference = np.minimum(reference, sdf)
        reference -= config.drone_radius + config.safety_margin
        np.testing.assert_array_equal(obstacle_clearance(config, maze), reference)

    def test_dfs_geometry_and_hash_ignore_obstacle_settings(self):
        """切换场景类型不得改变 DFS 的几何、内容散列与 TOA 缓存键来源。"""

        plain = self.config(scene_type="dfs", maze_size=7)
        tweaked = self.config(
            scene_type="dfs",
            maze_size=7,
            arena_size=99.0,
            cylinder_count=7,
            cylinder_radius=1.5,
            u_shape_count=4,
            obstacle_min_separation=3.0,
        )
        self.assertEqual(plain.cell_size, tweaked.cell_size)
        self.assertEqual(plain.arena_half_extent, tweaked.arena_half_extent)
        for first, second in zip(
            build_map_pools(plain)["train"], build_map_pools(tweaked)["train"]
        ):
            np.testing.assert_array_equal(first.occupied, second.occupied)
            self.assertEqual(first.content_hash, second.content_hash)
            self.assertEqual(first.obstacles, ())

    def test_layouts_are_seed_reproducible_and_pools_unique(self):
        config = self.config()
        first = generate_obstacle_field(config, 42)
        again = generate_obstacle_field(config, 42)
        self.assertEqual(first.content_hash, again.content_hash)
        self.assertNotEqual(
            generate_obstacle_field(config, 43).content_hash, first.content_hash
        )
        pools = build_map_pools(config)
        hashes = [maze.content_hash for maze in pools["train"]]
        self.assertEqual(len(set(hashes)), len(hashes))

    def test_default_layout_is_reachable_for_many_seeds(self):
        config = self.config()
        for seed in (1, 2, 3, 4, 5, 6, 7, 8):
            maze = generate_obstacle_field(config, seed)
            self.assertEqual(
                len(maze.obstacles), config.cylinder_count + config.u_shape_count
            )

    def test_mesh_covers_every_solid_and_stays_inside_arena(self):
        config = self.config()
        maze = generate_obstacle_field(config, 5)
        vertices, faces = pool_mesh([maze], pool_origins(1, maze.extent, 22.0))
        expected = 8 + 2 * sum(
            len(solid.corners()) for solid in maze.solids()
        )
        self.assertEqual(len(vertices), expected)
        self.assertTrue((faces >= 0).all() and (faces < len(vertices)).all())
        self.assertTrue((np.abs(vertices[:, :2]) <= maze.extent + 1e-4).all())
        self.assertEqual(float(vertices[:, 2].max()), config.arena_height)

    def test_scene_versions_gate_checkpoints_per_scene(self):
        self.assertEqual(scene_version("obstacles"), OBSTACLE_SCENE_VERSION)
        self.assertEqual(scene_version("dfs"), SCENE_VERSION)
        self.assertNotEqual(OBSTACLE_SCENE_VERSION, SCENE_VERSION)
        with self.assertRaisesRegex(ValueError, "未知场景类型"):
            scene_version("walls")
        require_scene_checkpoint(
            {
                "config": {"scene_type": "obstacles"},
                "scene_manifest": {"scene_version": OBSTACLE_SCENE_VERSION},
            }
        )
        # 旧 checkpoint 没有 scene_type 字段时按 DFS 处理。
        require_scene_checkpoint({"scene_manifest": {"scene_version": SCENE_VERSION}})
        with self.assertRaisesRegex(ValueError, "场景"):
            require_scene_checkpoint(
                {
                    "config": {"scene_type": "obstacles"},
                    "scene_manifest": {"scene_version": SCENE_VERSION},
                }
            )

    def test_manifests_only_extend_the_obstacle_scene(self):
        """DFS 清单保持历史键集合，只有障碍场地追加参数。"""

        dfs = build_toa_bank(self.config(scene_type="dfs", maze_size=7, train_map_count=1, eval_map_count=1))
        obstacle = build_toa_bank(self.config(train_map_count=1, eval_map_count=1))
        self.assertEqual(
            set(dfs.manifest),
            {
                "scene_version",
                "generator",
                "toa_version",
                "solver",
                "toa_reward_normalization",
                "sampling",
                "pools",
                "grid_size",
                "spacing",
                "safety",
                "speed",
                "success_radius",
            },
        )
        self.assertNotIn("obstacle_field", dfs.manifest)
        self.assertEqual(obstacle.manifest["scene_version"], OBSTACLE_SCENE_VERSION)
        self.assertEqual(obstacle.manifest["obstacle_field"]["cylinders"]["count"], 60)

    def test_obstacle_arena_reaches_thirty_metres(self):
        config = self.config()
        self.assertEqual(config.cell_size, 2.0)
        self.assertEqual(config.arena_half_extent, 15.0)
        maze = generate_obstacle_field(config, 3)
        self.assertEqual(2 * maze.extent, config.arena_size)

    def test_old_checkpoints_resume_with_default_scene_fields(self):
        """旧 checkpoint 没有场景字段时按默认 DFS 恢复，切换场景则必须报错。"""

        saved = asdict(EnvConfig())
        for key in (
            "scene_type",
            "arena_size",
            "cylinder_count",
            "cylinder_radius",
            "u_shape_count",
            "u_shape_arm_length",
            "u_shape_arm_thickness",
            "u_shape_opening",
            "obstacle_min_separation",
        ):
            saved.pop(key)
        restored = training_env_config(saved, {"scene_type": None, "num_envs": 8})
        self.assertEqual(restored.scene_type, "dfs")
        validate_resume_config(saved, restored)
        with self.assertRaisesRegex(ValueError, "scene_type"):
            training_env_config(saved, {"scene_type": "obstacles", "num_envs": 8})
        with self.assertRaisesRegex(ValueError, "obstacle_min_separation"):
            validate_resume_config(
                saved, EnvConfig(**{**saved, "obstacle_min_separation": 0.5})
            )

    def test_invalid_scene_settings_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "场景类型"):
            self.config(scene_type="walls").validate()
        with self.assertRaisesRegex(ValueError, "障碍物数量"):
            self.config(cylinder_count=-1).validate()
        with self.assertRaisesRegex(ValueError, "obstacle_min_separation"):
            self.config(obstacle_min_separation=-0.1).validate()
        with self.assertRaisesRegex(ValueError, "栅格宽度"):
            self.config(arena_size=8.0).validate()
        # 障碍物尺寸远大于场地时必须给出可读的失败信息。
        with self.assertRaisesRegex(ValueError, "摆不下"):
            generate_obstacle_field(
                self.config(arena_size=20.0, u_shape_arm_length=20.0), 3
            )


def _outline_points(solid, spacing: float = 0.05) -> np.ndarray:
    """沿图元水平截面边界等距采样，用于独立复核表面间距。"""

    if isinstance(solid, Disc):
        angles = np.arange(0.0, 2.0 * np.pi, spacing / solid.radius)
        return np.column_stack(
            (
                solid.x + solid.radius * np.cos(angles),
                solid.y + solid.radius * np.sin(angles),
            )
        )
    corners = solid.corners()
    points = []
    for index in range(len(corners)):
        start, end = corners[index], corners[(index + 1) % len(corners)]
        steps = max(int(np.linalg.norm(end - start) / spacing), 1)
        points.append(np.linspace(start, end, steps, endpoint=False))
    return np.concatenate(points)


if __name__ == "__main__":
    unittest.main()
