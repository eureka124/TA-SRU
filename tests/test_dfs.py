"""DFS 几何、可达采样、缓存和奖励语义的独立 CPU 验证。"""

import unittest
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from ta_sru.config import EnvConfig
from ta_sru.envs.maze import (
    SCENE_VERSION,
    MazeDefinition,
    build_map_pools,
    generate_maze,
    pool_mesh,
    pool_origins,
)
from ta_sru.envs.toa import (
    ToaBank,
    build_toa_bank,
    build_toa_map,
    components,
    obstacle_clearance,
)
from ta_sru.envs.toa_sampling import normalized_progress, sample_toa
from ta_sru.evaluation import EvaluationStats, MapQuotaScheduler, require_dfs_checkpoint
from ta_sru.scene_config import training_env_config, validate_resume_config


class DfsTests(unittest.TestCase):
    def config(self, **kwargs):
        values = {
            "train_map_count": 2,
            "eval_map_count": 2,
            "goals_per_map": 2,
            "maze_size": 7,
            "toa_resolution": 0.2,
            "toa_cache_dir": None,
        }
        values.update(kwargs)
        return EnvConfig(**values)

    def test_dfs_tree_boundaries_and_reproducible_pools(self):
        config = self.config(maze_wall_removal_probability=0)
        pools = build_map_pools(config)
        again = build_map_pools(replace(config, num_envs=99, seed=99))
        hashes = []
        for split, maps in pools.items():
            for maze, other in zip(maps, again[split]):
                np.testing.assert_array_equal(maze.occupied, other.occupied)
                self.assertTrue(maze.occupied[[0, -1], :].all())
                self.assertTrue(maze.occupied[:, [0, -1]].all())
                self.assertEqual(set(components(~maze.occupied)[~maze.occupied]), {0})
                nodes = ((config.maze_size - 1) // 2) ** 2
                self.assertEqual((~maze.occupied).sum(), nodes * 2 - 1)
                hashes.append(maze.content_hash)
        self.assertEqual(len(set(hashes)), len(hashes))
        opened = generate_maze(replace(config, maze_wall_removal_probability=1), 99)
        self.assertGreater((~opened.occupied).sum(), nodes * 2 - 1)

    def test_wall_rectangles_and_mesh_preserve_occupancy(self):
        config = self.config()
        maze = generate_maze(config, 42)
        reconstructed = np.zeros_like(maze.occupied)
        for row in range(config.maze_size):
            for col in range(config.maze_size):
                x = -maze.extent + (col + 0.5) * maze.cell_size
                y = -maze.extent + (row + 0.5) * maze.cell_size
                reconstructed[row, col] = any(
                    abs(x - cx) < sx / 2 and abs(y - cy) < sy / 2
                    for cx, cy, sx, sy in maze.rectangles()
                )
        np.testing.assert_array_equal(reconstructed, maze.occupied)
        vertices, faces = pool_mesh([maze], pool_origins(1, maze.extent, 22))
        self.assertEqual(len(vertices), 8 * (len(maze.rectangles()) + 1))
        self.assertTrue((faces >= 0).all() and (faces < len(vertices)).all())
        self.assertEqual(float(vertices[:, 2].max()), config.arena_height)

    def test_point_zero_reachable_sampling_and_cache_repair(self):
        with TemporaryDirectory() as folder:
            config = self.config(toa_cache_dir=folder)
            bank = build_toa_bank(config)
            cached = build_toa_bank(config)
            np.testing.assert_array_equal(bank.values, cached.values)
            for index, values in enumerate(bank.values):
                self.assertEqual(np.count_nonzero(values == 0), 1)
                y, x = np.argwhere(values == 0)[0]
                coords = np.linspace(
                    -config.arena_half_extent,
                    config.arena_half_extent,
                    config.toa_grid_size,
                )
                np.testing.assert_allclose(
                    bank.goals[index], [coords[x], coords[y]], atol=1e-6
                )
                for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                    if np.isfinite(values[y + dy, x + dx]):
                        self.assertGreater(values[y + dy, x + dx], 0)
            for episode in range(30):
                mid, gid, start, t0 = bank.sample(config, 0, episode)
                self.assertTrue(np.isfinite(t0) and t0 > 0)
                self.assertGreaterEqual(
                    np.linalg.norm(start - bank.goals[mid * config.goals_per_map + gid])
                    + 1e-5,
                    config.min_start_goal_distance,
                )
                cells = np.floor(
                    (start + config.arena_half_extent) / config.maze_cell_size
                ).astype(int)
                self.assertTrue(np.any((cells == 1) | (cells == config.maze_size - 2)))
                same = bank.sample(replace(config, num_envs=12), 0, episode)
                np.testing.assert_array_equal(start, same[2])
                values, valid = sample_toa(
                    torch.from_numpy(bank.values),
                    torch.tensor([mid * config.goals_per_map + gid]),
                    torch.tensor(start).reshape(1, 1, 2),
                    config.arena_half_extent,
                )
                self.assertTrue(valid.item())
                self.assertAlmostEqual(values.item(), t0, places=4)
            first = next(Path(folder).glob("*.npz"))
            first.write_bytes(b"broken")
            repaired = build_toa_bank(config)
            np.testing.assert_array_equal(bank.values, repaired.values)
            count = len(list(Path(folder).glob("*.npz")))
            build_toa_bank(replace(config, toa_slow_speed=0.3))
            self.assertGreater(len(list(Path(folder).glob("*.npz"))), count)
            evaluated = build_toa_bank(replace(config, map_split="eval"))
            self.assertEqual(bank.manifest, evaluated.manifest)
            for pool_config, pool in (
                (config, bank),
                (replace(config, map_split="eval"), evaluated),
            ):
                for mid, choices in enumerate(pool.starts):
                    xy = np.linspace(
                        -config.arena_half_extent,
                        config.arena_half_extent,
                        config.toa_grid_size,
                    )[choices[:, ::-1]]
                    cells = np.floor(
                        (xy + config.arena_half_extent) / config.maze_cell_size
                    ).astype(int)
                    self.assertTrue(
                        np.any(
                            (cells == 1) | (cells == config.maze_size - 2), axis=1
                        ).all()
                    )
                    for episode in range(30):
                        _, gid, start, t0 = pool.sample(pool_config, 3, episode, mid)
                        self.assertGreaterEqual(
                            np.linalg.norm(
                                start - pool.goals[mid * config.goals_per_map + gid]
                            ),
                            8.0,
                        )
                        self.assertTrue(np.isfinite(t0) and t0 > 0)

    def test_start_is_fixed_while_near_and_unreachable_goals_are_rejected(self):
        config = self.config(goals_per_map=3)
        starts = np.array([[15, 15]], dtype=np.int32)
        values = np.full(
            (3, config.toa_grid_size, config.toa_grid_size), np.inf, dtype=np.float32
        )
        values[0, 15, 15] = 1.0
        values[2, 15, 15] = 12.0
        bank = ToaBank(
            [generate_maze(config, 42)],
            np.array([[-3.0, -4.0], [4.0, 4.0], [4.0, -4.0]], dtype=np.float32),
            values,
            [starts],
            {},
        )
        with patch("ta_sru.envs.toa.np.random.default_rng") as create_rng:
            rng = create_rng.return_value
            rng.integers.return_value = 0
            rng.permutation.return_value = np.array([0, 1, 2])
            mid, gid, start, t0 = bank.sample(config, 0, 0, map_id=0)
            self.assertEqual((mid, gid, t0), (0, 2, 12.0))
            np.testing.assert_array_equal(start, [-4.0, -4.0])
            # 距离恰好 8 m 可接受；近目标和不可达目标不会触发重新采样起点。
            rng.integers.assert_called_once_with(1)

    def test_impossible_outer_start_distance_reports_error(self):
        with self.assertRaisesRegex(ValueError, "最外围.*至少 100 m"):
            build_toa_bank(self.config(min_start_goal_distance=100.0))

    def test_toa_routes_around_wall_and_unwritable_cache_falls_back(self):
        config = self.config(toa_slow_speed=1)
        occupied = np.zeros((7, 7), dtype=bool)
        occupied[[0, -1], :] = True
        occupied[:, [0, -1]] = True
        occupied[1:5, 3] = True
        maze = MazeDefinition("detour", 0, occupied, 2.0, 4.0)
        clearance = obstacle_clearance(config, maze)
        values = build_toa_map(config, clearance, np.array([15, 45]))
        # 起终点直线相距 4 m，墙体使点目标最短到达时间显著增加。
        self.assertGreater(values[15, 25], 15.0)
        with TemporaryDirectory() as folder:
            blocked = Path(folder) / "file"
            blocked.write_text("not a directory")
            with self.assertWarns(UserWarning):
                bank = build_toa_bank(replace(config, toa_cache_dir=str(blocked)))
            self.assertTrue(np.isfinite(bank.values).any())

    def test_sampling_does_not_clamp_outside_or_interpolate_through_wall(self):
        maps = torch.tensor([[[0.0, 1.0, 2.0], [1.0, torch.inf, 3.0], [2.0, 3.0, 4.0]]])
        points = torch.tensor(
            [[[-2.0, 0.0], [0.0, 0.0], [-0.5, -0.5], [-1.0, -1.0], [0.0, -1.0]]]
        )
        values, valid = sample_toa(maps, torch.tensor([0]), points, 1.0)
        self.assertEqual(valid.tolist(), [[False, False, False, True, True]])
        torch.testing.assert_close(values[0, 3:], torch.tensor([0.0, 1.0]))

    def test_normalized_reward_fraction_backtrack_and_invalid(self):
        torch.testing.assert_close(
            normalized_progress(
                torch.tensor([100.0, 10.0]),
                torch.tensor([98.0, 9.8]),
                torch.tensor([100.0, 10.0]),
            ),
            torch.tensor([0.02, 0.02]),
        )
        path = torch.tensor([100.0, 90.0, 95.0, 50.0, 0.0])
        rewards = normalized_progress(path[:-1], path[1:], torch.full((4,), 100.0))
        self.assertAlmostEqual(rewards.sum().item(), 1.0, places=6)
        self.assertLess(rewards[1].item(), 0)
        torch.testing.assert_close(
            normalized_progress(
                torch.tensor([torch.inf, 5.0, 5.0]),
                torch.tensor([4.0, torch.inf, 4.0]),
                torch.tensor([10.0, 10.0, 0.0]),
            ),
            torch.zeros(3),
        )
        # 在成功半径内尚有剩余到达时间，不补发剩余进度。
        self.assertAlmostEqual(
            normalized_progress(
                torch.tensor([2.0]), torch.tensor([1.0]), torch.tensor([10.0])
            ).item(),
            0.1,
        )

    def test_resume_ignores_removed_threshold_curriculum_field(self):
        saved = asdict(self.config(collision_force_threshold=12.0))
        saved["minimum_contact_force_threshold"] = 0.1
        restored = training_env_config(saved, {})
        self.assertEqual(restored.collision_force_threshold, 0.1)
        self.assertNotIn("minimum_contact_force_threshold", asdict(restored))
        self.assertIn("minimum_contact_force_threshold", saved)
        validate_resume_config(saved, restored)

    def test_fixed_collision_threshold_requires_finite_positive_force(self):
        for threshold in (0.0, -1.0, float("inf"), float("nan")):
            with (
                self.subTest(threshold=threshold),
                self.assertRaisesRegex(ValueError, "collision_force_threshold"),
            ):
                self.config(collision_force_threshold=threshold).validate()

    def test_single_env_can_finish_all_quotas_and_resume_rejects_changes(self):
        names = [f"eval_{i}" for i in range(16)]
        scheduler = MapQuotaScheduler(16, 3)
        stats = EvaluationStats(3, names)
        for _ in range(48):
            mid = scheduler.assign()
            stats.record(names[mid], {"success": True})
        self.assertTrue(stats.complete)
        self.assertEqual(scheduler.assign(), -1)
        config = self.config(toa_normalization_max=100.0)
        restored = training_env_config(
            asdict(config), {"num_envs": 17, "maze_size": None}
        )
        self.assertEqual(restored.maze_size, config.maze_size)
        validate_resume_config(asdict(config), restored)
        with self.assertRaises(ValueError):
            training_env_config(asdict(config), {"maze_size": 9})
        with self.assertRaises(ValueError):
            require_dfs_checkpoint({"config": {}})
        require_dfs_checkpoint({"scene_manifest": {"scene_version": SCENE_VERSION}})


if __name__ == "__main__":
    unittest.main()
