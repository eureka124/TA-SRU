"""验证原始深度保真、视频编码以及并行回合的隔离。"""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from ta_sru.debug.playback import PlayDebugRecorder, depth_to_gray


class _Scene(dict):
    env_origins = torch.tensor([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]])


def _env():
    scene = _Scene()
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0], [2**-0.5, 0.0, 0.0, 2**-0.5]])
    wall = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.tensor([[1.0, 2.0, 2.0], [51.0, 2.0, 2.0]]),
            root_quat_w=quaternion.clone(),
        ),
        cfg=SimpleNamespace(spawn=SimpleNamespace(size=(12.0, 0.7, 4.0))),
    )
    for side in ("north", "south", "east", "west"):
        scene[f"boundary_{side}"] = wall
    depth = torch.arange(2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8, 1) / 2
    depth[0, 0, 0, 0] = torch.inf
    depth[0, 0, 1, 0] = torch.nan
    return SimpleNamespace(
        num_envs=2,
        scene=scene,
        task=SimpleNamespace(arena_half_extent=20.0, depth_max_distance=10.0),
        inner_walls=[wall],
        maze_ids=torch.tensor([0, 1]),
        goal_positions=torch.tensor([[5.0, 8.0, 2.0], [55.0, 8.0, 2.0]]),
        cylinder_radii=torch.tensor([0.3, 0.6]),
        random_cylinders=SimpleNamespace(
            data=SimpleNamespace(
                object_pos_w=torch.tensor(
                    [
                        [[3.0, 4.0, 1.0], [0.0, 0.0, -10.0]],
                        [[53.0, 4.0, 1.0], [50.0, 0.0, -10.0]],
                    ]
                ),
            )
        ),
        camera=SimpleNamespace(
            data=SimpleNamespace(output={"distance_to_image_plane": depth})
        ),
        robot=SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=torch.tensor([[0.0, 0.0, 2.0], [50.0, 0.0, 2.0]]),
                root_quat_w=quaternion,
                root_lin_vel_w=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                root_ang_vel_w=torch.tensor([[0.0, 0.0, 0.4], [0.0, 0.0, 0.4]]),
            )
        ),
        actions=torch.tensor([[2.0, 0.0, 0.3], [2.0, 0.0, 0.3]]),
        extras={
            key: torch.tensor([False, False])
            for key in ("success", "collided", "outside", "time_out")
        },
    )


class PlaybackTests(unittest.TestCase):
    def test_gray_mapping_does_not_modify_raw_depth(self):
        raw = np.array(
            [0.0, 5.0, 10.0, 30.0, np.inf, -np.inf, np.nan], dtype=np.float32
        )
        expected = raw.copy()
        np.testing.assert_array_equal(
            depth_to_gray(raw, 10), [0, 128, 255, 255, 255, 0, 255]
        )
        np.testing.assert_array_equal(raw, expected)

    def test_video_and_parallel_episode_boundaries(self):
        env = _env()
        requested = torch.tensor([[3.0, 0.0, 0.3], [3.0, 0.0, 0.3]])
        with TemporaryDirectory() as temporary:
            recorder = PlayDebugRecorder(temporary, 1 / 24)
            recorder.begin_step(env, requested)
            env.robot.data.root_pos_w[:, 0] += 1
            terminal_depth = (
                env.camera.data.output["distance_to_image_plane"][1, ..., 0]
                .numpy()
                .copy()
            )
            env.extras["collided"][1] = True
            recorder.capture(env, torch.tensor([False, True]))
            self.assertEqual(len(recorder.entries), 1)
            # 第二个环境已重置，第一个环境仍处于原回合。
            env.robot.data.root_pos_w[1, :2] = torch.tensor([45.0, -2.0])
            env.camera.data.output["distance_to_image_plane"][1] = 7
            env.random_cylinders.data.object_pos_w[1, 0, :2] = torch.tensor([48.0, 6.0])
            env.extras["collided"][1] = False
            recorder.begin_step(env, requested)
            env.robot.data.root_pos_w[:, 0] += 1
            recorder.capture(env, torch.tensor([False, False]))
            recorder.close()
            recorder.close()
            self.assertEqual(len(recorder.entries), 3)
            first = recorder.output_dir / "env_000_episode_00000"
            terminal = recorder.output_dir / "env_001_episode_00000"
            reset = recorder.output_dir / "env_001_episode_00001"
            with np.load(first / "depth_raw.npz") as data:
                self.assertEqual(data["depth_m"].shape, (2, 4, 8))
                self.assertTrue(np.isinf(data["depth_m"][0, 0, 0]))
                self.assertTrue(np.isnan(data["depth_m"][0, 0, 1]))
                np.testing.assert_allclose(data["time_s"], [1 / 24, 2 / 24])
                np.testing.assert_allclose(data["position_xy"], [[1, 0], [2, 0]])
                self.assertEqual(data["action"][0, 0], 3)
                self.assertEqual(data["applied_action"][0, 0], 2)
            with np.load(terminal / "depth_raw.npz") as data:
                np.testing.assert_array_equal(data["depth_m"][0], terminal_depth)
                np.testing.assert_allclose(data["position_xy"], [[1, 0]])
                np.testing.assert_allclose(
                    data["state_body"], [[0, -1, 0.4]], atol=1e-6
                )
            reset_data = json.loads((reset / "telemetry.json").read_text())
            self.assertEqual(reset_data["start"], [-5, -2])
            self.assertEqual(reset_data["cylinders"][0]["xy"], [-2, 6])
            self.assertEqual(len(reset_data["cylinders"]), 1)
            self.assertEqual(reset_data["samples"][0]["xy"], [-4, -2])
            self.assertEqual(reset_data["samples"][0]["t"], 1 / 24)
            self.assertEqual(reset_data["outcome"], "interrupted")
            self.assertIn('src="depth.mp4"', (first / "index.html").read_text())
            decoded = subprocess.run(
                [
                    recorder.ffmpeg,
                    "-i",
                    str(first / "depth.mp4"),
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "gray",
                    "pipe:1",
                ],
                capture_output=True,
                check=True,
            )
            self.assertEqual(len(decoded.stdout), 2 * 4 * 8)
            self.assertIn(b"h264 (High)", decoded.stderr)
            self.assertIn(b"24 fps", decoded.stderr)
            self.assertIn(b"8x4", decoded.stderr)

    def test_no_steps_creates_only_index(self):
        with TemporaryDirectory() as temporary:
            recorder = PlayDebugRecorder(temporary, 0.1)
            recorder.close()
            self.assertEqual(
                [p.name for p in Path(recorder.output_dir).iterdir()], ["index.html"]
            )

    def test_shared_layout_quota_and_collision_precedence(self):
        env = _env()
        env.maze_ids[:] = 0
        env.extras["collided"][:] = True
        env.extras["success"][:] = True
        with TemporaryDirectory() as temporary:
            recorder = PlayDebugRecorder(temporary, 0.1, max_episodes_per_layout=3)
            for _ in range(4):
                recorder.begin_step(env, env.actions)
                recorder.capture(env, torch.tensor([True, True]))
            recorder.close()
            self.assertEqual(len(recorder.entries), 3)
            self.assertEqual(recorder.layout_counts, {"maze_01": 3})
            self.assertEqual(recorder.episodes, {})
            self.assertTrue(
                all(entry["outcome"] == "collision" for entry in recorder.entries)
            )

    def test_limited_recorder_does_not_save_partial_episode(self):
        env = _env()
        with TemporaryDirectory() as temporary:
            recorder = PlayDebugRecorder(temporary, 0.1, max_episodes_per_layout=4)
            recorder.begin_step(env, env.actions)
            recorder.capture(env, torch.tensor([False, False]))
            recorder.close()
            self.assertEqual(recorder.entries, [])
            self.assertEqual(recorder.episodes, {})

    def test_middle_selection_follows_starts_not_completions(self):
        env = _env()
        env.maze_ids[:] = 0
        env.extras["success"][:] = True
        with TemporaryDirectory() as temporary:
            recorder = PlayDebugRecorder(
                temporary, 0.1, max_episodes_per_layout=4, episodes_per_layout=6
            )
            self.assertEqual(recorder.selection_range, (2, 5))
            # 第 2 个回合持续三步，第 3 个先结束，第 4 个与它同时结束。
            for done in ([True, False], [True, False], [True, True], [True, True]):
                recorder.begin_step(env, env.actions)
                recorder.capture(env, torch.tensor(done))
            recorder.begin_step(env, env.actions)
            recorder.capture(env, torch.tensor([True, True]))
            recorder.close()
            self.assertEqual(len(recorder.entries), 4)
            self.assertEqual(
                [entry["layout_start_episode"] for entry in recorder.entries],
                [3, 2, 4, 5],
            )
            selected = next(
                entry
                for entry in recorder.entries
                if entry["layout_start_episode"] == 2
            )
            self.assertEqual(selected["frames"], 3)
            data = json.loads(
                (recorder.output_dir / selected["path"] / "telemetry.json").read_text()
            )
            self.assertEqual(data["layout_start_episode"], 2)
            self.assertEqual(recorder.episodes, {})

    def test_middle_selection_skips_early_episodes_without_recording(self):
        env = _env()
        env.maze_ids[:] = 0
        env.extras["success"][:] = True
        with TemporaryDirectory() as temporary:
            recorder = PlayDebugRecorder(
                temporary, 0.1, max_episodes_per_layout=4, episodes_per_layout=1000
            )
            self.assertEqual(recorder.selection_range, (499, 502))
            for _ in range(249):
                recorder.begin_step(env, env.actions)
                self.assertEqual(recorder.episodes, {})
                recorder.capture(env, torch.tensor([True, True]))
            for _ in range(2):
                recorder.begin_step(env, env.actions)
                recorder.capture(env, torch.tensor([True, True]))
            recorder.close()
            self.assertEqual(
                [entry["layout_start_episode"] for entry in recorder.entries],
                [499, 500, 501, 502],
            )


if __name__ == "__main__":
    unittest.main()
