"""按环境、按回合保存原始深度、视频及同步 HTML 回放。"""

from __future__ import annotations

import html
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import imageio_ffmpeg
import numpy as np
import torch

from ta_sru.envs.layouts import MAZE_LAYOUTS
from ta_sru.models.hummingbird import rotate_inverse, yaw_from_quaternion

if TYPE_CHECKING:
    from ta_sru.envs.navigation import NavigationEnv


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def depth_to_gray(depth: np.ndarray, max_distance: float) -> np.ndarray:
    """只对视频副本作固定量程映射，原始米制深度保留 NaN/Inf。"""

    finite = np.nan_to_num(depth, nan=max_distance, posinf=max_distance, neginf=0.0)
    return np.rint(np.clip(finite / max_distance, 0, 1) * 255).astype(np.uint8)


@dataclass
class _Episode:
    metadata: dict
    depth: list[np.ndarray] = field(default_factory=list)
    samples: list[dict] = field(default_factory=list)


class PlayDebugRecorder:
    def __init__(
        self,
        output_dir: str | Path,
        dt: float,
        *,
        max_episodes_per_layout: int | None = None,
    ) -> None:
        if dt <= 0:
            raise ValueError("回放时间步长必须为正数")
        if max_episodes_per_layout is not None and max_episodes_per_layout <= 0:
            raise ValueError("每种布局的保存上限必须为正数")
        self.max_episodes_per_layout = max_episodes_per_layout
        self.layout_counts: dict[str, int] = {}
        self.dt = dt
        self.output_dir = Path(output_dir) / datetime.now(
            timezone.utc
        ).astimezone().strftime("play_%Y%m%d_%H%M%S_%f")
        self.output_dir.mkdir(parents=True)
        self.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        self.episodes: dict[int, _Episode] = {}
        self.episode_counts: dict[int, int] = {}
        self.entries: list[dict] = []
        self.requested_actions: np.ndarray | None = None
        self._write_index()

    @staticmethod
    def _scene(env: NavigationEnv, env_id: int) -> dict:
        """使用本回合的实际障碍物姿态，并去掉并行环境的平移量。"""

        origin = _numpy(env.scene.env_origins[env_id, :2])
        rectangles = []
        walls = [
            env.scene[f"boundary_{side}"] for side in ("north", "south", "east", "west")
        ]
        for wall in [*walls, *env.inner_walls]:
            position = _numpy(wall.data.root_pos_w[env_id])
            if position[2] < 0:
                continue
            rectangles.append(
                {
                    "xy": (position[:2] - origin).tolist(),
                    "size": list(wall.cfg.spawn.size[:2]),
                    "yaw": float(
                        yaw_from_quaternion(
                            wall.data.root_quat_w[env_id : env_id + 1]
                        ).item()
                    ),
                }
            )
        cylinders = []
        positions = _numpy(env.random_cylinders.data.object_pos_w[env_id])
        radii = _numpy(env.cylinder_radii)
        for position, radius in zip(positions, radii):
            if position[2] >= 0:
                cylinders.append(
                    {"xy": (position[:2] - origin).tolist(), "radius": float(radius)}
                )
        return {
            "env_id": env_id,
            "maze": MAZE_LAYOUTS[int(env.maze_ids[env_id].item())].name,
            "extent": env.task.arena_half_extent,
            "depth_max": env.task.depth_max_distance,
            "goal": (_numpy(env.goal_positions[env_id, :2]) - origin).tolist(),
            "start": (_numpy(env.robot.data.root_pos_w[env_id, :2]) - origin).tolist(),
            "rectangles": rectangles,
            "cylinders": cylinders,
        }

    def begin_step(self, env: NavigationEnv, actions: torch.Tensor) -> None:
        self.requested_actions = _numpy(actions)
        for env_id in range(env.num_envs):
            maze = MAZE_LAYOUTS[int(env.maze_ids[env_id].item())].name
            if self._layout_full(maze):
                self.episodes.pop(env_id, None)
                continue
            if env_id not in self.episodes:
                episode_id = self.episode_counts.get(env_id, 0)
                metadata = self._scene(env, env_id)
                metadata.update(episode_id=episode_id, dt=self.dt)
                self.episodes[env_id] = _Episode(metadata)

    def capture(self, env: NavigationEnv, done: torch.Tensor) -> None:
        """每个策略步结束、自动重置之前，统一采样相机和机体状态。"""

        if not self.episodes:
            return
        depth = _numpy(env.camera.data.output["distance_to_image_plane"])
        if depth.ndim == 4:
            depth = depth[..., 0]
        quaternion = env.robot.data.root_quat_w
        velocity = rotate_inverse(quaternion, env.robot.data.root_lin_vel_w)
        angular = rotate_inverse(quaternion, env.robot.data.root_ang_vel_w)
        actual = _numpy(torch.cat((velocity[:, :2], angular[:, 2:3]), dim=-1))
        position = _numpy(
            env.robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
        )
        yaw = _numpy(yaw_from_quaternion(quaternion))
        applied = _numpy(env.actions)
        done_host = _numpy(done)
        outcomes = {
            key: _numpy(env.extras[key])
            for key in ("success", "collided", "outside", "time_out")
        }
        for env_id, episode in list(self.episodes.items()):
            if self._layout_full(episode.metadata["maze"]):
                del self.episodes[env_id]
                continue
            episode.depth.append(depth[env_id].copy())
            episode.samples.append(
                {
                    "t": len(episode.depth) * self.dt,
                    "action": self.requested_actions[env_id].tolist(),
                    "applied": applied[env_id].tolist(),
                    "actual": actual[env_id].tolist(),
                    "xy": position[env_id].tolist(),
                    "yaw": float(yaw[env_id]),
                }
            )
            if done_host[env_id]:
                if self.max_episodes_per_layout is not None:
                    # 回放独立于评估脚本，使用相同的碰撞优先顺序。
                    if outcomes["collided"][env_id] or outcomes["outside"][env_id]:
                        reason = "collision"
                    elif outcomes["success"][env_id]:
                        reason = "success"
                    elif outcomes["time_out"][env_id]:
                        reason = "timeout"
                    else:
                        raise ValueError("已结束的回放回合缺少终止原因")
                else:
                    reason = next(
                        key for key, values in outcomes.items() if values[env_id]
                    )
                self._finish(env_id, reason)

    def _layout_full(self, maze: str) -> bool:
        return (
            self.max_episodes_per_layout is not None
            and self.layout_counts.get(maze, 0) >= self.max_episodes_per_layout
        )

    def _finish(self, env_id: int, reason: str) -> None:
        episode = self.episodes[env_id]
        if not episode.samples:
            del self.episodes[env_id]
            return
        metadata = episode.metadata
        name = f"env_{env_id:03d}_episode_{metadata['episode_id']:05d}"
        directory = self.output_dir / name
        directory.mkdir(exist_ok=True)
        depth = np.stack(episode.depth)
        # NPZ 保留相机的原始分辨率、米制浮点数以及无效值，视频仅用于观看。
        np.savez_compressed(
            directory / "depth_raw.npz",
            depth_m=depth,
            time_s=np.asarray([s["t"] for s in episode.samples]),
            action=np.asarray([s["action"] for s in episode.samples]),
            applied_action=np.asarray([s["applied"] for s in episode.samples]),
            state_body=np.asarray([s["actual"] for s in episode.samples]),
            position_xy=np.asarray([s["xy"] for s in episode.samples]),
            yaw_rad=np.asarray([s["yaw"] for s in episode.samples]),
        )
        metadata.update(
            outcome=reason, frame_count=len(depth), depth_shape=list(depth.shape[1:])
        )
        data = {**metadata, "samples": episode.samples}
        serialized = json.dumps(data, ensure_ascii=False, allow_nan=False)
        (directory / "telemetry.json").write_text(serialized, encoding="utf-8")
        fps = Fraction(1 / self.dt).limit_denominator(1_000_000)
        result = subprocess.run(
            [
                self.ffmpeg,
                "-y",
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "gray",
                "-video_size",
                f"{depth.shape[2]}x{depth.shape[1]}",
                "-framerate",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "1",
                "-crf",
                "12",
                "-profile:v",
                "high",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(directory / "depth.mp4"),
            ],
            input=depth_to_gray(depth, metadata["depth_max"]).tobytes(),
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"深度视频编码失败：{result.stderr.decode(errors='replace')}"
            )
        template = Path(__file__).with_name("playback.html").read_text(encoding="utf-8")
        (directory / "index.html").write_text(
            template.replace("__PLAYBACK_DATA__", serialized.replace("<", "\\u003c")),
            encoding="utf-8",
        )
        self.entries.append(
            {
                "path": name,
                "env_id": env_id,
                "episode_id": metadata["episode_id"],
                "outcome": reason,
                "frames": len(depth),
            }
        )
        self.episode_counts[env_id] = metadata["episode_id"] + 1
        maze = metadata["maze"]
        self.layout_counts[maze] = self.layout_counts.get(maze, 0) + 1
        del self.episodes[env_id]
        self._write_index()
        print(f"[DEBUG] 已保存 {name}（{reason}，{len(depth)} 帧）", flush=True)

    def _write_index(self) -> None:
        rows = "\n".join(
            f'<li><a href="{entry["path"]}/index.html">环境 {entry["env_id"]} / 回合 {entry["episode_id"]}</a>'
            f" — {html.escape(entry['outcome'])}，{entry['frames']} 帧</li>"
            for entry in self.entries
        )
        content = (
            '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            "<title>导航调试回放</title><h1>导航调试回放</h1>"
            "<p>每个回合结束后更新此列表；刷新页面查看新回放。</p>"
            f"<ul>{rows}</ul></html>"
        )
        temporary = self.output_dir / "index.html.tmp"
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(self.output_dir / "index.html")

    def close(self) -> None:
        """限额评估只保存完整回合；原有无限额回放仍保存中断回合。"""

        if self.max_episodes_per_layout is not None:
            self.episodes.clear()
            return
        errors = []
        for env_id in list(self.episodes):
            try:
                self._finish(env_id, "interrupted")
            except Exception as error:  # noqa: BLE001
                errors.append(str(error))
        if errors:
            raise RuntimeError("；".join(errors))
