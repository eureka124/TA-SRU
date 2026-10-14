"""无需 Isaac Sim 的采样输出、失败标记与旧 CSV 绘图检查。"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts/controller_step_response.py"
)
SPEC = importlib.util.spec_from_file_location("controller_step_response", SCRIPT_PATH)
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


class StepResponseOutputTests(unittest.TestCase):
    def rig(self) -> SimpleNamespace:
        state = torch.zeros((3, 13))
        state[:, 2] = 2.0
        state[:, 3] = 1.0
        return SimpleNamespace(
            robot=SimpleNamespace(data=SimpleNamespace(root_state_w=state)),
            actions=torch.eye(3),
            last_wrench=torch.zeros((3, 4)),
            physics_steps=4,
            physics_dt=1 / 128,
        )

    def test_contact_failure_is_saved_with_peer_status_and_incomplete_page(
        self,
    ) -> None:
        samples = []
        SCRIPT.record_response(
            samples, self.rig(), "step", torch.tensor([0.2, 0.0, 0.0]), "contact", (0,)
        )
        summary = {
            "completed": False,
            "stop_reason": "contact",
            "stop_env_ids": [0],
            "actual_duration_s": 0.03125,
            "planned_duration_s": 13.0,
        }
        with TemporaryDirectory() as directory:
            path = Path(directory)
            SCRIPT.save_results(path, samples, summary)
            with (path / "response.csv").open() as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual(saved[0]["stop_reason"], "contact")
            self.assertEqual(saved[1]["stop_reason"], "peer_stopped")
            self.assertEqual(saved[0]["phase"], "step")
            self.assertIn("采集未完成", (path / "index.html").read_text())
            self.assertEqual(json.loads((path / "summary.json").read_text()), summary)

    def test_nonfinite_state_is_recorded_but_does_not_corrupt_svg(self) -> None:
        samples = []
        rig = self.rig()
        SCRIPT.record_response(samples, rig, "baseline", torch.zeros(3))
        rig.robot.data.root_state_w[0, 7] = math.nan
        SCRIPT.record_response(
            samples, rig, "step", torch.zeros(3), "nonfinite_state", (0,)
        )
        self.assertTrue(math.isnan(samples[3]["response"]))
        with TemporaryDirectory() as directory:
            page = Path(directory) / "index.html"
            SCRIPT.write_response_page(page, samples)
            self.assertNotIn("nan", page.read_text().lower())

    def test_legacy_csv_can_be_replotted_without_simulator_import(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory)
            csv_path = path / "response.csv"
            with csv_path.open("w", newline="") as stream:
                fields = (
                    "time_s",
                    "env_id",
                    "command",
                    "body_vx",
                    "body_vy",
                    "body_wz",
                )
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for env_id in range(3):
                    writer.writerow(
                        dict(
                            zip(fields, (0.5, env_id, 1.0, 0.2, 0.2, 0.2), strict=True)
                        )
                    )
            with patch(
                "sys.argv", ["controller_step_response.py", "--plot-csv", str(csv_path)]
            ):
                self.assertEqual(SCRIPT.main(), 0)
            page = (path / "index.html").read_text()
            self.assertEqual(page.count("<svg "), 3)
            self.assertIn("时间 (s)", page)

    def test_interrupt_saves_partial_data_and_preserves_exit_code(self) -> None:
        initial = self.rig()

        class FakeRig:
            def __init__(self, _device, _spacing):
                self.__dict__.update(initial.__dict__)
                self.physics_steps = 0
                self.sample_dt = 1 / 32
                self.device = "cpu"
                self.task = SimpleNamespace(
                    action_high=(2.0, 0.5, math.pi / 3),
                    control_decimation=4,
                    collision_force_threshold=0.1,
                )

            def metadata(self):
                return {}

            def step(self, commands):
                if self.physics_steps == 5:
                    raise KeyboardInterrupt
                self.actions.copy_(commands)
                self.physics_steps += 1

            def contact_force(self):
                return torch.zeros(3)

        class FakeLauncher:
            def __init__(self, _args, **_kwargs):
                self.app = SimpleNamespace(is_running=lambda: True)

            @staticmethod
            def add_app_launcher_args(parser):
                parser.add_argument("--device", default="cpu")

        modules = {
            "isaaclab.app": SimpleNamespace(AppLauncher=FakeLauncher),
            "ta_sru.envs.controller_test": SimpleNamespace(
                ControllerStepTestRig=FakeRig
            ),
        }
        with TemporaryDirectory() as directory:
            with (
                patch.dict("sys.modules", modules),
                patch(
                    "sys.argv",
                    ["controller_step_response.py", "--output-dir", directory],
                ),
                patch.object(
                    SCRIPT.os, "_exit", side_effect=SystemExit
                ) as process_exit,
                self.assertRaises(SystemExit),
            ):
                SCRIPT.main()
            process_exit.assert_called_once_with(130)
            path = Path(directory)
            summary = json.loads((path / "summary.json").read_text())
            self.assertFalse(summary["completed"])
            self.assertEqual(summary["stop_reason"], "interrupted")
            with (path / "response.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertGreater(len(rows), 3)
            self.assertEqual(rows[-1]["stop_reason"], "interrupted")


if __name__ == "__main__":
    unittest.main()
