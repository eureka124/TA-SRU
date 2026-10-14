"""共享动作转换的坐标、积分与悬停契约检查。"""

from __future__ import annotations

import math
import unittest
from unittest.mock import Mock

import torch

from ta_sru.models.hummingbird import HummingbirdParameters, yaw_to_quaternion
from ta_sru.models.lee_controller import LeePositionController
from ta_sru.models.navigation_control import compute_velocity_yaw_wrench


def root_state(yaw: float = 0.0) -> torch.Tensor:
    state = torch.zeros((1, 13))
    state[:, 2] = 2.0
    state[:, 3:7] = yaw_to_quaternion(torch.tensor([yaw]))
    return state


class NavigationControlTests(unittest.TestCase):
    def test_zero_action_hovers_without_moving_setpoints(self) -> None:
        parameters = HummingbirdParameters()
        state = root_state()
        position = state[:, :3].clone()
        yaw = torch.zeros((1, 1))
        wrench = compute_velocity_yaw_wrench(
            LeePositionController(parameters),
            state,
            torch.zeros((1, 3)),
            position,
            yaw,
            1 / 128,
        )
        torch.testing.assert_close(position, state[:, :3])
        torch.testing.assert_close(yaw, torch.zeros_like(yaw))
        torch.testing.assert_close(
            wrench, torch.tensor([[parameters.mass * 9.81, 0.0, 0.0, 0.0]])
        )

    def test_velocity_axes_rotate_and_integrate_once_per_physics_step(self) -> None:
        cases = (
            (0.0, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            (0.0, (0.0, 0.25, 0.0), (0.0, 0.25, 0.0)),
            (math.pi / 2, (1.0, 0.25, 0.0), (-0.25, 1.0, 0.0)),
        )
        for heading, action, expected in cases:
            with self.subTest(heading=heading, action=action):
                state = root_state(heading)
                position = state[:, :3].clone()
                yaw = torch.tensor([[heading]])
                controller = Mock(return_value=torch.zeros((1, 4)))
                compute_velocity_yaw_wrench(
                    controller,
                    state,
                    torch.tensor([action]),
                    position,
                    yaw,
                    1 / 128,
                )
                velocity = torch.tensor([expected])
                torch.testing.assert_close(controller.call_args.args[2], velocity)
                torch.testing.assert_close(
                    controller.call_args.args[3], torch.zeros((1, 3))
                )
                torch.testing.assert_close(position, state[:, :3] + velocity / 128)

    def test_yaw_wraps_at_both_pi_boundaries_without_translating(self) -> None:
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                initial = direction * (math.pi - 0.002)
                state = root_state(initial)
                position = state[:, :3].clone()
                yaw = torch.tensor([[initial]])
                controller = Mock(return_value=torch.zeros((1, 4)))
                compute_velocity_yaw_wrench(
                    controller,
                    state,
                    torch.tensor([[0.0, 0.0, float(direction)]]),
                    position,
                    yaw,
                    1 / 128,
                )
                expected = direction * (-math.pi + 1 / 128 - 0.002)
                self.assertAlmostEqual(float(yaw[0, 0]), expected, places=5)
                torch.testing.assert_close(position, state[:, :3])
                torch.testing.assert_close(controller.call_args.args[4], yaw)


if __name__ == "__main__":
    unittest.main()
