"""验证参考项目的 KL 学习率调度、掩码及完整更新次数。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from ta_sru.algorithms.ppo import RecurrentPPO
from ta_sru.config import PPOConfig


class SmallPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))
        self.log_std = nn.Parameter(torch.zeros(1))

    def evaluate_sequences(self, *args):
        # 有效位置的均值偏移 1、标准差不变，解析 KL 应为 0.5。
        self.action_distribution = Normal(torch.ones(2, 1), torch.ones(2, 1))
        return self.weight.expand(2), self.weight.expand(2) * 0, self.log_std.expand(2)


class PpoUpdateTests(unittest.TestCase):
    def make_agent(self, **kwargs):
        agent = RecurrentPPO.__new__(RecurrentPPO)
        agent.config = SimpleNamespace(ppo=PPOConfig(epochs=3, **kwargs))
        agent.policy = SmallPolicy()
        agent.optimizer = torch.optim.SGD(agent.policy.parameters(), lr=1e-3)
        batch = SimpleNamespace(
            mask=torch.tensor([True, False]),
            observations={},
            actions=None,
            recurrent_state=None,
            episode_starts=None,
            advantages=torch.ones(2),
            old_log_probabilities=torch.zeros(2),
            old_values=torch.zeros(2),
            returns=torch.ones(2),
            # 填充位置使用非法分布参数，验证 KL 只读取有效位置。
            old_action_means=torch.tensor([[0.0], [float("nan")]]),
            old_action_stds=torch.tensor([[1.0], [0.0]]),
        )
        agent.buffer = SimpleNamespace(
            iterate_batches=lambda *args: iter((batch, batch)),
            returns=np.array([1.0, 2.0]),
            values=np.zeros(2),
        )
        agent.device = torch.device("cpu")
        agent.rng = np.random.default_rng(123)
        agent.updates = 0
        return agent

    def test_high_kl_adjusts_before_step_without_early_stop(self):
        agent = self.make_agent()
        rates = []
        original_step = agent.optimizer.step

        def step():
            rates.append(agent.optimizer.param_groups[0]["lr"])
            return original_step()

        with patch.object(agent.optimizer, "step", side_effect=step):
            metrics = agent.update()
        self.assertEqual(metrics["optimizer_steps"], 6)
        self.assertAlmostEqual(metrics["analytic_kl"], 0.5, places=6)
        for i, rate in enumerate(rates, 1):
            self.assertAlmostEqual(rate, 1e-3 / 1.5**i)
        self.assertAlmostEqual(metrics["learning_rate"], rates[-1])
        self.assertEqual(agent.update()["optimizer_steps"], 6)
        self.assertEqual(agent.updates, 2)

    def test_schedule_boundaries_and_limits(self):
        agent = self.make_agent()
        for initial, kl, expected in (
            (3e-4, 0.003, 4.5e-4),
            (3e-4, 0.01, 3e-4),
            (3e-4, 0.005, 3e-4),
            (3e-4, 0.02, 3e-4),
            (3e-4, 0.0, 3e-4),
            (1e-5, 0.1, 1e-5),
            (1e-3, 0.001, 1e-3),
        ):
            with self.subTest(initial=initial, kl=kl):
                agent.optimizer.param_groups[0]["lr"] = initial
                agent._adapt_learning_rate(kl)
                self.assertAlmostEqual(agent.optimizer.param_groups[0]["lr"], expected)

    def test_fixed_or_disabled_target_keeps_learning_rate(self):
        for config in ({"schedule": "fixed"}, {"target_kl": None}):
            with self.subTest(config=config):
                metrics = self.make_agent(**config).update()
                self.assertEqual(metrics["optimizer_steps"], 6)
                self.assertEqual(metrics["learning_rate"], 1e-3)


if __name__ == "__main__":
    unittest.main()
