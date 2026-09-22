"""验证评估配置固定、互斥终局及并行完成时的布局配额。"""

import unittest
from dataclasses import asdict
from unittest.mock import patch

import torch

from ta_sru.config import EnvConfig, NetworkConfig
from ta_sru.envs.curriculum import select_training_maze_curriculum_stage
from ta_sru.envs.layouts import MAZE_LAYOUTS
from ta_sru.evaluation import (
    EvaluationStats,
    configure_evaluation,
    evaluation_outcome,
    load_evaluation_policy,
)
from ta_sru.models.actor_critic import (
    ACTION_TRANSFORM,
    AsymmetricRecurrentActorCritic,
)


class EvaluationTests(unittest.TestCase):
    def test_legacy_checkpoint_without_action_transform_is_rejected(self):
        # clamp 版本的 action_mean 输出物理量纲的无界均值，直接按 tanh 解释会得到
        # 完全不同的动作，必须在加载阶段就报错。
        with self.assertRaises(ValueError):
            load_evaluation_policy({"config": {}}, "cpu")

    def test_actor_only_inference_matches_full_policy_and_resets(self):
        for recurrent_type in ("none", "sru-lstm", "sru-gru", "sru-lstm-gate", "lstm"):
            with self.subTest(recurrent_type=recurrent_type):
                config = NetworkConfig(
                    recurrent_type=recurrent_type,
                    feature_dim=16,
                    recurrent_hidden_size=8,
                    actor_hidden_sizes=(8,),
                    critic_hidden_sizes=(8,),
                )
                reference = AsymmetricRecurrentActorCritic(config).eval()
                checkpoint = {
                    "action_transform": ACTION_TRANSFORM,
                    "config": {
                        "network": asdict(config),
                        "algorithm": "ppo"
                        if recurrent_type == "none"
                        else "recurrent_ppo",
                    },
                    "policy": reference.state_dict(),
                }
                # 加载推理网络期间任何训练资源分配都视为失败。
                with (
                    patch(
                        "ta_sru.algorithms.ppo.RecurrentPPO",
                        side_effect=AssertionError("不应创建训练器"),
                    ),
                    patch(
                        "torch.optim.Adam", side_effect=AssertionError("不应创建优化器")
                    ),
                ):
                    policy = load_evaluation_policy(checkpoint, "cpu")
                state = policy.initial_state(2)
                reference_state = reference.initial_state(2)
                observation = {
                    "camera": torch.randn(2, 1, 12, 16),
                    "robot_state": torch.randn(2, 8),
                    "critic_toa": torch.randn(2, 1, 16, 16),
                }
                actor_observation = {
                    key: value
                    for key, value in observation.items()
                    if key != "critic_toa"
                }
                for starts in ([True, True], [False, False], [True, False]):
                    starts = torch.tensor(starts)
                    expected, _, _, reference_state = reference.act(
                        observation, reference_state, starts, deterministic=True
                    )
                    with patch.object(
                        policy.critic_encoder,
                        "forward",
                        side_effect=AssertionError("不应执行 Critic"),
                    ):
                        actual, state = policy.act_actor(
                            actor_observation, state, starts
                        )
                    torch.testing.assert_close(actual, expected)

    def test_fixed_curriculum_at_all_training_steps(self):
        original = {"training_curriculum_cylinder_counts": (10, 30, 60)}
        config = EnvConfig(**configure_evaluation(original))
        self.assertEqual(original["training_curriculum_cylinder_counts"], (10, 30, 60))
        for steps in (0, 100, config.total_training_steps * 2):
            stage = select_training_maze_curriculum_stage(
                steps,
                config.total_training_steps,
                config.training_curriculum_stage_fractions,
                config.training_curriculum_maze_counts,
                config.training_curriculum_cylinder_counts,
                config.training_curriculum_maze_start_indices,
            )
            self.assertEqual(
                (stage.maze_count, stage.maze_start_index, stage.cylinder_count),
                (6, 0, 60),
            )
        self.assertEqual(config.collision_force_threshold, 0.1)
        self.assertEqual(config.minimum_contact_force_threshold, 0.1)

    def test_outcome_precedence(self):
        self.assertEqual(
            evaluation_outcome({"collided": True, "success": True, "time_out": True}),
            "collision",
        )
        self.assertEqual(
            evaluation_outcome({"outside": True, "success": True}), "collision"
        )
        self.assertEqual(
            evaluation_outcome({"success": True, "time_out": True}), "success"
        )
        self.assertEqual(evaluation_outcome({"time_out": True}), "timeout")
        with self.assertRaises(ValueError):
            evaluation_outcome({})

    def test_quota_ignores_extra_parallel_completions(self):
        stats = EvaluationStats(2)
        self.assertIsNone(stats.summary()["overall"]["success_rate"])
        for layout in MAZE_LAYOUTS:
            self.assertFalse(stats.complete)
            self.assertEqual(stats.record(layout.name, {"success": True}), "success")
            self.assertEqual(stats.record(layout.name, {"collided": True}), "collision")
            self.assertIsNone(stats.record(layout.name, {"time_out": True}))
        self.assertTrue(stats.complete)
        summary = stats.summary()
        self.assertEqual(summary["overall"]["episodes"], 12)
        self.assertEqual(summary["overall"]["success_rate"], 0.5)
        for counts in summary["layouts"].values():
            self.assertEqual(counts["episodes"], 2)
            self.assertEqual(
                sum(
                    counts[f"{key}_rate"] for key in ("success", "collision", "timeout")
                ),
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
