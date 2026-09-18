"""验证评估配置固定、互斥终局及并行完成时的布局配额。"""

import unittest

from scripts.evaluation import EvaluationStats, configure_evaluation, evaluation_outcome
from ta_sru.config import EnvConfig
from ta_sru.envs.curriculum import select_training_maze_curriculum_stage
from ta_sru.envs.layouts import MAZE_LAYOUTS


class EvaluationTests(unittest.TestCase):
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
