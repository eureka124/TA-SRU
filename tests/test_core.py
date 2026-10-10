from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from ta_sru.algorithms import RecurrentPPO
from ta_sru.config import EnvConfig, NetworkConfig, PPOConfig, TrainConfig
from ta_sru.envs.contact import peak_contact_force
from ta_sru.envs.toa import resolve_toa_normalization_max
from ta_sru.models import (
    AsymmetricRecurrentActorCritic,
    HummingbirdParameters,
    LeePositionController,
)
from ta_sru.models.actor_critic import require_supported_action_transform
from ta_sru.models.recurrent import build_recurrent


def _observation(num_envs: int) -> dict[str, np.ndarray]:
    return {
        "camera": np.zeros((num_envs, 1, 12, 16), dtype=np.float32),
        "robot_state": np.zeros((num_envs, 8), dtype=np.float32),
        "critic_toa": np.zeros((num_envs, 1, 16, 16), dtype=np.float32),
    }


class FakeVectorEnv:
    """无需 Isaac Sim 的 PPO 接口桩，只用于算法单元测试。"""

    num_envs = 2

    def __init__(self) -> None:
        self.steps = np.zeros(self.num_envs, dtype=np.int32)

    def reset(self):
        self.steps[:] = 0
        return _observation(self.num_envs), [{}, {}]

    def step(self, actions):
        self.steps += 1
        terminal = _observation(self.num_envs)
        truncated = self.steps % 2 == 0
        infos = [{} for _ in range(self.num_envs)]
        for env_id in np.flatnonzero(truncated):
            infos[env_id]["terminal_observation"] = {
                key: value[env_id] for key, value in terminal.items()
            }
            infos[env_id]["success"] = False
            infos[env_id]["collided"] = False
            self.steps[env_id] = 0
        return (
            _observation(self.num_envs),
            np.ones(self.num_envs, dtype=np.float32),
            np.zeros(self.num_envs, dtype=bool),
            truncated,
            infos,
        )


class ModelTests(unittest.TestCase):
    def test_gyroscopic_torque_compensates_rigid_body_coupling(self) -> None:
        parameters = HummingbirdParameters(inertia=(0.007, 0.009, 0.012))
        controller = LeePositionController(
            parameters, angular_rate_gain=(0.0, 0.0, 0.0)
        )
        state = torch.tensor(
            [[0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 2, 3, 4]], dtype=torch.float32
        )
        torque = controller(state)[0, 1:]
        # 按欧拉刚体方程逐分量构造耦合力矩，避免复用控制器的叉乘表达式。
        jx, jy, jz = parameters.inertia
        expected = torch.tensor(
            [(jz - jy) * 3 * 4, (jx - jz) * 4 * 2, (jy - jx) * 2 * 3]
        )
        torch.testing.assert_close(torque, expected)
        torch.testing.assert_close(
            (torque - expected) / controller.inertia, torch.zeros(3)
        )

    def test_hover_wrench(self) -> None:
        parameters = HummingbirdParameters()
        controller = LeePositionController(parameters)
        state = torch.tensor(
            [[0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32
        )
        command = controller(state)
        self.assertAlmostEqual(float(command[0, 0]), parameters.mass * 9.81, places=4)
        torch.testing.assert_close(command[0, 1:], torch.zeros(3), atol=1e-6, rtol=0)

    def test_actor_cannot_see_privileged_toa(self) -> None:
        torch.manual_seed(1)
        model = AsymmetricRecurrentActorCritic(
            NetworkConfig(
                feature_dim=256,
                recurrent_hidden_size=8,
                actor_hidden_sizes=(8,),
                critic_hidden_sizes=(8,),
            )
        )
        observation = {
            "camera": torch.randn(2, 1, 12, 16),
            "robot_state": torch.randn(2, 8),
            "critic_toa": torch.randn(2, 1, 16, 16),
        }
        changed = dict(observation)
        changed["critic_toa"] = observation["critic_toa"] + 10.0
        state = model.initial_state(2)
        starts = torch.zeros(2, dtype=torch.bool)
        # 比较 Actor 的完整前向，比只比较编码器输出更强：深度图不经过 TOA，
        # 而特权观测一旦泄漏进 Actor 的任何一层都会让动作发生变化。
        torch.testing.assert_close(
            model.act_actor(observation, state, starts)[0],
            model.act_actor(changed, state, starts)[0],
        )
        depth_feature = model.depth_encoder(observation["camera"])
        self.assertFalse(
            torch.allclose(
                model.critic_encoder(observation, depth_feature),
                model.critic_encoder(changed, depth_feature),
            )
        )

    def test_action_log_probability_round_trips(self) -> None:
        torch.manual_seed(3)
        model = AsymmetricRecurrentActorCritic(
            NetworkConfig(
                feature_dim=256,
                recurrent_hidden_size=8,
                actor_hidden_sizes=(8,),
                critic_hidden_sizes=(8,),
            )
        )
        observation = {
            "camera": torch.randn(4, 1, 12, 16),
            "robot_state": torch.randn(4, 8),
            "critic_toa": torch.randn(4, 1, 16, 16),
        }
        starts = torch.zeros(4, dtype=torch.bool)
        state = model.initial_state(4)
        action, _, log_probability, _ = model.act(observation, state, starts)

        self.assertFalse(torch.allclose(action, torch.zeros_like(action)))

        # 策略输出无界高斯采样，物理边界由环境 clamp 施加，因此缓冲里存的就是采样
        # 值本身。对数概率必须与采样时刻一致，否则 PPO 的概率比会带上一个与动作
        # 相关的系统偏差。
        _, replayed, _ = model.evaluate_sequences(
            {key: value.unsqueeze(0) for key, value in observation.items()},
            action.unsqueeze(0),
            state,
            starts.unsqueeze(0),
        )
        torch.testing.assert_close(replayed[0], log_probability, atol=1e-4, rtol=1e-4)

    def test_checkpoint_action_transform_is_validated(self) -> None:
        # 当前参数化直接放行。
        require_supported_action_transform("clamp")
        # 引入该标识之前的 checkpoint 没有此字段，属于 clamp 版本，可以加载。
        require_supported_action_transform(None)
        # main 分支的 tanh 版本权重含义不同，必须显式拒绝而不是静默加载。
        with self.assertRaisesRegex(ValueError, "tanh"):
            require_supported_action_transform("tanh")
        with self.assertRaisesRegex(ValueError, "不一致"):
            require_supported_action_transform("beta")

    def test_all_recurrent_types_and_episode_reset(self) -> None:
        torch.manual_seed(2)
        sequence = torch.randn(4, 2, 5)
        episode_starts = torch.zeros(4, 2, dtype=torch.bool)
        episode_starts[2, 0] = True
        for recurrent_type in ("sru-lstm", "sru-gru", "sru-lstm-gate", "lstm"):
            with self.subTest(recurrent_type=recurrent_type):
                recurrent = build_recurrent(recurrent_type, 5, 7, 2)
                output, state = recurrent(sequence, episode_starts=episode_starts)
                standalone, _ = recurrent(sequence[2:, 0:1])
                self.assertEqual(output.shape, (4, 2, 7))
                self.assertEqual(state[0].shape, (2, 2, 7))
                self.assertEqual(state[1].shape, (2, 2, 7))
                torch.testing.assert_close(output[2:, 0:1], standalone)
                if recurrent_type == "sru-gru":
                    torch.testing.assert_close(state[1], torch.zeros_like(state[1]))


class TaskTests(unittest.TestCase):
    def test_contact_peak_captures_earlier_substeps_and_excludes_previous_step(self):
        history = torch.zeros(2, 6, 5, 3)
        history[0, 4, 3] = torch.tensor([3.0, 4.0, 0.0])
        history[1, 1, 0, 2] = 0.2
        history[:, 5, :, :] = 100.0
        torch.testing.assert_close(
            peak_contact_force(history, 5), torch.tensor([5.0, 0.2])
        )
        history[0] = 0.0
        torch.testing.assert_close(
            peak_contact_force(history, 5), torch.tensor([0.0, 0.2])
        )
        with self.assertRaises(ValueError):
            peak_contact_force(history, 7)

    def test_toa_scale_excludes_unreachable(self):
        maps = np.array([0.0, 12.0, 48.7, np.nan, np.inf])
        self.assertAlmostEqual(resolve_toa_normalization_max(maps, None), 48.7)
        self.assertEqual(resolve_toa_normalization_max(maps, 255.0), 255.0)
        with self.assertRaises(ValueError):
            resolve_toa_normalization_max(np.array([np.inf]), None)
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                resolve_toa_normalization_max(maps, bad)


class AlgorithmTests(unittest.TestCase):
    def test_cpu_buffer_and_ppo_update(self) -> None:
        for recurrent_type in ("sru-lstm", "sru-gru", "sru-lstm-gate", "lstm"):
            with self.subTest(recurrent_type=recurrent_type):
                config = TrainConfig(
                    env=EnvConfig(num_envs=2, depth_height=12, depth_width=16),
                    network=NetworkConfig(
                        feature_dim=256,
                        recurrent_type=recurrent_type,
                        recurrent_hidden_size=8,
                        actor_hidden_sizes=(8,),
                        critic_hidden_sizes=(8,),
                    ),
                    ppo=PPOConfig(
                        rollout_steps=4,
                        batch_size=4,
                        recurrent_sequence_length=2,
                        epochs=1,
                    ),
                    total_timesteps=8,
                    device="cpu",
                )
                agent = RecurrentPPO(FakeVectorEnv(), config)
                self.assertEqual(agent.buffer.storage_device, "cpu")
                self.assertTrue(
                    all(
                        isinstance(value, np.ndarray)
                        for value in agent.buffer.observations.values()
                    )
                )
                agent.collect_rollout()
                metrics = agent.update()
                self.assertTrue(np.isfinite(metrics["policy_loss"]))
                self.assertTrue(np.isfinite(metrics["value_loss"]))

    def test_checkpoint_round_trip_loads_and_rejects_other_parameterization(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            config = TrainConfig(
                env=EnvConfig(num_envs=2, depth_height=12, depth_width=16),
                network=NetworkConfig(
                    feature_dim=256,
                    recurrent_hidden_size=8,
                    actor_hidden_sizes=(8,),
                    critic_hidden_sizes=(8,),
                ),
                ppo=PPOConfig(
                    rollout_steps=4,
                    batch_size=4,
                    recurrent_sequence_length=2,
                    epochs=1,
                ),
                total_timesteps=8,
                device="cpu",
                checkpoint_dir=str(Path(directory) / "checkpoints"),
            )
            path = Path(directory) / "checkpoints" / "model.pt"
            RecurrentPPO(FakeVectorEnv(), config).save(path)
            RecurrentPPO(FakeVectorEnv(), config).load(path)

            # 写出的 checkpoint 必须带动作参数化标识，否则新 checkpoint 会被当成
            # 旧版本，在另一个分支上静默加载。
            saved = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(saved["action_transform"], "clamp")

            # 把标识改成 main 分支的 tanh 版本，加载必须失败而不是产生错误动作。
            saved["action_transform"] = "tanh"
            torch.save(saved, path)
            with self.assertRaisesRegex(ValueError, "tanh"):
                RecurrentPPO(FakeVectorEnv(), config).load(path)

    def test_training_log_contains_recurrent_type(self) -> None:
        with TemporaryDirectory() as directory:
            config = TrainConfig(
                env=EnvConfig(num_envs=2, depth_height=12, depth_width=16),
                network=NetworkConfig(
                    feature_dim=256,
                    recurrent_type="sru-gru",
                    recurrent_hidden_size=8,
                    actor_hidden_sizes=(8,),
                    critic_hidden_sizes=(8,),
                ),
                ppo=PPOConfig(
                    rollout_steps=4,
                    batch_size=4,
                    recurrent_sequence_length=2,
                    epochs=1,
                ),
                total_timesteps=8,
                device="cpu",
                log_interval=1,
                log_dir=directory,
                checkpoint_dir=str(Path(directory) / "checkpoints"),
            )
            RecurrentPPO(FakeVectorEnv(), config).learn()
            progress = (Path(directory) / "progress.csv").read_text(encoding="utf-8")
            self.assertIn("recurrent_type", progress)
            self.assertIn("sru-gru", progress)
            accumulator = EventAccumulator(directory)
            accumulator.Reload()
            scalar_tags = set(accumulator.Tags()["scalars"])
            self.assertTrue(
                {
                    "progress/policy_loss",
                    "progress/cpu_buffer_mib",
                    "rollout/ep_len_mean",
                    "rollout/ep_rew_mean",
                    "time/fps",
                    "Metrics/Collision_Rate",
                    "Metrics/Success_Rate",
                    "Metrics/Timeout_Rate",
                    "train/explained_variance",
                    "train/loss",
                    "train/std",
                }.issubset(scalar_tags)
            )
            self.assertEqual(len(accumulator.Scalars("progress/update")), 1)
            self.assertAlmostEqual(
                accumulator.Scalars("progress/contact_force_threshold")[0].value,
                config.env.collision_force_threshold,
            )


if __name__ == "__main__":
    unittest.main()
