"""验证空间交叉注意力的时序边界、梯度、旧架构兼容和配置恢复。"""

import unittest
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from ta_sru.config import NetworkConfig, TrainConfig
from ta_sru.export import export_actor
from ta_sru.models.actor_critic import (
    ActorEncoder,
    AsymmetricRecurrentActorCritic,
    CrossAttentionActorEncoder,
)


class CrossAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.config = NetworkConfig(
            feature_dim=24,
            cnn_chunk_size=4,
            recurrent_hidden_size=8,
            actor_hidden_sizes=(8,),
            critic_hidden_sizes=(8,),
        )
        self.observation = {
            "camera": torch.rand(3, 3, 1, 12, 16),
            "robot_state": torch.randn(3, 3, 8),
        }

    def test_zero_attention_scale_recovers_original_fusion(self):
        legacy = ActorEncoder(8, self.config)
        encoder = CrossAttentionActorEncoder(
            8, replace(self.config, attention_residual_init=0.0)
        )
        encoder.camera.load_state_dict(legacy.camera.state_dict(), strict=True)
        encoder.robot_state.load_state_dict(
            legacy.robot_state.state_dict(), strict=True
        )
        torch.testing.assert_close(encoder(self.observation), legacy(self.observation))

    def test_query_changes_spatial_weights(self):
        encoder = CrossAttentionActorEncoder(8, self.config)
        captured = []
        handle = encoder.attention.register_forward_pre_hook(
            lambda module, args: captured.append(args)
        )
        observation = {key: value[0, :1] for key, value in self.observation.items()}
        with torch.no_grad():
            encoder(observation)
            encoder({**observation, "robot_state": -observation["robot_state"]})
        handle.remove()
        with torch.no_grad():
            _, first = encoder.attention(*captured[0], average_attn_weights=False)
            _, second = encoder.attention(*captured[1], average_attn_weights=False)
        self.assertEqual(first.shape, (1, 4, 1, 24))
        self.assertFalse(torch.allclose(first, second))

    def test_checkpointed_chunks_match_outputs_and_all_gradients(self):
        whole = CrossAttentionActorEncoder(
            8,
            replace(self.config, cnn_chunk_size=100, cnn_gradient_checkpointing=False),
        )
        chunked = CrossAttentionActorEncoder(8, self.config)
        chunked.load_state_dict(whole.state_dict(), strict=True)
        expected = whole(self.observation)
        actual = chunked(self.observation)
        self.assertEqual(actual.shape, (3, 3, 24))
        torch.testing.assert_close(actual, expected)
        weight = torch.randn_like(expected)
        (expected * weight).sum().backward()
        (actual * weight).sum().backward()
        for name, parameter in whole.named_parameters():
            with self.subTest(parameter=name):
                gradient = dict(chunked.named_parameters())[name].grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient.abs().sum().item(), 0.0)
                torch.testing.assert_close(
                    gradient, parameter.grad, atol=2e-5, rtol=1e-4
                )

    def test_sequence_matches_online_actions_and_resets_without_toa(self):
        model = AsymmetricRecurrentActorCritic(self.config)
        starts = torch.zeros(3, 3, dtype=torch.bool)
        starts[1, 0] = True
        with torch.no_grad():
            features = model.actor_encoder(self.observation)
            latent, _ = model.actor_recurrent(
                features, model.initial_state(3).actor, starts
            )
            expected = model.action_mean(model.actor_mlp(latent))
            state = model.initial_state(3)
            online = []
            for step in range(3):
                action, state = model.act_actor(
                    {key: value[step] for key, value in self.observation.items()},
                    state,
                    starts[step],
                )
                online.append(action)
            torch.testing.assert_close(torch.stack(online), expected)
            # 新回合首步与独立零状态前向一致。
            reset_action, _ = model.act_actor(
                {key: value[1, :1] for key, value in self.observation.items()},
                model.initial_state(1),
                torch.ones(1, dtype=torch.bool),
            )
            torch.testing.assert_close(reset_action[0], expected[1, 0])
            # 修改未来帧及其他环境，不得影响当前样本的空间融合。
            changed = {key: value.clone() for key, value in self.observation.items()}
            for value in changed.values():
                value[1:] += 10
                value[0, 1:] -= 10
            torch.testing.assert_close(
                model.actor_encoder(changed)[0, 0], features[0, 0]
            )

    def test_legacy_checkpoint_keeps_original_encoder_and_keys(self):
        legacy_config = replace(
            self.config, architecture="tutorial_cnn_v1", feature_dim=13
        )
        legacy = AsymmetricRecurrentActorCritic(legacy_config)
        saved_config = asdict(legacy_config)
        for key in ("attention_heads", "attention_grid", "attention_residual_init"):
            saved_config.pop(key)
        restored = AsymmetricRecurrentActorCritic(
            NetworkConfig.from_checkpoint(saved_config)
        )
        self.assertIs(type(restored.actor_encoder), ActorEncoder)
        self.assertFalse(any("attention" in key for key in legacy.state_dict()))
        restored.load_state_dict(legacy.state_dict(), strict=True)
        torch.testing.assert_close(
            restored.actor_encoder(self.observation),
            legacy.actor_encoder(self.observation),
        )

    def test_attention_config_roundtrip_and_validation(self):
        restored = NetworkConfig.from_checkpoint(asdict(self.config))
        self.assertEqual(restored, self.config)
        for values in (
            {"attention_heads": 0},
            {"attention_heads": 5},
            {"attention_grid": (1, 1)},
            {"attention_grid": (0, 6)},
            {"attention_grid": (4,)},
            {"attention_grid": (4, 2.5)},
            {"attention_residual_init": float("nan")},
        ):
            with (
                self.subTest(values=values),
                self.assertRaisesRegex(ValueError, "attention"),
            ):
                replace(self.config, **values).validate()

    def test_export_roundtrip_for_both_architectures(self):
        for architecture in ("tutorial_cnn_v1", "cross_attention_cnn_v1"):
            with (
                self.subTest(architecture=architecture),
                TemporaryDirectory() as directory,
            ):
                config = replace(self.config, architecture=architecture)
                model = AsymmetricRecurrentActorCritic(config)
                saved_config = asdict(TrainConfig(network=config))
                if architecture == "tutorial_cnn_v1":
                    # 旧配置没有注意力字段，导出后仍必须构造原有网络。
                    for key in (
                        "attention_heads",
                        "attention_grid",
                        "attention_residual_init",
                    ):
                        saved_config["network"].pop(key)
                checkpoint = Path(directory) / "model.pt"
                torch.save(
                    {
                        "config": saved_config,
                        "policy": model.state_dict(),
                        "action_transform": "clamp",
                    },
                    checkpoint,
                )
                artifact = torch.load(
                    export_actor(checkpoint, Path(directory) / "actor.pt"),
                    weights_only=True,
                )
                restored = AsymmetricRecurrentActorCritic(
                    NetworkConfig.from_checkpoint(artifact["network"])
                )
                incompatible = restored.load_state_dict(artifact["actor"], strict=False)
                self.assertEqual(incompatible.unexpected_keys, [])
                self.assertTrue(
                    all(
                        key == "log_std" or key.startswith(("critic_", "value_head."))
                        for key in incompatible.missing_keys
                    )
                )
                observation = {key: value[0] for key, value in self.observation.items()}
                starts = torch.zeros(3, dtype=torch.bool)
                torch.testing.assert_close(
                    model.act_actor(observation, model.initial_state(3), starts)[0],
                    restored.act_actor(observation, restored.initial_state(3), starts)[
                        0
                    ],
                )


if __name__ == "__main__":
    unittest.main()
