"""冻结深度编码器与深度观测预处理的回归测试。"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch

from ta_sru.config import EnvConfig, NetworkConfig, TrainConfig
from ta_sru.envs.depth import (
    downsample_depth,
    mask_invalid_depth,
    sanitize_depth,
)
from ta_sru.models.actor_critic import AsymmetricRecurrentActorCritic
from ta_sru.models.depth_encoder import (
    DepthFeatureEncoder,
    default_weights_path,
    depth_feature_size,
    load_pretrained_weights,
    resolve_weights_path,
)

# 测试用的最小网络：12×16 深度图对应 64×2×2 = 256 维特征。
SMALL_FEATURE_DIM = depth_feature_size(12, 16)


def _small_network() -> NetworkConfig:
    return NetworkConfig(
        feature_dim=SMALL_FEATURE_DIM,
        recurrent_hidden_size=8,
        actor_hidden_sizes=(8,),
        critic_hidden_sizes=(8,),
    )


def _pretrained_state() -> dict[str, torch.Tensor]:
    return torch.load(
        default_weights_path(), map_location="cpu", weights_only=True
    )


class DepthEncoderTests(unittest.TestCase):
    def test_module_keys_match_pretrained_file(self) -> None:
        """键集必须与预训练文件逐字对应。

        参考实现里 ``enc_1`` 用切片、``enc_2``/``enc_3`` 用整数索引，两者产生的
        键名不同；这个断言把结构钉死，防止后续重构悄悄改坏。
        """

        encoder = DepthFeatureEncoder()
        expected = {
            key
            for key in _pretrained_state()
            if key.startswith(("depth_encoder.", "vae_sampler."))
        }
        self.assertEqual(set(encoder.state_dict()), expected)

    def test_feature_size_follows_formula(self) -> None:
        encoder = DepthFeatureEncoder()
        for height, width in ((48, 64), (40, 64), (24, 32)):
            with self.subTest(size=(height, width)):
                with torch.no_grad():
                    feature = encoder(torch.zeros(2, 1, height, width))
                self.assertEqual(feature.shape[1], depth_feature_size(height, width))
        self.assertEqual(depth_feature_size(48, 64), 3072)

    def test_supports_leading_dimensions_and_chunking(self) -> None:
        encoder = DepthFeatureEncoder()
        with torch.no_grad():
            batched = encoder(torch.zeros(2, 3, 1, 24, 32))
        self.assertEqual(batched.shape, (2, 3, depth_feature_size(24, 32)))
        self.assertFalse(batched.requires_grad)

        # 分批前向只为限制显存峰值，不改变结果。批量形状不同会让底层卷积挑选不同
        # 的实现，因此这里是数值等价而不是逐位相等。
        images = torch.rand(5, 1, 24, 32)
        with (
            mock.patch("ta_sru.models.depth_encoder.FORWARD_CHUNK", 2),
            torch.no_grad(),
        ):
            chunked = encoder(images)
            whole = encoder._encode(images)
        torch.testing.assert_close(whole, chunked, atol=1e-5, rtol=1e-4)

    def test_encoder_is_frozen_and_always_in_eval(self) -> None:
        encoder = DepthFeatureEncoder()
        self.assertFalse(any(p.requires_grad for p in encoder.parameters()))
        # 策略整体的 train() 不能把编码器切回训练态，否则 BatchNorm 会用批次统计
        # 并污染预训练的 running_mean／running_var。
        encoder.train()
        self.assertFalse(encoder.training)

    def test_orthogonal_initialization_keeps_pretrained_weights(self) -> None:
        """策略的正交初始化直接写 .data，必须显式跳过冻结编码器。"""

        model = AsymmetricRecurrentActorCritic(_small_network())
        state = _pretrained_state()
        for key, value in model.depth_encoder.state_dict().items():
            self.assertTrue(torch.equal(value, state[key]), key)

    def test_missing_weights_fail_loudly(self) -> None:
        encoder = DepthFeatureEncoder()
        with self.assertRaisesRegex(FileNotFoundError, "预训练权重不存在"):
            load_pretrained_weights(encoder, Path("/nonexistent/weights.pth"))

    def test_lfs_pointer_and_wrong_file_fail_loudly(self) -> None:
        encoder = DepthFeatureEncoder()
        with tempfile.TemporaryDirectory() as directory:
            pointer = Path(directory) / "pointer.pth"
            pointer.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:x\n")
            with self.assertRaisesRegex(ValueError, "git lfs pull"):
                load_pretrained_weights(encoder, pointer)

            wrong = Path(directory) / "wrong.pth"
            torch.save({"unrelated.weight": torch.zeros(1)}, wrong)
            with self.assertRaisesRegex(ValueError, "不是深度编码器预训练权重"):
                load_pretrained_weights(encoder, wrong)

    def test_configured_path_overrides_default(self) -> None:
        self.assertEqual(resolve_weights_path(None), default_weights_path())
        self.assertEqual(
            resolve_weights_path("~/custom.pth"), Path("~/custom.pth").expanduser()
        )

    def test_shared_encoder_is_single_copy(self) -> None:
        shared = AsymmetricRecurrentActorCritic(_small_network())
        self.assertIsNone(shared.critic_depth_encoder)
        separate = AsymmetricRecurrentActorCritic(
            replace(_small_network(), share_depth_encoder=False)
        )
        frozen = sum(
            p.numel() for p in shared.parameters() if not p.requires_grad
        )
        self.assertEqual(
            sum(p.numel() for p in separate.parameters() if not p.requires_grad),
            2 * frozen,
        )


class DepthObservationTests(unittest.TestCase):
    def test_downsample_takes_nearest_surface(self) -> None:
        depth = torch.full((1, 1, 4, 8), 9.0)
        depth[0, 0, 0, 0] = 0.5
        pooled = downsample_depth(depth)
        self.assertEqual(pooled.shape, (1, 1, 1, 2))
        self.assertAlmostEqual(float(pooled[0, 0, 0, 0]), 0.5, places=6)

    def test_invalid_and_out_of_range_become_zero(self) -> None:
        depth = torch.tensor(
            [[[[0.1, 1.0, 9.9, 10.0], [float("nan"), float("inf"), -1.0, 0.0]]]]
        )
        observed = mask_invalid_depth(
            sanitize_depth(depth, invalid_distance=50.0),
            min_distance=0.25,
            max_distance=10.0,
        )
        expected = torch.tensor([[[[0.0, 1.0, 9.9, 0.0], [0.0, 0.0, 0.0, 0.0]]]])
        torch.testing.assert_close(observed, expected)

    def test_nan_is_replaced_before_pooling(self) -> None:
        """一次 NaN 不能污染整个池化块。

        最小池化遇到 NaN 会返回 NaN，若把替换放到池化之后，块内其他像素上真实的
        近距读数会被整块抹掉。
        """

        invalid = float("nan")
        depth = torch.tensor(
            [
                [[[0.5, 0.5], [0.5, invalid]]],
                [[[invalid, invalid], [invalid, invalid]]],
            ]
        )
        pooled = downsample_depth(
            sanitize_depth(depth, invalid_distance=50.0), factor=2
        )
        self.assertAlmostEqual(float(pooled[0, 0, 0, 0]), 0.5, places=6)
        self.assertAlmostEqual(float(pooled[1, 0, 0, 0]), 50.0, places=6)
        observed = mask_invalid_depth(pooled, min_distance=0.25, max_distance=10.0)
        self.assertAlmostEqual(float(observed[0, 0, 0, 0]), 0.5, places=6)
        # 整块都无效时仍然是无效。
        self.assertAlmostEqual(float(observed[1, 0, 0, 0]), 0.0, places=6)


class DepthEncoderConfigTests(unittest.TestCase):
    def test_default_config_is_self_consistent(self) -> None:
        config = TrainConfig()
        config.validate()
        self.assertEqual(
            config.network.feature_dim,
            depth_feature_size(config.env.depth_height, config.env.depth_width),
        )

    def test_feature_dim_must_match_encoder_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "feature_dim 必须等于深度编码器输出"):
            TrainConfig(network=NetworkConfig(feature_dim=512)).validate()

    def test_depth_range_order_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "深度量程"):
            EnvConfig(depth_min_distance=20.0).validate()
        with self.assertRaisesRegex(ValueError, "深度量程"):
            EnvConfig(depth_invalid_distance=5.0).validate()


if __name__ == "__main__":
    unittest.main()
