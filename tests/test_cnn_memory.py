"""验证 CNN 显存优化保持输出与梯度，并减少保留的反向传播张量。"""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from ta_sru.config import NetworkConfig
from ta_sru.models.actor_critic import (
    AsymmetricRecurrentActorCritic,
    DeterministicAdaptiveAvgPool2d,
    ImageEncoder,
)


class DeterministicPoolingTests(unittest.TestCase):
    def setUp(self) -> None:
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        self.addCleanup(
            torch.use_deterministic_algorithms, enabled, warn_only=warn_only
        )
        torch.use_deterministic_algorithms(True)

    def test_matches_native_outputs_and_gradients(self) -> None:
        # 包含整除、重叠分区、输出大于输入，以及实际 CNN 的空间尺寸。
        for height, width, output in (
            (8, 12, (2, 3)),
            (7, 11, (4, 6)),
            (1, 2, (4, 6)),
            (7, 11, (2, 3)),
            (43, 59, (4, 6)),
        ):
            with self.subTest(size=(height, width), output=output):
                original = torch.randn(
                    2, 3, height, width, dtype=torch.float64, requires_grad=True
                )
                image = original.detach().clone().requires_grad_()
                expected = torch.nn.functional.adaptive_avg_pool2d(original, output)
                actual = DeterministicAdaptiveAvgPool2d(output)(image)
                torch.testing.assert_close(actual, expected)
                weight = torch.randn_like(expected)
                (expected * weight).sum().backward()
                (actual * weight).sum().backward()
                torch.testing.assert_close(image.grad, original.grad)

    def test_actor_and_critic_backward_avoid_native_pooling(self) -> None:
        for architecture in ("tutorial_cnn_v1", "cross_attention_cnn_v1"):
            with self.subTest(architecture=architecture):
                model = AsymmetricRecurrentActorCritic(
                    NetworkConfig(
                        architecture=architecture,
                        feature_dim=24,
                        recurrent_hidden_size=8,
                        actor_hidden_sizes=(8,),
                        critic_hidden_sizes=(8,),
                    )
                )
                observation = {
                    "camera": torch.rand(2, 1, 12, 16),
                    "robot_state": torch.rand(2, 8),
                    "critic_toa": torch.rand(2, 1, 16, 16),
                }
                with mock.patch(
                    "torch.nn.functional.adaptive_avg_pool2d",
                    side_effect=AssertionError("不能调用原生自适应平均池化"),
                ):
                    loss = model.actor_encoder(observation).square().sum()
                    loss = loss + model.critic_encoder(observation).square().sum()
                    loss.backward()
                for encoder in (model.actor_encoder, model.critic_encoder):
                    for parameter in encoder.parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_cuda_backward_is_deterministic(self) -> None:
        original = torch.randn(2, 3, 7, 11, requires_grad=True)
        expected = torch.nn.functional.adaptive_avg_pool2d(original, (4, 6))
        weight = torch.randn_like(expected)
        (expected * weight).sum().backward()
        gradients = []
        for _ in range(2):
            image = original.detach().cuda().requires_grad_()
            actual = DeterministicAdaptiveAvgPool2d((4, 6))(image)
            (actual * weight.cuda()).sum().backward()
            torch.testing.assert_close(actual.cpu(), expected)
            torch.testing.assert_close(image.grad.cpu(), original.grad)
            gradients.append(image.grad)
        self.assertTrue(torch.equal(*gradients))


class ImageEncoderMemoryTests(unittest.TestCase):
    def test_chunked_checkpoint_matches_outputs_and_gradients(self) -> None:
        for input_requires_grad in (False, True):
            for channels in ((8, 16), (8, 16, 32)):
                with self.subTest(input_grad=input_requires_grad, channels=channels):
                    whole = ImageEncoder(channels, 12, gradient_checkpointing=False)
                    chunked = ImageEncoder(channels, 12, chunk_size=4)
                    chunked.load_state_dict(whole.state_dict(), strict=True)
                    # 同时覆盖时间维度、末尾不足一个块及原始观测不求梯度的情况。
                    image = torch.rand(3, 3, 1, 12, 16)
                    original = image.clone().requires_grad_(input_requires_grad)
                    limited = image.clone().requires_grad_(input_requires_grad)
                    expected = whole(original)
                    actual = chunked(limited)
                    self.assertEqual(actual.shape, (3, 3, 12))
                    torch.testing.assert_close(actual, expected)
                    weight = torch.randn_like(expected)
                    (expected * weight).sum().backward()
                    (actual * weight).sum().backward()
                    for name, parameter in whole.named_parameters():
                        gradient = dict(chunked.named_parameters())[name].grad
                        self.assertIsNotNone(gradient)
                        torch.testing.assert_close(
                            gradient, parameter.grad, atol=1e-5, rtol=1e-4
                        )
                    if input_requires_grad:
                        torch.testing.assert_close(limited.grad, original.grad)

    def test_no_grad_and_eval_skip_checkpoint_but_still_chunk(self) -> None:
        encoder = ImageEncoder((8, 16, 32), 12, chunk_size=4)
        image = torch.rand(9, 1, 12, 16)
        with (
            mock.patch("ta_sru.models.actor_critic.checkpoint") as checkpoint,
            mock.patch.object(encoder, "_encode", wraps=encoder._encode) as encode,
            torch.no_grad(),
        ):
            expected = encoder(image)
            checkpoint.assert_not_called()
            self.assertEqual(encode.call_count, 3)
            self.assertEqual(
                [call.args[0].shape[0] for call in encode.call_args_list], [4, 4, 1]
            )
        encoder.eval()
        with mock.patch("ta_sru.models.actor_critic.checkpoint") as checkpoint:
            actual = encoder(image)
            checkpoint.assert_not_called()
        torch.testing.assert_close(actual, expected)

    def test_checkpoint_saves_less_than_one_tenth_of_backward_tensors(self) -> None:
        sizes = []
        for checkpointing in (False, True):
            encoder = ImageEncoder(
                (8, 16, 32),
                12,
                chunk_size=4,
                gradient_checkpointing=checkpointing,
            )
            saved_sizes = []

            def pack(tensor, saved_sizes=saved_sizes):
                saved_sizes.append(tensor.numel() * tensor.element_size())
                return tensor

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                encoded = encoder(torch.rand(9, 1, 12, 16))
            encoded.sum().backward()
            sizes.append(sum(saved_sizes))
        self.assertLess(sizes[1], sizes[0] / 10)

    def test_chunk_size_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            ImageEncoder((8, 16), 12, chunk_size=0)
        with self.assertRaisesRegex(ValueError, "cnn_chunk_size"):
            NetworkConfig(cnn_chunk_size=0).validate()

    def test_existing_tutorial_checkpoint_uses_memory_defaults(self) -> None:
        restored = NetworkConfig.from_checkpoint({"architecture": "tutorial_cnn_v1"})
        self.assertEqual(restored.cnn_chunk_size, 128)
        self.assertTrue(restored.cnn_gradient_checkpointing)


if __name__ == "__main__":
    unittest.main()
