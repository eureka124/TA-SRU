"""验证 CNN 显存优化保持输出与梯度，并减少保留的反向传播张量。"""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from ta_sru.config import NetworkConfig
from ta_sru.models.actor_critic import ImageEncoder


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
