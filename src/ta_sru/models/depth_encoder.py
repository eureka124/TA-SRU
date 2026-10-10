"""深度图编码器：RegNetX-400MF 主干 + FPN + VAE 均值头。

结构与权重来自 SRU 参考工程：导航侧实现见 ``sru-navigation-sim`` 的
``mdp/depth_utils/depth_noise_encoder.py``，训练侧同构实现见
``sru-depth-pretraining`` 的 ``network/encoder.py``。单通道深度图先经过
RegNetX-400MF 的三个尺度，由 FPN 融合后只取最高分辨率的 ``feat1``，再经过 VAE
的均值头输出 ``latent_dim`` 通道的特征图。

``state_dict`` 的键名必须与预训练文件逐字对应，因此分支的索引／切片写法照抄参考
实现：切片保留 stage 名（``enc_1.block1.*``），整数索引丢掉包装
（``enc_2.block3-0.*``）。改动任何一处都会让 ``strict=True`` 加载失败。

编码器加载预训练权重后整体冻结：恒为 eval（BatchNorm 使用预训练运行统计），参数
不参与反向，前向不建立计算图。
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import regnet_x_400mf
from torchvision.ops import Conv2dNormActivation, FeaturePyramidNetwork

# 均值头的通道数，同时是参考工程里的 ``latent_dim``。
LATENT_CHANNELS = 64
# 主干的总下采样倍率：stem 与四个 stage 各下采样一次。特征图高宽为 ceil(H/8)、ceil(W/8)。
DOWNSAMPLE_FACTOR = 8
# 预训练文件中属于本模块的前缀；``depth_decoder.`` 是预训练专有部分，加载时丢弃。
WEIGHT_PREFIXES = ("depth_encoder.", "vae_sampler.")
# 预训练文件可能出现的前缀全集，用于识别误传的权重文件。
KNOWN_PREFIXES = frozenset({"depth_encoder", "vae_sampler", "depth_decoder"})
# 仓库内预训练权重的相对路径。
DEFAULT_WEIGHTS = Path("assets/depth_encoder/vae_pretrain_new.pth")
# 冻结编码器一次前向的图像数上限。中间激活随批量线性增长，分批前向把峰值限制在
# 可预期范围内，避免与仿真争抢显存。
FORWARD_CHUNK = 1024


def depth_feature_size(height: int, width: int) -> int:
    """编码器在 ``height × width`` 深度图上的展平输出维度。"""

    return (
        LATENT_CHANNELS
        * -(-height // DOWNSAMPLE_FACTOR)
        * -(-width // DOWNSAMPLE_FACTOR)
    )


def default_weights_path() -> Path:
    """仓库内预训练权重的绝对路径。"""

    return Path(__file__).resolve().parents[3] / DEFAULT_WEIGHTS


def resolve_weights_path(configured: str | None) -> Path:
    """命令行／配置未指定时回退到仓库内权重。"""

    return Path(configured).expanduser() if configured else default_weights_path()


def load_pretrained_weights(module: nn.Module, path: Path) -> None:
    """把预训练文件里属于 ``module`` 的子集严格加载进去。

    预训练文件同时包含 VAE 解码器，这里按前缀过滤后仍用 ``strict=True``，缺键和
    多键都会立刻报错。
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"深度编码器预训练权重不存在：{path}。"
            f"仓库默认权重位于 {DEFAULT_WEIGHTS}，配置项为 NetworkConfig.depth_encoder_weights。"
        )
    # Git LFS 指针文件同样是"存在"的，但内容不是权重；这里提前给出可操作的报错。
    if path.stat().st_size < 1024:
        head = path.read_bytes()[:64]
        if head.startswith(b"version https://git-lfs"):
            raise ValueError(
                f"{path} 是 Git LFS 指针文件而不是真实权重，请先执行 git lfs pull"
            )

    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{path} 不是有效的深度编码器权重文件")

    prefixes = {key.split(".")[0] for key in state}
    if prefixes - KNOWN_PREFIXES:
        raise ValueError(
            f"{path} 不是深度编码器预训练权重（发现前缀 {sorted(prefixes)}）"
        )

    weights = {k: v for k, v in state.items() if k.startswith(WEIGHT_PREFIXES)}
    module.load_state_dict(weights, strict=True)


class VAESampler(nn.Module):
    """VAE 采样器的均值分支。

    参考实现同时定义了 logvar 分支与重参数化函数，但前向只走均值分支；这里保留
    logvar 分支以满足 ``strict=True`` 加载（41,152 个死权重），前向不执行它。
    """

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.conv = Conv2dNormActivation(
            input_dim, latent_dim, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.mean_layers = nn.Sequential(
            Conv2dNormActivation(
                latent_dim, latent_dim, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, stride=1, padding=0),
        )
        self.logvar_layers = nn.Sequential(
            Conv2dNormActivation(
                latent_dim, latent_dim, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.mean_layers(self.conv(feature))


class EncoderFPN(nn.Module):
    """RegNetX-400MF 主干加 FPN，输出 ``out_channel`` 通道的 ``feat1``。"""

    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()
        # children 的顺序是 [stem, trunk, avgpool, fc]，去掉最后两项即分类头。
        # weights=None：权重一律来自预训练文件，避免训练机联网下载 ImageNet 权重。
        encoder = regnet_x_400mf(weights=None)
        encoder = nn.Sequential(*list(encoder.children())[:-2])
        # 只替换 stem 的卷积，保留其 BatchNorm 与 ReLU。
        encoder[0][0] = nn.Conv2d(
            in_channel, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.enc = encoder[0]
        self.enc_1 = encoder[1][:2]
        self.enc_2 = encoder[1][2]
        self.enc_3 = encoder[1][3]
        self.fpn = FeaturePyramidNetwork([64, 160, 400], out_channel)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat1 = self.enc_1(self.enc(image))
        feat2 = self.enc_2(feat1)
        feat3 = self.enc_3(feat2)
        out = self.fpn({"feat1": feat1, "feat2": feat2, "feat3": feat3})
        return out["feat1"]


class DepthEncoder(EncoderFPN):
    """单通道深度图编码器，输出 64 通道、边长 1/8 的特征图。"""

    def __init__(self, out_channel: int) -> None:
        super().__init__(1, out_channel)


class DepthFeatureEncoder(nn.Module):
    """冻结的深度特征提取器：米制深度图 ``(..., 1, H, W)`` → ``(..., 特征维度)``。

    子模块名 ``depth_encoder`` 与 ``vae_sampler`` 直接构成预训练文件的键前缀，不能
    改名。对外契约与旧的 ``ImageEncoder`` 一致：支持任意前导维度，输出已展平。
    """

    def __init__(self, weights: str | None = None) -> None:
        super().__init__()
        self.depth_encoder = DepthEncoder(LATENT_CHANNELS)
        self.vae_sampler = VAESampler(LATENT_CHANNELS, LATENT_CHANNELS)
        load_pretrained_weights(self, resolve_weights_path(weights))
        self.requires_grad_(False)
        self.eval()

    @property
    def feature_size(self) -> int:
        return LATENT_CHANNELS

    def train(self, mode: bool = True) -> DepthFeatureEncoder:
        """冻结的预训练编码器恒为 eval。

        策略整体的 ``train()`` 会递归切到训练态；若不拦住，BatchNorm 会用当前批次
        统计并污染预训练的 ``running_mean``／``running_var``。
        """

        del mode
        return super().train(False)

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.vae_sampler(self.depth_encoder(images)).flatten(1)

    @torch.no_grad()
    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        prefix = depth.shape[:-3]
        images = depth.reshape(-1, *depth.shape[-3:]).float()
        if images.shape[0] <= FORWARD_CHUNK:
            encoded = self._encode(images)
        else:
            encoded = torch.cat(
                [self._encode(chunk) for chunk in images.split(FORWARD_CHUNK)], dim=0
            )
        return encoded.reshape(*prefix, -1)
