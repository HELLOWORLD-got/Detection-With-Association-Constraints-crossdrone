"""Single-view feature extraction, Eq. (5): ``F_k = Backbone(I_k)``.

The paper evaluates ResNet-50 (default) and ViT-B/16, both ImageNet pre-trained,
and uses a D = 256 channel feature map for every view. torchvision backbones
produce 2048 / 768 channels, so a 1x1 projection to 256 is attached to each,
which is also what the reference implementation does.

Spatial reduction: ResNet-50 -> /32, ViT-B/16 -> /16.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBackbone(nn.Module):
    """ResNet-50 truncated after ``layer4`` + 1x1 projection to ``dim``."""

    stride = 32

    def __init__(self, dim: int = 256, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = resnet50(weights=weights)
        # Everything up to and including layer4: stride 32, 2048 channels.
        self.stem = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
        )
        self.channels = 2048
        self.projection = nn.Conv2d(self.channels, dim, kernel_size=1)
        self.dim = dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(self.stem(images))


class ViTBackbone(nn.Module):
    """ViT-B/16 patch tokens reshaped to a 2D grid + linear projection.

    torchvision fixes the positional embedding to 14x14 patches, so it is
    interpolated for other input sizes instead of forcing 224x224 inputs.
    """

    stride = 16

    def __init__(self, dim: int = 256, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision.models import ViT_B_16_Weights, vit_b_16

        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = vit_b_16(weights=weights)
        self.vit.heads.requires_grad_(False)  # The patch-feature path bypasses the classifier.
        self.channels = 768
        self.patch = 16
        self.projection = nn.Conv2d(self.channels, dim, kernel_size=1)
        self.dim = dim

    def _position_embedding(self, gh: int, gw: int) -> torch.Tensor:
        pos = self.vit.encoder.pos_embedding  # (1, 1 + 14*14, 768)
        cls_pos, patch_pos = pos[:, :1], pos[:, 1:]
        side = int(round(patch_pos.shape[1] ** 0.5))
        patch_pos = patch_pos.reshape(1, side, side, -1).permute(0, 3, 1, 2)
        if side != gh or side != gw:
            patch_pos = F.interpolate(
                patch_pos, size=(gh, gw), mode="bicubic", align_corners=False
            )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, gh * gw, -1)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        tokens = self.vit.conv_proj(images)  # (B, 768, gh, gw)
        gh, gw = tokens.shape[-2:]
        tokens = tokens.flatten(2).transpose(1, 2)
        cls = self.vit.class_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self._position_embedding(gh, gw).to(tokens.dtype)
        encoder = self.vit.encoder
        tokens = encoder.ln(encoder.layers(encoder.dropout(tokens)))
        patches = tokens[:, 1:]
        grid = patches.transpose(1, 2).reshape(batch, self.channels, gh, gw)
        return self.projection(grid)


def build_backbone(
    name: str = "resnet50", dim: int = 256, pretrained: bool = True
) -> nn.Module:
    if name in ("resnet50", "resnet", "resnet-50"):
        return ResNetBackbone(dim=dim, pretrained=pretrained)
    if name in ("vit_b_16", "vit", "vit-b/16", "vit_b16"):
        return ViTBackbone(dim=dim, pretrained=pretrained)
    raise ValueError(f"unknown backbone {name!r}; expected resnet50 or vit_b_16")
