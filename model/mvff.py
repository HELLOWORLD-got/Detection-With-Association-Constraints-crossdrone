"""Multi-View Feature Fusion Module (MVFF), Sec. IV-B, Eq. (5)-(8).

Two stages:

1. Cross-view attention (Eq. 6).  For every view k, that view's tokens are the
   queries and the tokens of *all other* views concatenated along the token
   axis are keys/values.  Three layers, 8 heads, d = 256.
2. Global self-attention (Eq. 7-8).  The K cross-view feature maps are
   concatenated along the channel axis into a ``256*K x H' x W'`` map, then
   three layers of self-attention produce ``F_G`` with 256 channels.

The paper does not mention the channel reduction that stage 2 needs: attention
with 8 heads cannot consume 256*K channels and still output 256.  A 1x1
convolution is therefore applied to the concatenated map before the SA layers
(one linear projection, no non-linearity).  Everything else follows the text
literally.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .blocks import CrossAttentionBlock, SelfAttentionBlock


class MultiViewFeatureFusion(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        num_views: int = 4,
        num_layers: int = 3,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_views = num_views
        self.num_layers = num_layers

        self.ca_layers = nn.ModuleList(
            [
                CrossAttentionBlock(dim, num_heads, ffn_ratio, dropout)
                for _ in range(num_layers)
            ]
        )
        # 256*K -> 256, required before self-attention (see module docstring).
        self.fuse_projection = nn.Conv2d(dim * num_views, dim, kernel_size=1)
        self.sa_layers = nn.ModuleList(
            [
                SelfAttentionBlock(dim, num_heads, ffn_ratio, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        """``features``: K tensors of shape ``(B, dim, H', W')``.

        Returns ``F_G`` of shape ``(B, dim, H', W')`` (Eq. 2).
        """
        if len(features) != self.num_views:
            raise ValueError(
                f"MVFF was built for K={self.num_views} views, got {len(features)}"
            )
        batch, dim, height, width = features[0].shape
        tokens = [f.flatten(2).transpose(1, 2) for f in features]  # (B, H'W', dim)

        # --- stage 1: cross-view attention, Eq. (6) -------------------------
        cross_view = []
        for k in range(self.num_views):
            others = [tokens[j] for j in range(self.num_views) if j != k]
            # K=1 degenerates to self-attention on the only view.
            keys = torch.cat(others, dim=1) if others else tokens[k]
            x = tokens[k]
            for layer in self.ca_layers:
                x = layer(x, keys)
            cross_view.append(x)

        # --- stage 2: global self-attention, Eq. (7)-(8) --------------------
        concatenated = torch.cat(
            [x.transpose(1, 2).reshape(batch, dim, height * width) for x in cross_view],
            dim=1,
        )
        fused = self.fuse_projection(concatenated.reshape(batch, dim * self.num_views, height, width))
        tokens_out = fused.flatten(2).transpose(1, 2)  # (B, H'W', dim)
        for layer in self.sa_layers:
            tokens_out = layer(tokens_out)

        return tokens_out.transpose(1, 2).reshape(batch, dim, height, width)
