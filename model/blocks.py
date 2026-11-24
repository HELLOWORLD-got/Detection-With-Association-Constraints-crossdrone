"""Transformer building blocks shared by the three modules.

The paper fixes one configuration everywhere (Sec. IV-B2):
    hidden dimension d = 256, 8 attention heads, q/k/v projected to 256,
    and every sublayer is followed by a residual connection and layer
    normalization ("layer normalization is applied after each layer with a
    residual connection").

All blocks below therefore use post-normalization, i.e. ``LN(x + f(x))``,
which is what that sentence describes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FFN(nn.Module):
    """Two-layer feed-forward network used inside every transformer stage.

    In the paper the FFNs of the classification / regression / confidence
    heads are described only as "FFN" (Eq. 10, 12, 13); the internal ratio of
    four is the standard DETR/ViT choice and is exposed as ``ratio``.
    """

    def __init__(self, dim: int, ratio: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = int(dim * ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    """Eq. (6): ``F_k,cross = CA(F_k, F_{-k})``.

    ``queries`` come from one view, ``keys``/``values`` from the other views.
    Post-norm residual, then the position-wise FFN (again post-norm).
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.drop1 = nn.Dropout(dropout)
        self.ffn = FFN(dim, ffn_ratio, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = keys if values is None else values
        attn, _ = self.attn(
            queries,
            keys,
            values,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(queries + self.drop1(attn))
        x = self.norm2(x + self.drop2(self.ffn(x)))
        return x


class SelfAttentionBlock(nn.Module):
    """Eq. (8): ``F_G = SA(F_concat)`` — self-attention over a token set."""

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.drop1 = nn.Dropout(dropout)
        self.ffn = FFN(dim, ffn_ratio, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.drop1(attn))
        x = self.norm2(x + self.drop2(self.ffn(x)))
        return x


class DecoderBlock(nn.Module):
    """Transformer decoder with query self-attention and memory cross-attention."""

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_sa = nn.LayerNorm(dim)
        self.drop_sa = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_ca = nn.LayerNorm(dim)
        self.drop_ca = nn.Dropout(dropout)

        self.ffn = FFN(dim, ffn_ratio, dropout)
        self.norm_ffn = nn.LayerNorm(dim)
        self.drop_ffn = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        sa, _ = self.self_attn(queries, queries, queries, need_weights=False)
        x = self.norm_sa(queries + self.drop_sa(sa))
        ca, _ = self.cross_attn(x, memory, memory, need_weights=False)
        x = self.norm_ca(x + self.drop_ca(ca))
        x = self.norm_ffn(x + self.drop_ffn(self.ffn(x)))
        return x


class PositionEncoding2D(nn.Module):
    """Fixed 2D sine-cosine position encoding added to ``F_G`` (Sec. IV-C1).

    The first half of the channels encodes the vertical coordinate, the second
    half the horizontal one; every pair is ``sin``/``cos`` of the same
    frequency. Coordinates are normalized to ``[0, 1]`` so the encoding does
    not depend on the feature-map size.
    """

    def __init__(
        self,
        dim: int = 256,
        temperature: float = 10000.0,
        scale: float = 2.0 * math.pi,
    ) -> None:
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f"position encoding needs dim divisible by 4, got {dim}")
        self.dim = dim
        self.temperature = temperature
        self.scale = scale

    def _axis_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (H, W) in [0, 1] -> (H, W, dim/2)."""
        half = self.dim // 2
        num_freqs = half // 2
        idx = torch.arange(num_freqs, dtype=torch.float32, device=coords.device)
        omega = 1.0 / (self.temperature ** (idx / num_freqs))
        angles = self.scale * coords.unsqueeze(-1) * omega  # (H, W, num_freqs)
        enc = torch.empty(
            *coords.shape, half, dtype=coords.dtype, device=coords.device
        )
        enc[..., 0::2] = torch.sin(angles)
        enc[..., 1::2] = torch.cos(angles)
        return enc

    def forward(
        self,
        height: int,
        width: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Returns ``(height * width, dim)``."""
        ys = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        enc_y = self._axis_encoding(yy.to(torch.float32)).to(dtype)
        enc_x = self._axis_encoding(xx.to(torch.float32)).to(dtype)
        enc = torch.cat([enc_y, enc_x], dim=-1)  # (H, W, dim)
        return enc.reshape(height * width, self.dim)
