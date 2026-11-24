"""Global Instance Query Module (GIQ), Sec. IV-C, Eq. (9)-(10).

N = 25 learnable instance queries (paper: "set to 25", a little more than the
largest number of objects per scene).  A fixed 2D sine-cosine position encoding
is added to ``F_G``, the queries attend to the encoded map, and the resulting
instance features carry the identity of one physical object each.  The class
head maps every instance feature to ``C + 1`` probabilities, the extra class
being the no-object symbol of Eq. (10).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import CrossAttentionBlock, PositionEncoding2D


class GlobalInstanceQuery(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        num_queries: int = 25,
        num_classes: int = 10,
        num_layers: int = 3,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries
        self.num_classes = num_classes

        self.instance_queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.position_encoding = PositionEncoding2D(dim)
        self.layers = nn.ModuleList(
            [CrossAttentionBlock(dim, num_heads, ffn_ratio, dropout) for _ in range(num_layers)]
        )

        # Eq. (10): FFN + Softmax over C + 1 classes.
        self.class_head = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, num_classes + 1),
        )

    def forward(self, global_feature: torch.Tensor) -> dict[str, torch.Tensor]:
        """``global_feature``: ``F_G`` of shape ``(B, dim, H', W')``.

        Returns instance features ``(B, N, dim)`` and the class distribution
        ``(B, N, C+1)``.  The no-object column is the last one, ``index C``.
        """
        batch, _, height, width = global_feature.shape
        memory = global_feature.flatten(2).transpose(1, 2)  # (B, H'W', dim)
        pe = self.position_encoding(
            height, width, device=global_feature.device, dtype=global_feature.dtype
        )
        memory = memory + pe.unsqueeze(0)

        queries = self.instance_queries.unsqueeze(0).expand(batch, -1, -1)
        for layer in self.layers:
            queries = layer(queries, memory)

        class_logits = self.class_head(queries)  # (B, N, C+1)
        class_probs = class_logits.softmax(dim=-1)
        return {
            "instance_features": queries,
            "class_logits": class_logits,
            "class_probs": class_probs,
        }
