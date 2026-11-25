"""Multi-View Object Query Module (MVOQ), Sec. IV-D, Eq. (11)-(14).

The i-th instance feature owns K object queries, one per view (Eq. 11).  Those
queries are learnable and fixed per view, so instance i is bound to view k by
construction — that binding is what makes the association conflict-free.

Eq. (11) uses the object query as Q, its global instance feature as the
source of K, and the single-view tokens as V. We multiply the instance
feature by a fixed 2D position encoding before key projection to obtain
one instance-specific key per value token.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .blocks import FFN, PositionEncoding2D


class TripletAttentionBlock(nn.Module):
    """One layer of the instance-conditioned query / view-feature attention."""

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_heads < 1 or dim % num_heads:
            raise ValueError("dim must be divisible by a positive num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.position_encoding = PositionEncoding2D(dim)
        self.query_projection = nn.Linear(dim, dim)
        self.key_projection = nn.Linear(dim, dim, bias=False)
        self.value_projection = nn.Linear(dim, dim)
        self.output_projection = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.drop1 = nn.Dropout(dropout)
        self.ffn = FFN(dim, ffn_ratio, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        object_queries: torch.Tensor,
        instance_features: torch.Tensor,
        view_tokens: torch.Tensor,
        spatial_shape: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        batch, instances, dim = object_queries.shape
        tokens = view_tokens.shape[1]
        height, width = spatial_shape if spatial_shape is not None else (1, tokens)
        if height * width != tokens:
            raise ValueError("spatial shape must match the number of view tokens")

        query = self.query_projection(object_queries).view(
            batch, instances, self.num_heads, self.head_dim
        )
        value = self.value_projection(view_tokens).view(
            batch, tokens, self.num_heads, self.head_dim
        )
        positions = self.position_encoding(height, width, query.device, query.dtype)

        # q · W_k(f_i ⊙ p_t) = (q · W_k ⊙ f_i) · p_t. Factoring the dot
        # product avoids allocating a separate full key map for every instance.
        key_weights = self.key_projection.weight.view(self.num_heads, self.head_dim, dim)
        query_in_key_space = torch.einsum("bnhd,hdc->bnhc", query, key_weights)
        modulated = query_in_key_space * instance_features.to(query.dtype).unsqueeze(2)
        scores = torch.einsum("bnhc,tc->bhnt", modulated, positions) * self.head_dim ** -0.5
        weights = self.attention_dropout(scores.softmax(dim=-1))
        context = torch.einsum("bhnt,bthd->bnhd", weights, value)
        attended = self.output_projection(context.reshape(batch, instances, dim))
        x = self.norm1(object_queries + self.drop1(attended))
        x = self.norm2(x + self.drop2(self.ffn(x)))
        return x


class MultiViewObjectQuery(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        num_views: int = 4,
        num_queries: int = 25,
        num_layers: int = 2,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        share_attention: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_views = num_views
        self.num_queries = num_queries

        # {q_obj,k,i}: one object query per (view, instance).
        self.object_queries = nn.Parameter(
            torch.randn(num_views, num_queries, dim) * 0.02
        )
        def attention_stack():
            return nn.ModuleList(
                TripletAttentionBlock(dim, num_heads, ffn_ratio, dropout)
                for _ in range(num_layers)
            )

        self.share_attention = share_attention
        if share_attention:
            # Preserve the original parameter names for legacy checkpoint evaluation.
            self.layers = attention_stack()
        else:
            # Fig. 3 puts the CA stack inside each view's regression branch.
            self.view_layers = nn.ModuleList(attention_stack() for _ in range(num_views))

        # Eq. (12): box head per view, Sigmoid to keep the box in [0, 1]^4.
        # Fig. 3 labels them "View 1 Reg Head" ... "View n Reg Head", and the
        # text says box prediction is performed independently for each view.
        self.reg_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, 2 * dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(2 * dim, 4),
                )
                for _ in range(num_views)
            ]
        )
        # Eq. (13): separate confidence head, per view as well.
        self.conf_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, 2 * dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(2 * dim, 1),
                )
                for _ in range(num_views)
            ]
        )

    def forward(
        self,
        instance_features: torch.Tensor,
        single_view_features: Sequence[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """``instance_features``: ``(B, N, dim)``; K view maps ``(B, dim, H', W')``.

        Returns boxes ``(B, K, N, 4)`` in ``(x, y, w, h)`` normalized form and
        confidences ``(B, K, N)`` — Eq. (14)'s per-instance payload.
        """
        if len(single_view_features) != self.num_views:
            raise ValueError(
                f"MVOQ was built for K={self.num_views} views, got {len(single_view_features)}"
            )
        batch, num_queries, _ = instance_features.shape
        if num_queries != self.num_queries:
            raise ValueError(
                f"MVOQ was built for N={self.num_queries} queries, got {num_queries}"
            )

        per_view = []
        for k in range(self.num_views):
            view = single_view_features[k]
            view_tokens = view.flatten(2).transpose(1, 2)
            queries = self.object_queries[k].unsqueeze(0).expand(batch, -1, -1)
            layers = self.layers if self.share_attention else self.view_layers[k]
            for layer in layers:
                queries = layer(queries, instance_features, view_tokens, view.shape[-2:])
            per_view.append(queries)

        object_features = torch.stack(per_view, dim=1)  # (B, K, N, dim)
        boxes = torch.stack(
            [self.reg_heads[k](object_features[:, k]).sigmoid() for k in range(self.num_views)],
            dim=1,
        )
        confidence_logits = torch.stack(
            [
                self.conf_heads[k](object_features[:, k]).squeeze(-1)
                for k in range(self.num_views)
            ],
            dim=1,
        )
        return {
            "boxes": boxes,
            "confidence_logits": confidence_logits,
            "confidences": confidence_logits.sigmoid(),
            "object_features": object_features,
        }
