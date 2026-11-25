"""Top-level model, Eq. (1)-(4) and Eq. (14).

    Y_hat = { ( p_hat_i, { (b_hat_1i, s_hat_1i), ..., (b_hat_Ki, s_hat_Ki) } ) }

Why the association cannot conflict: instance query i is a single slot of the
whole scene. The K per-view predictions are reported under the same global
ID i, with at most one reported box per (view, ID) pair. Hungarian supervision
trains distinct slots to represent distinct physical objects.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import build_backbone
from .config import ModelConfig
from .giq import GlobalInstanceQuery
from .loss import SetCriterion
from .matcher import HungarianMatcher
from .mvoq import MultiViewObjectQuery
from .mvff import MultiViewFeatureFusion


class ConflictFreeMultiViewAssociation(nn.Module):
    def __init__(self, config: ModelConfig | None = None, **overrides) -> None:
        super().__init__()
        config = config or ModelConfig()
        if overrides:
            from dataclasses import replace

            config = replace(config, **overrides)
        self.config = config

        self.backbone = build_backbone(
            config.backbone, config.hidden_dim, config.pretrained
        )
        if config.freeze_backbone:
            self.backbone.eval()
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

        self.mvff = MultiViewFeatureFusion(
            dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_views=config.num_views,
            num_layers=config.mvff_layers,
            ffn_ratio=config.ffn_ratio,
            dropout=config.dropout,
        )
        self.giq = GlobalInstanceQuery(
            dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_queries=config.num_queries,
            num_classes=config.num_classes,
            num_layers=config.giq_layers,
            ffn_ratio=config.ffn_ratio,
            dropout=config.dropout,
        )
        self.mvoq = MultiViewObjectQuery(
            dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_views=config.num_views,
            num_queries=config.num_queries,
            num_layers=config.mvoq_layers,
            share_attention=config.mvoq_share_attention,
            content_key=config.mvoq_content_key,
            ffn_ratio=config.ffn_ratio,
            dropout=config.dropout,
        )

    # ------------------------------------------------------------------ #
    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_backbone:
            self.backbone.eval()
        return self

    def _as_view_list(self, images) -> list[torch.Tensor]:
        if isinstance(images, torch.Tensor):
            if images.dim() != 5:
                raise ValueError(
                    "a single tensor input must be (B, K, 3, H, W); got shape "
                    f"{tuple(images.shape)}"
                )
            return list(images.unbind(dim=1))
        views = list(images)
        if not views:
            raise ValueError("expected at least one view")
        return views

    def forward(self, images) -> dict[str, torch.Tensor]:
        """Args:
            images: list of K tensors, each ``(B, 3, H, W)``, or one tensor
                ``(B, K, 3, H, W)``.

        Returns:
            ``class_logits`` ``(B, N, C+1)``, ``class_probs`` ``(B, N, C+1)``,
            ``boxes`` ``(B, K, N, 4)`` normalized center-size,
            ``confidences`` ``(B, K, N)``, plus the intermediate features.
        """
        views = self._as_view_list(images)

        # Eq. (5): per-view features.
        view_features = [self.backbone(view) for view in views]
        # Eq. (2): multi-view fusion.
        global_feature = self.mvff(view_features)
        # Eq. (3): global instance queries.
        giq_output = self.giq(global_feature)
        # Eq. (4): per-view object queries.
        mvoq_output = self.mvoq(giq_output["instance_features"], view_features)

        return {
            "class_logits": giq_output["class_logits"],
            "class_probs": giq_output["class_probs"],
            "instance_features": giq_output["instance_features"],
            "boxes": mvoq_output["boxes"],
            "confidences": mvoq_output["confidences"],
            "confidence_logits": mvoq_output["confidence_logits"],
            "global_feature": global_feature,
            "view_features": view_features,
        }

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(
        self,
        images,
        score_threshold: float = 0.5,
        class_threshold: float | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """Decode the output set into per-view detections with global IDs.

        Each detection carries the instance index ``i`` as its global ID, which
        is the model's identity output (Eq. 14).  Sorted by confidence.

        Returns one dict per batch element::

            {"labels": (N,), "boxes": (K, N, 4), "scores": (K, N),
             "valid": (K, N), "detections": [K x list of dict]}
        """
        outputs = self.forward(images)
        class_probs = outputs["class_probs"]
        boxes = outputs["boxes"]
        confidences = outputs["confidences"]
        num_classes = self.config.num_classes
        batch, num_views, _ = confidences.shape

        labels = class_probs.argmax(dim=-1)  # (B, N)
        class_scores = class_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        keep = labels != num_classes
        if class_threshold is not None:
            keep = keep & (class_scores >= class_threshold)
        valid = keep.unsqueeze(1) & (confidences >= score_threshold)

        results = []
        for b in range(batch):
            per_view = []
            for k in range(num_views):
                ids = torch.nonzero(valid[b, k], as_tuple=False).flatten()
                order = ids[torch.argsort(confidences[b, k, ids], descending=True)]
                per_view.append(
                    {
                        "global_ids": order,
                        "labels": labels[b, order],
                        "boxes": boxes[b, k, order],
                        "scores": confidences[b, k, order],
                    }
                )
            results.append(
                {
                    "labels": labels[b],
                    "boxes": boxes[b],
                    "scores": confidences[b],
                    "valid": valid[b],
                    "detections": per_view,
                }
            )
        return results

    # ------------------------------------------------------------------ #
    def build_criterion(self) -> SetCriterion:
        """Loss + matcher with the weights of ``self.config``."""
        matcher = HungarianMatcher(
            cost_class=self.config.cost_class,
            cost_bbox=self.config.cost_bbox,
            weight_l1=self.config.weight_l1,
            weight_giou=self.config.weight_giou,
        )
        return SetCriterion(
            num_classes=self.config.num_classes,
            matcher=matcher,
            weight_cls=self.config.weight_cls,
            weight_reg=self.config.weight_reg,
            weight_conf=self.config.weight_conf,
            weight_l1=self.config.weight_l1,
            weight_giou=self.config.weight_giou,
        )

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone = sum(p.numel() for p in self.backbone.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "backbone": backbone,
            "heads": total - backbone,
        }


def build_model(config: ModelConfig | None = None, **overrides) -> ConflictFreeMultiViewAssociation:
    return ConflictFreeMultiViewAssociation(config, **overrides)
