"""Conflict-free multi-view object association — model reproduction of the TGRS paper.

Modules follow the paper section by section:

    model.backbone   Eq. (5)        F_k = Backbone(I_k)
    model.mvff       Eq. (6)-(8)    MVFF, cross-view attention + global self-attention
    model.giq        Eq. (9)-(10)   GIQ, instance queries + class head
    model.mvoq       Eq. (11)-(14)  MVOQ, per-view object queries + box / confidence heads
    model.targets    Eq. (15)-(19)  padded ground truth and visibility labels
    model.matcher    Eq. (20)       Hungarian matching
    model.loss       Eq. (21)-(25)  global joint loss
    model.association Eq. (1)-(4)   the assembled model

The Cross-Drone data and training entry points are separate from these modules.
"""

from __future__ import annotations

from .association import ConflictFreeMultiViewAssociation, build_model
from .backbone import ResNetBackbone, ViTBackbone, build_backbone
from .blocks import (
    CrossAttentionBlock,
    DecoderBlock,
    FFN,
    PositionEncoding2D,
    SelfAttentionBlock,
)
from .box_ops import (
    box_l1,
    box_loss,
    box_loss_pairwise,
    box_xywh_to_xyxy,
    generalized_iou_pairwise,
)
from .config import ModelConfig, TrainConfig
from .giq import GlobalInstanceQuery
from .loss import HungarianLoss, SetCriterion
from .matcher import HungarianMatcher, linear_sum_assignment
from .mvoq import MultiViewObjectQuery, TripletAttentionBlock
from .mvff import MultiViewFeatureFusion
from .targets import pad_targets

__all__ = [
    "ConflictFreeMultiViewAssociation",
    "build_model",
    "build_backbone",
    "ResNetBackbone",
    "ViTBackbone",
    "MultiViewFeatureFusion",
    "GlobalInstanceQuery",
    "MultiViewObjectQuery",
    "TripletAttentionBlock",
    "FFN",
    "CrossAttentionBlock",
    "SelfAttentionBlock",
    "DecoderBlock",
    "PositionEncoding2D",
    "HungarianMatcher",
    "HungarianLoss",
    "SetCriterion",
    "linear_sum_assignment",
    "pad_targets",
    "box_xywh_to_xyxy",
    "box_l1",
    "box_loss",
    "box_loss_pairwise",
    "generalized_iou_pairwise",
    "ModelConfig",
    "TrainConfig",
]
