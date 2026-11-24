"""Configuration and factory helpers.

Defaults are the values stated in the paper: d = 256, 8 heads, 3 CA + 3 SA
layers, N = 25 instance queries, ResNet-50 backbone, AdamW with lr 1e-4 and
weight decay 1e-5 for 100 epochs (the optimizer block is for the training
script, not the model).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    # data
    num_classes: int = 10
    num_views: int = 4
    # backbone, Eq. (5)
    backbone: str = "resnet50"
    pretrained: bool = True
    freeze_backbone: bool = False
    # shared transformer geometry (Sec. IV-B2)
    hidden_dim: int = 256
    num_heads: int = 8
    ffn_ratio: int = 4
    dropout: float = 0.1
    # MVFF
    mvff_layers: int = 3
    # GIQ
    num_queries: int = 25
    giq_layers: int = 3
    # MVOQ
    mvoq_layers: int = 2
    # matching / loss, Eq. (20)-(25)
    cost_class: float = 1.0
    cost_bbox: float = 1.0
    weight_l1: float = 5.0
    weight_giou: float = 2.0
    weight_cls: float = 1.0
    weight_reg: float = 1.0
    weight_conf: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
