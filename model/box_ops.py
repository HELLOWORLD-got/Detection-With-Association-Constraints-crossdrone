"""Box helpers and the pairwise box loss of Eq. (25).

Box convention follows Eq. (12): ``b = (x, y, w, h)``, every coordinate
normalized to ``[0, 1]`` and ``(x, y)`` the box center (DETR convention).

Eq. (25)::

    L_box(b, b_hat) = lam_l1 * || b - b_hat ||_1 + lam_giou * (1 - GIoU(b, b_hat))

The paper does not state ``lam_l1`` / ``lam_giou``; the DETR values 5 and 2 are
used and kept configurable.
"""

from __future__ import annotations

import torch

EPS = torch.finfo(torch.float32).tiny


def box_xywh_to_xyxy(boxes: torch.Tensor, *, clip: bool = True) -> torch.Tensor:
    """Convert (..., 4) center-size boxes to corner form."""
    x, y, w, h = boxes.unbind(-1)
    x1, y1 = x - 0.5 * w, y - 0.5 * h
    x2, y2 = x + 0.5 * w, y + 0.5 * h
    if clip:
        x1, y1 = x1.clamp(min=0.0), y1.clamp(min=0.0)
        x2, y2 = x2.clamp(max=1.0), y2.clamp(max=1.0)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_area(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    return (boxes_xyxy[..., 2] - boxes_xyxy[..., 0]).clamp(min=0.0) * (
        boxes_xyxy[..., 3] - boxes_xyxy[..., 1]
    ).clamp(min=0.0)


def box_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``|| pred - target ||_1`` over the four coordinates (same shapes)."""
    return (pred - target).abs().sum(dim=-1)


def _giou_same(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """GIoU of corresponding boxes, shapes ``(..., 4)`` in corner form."""
    area_a, area_b = box_area(a), box_area(b)
    lt = torch.max(a[..., :2], b[..., :2])
    rb = torch.min(a[..., 2:], b[..., 2:])
    wh = (rb - lt).clamp(min=0.0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a + area_b - inter
    iou = inter / union.clamp(min=EPS)

    lt_enclosing = torch.min(a[..., :2], b[..., :2])
    rb_enclosing = torch.max(a[..., 2:], b[..., 2:])
    wh_enclosing = (rb_enclosing - lt_enclosing).clamp(min=0.0)
    enclosing = wh_enclosing[..., 0] * wh_enclosing[..., 1]
    return iou - (enclosing - union) / enclosing.clamp(min=EPS)


def generalized_iou_pairwise(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """GIoU between two corner-form sets: ``(N, 4)`` x ``(M, 4)`` -> ``(N, M)``."""
    area1 = box_area(boxes1)[:, None]
    area2 = box_area(boxes2)[None, :]
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0.0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=EPS)

    lt_enclosing = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_enclosing = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_enclosing = (rb_enclosing - lt_enclosing).clamp(min=0.0)
    enclosing = wh_enclosing[..., 0] * wh_enclosing[..., 1]
    return iou - (enclosing - union) / enclosing.clamp(min=EPS)


def box_loss_pairwise(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight_l1: float = 5.0,
    weight_giou: float = 2.0,
) -> torch.Tensor:
    """Eq. (25) for ``pred (N, 4)`` against ``target (M, 4)`` -> ``(N, M)``."""
    target_xyxy = box_xywh_to_xyxy(target, clip=False)
    pred_xyxy = box_xywh_to_xyxy(pred, clip=False)
    giou = generalized_iou_pairwise(pred_xyxy, target_xyxy)
    l1 = (pred[:, None, :] - target[None, :, :]).abs().sum(dim=-1)
    return weight_l1 * l1 + weight_giou * (1.0 - giou)


def box_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight_l1: float = 5.0,
    weight_giou: float = 2.0,
) -> torch.Tensor:
    """Eq. (25) on corresponding boxes (same shape, e.g. matched pairs)."""
    giou = _giou_same(
        box_xywh_to_xyxy(pred, clip=False), box_xywh_to_xyxy(target, clip=False)
    )
    return weight_l1 * box_l1(pred, target) + weight_giou * (1.0 - giou)
