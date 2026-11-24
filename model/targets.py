"""Ground-truth padding, Eq. (15)-(19).

The paper's ground truth is per scene: ``M`` real instances plus N - M padded
no-object instances (Eq. 16-17).  Each real instance carries a class label and
K boxes, and every (instance, view) pair carries a visibility bit ``v_k,m``
(Eq. 19) that separates "object is background" from "object is inside the
frustum but fully occluded".  Visibility decides whether a box enters the
regression loss and whether the confidence target is 1 or 0.
"""

from __future__ import annotations

from typing import Sequence

import torch


def pad_targets(
    labels: Sequence[torch.Tensor],
    boxes: Sequence[torch.Tensor],
    visibility: Sequence[torch.Tensor],
    num_queries: int,
    no_object_index: int,
    num_views: int | None = None,
) -> dict[str, torch.Tensor]:
    """Pack per-scene annotations into the fixed-size padded ground truth.

    Args:
        labels: B sequences of class labels, ``(M_b,)``.
        boxes: B tensors ``(K, M_b, 4)``, normalized ``(x, y, w, h)``.
        visibility: B tensors ``(K, M_b)`` of 0/1.
        num_queries: N, the padded instance count.
        no_object_index: label written into padded slots, i.e. ``C``.
        num_views: K; inferred from ``boxes`` when omitted.

    Returns a dict with padded tensors and ``num_instances`` (the real M per
    scene, i.e. the indicator ``I(m)`` of Eq. 18 in count form).
    """
    batch = len(labels)
    num_views = num_views if num_views is not None else int(boxes[0].shape[0])
    device = boxes[0].device

    padded_labels = torch.full(
        (batch, num_queries), no_object_index, dtype=torch.long, device=device
    )
    padded_boxes = torch.zeros(
        (batch, num_views, num_queries, 4), dtype=torch.float32, device=device
    )
    padded_visibility = torch.zeros(
        (batch, num_views, num_queries), dtype=torch.float32, device=device
    )
    instance_count = torch.zeros(batch, dtype=torch.long, device=device)

    for b in range(batch):
        count = int(labels[b].shape[0])
        if count > num_queries:
            raise ValueError(
                f"scene {b} has {count} instances but only N={num_queries} queries; "
                "raise num_queries"
            )
        padded_labels[b, :count] = torch.as_tensor(labels[b], dtype=torch.long, device=device)
        padded_boxes[b, :, :count] = torch.as_tensor(boxes[b], dtype=torch.float32, device=device)
        padded_visibility[b, :, :count] = torch.as_tensor(
            visibility[b], dtype=torch.float32, device=device
        )
        instance_count[b] = count

    mask = torch.arange(num_queries, device=device).unsqueeze(0) < instance_count.unsqueeze(1)
    return {
        "labels": padded_labels,
        "boxes": padded_boxes,
        "visibility": padded_visibility,
        "num_instances": instance_count,
        "mask": mask,
    }
