"""Optimal bipartite matching, Eq. (20).

Every scene has exactly N padded ground-truth slots and N predictions, so the
assignment is a permutation over N slots.  Real slots (m <= M) cost the
classification term plus the visibility-masked box term; padded slots
(m > M) cost only ``-log p_hat_i(c_empty)``.

`scipy.optimize.linear_sum_assignment` is used when available.  scipy is not on
every deployment box, so an O(n^3) Hungarian implementation in numpy is
included as a fallback and checked against the optimum in the tests.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .box_ops import box_loss_pairwise


def _hungarian_numpy(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-cost full assignment for a square/rectangular cost matrix.

    Shortest augmenting path form of the Kuhn-Munkres algorithm with
    potentials (e-maxx formulation); n <= m required internally, the
    transposed case is handled by solving on the transpose.
    """
    cost = np.asarray(cost_matrix, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError("cost matrix must be 2-D")
    transposed = False
    if cost.shape[0] > cost.shape[1]:
        cost = cost.T
        transposed = True
    n, m = cost.shape

    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    match = np.zeros(m + 1, dtype=np.int64)  # match[j] = row index (1-based)
    way = np.zeros(m + 1, dtype=np.int64)

    for i in range(1, n + 1):
        match[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = match[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[match[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if match[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            match[j0] = match[j1]
            j0 = j1

    cols = np.full(n, -1, dtype=np.int64)
    for j in range(1, m + 1):
        if match[j] > 0:
            cols[match[j] - 1] = j - 1
    rows = np.arange(n, dtype=np.int64)

    if transposed:
        return cols, rows
    return rows, cols


def linear_sum_assignment(cost_matrix: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Assignment minimising the total cost; scipy if importable, else fallback."""
    array = cost_matrix.detach().cpu().numpy()
    try:
        from scipy.optimize import linear_sum_assignment as scipy_assignment

        return scipy_assignment(array)
    except ImportError:  # pragma: no cover - depends on the environment
        return _hungarian_numpy(array)


class HungarianMatcher(torch.nn.Module):
    """Eq. (20) as a module so the cost weights follow the model around."""

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 1.0,
        weight_l1: float = 5.0,
        weight_giou: float = 2.0,
    ) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.weight_l1 = weight_l1
        self.weight_giou = weight_giou

    @torch.no_grad()
    def forward(
        self,
        class_logits: torch.Tensor,
        boxes: torch.Tensor,
        targets: dict[str, torch.Tensor],
    ) -> list[torch.Tensor]:
        """Args:
            class_logits: ``(B, N, C+1)``.
            boxes: ``(B, K, N, 4)`` predicted, normalized center-size.
            targets: output of :func:`pad_targets`.

        Returns B tensors ``(N, 2)`` of ``(prediction index, ground-truth slot)``.
        """
        batch, num_queries, num_classes_p1 = class_logits.shape
        num_classes = num_classes_p1 - 1
        num_views = boxes.shape[1]
        device = class_logits.device

        assignments: list[torch.Tensor] = []
        for b in range(batch):
            count = int(targets["num_instances"][b])
            log_probs = F.log_softmax(class_logits[b].float(), dim=-1)

            cost = torch.zeros(
                num_queries, num_queries, dtype=torch.float32, device=device
            )
            if count > 0:
                labels = targets["labels"][b, :count]
                gt_boxes = targets["boxes"][b, :, :count]  # (K, M, 4)
                visibility = targets["visibility"][b, :, :count]  # (K, M)

                cls_cost = -log_probs[:, labels]  # (N, M)
                box_cost = torch.zeros(
                    num_queries, count, dtype=torch.float32, device=device
                )
                for k in range(num_views):
                    pair = box_loss_pairwise(
                        boxes[b, k], gt_boxes[k], self.weight_l1, self.weight_giou
                    )
                    box_cost = box_cost + visibility[k].unsqueeze(0) * pair
                cost[:, :count] = self.cost_class * cls_cost + self.cost_bbox * box_cost

            if count < num_queries:
                empty_cost = -log_probs[:, num_classes].unsqueeze(1)
                cost[:, count:] = self.cost_class * empty_cost.expand(
                    num_queries, num_queries - count
                )

            rows, cols = linear_sum_assignment(cost)
            assignment = torch.stack(
                [
                    torch.as_tensor(rows, dtype=torch.long, device=device),
                    torch.as_tensor(cols, dtype=torch.long, device=device),
                ],
                dim=1,
            )
            assignments.append(assignment)
        return assignments
