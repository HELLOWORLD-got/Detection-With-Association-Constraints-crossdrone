"""Equations (15)–(25), with normalized center-size boxes."""
from dataclasses import dataclass

from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F
from torchvision.ops import box_convert, generalized_box_iou


@dataclass
class LossWeights:
    match_box: float = 1.
    regression: float = 1.
    confidence: float = 1.
    l1: float = 5.
    giou: float = 2.


def corners(boxes):
    return box_convert(boxes, "cxcywh", "xyxy")


def box_cost(predicted, truth, weights):
    return (weights.l1 * torch.cdist(predicted, truth, p=1)
            + weights.giou * (1 - generalized_box_iou(corners(predicted), corners(truth))))


def scene_loss(category, predicted, confidence, target, weights):
    n, classes = category.shape
    classes -= 1
    m = len(target["labels"])
    if m > n:
        raise ValueError(f"{m} instances exceed the {n} query capacity")
    labels = F.pad(target["labels"].to(category.device), (0, n - m), value=classes)
    gt = target["boxes"].to(predicted)
    visible = target["visible"].to(device=predicted.device)
    # Build only valid geometry. A padded/invisible box has no GIoU.
    log_prob = category.float().log_softmax(-1)
    with torch.no_grad():
        assignment_cost = -log_prob[:, labels]
        for view in range(predicted.shape[0]):
            indices = visible[view].nonzero(as_tuple=True)[0]
            if len(indices):
                assignment_cost[:, indices] += weights.match_box * box_cost(
                    predicted[view].float(), gt[view, indices].float(), weights
                )
        rows, columns = linear_sum_assignment(assignment_cost.cpu().numpy())
        rows = torch.as_tensor(rows, device=category.device)
        columns = torch.as_tensor(columns, device=category.device)
    classification = -log_prob[rows, labels[columns]].mean()
    matched_visibility = F.pad(visible, (0, n - m))[:, columns]
    aligned_boxes = F.pad(gt, (0, 0, 0, n - m))[:, columns]
    predicted = predicted[:, rows].float()
    if matched_visibility.any():
        actual = predicted[matched_visibility]
        expected = aligned_boxes[matched_visibility].float()
        regression = box_cost(actual, expected, weights).diagonal().mean()
    else:
        regression = predicted.sum() * 0
    visibility = F.binary_cross_entropy_with_logits(confidence[:, rows].float(), matched_visibility.float())
    return torch.stack((classification, regression, visibility))


def objective(output, targets, weights):
    values = torch.stack([
        scene_loss(category, boxes, confidence, target, weights)
        for category, boxes, confidence, target in zip(
            output["category"], output["box"], output["visibility"], targets)
    ]).mean(0)
    return {"loss": values[0] + weights.regression * values[1] + weights.confidence * values[2],
            "classification": values[0], "regression": values[1], "visibility": values[2]}
