"""Frame-local metrics from Sec. V-B and COCO bounding-box AP."""

from contextlib import redirect_stdout
from io import StringIO
from itertools import combinations

import torch
from torchvision.ops import box_iou

from model import box_xywh_to_xyxy, linear_sum_assignment


def _match(valid, iou=None):
    if not valid.numel():
        return {}
    # Find as many valid matches as possible, with IoU breaking ties.
    value = valid.float() * (min(valid.shape) + 1)
    if iou is not None:
        value += iou * valid
    rows, cols = linear_sum_assignment(-value)
    return {int(r): int(c) for r, c in zip(rows, cols) if valid[r, c]}


def decode(outputs, threshold):
    vehicle = outputs["class_logits"].float().softmax(dim=-1)[..., 0].cpu()
    boxes = box_xywh_to_xyxy(outputs["boxes"].float().cpu())
    confidence = outputs["confidences"].float().cpu()
    scores = confidence * vehicle[:, None]
    present = (vehicle[:, None] >= 0.5) & (confidence >= threshold)
    if not all(torch.isfinite(t).all() for t in (boxes, confidence, vehicle)):
        raise FloatingPointError("non-finite Cross-Drone predictions")
    return list(zip(boxes, scores, present))


class AssociationMetrics:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.mda_sum = 0.0
        self.nonempty = self.empty = 0
        self.pair_counts = {}
        self.correct = self.total = 0

    def update(self, boxes, present, target):
        truth = box_xywh_to_xyxy(target["boxes"])
        visible = target["visibility"]
        ious = [box_iou(pred, gt).nan_to_num(0.0) for pred, gt in zip(boxes, truth)]
        maps = [
            _match((iou >= self.iou_threshold) & current[:, None] & annotated[None], iou)
            for iou, current, annotated in zip(ious, present, visible)
        ]
        for a, b in combinations(range(len(maps)), 2):
            linked = (present[a] & present[b]).nonzero().flatten().tolist()
            ga = int((visible[a] & visible[b]).sum())
            ta = sum(q in maps[a] and q in maps[b] and maps[a][q] == maps[b][q] for q in linked)
            fa, ma = len(linked) - ta, ga - ta
            denominator = ga + fa + ma
            if denominator:
                self.mda_sum += ta / denominator
                self.nonempty += 1
            else:
                self.empty += 1
            counts = self.pair_counts.setdefault((a, b), [0, 0, 0])
            counts[0] += ta
            counts[1] += len(linked)
            counts[2] += ga
        valid = ((present[:, :, None] == visible[:, None, :])
                 & (~visible[:, None, :] | (torch.stack(ious) > 0.6))).all(dim=0)
        self.correct += len(_match(valid))
        self.total += visible.shape[1]

    def compute(self):
        precision, recall, f1 = [], [], []
        for ta, predicted, truth in self.pair_counts.values():
            precision.append(ta / predicted if predicted else 0.0)
            recall.append(ta / truth if truth else 0.0)
            f1.append(2 * ta / (predicted + truth) if predicted + truth else 0.0)
        return {
            "MDA": self.mda_sum / max(1, self.nonempty),
            "UCR": self.correct / max(1, self.total),
            "association_precision": sum(precision) / max(1, len(precision)),
            "association_recall": sum(recall) / max(1, len(recall)),
            "association_F1": sum(f1) / max(1, len(f1)),
            "empty_association_pairs": self.empty,
            "UCR_correct": self.correct,
            "UCR_total": self.total,
        }


class DetectionMetrics:
    def __init__(self):
        self.images = []
        self.annotations = []
        self.detections = []

    def update(self, boxes, scores, target):
        truth = box_xywh_to_xyxy(target["boxes"])
        for view, (width, height) in enumerate(target["sizes"]):
            image_id = len(self.images) + 1
            self.images.append({"id": image_id, "width": width, "height": height})
            scale = boxes.new_tensor([width, height, width, height])
            valid_truth = truth[view, target["visibility"][view]] * scale
            for left, top, right, bottom in valid_truth.tolist():
                w, h = right - left, bottom - top
                self.annotations.append({
                    "id": len(self.annotations) + 1, "image_id": image_id, "category_id": 1,
                    "bbox": [left, top, w, h], "area": w * h, "iscrowd": 0,
                })
            for corners, score in zip((boxes[view] * scale).tolist(), scores[view].tolist()):
                left, top, right, bottom = corners
                self.detections.append({
                    "image_id": image_id, "category_id": 1,
                    "bbox": [left, top, right - left, bottom - top], "score": score,
                })

    def compute(self):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        if not self.annotations:
            return {"mAP": None, "mAP50": None}
        with redirect_stdout(StringIO()):
            truth = COCO()
            truth.dataset = {
                "images": self.images, "annotations": self.annotations,
                "categories": [{"id": 1, "name": "vehicle"}], "info": {},
            }
            truth.createIndex()
            metric = COCOeval(truth, truth.loadRes(self.detections), "bbox")
            metric.evaluate()
            metric.accumulate()
            metric.summarize()
        return {"mAP": float(metric.stats[0]), "mAP50": float(metric.stats[1])}
