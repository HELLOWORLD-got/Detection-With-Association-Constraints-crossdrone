"""Frame-local association and single-class COCO detection scoring."""
from contextlib import redirect_stdout
from io import StringIO
from itertools import combinations

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment
import torch
from torchvision.ops import box_iou

from .objective import corners


def one_to_one(allowed, tie_break=None):
    if not allowed.numel():
        return {}
    reward = allowed.float() * (min(allowed.shape) + 1)
    if tie_break is not None:
        reward += tie_break * allowed
    row, col = linear_sum_assignment(-reward.numpy())
    return {int(r): int(c) for r, c in zip(row, col) if allowed[r, c]}


class LocalizationReport:
    """Counterfactual IoU with just one part of each candidate box corrected."""

    def __init__(self):
        self.values = {name: [] for name in (
            "best_iou", "center_error_px", "width_ratio", "height_ratio",
            "best_iou_if_center_fixed", "best_iou_if_size_fixed",
        )}
        self.gt = 0

    @staticmethod
    def corrected_iou(center, size, gt):
        lower = (center - size / 2).clamp(0, 1)
        upper = (center + size / 2).clamp(0, 1)
        gt_lower, gt_upper = corners(gt).clamp(0, 1).chunk(2, -1)
        intersection = (torch.minimum(upper, gt_upper) - torch.maximum(lower, gt_lower)).clamp_min(0).prod(-1)
        area = (upper - lower).prod(-1)
        gt_area = (gt_upper - gt_lower).prod(-1)
        return (intersection / (area + gt_area - intersection).clamp_min(torch.finfo(area.dtype).tiny)).amax(0)

    def add(self, candidates, gt, overlaps, width, height):
        best = overlaps.amax(0)
        center_delta = (candidates[:, None, :2] - gt[None, :, :2]) * gt.new_tensor([width, height])
        distances = center_delta.norm(dim=-1)
        # At equal IoU pick the closest center, including when all candidates are disjoint.
        index = distances.masked_fill(overlaps != best[None], torch.inf).argmin(0)
        self.values["best_iou"].extend(best.tolist())
        self.values["center_error_px"].extend(distances[index, torch.arange(len(gt))].tolist())
        ratios = candidates[index, 2:] / gt[:, 2:]
        self.values["width_ratio"].extend(ratios[:, 0].tolist())
        self.values["height_ratio"].extend(ratios[:, 1].tolist())
        self.values["best_iou_if_center_fixed"].extend(
            self.corrected_iou(gt[None, :, :2], candidates[:, None, 2:], gt).tolist())
        self.values["best_iou_if_size_fixed"].extend(
            self.corrected_iou(candidates[:, None, :2], gt[None, :, 2:], gt).tolist())
        self.gt += len(gt)

    def result(self):
        result = {"visible_gt": self.gt}
        for key, values in self.values.items():
            array = torch.tensor(values)
            result[key] = {"p50": array.quantile(.5).item(), "p90": array.quantile(.9).item(),
                           "max": array.max().item()} if len(values) else None
            if key.startswith("best_iou"):
                result[f"{key}_hits_at_50"] = int((array >= .5).sum())
        return result


class Evaluation:
    def __init__(self, visibility_threshold=.5, association_iou=.5, *, diagnostics=False):
        self.visibility_threshold = visibility_threshold
        self.association_iou = association_iou
        self.images, self.annotations, self.predictions = [], [], []
        self.frames = self.pairs = self.empty_pairs = self.correct = self.objects = 0
        self.mda_total = 0.
        self.counts = {}
        self.localization = LocalizationReport() if diagnostics else None
        self.localization_by_sequence = {}
        self.raw_iou_sum = self.raw_iou_count = self.raw_hits = self.kept_hits = 0

    def add(self, category, boxes, confidence, target):
        foreground = category.softmax(-1)[:, 0]
        confidence = confidence.sigmoid()
        keep = (foreground[None] >= .5) & (confidence >= self.visibility_threshold)
        scores = confidence * foreground[None]
        raw_boxes = boxes
        boxes = corners(boxes).clamp(0, 1)
        truth = corners(target["boxes"])
        visible = target["visible"]
        matches, overlaps, export = [], [], []
        for v, (width, height) in enumerate(target["sizes"]):
            iou = box_iou(boxes[v], truth[v]).nan_to_num()
            matches.append(one_to_one((iou >= self.association_iou)
                                      & keep[v, :, None] & visible[v, None], iou))
            overlaps.append(iou)
            if visible[v].any():
                overlaps_visible = iou[:, visible[v]]
                best = overlaps_visible.amax(0)
                if self.localization is not None:
                    truth_visible = target["boxes"][v, visible[v]]
                    sequence = target["key"].split("/", 1)[0]
                    if sequence not in self.localization_by_sequence:
                        self.localization_by_sequence[sequence] = LocalizationReport()
                    for group in (self.localization, self.localization_by_sequence[sequence]):
                        group.add(raw_boxes[v], truth_visible, overlaps_visible, width, height)
                self.raw_iou_sum += best.sum().item()
                self.raw_iou_count += len(best)
                self.raw_hits += int((best >= .5).sum())
                self.kept_hits += int((iou[keep[v]][:, visible[v]] >= .5).any(0).sum())
            scale = torch.tensor([width, height, width, height])
            image_id = len(self.images) + 1
            self.images.append({"id": image_id, "width": width, "height": height})
            for x1, y1, x2, y2 in (truth[v, visible[v]] * scale).tolist():
                w, h = x2 - x1, y2 - y1
                self.annotations.append(dict(id=len(self.annotations) + 1, image_id=image_id,
                                             category_id=1, bbox=[x1, y1, w, h], area=w*h, iscrowd=0))
            accepted = []
            for q, (x1, y1, x2, y2) in enumerate((boxes[v] * scale).tolist()):
                row = dict(image_id=image_id, category_id=1, bbox=[x1,y1,x2-x1,y2-y1],
                           score=scores[v,q].item())
                self.predictions.append(row)
                if keep[v,q]:
                    accepted.append({"query_id": q, "bbox_xywh": row["bbox"], "score": row["score"]})
            export.append(accepted)
        for a, b in combinations(range(len(matches)), 2):
            linked = (keep[a] & keep[b]).nonzero(as_tuple=True)[0].tolist()
            gt = int((visible[a] & visible[b]).sum())
            true = sum(q in matches[a] and q in matches[b] and matches[a][q] == matches[b][q] for q in linked)
            denominator = gt + (len(linked) - true) + (gt - true)
            self.mda_total += true / denominator if denominator else 0.
            self.pairs += 1
            self.empty_pairs += not denominator
            stats = self.counts.setdefault((a,b), [0,0,0])
            stats[0] += true
            stats[1] += len(linked)
            stats[2] += gt
        valid = ((keep[:, :, None] == visible[:, None])
                 & ((torch.stack(overlaps) > .6) | ~visible[:, None])).all(0)
        self.correct += len(one_to_one(valid))
        self.objects += visible.shape[1]
        self.frames += 1
        return {"pair": target["key"], "views": export}

    def result(self):
        ap, ap50 = None, None
        if self.annotations:
            with redirect_stdout(StringIO()):
                coco = COCO()
                coco.dataset = dict(info={}, images=self.images, annotations=self.annotations,
                                    categories=[dict(id=1, name="vehicle")])
                coco.createIndex()
                calc = COCOeval(coco, coco.loadRes(self.predictions), "bbox")
                calc.evaluate()
                calc.accumulate()
                calc.summarize()
                ap, ap50 = map(float, calc.stats[:2])
        ratios = [(t/max(1,p), t/max(1,g), 2*t/max(1,p+g)) for t,p,g in self.counts.values()]
        p,r,f = (tuple(sum(column) / len(column) for column in zip(*ratios)) if ratios else (0.,0.,0.))
        result = {"frames": self.frames, "mAP": ap, "mAP50": ap50,
                "MDA": self.mda_total / max(1,self.pairs), "UCR": self.correct / max(1,self.objects),
                "association_precision": p, "association_recall": r, "association_F1": f,
                "empty_association_terms": self.empty_pairs, "ucr_total": self.objects,
                "raw_best_iou_mean": self.raw_iou_sum / self.raw_iou_count if self.raw_iou_count else None,
                "iou50_targets": self.raw_iou_count, "raw_hits_iou50": self.raw_hits,
                "kept_hits_iou50": self.kept_hits}
        if self.localization is not None:
            result["localization"] = self.localization.result()
            result["localization_by_sequence"] = {
                sequence: meter.result() for sequence, meter in self.localization_by_sequence.items()
            }
        return result
