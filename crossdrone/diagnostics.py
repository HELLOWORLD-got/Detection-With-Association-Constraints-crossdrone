"""Threshold-free localization diagnostics. Oracle coverage is NOT detection recall."""
import torch
from torchvision.ops import box_iou
from model import box_xywh_to_xyxy
from .data import NUM_QUERIES


def evaluation_scenes(data, settings, split, limit=None, full=False):
    # Capacity filtering must precede slicing, exactly as in the training dataset.
    scenes = data[split]
    if split == "train":
        scenes = [s for s in scenes if len(s.ids) <= NUM_QUERIES]
    if settings.get("overfit", False) and split == "val":
        scenes = [s for s in data["train"] if len(s.ids) <= NUM_QUERIES]
        scenes = scenes[:settings["limit_train"]]
    inherited = settings.get(f"limit_{split}") if split in {"train", "val"} else None
    effective_limit = None if full else (limit if limit is not None else inherited)
    return scenes[:effective_limit]


def summary(values):
    x = torch.tensor(values, dtype=torch.float32)
    if not x.numel():
        return {"count": 0, "mean": None, "p50": None, "p90": None, "max": None}
    return {"count": x.numel(), "mean": x.mean().item(),
            "p50": x.quantile(.5).item(), "p90": x.quantile(.9).item(), "max": x.max().item()}


def geometry_oracles(pred, gt):
    """Best IoU after correcting just the center or size of each candidate."""
    centers, sizes = pred[:, None, :2], pred[:, None, 2:]
    gt_center, gt_size = gt[None, :, :2], gt[None, :, 2:]

    def best_iou(center, size):
        lo = (center - size / 2).clamp(0, 1)
        hi = (center + size / 2).clamp(0, 1)
        lo_gt = (gt_center - gt_size / 2).clamp(0, 1)
        hi_gt = (gt_center + gt_size / 2).clamp(0, 1)
        area = (hi - lo).prod(-1)
        area_gt = (hi_gt - lo_gt).prod(-1)
        inter = (torch.minimum(hi, hi_gt) - torch.maximum(lo, lo_gt)).clamp_min(0).prod(-1)
        return (inter / (area + area_gt - inter).clamp_min(torch.finfo(inter.dtype).tiny)).amax(0)

    return best_iou(gt_center, sizes), best_iou(centers, gt_size)


class Diagnostics:
    def __init__(self, threshold=.5, *, detailed=True):
        self.threshold = threshold
        self.detailed = detailed
        self.failures = dict(localization=0, vehicle=0, visibility=0, both=0,
                             disjoint_candidates=0, kept=0)
        self.values = {k: [] for k in (
            "raw_best_iou", "kept_best_iou", "foreground_probability", "visibility_probability",
            "confidence_abs_logit", "center_error_px", "size_error_px", "pred_width_px",
            "pred_height_px", "gt_width_px", "gt_height_px")}
        if detailed:
            self.values.update(oracle_center_fixed_iou=[], oracle_size_fixed_iou=[],
                               best_iou_vehicle_probability=[], best_iou_visibility_probability=[])
        self.shared_targets = 0
        self.shared_covered = 0
        self.frames = []

    def update(self, outputs, targets):
        boxes = outputs["boxes"].detach().float().cpu()
        fg = outputs["class_logits"].detach().float().softmax(-1).cpu()[..., 0]
        confidence = outputs["confidences"].detach().float().cpu()
        if not all(torch.isfinite(x).all() for x in (boxes, fg, confidence)):
            raise FloatingPointError("non-finite diagnostic predictions")
        self.values["foreground_probability"].extend(fg.flatten().tolist())
        self.values["visibility_probability"].extend(confidence.flatten().tolist())
        if "confidence_logits" in outputs:
            self.values["confidence_abs_logit"].extend(
                outputs["confidence_logits"].detach().float().cpu().abs().flatten().tolist())
        for b, target in enumerate(targets):
            record = {"sequence": target["sequence"], "timestamp": target["timestamp"], "views": []}
            shared = target["visibility"].bool().all(dim=0)
            if self.detailed:
                joint = torch.ones(boxes.shape[2], int(shared.sum()), dtype=torch.bool)
            for v, (width, height) in enumerate(target["sizes"]):
                pred = boxes[b, v]
                gt = target["boxes"][v][target["visibility"][v].bool()]
                vehicle = fg[b] >= .5
                visible = confidence[b, v] >= self.threshold
                keep = vehicle & visible
                scale = pred.new_tensor([width, height])
                for prefix, tensor in (("pred", pred), ("gt", gt)):
                    self.values[f"{prefix}_width_px"].extend((tensor[:, 2] * width).tolist())
                    self.values[f"{prefix}_height_px"].extend((tensor[:, 3] * height).tolist())
                item = {"gt_count": len(gt), "kept_predictions": int(keep.sum())}
                if len(gt):
                    iou = box_iou(box_xywh_to_xyxy(pred), box_xywh_to_xyxy(gt))
                    best, nearest = iou.max(0)
                    kept = iou[keep].max(0).values if keep.any() else torch.zeros_like(best)
                    self.values["raw_best_iou"].extend(best.tolist())
                    self.values["kept_best_iou"].extend(kept.tolist())
                    if self.detailed:
                        joint &= iou[:, shared[target["visibility"][v].bool()]] >= .5
                        center_fixed, size_fixed = geometry_oracles(pred, gt)
                        self.values["oracle_center_fixed_iou"].extend(center_fixed.tolist())
                        self.values["oracle_size_fixed_iou"].extend(size_fixed.tolist())
                        eligible = iou >= .5
                        raw_ok = eligible.any(0)
                        vehicle_ok = (eligible & vehicle[:, None]).any(0)
                        visibility_ok = (eligible & visible[:, None]).any(0)
                        kept_ok = (eligible & keep[:, None]).any(0)
                        for key, mask in {
                            "localization": ~raw_ok,
                            "vehicle": raw_ok & ~vehicle_ok & visibility_ok,
                            "visibility": raw_ok & vehicle_ok & ~visibility_ok,
                            "both": raw_ok & ~vehicle_ok & ~visibility_ok,
                            "disjoint_candidates": vehicle_ok & visibility_ok & ~kept_ok,
                            "kept": kept_ok,
                        }.items():
                            self.failures[key] += int(mask.sum())
                        self.values["best_iou_vehicle_probability"].extend(fg[b, nearest[raw_ok]].tolist())
                        self.values["best_iou_visibility_probability"].extend(
                            confidence[b, v, nearest[raw_ok]].tolist()
                        )
                    # Best-IoU query may repeat; if all IoUs are zero the choice is arbitrary.
                    error = (pred[nearest] - gt).abs()
                    self.values["center_error_px"].extend((error[:, :2] * scale).mean(-1).tolist())
                    self.values["size_error_px"].extend((error[:, 2:] * scale).mean(-1).tolist())
                    item.update(raw_best_iou_mean=best.mean().item(),
                                raw_coverage_iou50=(best >= .5).float().mean().item(),
                                kept_coverage_iou50=(kept >= .5).float().mean().item())
                record["views"].append(item)
            if self.detailed:
                record["shared_targets"] = int(shared.sum())
                record["same_query_iou50_covered"] = int(joint.any(0).sum())
                self.shared_targets += record["shared_targets"]
                self.shared_covered += record["same_query_iou50_covered"]
            self.frames.append(record)

    def compute(self):
        result = {key: summary(value) for key, value in self.values.items()}
        for kind in ("raw", "kept"):
            values = self.values[f"{kind}_best_iou"]
            result[f"{kind}_oracle_coverage_iou50"] = (
                sum(v >= .5 for v in values) / len(values) if values else None)
        if self.detailed:
            result["raw_same_query_iou50"] = {
                "covered": self.shared_covered, "shared_targets": self.shared_targets
            }
            result["coverage_failure_counts"] = self.failures
            result["vehicle_threshold"] = .5
            result["visibility_threshold"] = self.threshold
            for geometry in ("center", "size"):
                values = self.values[f"oracle_{geometry}_fixed_iou"]
                result[f"oracle_{geometry}_fixed_coverage_iou50"] = (
                    sum(v >= .5 for v in values) / len(values) if values else None)
        return {"summary": result, "frames": self.frames,
                "notes": "Pixel units use original image sizes. Oracle coverage allows repeated queries; "
                         "it is not AP or one-to-one recall. Center/size errors use the best-IoU query; "
                         "ties at zero IoU are arbitrary."}
