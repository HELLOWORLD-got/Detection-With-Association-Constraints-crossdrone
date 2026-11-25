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


class Diagnostics:
    def __init__(self, threshold=.5):
        self.threshold = threshold
        self.values = {k: [] for k in (
            "raw_best_iou", "kept_best_iou", "foreground_probability", "visibility_probability",
            "confidence_abs_logit", "center_error_px", "size_error_px", "pred_width_px",
            "pred_height_px", "gt_width_px", "gt_height_px")}
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
            for v, (width, height) in enumerate(target["sizes"]):
                pred = boxes[b, v]
                gt = target["boxes"][v][target["visibility"][v].bool()]
                keep = (fg[b] >= .5) & (confidence[b, v] >= self.threshold)
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
                    # Best-IoU query may repeat; if all IoUs are zero the choice is arbitrary.
                    error = (pred[nearest] - gt).abs()
                    self.values["center_error_px"].extend((error[:, :2] * scale).mean(-1).tolist())
                    self.values["size_error_px"].extend((error[:, 2:] * scale).mean(-1).tolist())
                    item.update(raw_best_iou_mean=best.mean().item(),
                                raw_coverage_iou50=(best >= .5).float().mean().item(),
                                kept_coverage_iou50=(kept >= .5).float().mean().item())
                record["views"].append(item)
            self.frames.append(record)

    def compute(self):
        result = {key: summary(value) for key, value in self.values.items()}
        for kind in ("raw", "kept"):
            values = self.values[f"{kind}_best_iou"]
            result[f"{kind}_oracle_coverage_iou50"] = (
                sum(v >= .5 for v in values) / len(values) if values else None)
        return {"summary": result, "frames": self.frames,
                "notes": "Pixel units use original image sizes. Oracle coverage allows repeated queries; "
                         "it is not AP or one-to-one recall. Center/size errors use the best-IoU query; "
                         "ties at zero IoU are arbitrary."}
