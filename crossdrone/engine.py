"""Shared Cross-Drone train and evaluation operations."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

from model import pad_targets

from .data import NUM_QUERIES, CrossDroneDataset, collate
from .metrics import AssociationMetrics, DetectionMetrics, decode


def make_loader(scenes, size, device, batch_size, workers, *,
                training=False, limit=None, rank=0, world_size=1, seed=42):
    dataset = CrossDroneDataset(scenes, size, training=training, limit=limit)
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, seed=seed)
        if training and world_size > 1 else None
    )
    return DataLoader(
        dataset, batch_size=batch_size, num_workers=workers, sampler=sampler,
        shuffle=training and sampler is None, pin_memory=device.type == "cuda",
        multiprocessing_context="spawn" if workers and device.type == "cuda" else None,
        collate_fn=collate,
    )


def train_epoch(model, criterion, optimizer, loader, device, epoch, rank, log_every):
    model.train()
    fields = ("loss", "loss_cls", "loss_reg", "loss_conf")
    totals = torch.zeros(5, dtype=torch.float32, device=device)
    for step, (images, targets) in enumerate(loader, 1):
        padded = pad_targets(
            [target["labels"] for target in targets],
            [target["boxes"] for target in targets],
            [target["visibility"] for target in targets],
            num_queries=NUM_QUERIES, no_object_index=1,
        )
        padded = {k: v.to(device, non_blocking=True) for k, v in padded.items()}
        optimizer.zero_grad(set_to_none=True)
        losses = criterion(model(images.to(device, non_blocking=True)), padded)
        loss = losses["loss"]
        if not torch.isfinite(loss):
            ids = ", ".join(f"{t['sequence']}:{t['timestamp']}" for t in targets)
            raise FloatingPointError(f"non-finite training loss on {ids}")
        loss.backward()
        optimizer.step()
        totals[:4] += torch.stack([losses[field] for field in fields]).detach() * len(targets)
        totals[4] += len(targets)
        if rank == 0 and log_every and (step % log_every == 0 or step == len(loader)):
            print(f"epoch {epoch} step {step}/{len(loader)} loss={loss.item():.4f}", flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(totals)
    return dict(zip(fields, (totals[:4] / totals[4]).cpu().tolist()))


@torch.inference_mode()
def evaluate(model, loader, device, confidence_threshold, iou_threshold, predictions=None):
    model.eval()
    association = AssociationMetrics(iou_threshold)
    detection = DetectionMetrics()
    frames = 0
    for images, targets in loader:
        outputs = model(images.to(device, non_blocking=True))
        for (boxes, scores, present), target in zip(decode(outputs, confidence_threshold), targets):
            association.update(boxes, present, target)
            detection.update(boxes, scores, target)
            frames += 1
            if predictions is not None:
                detections = []
                for view, (width, height) in enumerate(target["sizes"]):
                    scale = boxes.new_tensor([width, height, width, height])
                    detections.append([
                        {"id": query, "xyxy": (boxes[view, query] * scale).tolist(),
                         "score": scores[view, query].item()}
                        for query in present[view].nonzero().flatten().tolist()
                    ])
                predictions.write(json.dumps({
                    "sequence": target["sequence"], "timestamp": target["timestamp"],
                    "views": detections,
                }) + "\n")
    return {"frames": frames, **association.compute(), **detection.compute()}


def write_json(path: Path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path: Path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def resolve_device(spec: str):
    if spec == "auto":
        spec = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported device {device}")
    return device
