"""Train the two-view, 25-query Cross-Drone vehicle model."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import timedelta
import json
import os
from pathlib import Path
import random

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from crossdrone.data import NUM_QUERIES, NUM_VIEWS, SPLITS, index_dataset
from crossdrone.engine import evaluate, make_loader, resolve_device, save_checkpoint, train_epoch, write_json
from model import ModelConfig, TrainConfig, build_model


def parse_args():
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("数据集"))
    parser.add_argument("--output", type=Path, default=Path("runs/crossdrone"))
    parser.add_argument("--backbone", choices=("resnet50", "vit_b_16"), default="resnet50")
    parser.add_argument("--size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=(576, 1024))
    for split, sequences in SPLITS.items():
        parser.add_argument(f"--{split}-sequences", nargs="+", default=sequences)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size, help="global batch size")
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N or mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--association-iou", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=50,
                        help="progress interval in redirected logs (0 disables progress)")
    parser.add_argument("--resume", action="store_true", help="resume from OUTPUT/last.pt")
    parser.add_argument("--audit-only", action="store_true", help="report XML and split coverage")
    parser.add_argument("--random-init", action="store_true", help="initialize without ImageNet weights")
    parser.add_argument("--limit-train", type=int, help="limit training scenes for a smoke run")
    parser.add_argument("--limit-val", type=int, help="limit validation scenes for a smoke run")
    args = parser.parse_args()
    if min(args.size) < 64 or any(length % 16 for length in args.size):
        parser.error("--size dimensions must be at least 64 and divisible by 16")
    if min(args.batch_size, args.epochs) < 1 or args.lr <= 0 or args.weight_decay < 0:
        parser.error("invalid batch size, epochs, learning rate or weight decay")
    if min(args.workers, args.log_every) < 0 or any(
        limit is not None and limit < 1 for limit in (args.limit_train, args.limit_val)
    ):
        parser.error("invalid workers, logging interval or scene limit")
    if not (0 <= args.confidence_threshold <= 1 and 0 < args.association_iou <= 1):
        parser.error("invalid confidence or association IoU threshold")
    return args


def setup_device(spec, world_size):
    device = resolve_device(spec)
    if world_size > 1:
        if device.type == "mps" or device.index is not None:
            raise ValueError("torchrun requires --device cuda or --device cpu")
        if device.type == "cuda":
            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
            torch.cuda.set_device(device)
        dist.init_process_group(
            backend="nccl" if device.type == "cuda" else "gloo", timeout=timedelta(hours=2)
        )
    return device


def run(args, data, audit, splits, world_size, rank, device):
    if args.batch_size % world_size:
        raise ValueError("global --batch-size must be divisible by the process count")
    output = args.output
    last_path = output / "last.pt"
    if not args.resume and last_path.exists():
        raise FileExistsError(f"{last_path} exists. Use --resume or a new --output.")
    checkpoint = torch.load(last_path, map_location="cpu", weights_only=True) if args.resume else None

    model_config = ModelConfig(
        backbone=args.backbone, num_classes=1, num_views=NUM_VIEWS, num_queries=NUM_QUERIES,
        pretrained=not args.random_init,
    )
    settings = {
        "model": asdict(model_config), "size": list(args.size),
        "sequences": {split: list(names) for split, names in splits.items()},
        "annotation_and_availability_sha256": audit["annotation_and_availability_sha256"],
        "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
        "seed": args.seed, "confidence_threshold": args.confidence_threshold,
        "association_iou": args.association_iou, "limit_train": args.limit_train,
        "limit_val": args.limit_val, "world_size": world_size,
    }
    if checkpoint is not None and checkpoint["settings"] != settings:
        raise ValueError("resume settings or indexed data differ from the saved run")
    if checkpoint is not None and checkpoint["epoch"] >= args.epochs:
        raise ValueError(f"checkpoint already completed epoch {checkpoint['epoch']}; raise --epochs to continue")
    batch_size = args.batch_size // world_size
    loader = make_loader(data["train"], args.size, device, batch_size, args.workers,
                         training=True, limit=args.limit_train, rank=rank,
                         world_size=world_size, seed=args.seed)
    if rank == 0:
        val_loader = make_loader(data["val"], args.size, device, batch_size, args.workers,
                                 limit=args.limit_val)
        output.mkdir(parents=True, exist_ok=True)
        audit["training_eligible"] = audit["splits"]["train"]["complete"] - audit["splits"]["train"]["over_capacity"]
        audit["training_used"] = len(loader.dataset)
        audit["validation_used"] = len(val_loader.dataset)
        write_json(output / "data_audit.json", audit)
        write_json(output / "config.json", {**settings, "epochs": args.epochs})

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    initialization = replace(model_config, pretrained=model_config.pretrained and checkpoint is None and rank == 0)
    net = build_model(initialization).to(device)
    criterion = net.build_criterion()
    optimizer = torch.optim.AdamW(
        (p for p in net.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay
    )
    history = []
    best = (-1.0, -1.0)
    start_epoch = 1
    if checkpoint is not None:
        net.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best = tuple(checkpoint["best"])
        history = checkpoint["history"]
        del checkpoint
    model = DistributedDataParallel(
        net, device_ids=[device.index] if device.type == "cuda" else None
    ) if world_size > 1 else net

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_seed = args.seed + epoch * world_size + rank
        random.seed(epoch_seed)
        torch.manual_seed(epoch_seed)
        if isinstance(loader.sampler, torch.utils.data.DistributedSampler):
            loader.sampler.set_epoch(epoch)
        losses = train_epoch(
            model, criterion, optimizer, loader, device, epoch, rank, args.log_every, epochs=args.epochs
        )
        if rank == 0:
            validation = evaluate(
                net, val_loader, device, args.confidence_threshold, args.association_iou,
                progress_label=f"Val {epoch}/{args.epochs}", log_every=args.log_every,
            )
            record = {"epoch": epoch, "train": losses, "val": validation}
            history.append(record)
            score = (validation["MDA"], validation["mAP"] or 0.0)
            improved = score > best
            best = max(score, best)
            state = {
                "epoch": epoch, "settings": settings, "model": net.state_dict(),
                "optimizer": optimizer.state_dict(), "history": history, "best": best,
            }
            if improved:
                save_checkpoint(output / "best.pt", state)
            save_checkpoint(last_path, state)
            write_json(output / "metrics.json", history)
            print(json.dumps(record), flush=True)
        if world_size > 1:
            dist.barrier()


def main():
    args = parse_args()
    splits = {split: getattr(args, f"{split}_sequences") for split in SPLITS}
    data, audit = index_dataset(args.data, splits)
    if args.audit_only:
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    device = setup_device(args.device, world_size)
    try:
        run(args, data, audit, splits, world_size, rank, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
