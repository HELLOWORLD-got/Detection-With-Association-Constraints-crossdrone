"""Train the two-view, 25-query Cross-Drone vehicle model."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from crossdrone.data import NUM_QUERIES, NUM_VIEWS, SPLITS, index_dataset
from crossdrone.diagnostics import Diagnostics
from crossdrone.engine import evaluate, make_loader, resolve_device, save_checkpoint, train_epoch, write_json
from model import ModelConfig, TrainConfig, build_model


def report_startup(message):
    print(f"[Startup] {message}", file=sys.stderr, flush=True)


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
    parser.add_argument("--distributed-backend", choices=("nccl", "gloo"),
                        help="override the process group backend (default: NCCL for CUDA, Gloo for CPU)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--association-iou", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=50,
                        help="progress interval in redirected logs (0 disables progress)")
    parser.add_argument("--resume", action="store_true", help="resume from OUTPUT/last.pt")
    parser.add_argument("--audit-only", action="store_true", help="report XML and split coverage")
    parser.add_argument("--random-init", action="store_true", help="initialize without ImageNet weights")
    parser.add_argument("--validate-before-training", action="store_true",
                        help="evaluate the initial model on the same validation split before epoch 1")
    parser.add_argument("--overfit", action="store_true",
                        help="diagnostic only: disable augmentation and validate on the same limited training scenes")
    parser.add_argument("--box-size-prior", action="store_true",
                        help="ablation: initialize box width/height biases from training medians")
    parser.add_argument("--mvoq-content-key", action="store_true",
                        help="ablation: include local appearance in the MVOQ Key")
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
    if args.overfit and (args.limit_train is None or args.limit_val is not None):
        parser.error("--overfit requires --limit-train and no --limit-val")
    return args


def setup_device(spec, world_size, backend=None):
    device = resolve_device(spec)
    if world_size > 1:
        if device.type == "mps" or device.index is not None:
            raise ValueError("torchrun requires --device cuda or --device cpu")
        if backend == "nccl" and device.type != "cuda":
            raise ValueError("the NCCL backend requires CUDA")
        if device.type == "cuda":
            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
            torch.cuda.set_device(device)
        dist.init_process_group(
            backend=backend or ("nccl" if device.type == "cuda" else "gloo"),
            timeout=timedelta(hours=2),
        )
    return device


def training_box_size_medians(scenes):
    medians = []
    for view in range(NUM_VIEWS):
        sizes = [
            (x2 - x1, y2 - y1)
            for scene in scenes
            for x1, y1, x2, y2 in scene.boxes[view]
            if x2 > x1 and y2 > y1
        ]
        if not sizes:
            raise ValueError(f"--box-size-prior: view {view + 1} has no visible training boxes")
        medians.append(torch.tensor(sizes).quantile(0.5, dim=0).tolist())
    return medians


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
        pretrained=not args.random_init, mvoq_content_key=args.mvoq_content_key,
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
    if args.overfit:
        settings["overfit"] = True
    if args.box_size_prior:
        # Match the loader's capacity filtering and sample cap before reading labels.
        eligible = [scene for scene in data["train"] if len(scene.ids) <= NUM_QUERIES]
        settings["box_size_prior"] = training_box_size_medians(eligible[:args.limit_train])
    if checkpoint is not None and checkpoint["settings"]["model"].get("mvoq_share_attention", True):
        raise ValueError(
            "This checkpoint uses shared MVOQ attention. Use evaluate.py to inspect it, "
            "or choose a new --output and train the per-view architecture from initialization."
        )
    if checkpoint is not None:
        checkpoint["settings"]["model"].setdefault("mvoq_content_key", False)
    if checkpoint is not None and checkpoint["settings"] != settings:
        raise ValueError("resume settings or indexed data differ from the saved run")
    if checkpoint is not None and checkpoint["epoch"] >= args.epochs:
        raise ValueError(f"checkpoint already completed epoch {checkpoint['epoch']}; raise --epochs to continue")
    batch_size = args.batch_size // world_size
    loader = make_loader(data["train"], args.size, device, batch_size, args.workers,
                         training=True, limit=args.limit_train, rank=rank,
                         world_size=world_size, seed=args.seed)
    if args.overfit:
        loader.dataset.training = False
    if rank == 0:
        val_scenes = loader.dataset.scenes if args.overfit else data["val"]
        val_loader = make_loader(val_scenes, args.size, device, batch_size, args.workers,
                                 limit=args.limit_val)
        output.mkdir(parents=True, exist_ok=True)
        audit["training_eligible"] = audit["splits"]["train"]["complete"] - audit["splits"]["train"]["over_capacity"]
        audit["training_used"] = len(loader.dataset)
        audit["validation_used"] = len(val_loader.dataset)
        if args.overfit:
            report_startup("OVERFIT DIAGNOSTIC: augmentation disabled; validation uses training images")
        report_startup(
            f"Data ready: {len(loader.dataset)} training pairs, {len(val_loader.dataset)} validation pairs"
        )
        write_json(output / "data_audit.json", audit)
        write_json(output / "config.json", {
            **settings, "epochs": args.epochs,
            "distributed_backend": dist.get_backend() if dist.is_initialized() else None,
        })

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    initialization = replace(model_config, pretrained=model_config.pretrained and checkpoint is None and rank == 0)
    if rank == 0:
        source = str(last_path) if checkpoint is not None else (
            "ImageNet pretrained weights" if initialization.pretrained else "random initialization"
        )
        report_startup(f"Building {model_config.backbone} on {device} using {source}")
    net = build_model(initialization).to(device)
    if args.box_size_prior and checkpoint is None:
        sizes = settings["box_size_prior"]
        net.mvoq.initialize_box_size_bias(sizes)
        if rank == 0:
            report_startup(f"Initialized normalized width/height biases from: {sizes}")
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
    if rank == 0 and world_size > 1:
        report_startup(f"Synchronizing model across {world_size} processes")
    model = DistributedDataParallel(
        net, device_ids=[device.index] if device.type == "cuda" else None
    ) if world_size > 1 else net
    def validate(label):
        diagnostics = Diagnostics(args.confidence_threshold) if args.overfit else None
        result = evaluate(
            net, val_loader, device, args.confidence_threshold, args.association_iou,
            progress_label=label, log_every=args.log_every, diagnostics=diagnostics,
        )
        if diagnostics is not None:
            summary = diagnostics.compute()["summary"]
            result["raw_best_iou"] = summary["raw_best_iou"]
            result["raw_oracle_coverage_iou50"] = summary["raw_oracle_coverage_iou50"]
        return result

    if args.validate_before_training and start_epoch == 1:
        if rank == 0:
            report_startup("Evaluating the initial model before training")
            baseline = validate("Val baseline")
            record = {"epoch": 0, "val": baseline}
            write_json(output / "baseline.json", record)
            print(json.dumps(record), flush=True)
        if world_size > 1:
            dist.barrier()
    if rank == 0:
        report_startup(f"Starting epoch {start_epoch}/{args.epochs} and loading its first batch")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_seed = args.seed + epoch * world_size + rank
        random.seed(epoch_seed)
        torch.manual_seed(epoch_seed)
        if isinstance(loader.sampler, torch.utils.data.DistributedSampler):
            loader.sampler.set_epoch(epoch)
        losses = train_epoch(
            model, criterion, optimizer, loader, device, epoch, rank, args.log_every, epochs=args.epochs,
            inspect_gradients=args.overfit,
        )
        if rank == 0:
            validation = validate(f"Val {epoch}/{args.epochs}")
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
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    splits = {split: getattr(args, f"{split}_sequences") for split in SPLITS}
    if rank == 0 and not args.audit_only:
        report_startup(f"Indexing XML annotations and image pairs in {args.data.resolve()}")
    data, audit = index_dataset(args.data, splits)
    if args.audit_only:
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        return

    if rank == 0 and world_size > 1:
        backend = args.distributed_backend or "default backend"
        report_startup(f"Initializing the distributed process group ({world_size} processes, {backend})")
    device = setup_device(args.device, world_size, args.distributed_backend)
    try:
        run(args, data, audit, splits, world_size, rank, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
