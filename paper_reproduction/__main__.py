"""Train and evaluate the independent paper reproduction."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from rich.progress import track

from .dataset import CrossDrone, collate, index_data
from .network import Architecture, AssociationNetwork
from .objective import LossWeights, objective
from .scoring import Evaluation


SPLITS = {"train": ["1","3","4","5","6","7","8"], "val": ["2"], "test": ["0"]}


def atomic(path, write):
    temporary = path.with_name(path.name + ".tmp")
    write(temporary)
    temporary.replace(path)


def json_file(path, payload):
    atomic(path, lambda p: p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8"))


def loader(pairs, cfg, batch, workers, device, training=False, rank=0, world=1):
    dataset = CrossDrone(pairs, cfg["size"], augment=training and not cfg["overfit"] and not cfg.get("no_augmentation", False))
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=cfg["seed"]) if training and world > 1 else None
    return DataLoader(dataset, batch_size=batch, shuffle=training and sampler is None, sampler=sampler,
                      num_workers=workers, collate_fn=collate, pin_memory=device.type == "cuda",
                      multiprocessing_context="spawn" if workers and device.type == "cuda" else None)


def selected(data, cfg, split, limit=None, full=False):
    train = [pair for pair in data["train"] if len(pair.ids) <= cfg["architecture"]["queries"]]
    if split == "train":
        return train[:None if full else (limit if limit is not None else cfg["limit_train"])]
    if split == "val" and cfg["overfit"]:
        return train[:cfg["limit_train"]][:limit]
    cap = cfg["limit_val"] if split == "val" else None
    return data[split][:None if full else (limit if limit is not None else cap)]


@torch.inference_mode()
def assess(net, batches, device, cfg, label, output=None, *, diagnostics=False):
    net.eval()
    meter = Evaluation(cfg["confidence_threshold"], cfg["association_iou"], diagnostics=diagnostics)
    with (output.open("w", encoding="utf-8") if output else nullcontext()) as stream:
        for images, targets in track(batches, description=label):
            views = {k: v.float().cpu() for k,v in net(images.to(device)).items()}
            for cat, box, confidence, truth in zip(views["category"], views["box"], views["visibility"], targets):
                row = meter.add(cat, box, confidence, truth)
                if stream:
                    stream.write(json.dumps(row) + "\n")
    return meter.result()


def seed(value):
    random.seed(value)
    torch.manual_seed(value)


def train(args, device, rank, world):
    architecture = Architecture(backbone=args.backbone)
    weights = LossWeights(match_box=args.lambda_box, regression=args.lambda_reg, confidence=args.lambda_conf,
                          l1=args.lambda_l1, giou=args.lambda_giou)
    splits = {s: getattr(args, f"{s}_sequences") for s in SPLITS}
    print(f"[rank {rank}] Indexing {args.data}", flush=True)
    data, audit, fingerprint = index_data(args.data, splits)
    cfg = dict(architecture=asdict(architecture), loss=asdict(weights), splits=splits, fingerprint=fingerprint,
               size=args.size, seed=args.seed, overfit=args.overfit, limit_train=args.limit_train,
               limit_val=args.limit_val, confidence_threshold=args.confidence_threshold,
               association_iou=args.association_iou, batch_size=args.batch_size, world_size=world,
               lr=args.lr, weight_decay=args.weight_decay, pretrained=not args.random_init,
               workers=args.workers)
    if args.no_augmentation:
        cfg["no_augmentation"] = True
    last = args.output / "last.pt"
    checkpoint = torch.load(last, map_location="cpu", weights_only=True) if args.resume else None
    if checkpoint is not None and checkpoint["config"] != cfg:
        raise ValueError("Saved training configuration or data index differs")
    if not args.resume and last.exists():
        raise FileExistsError(f"{last} exists; use --resume or a fresh output directory")
    start = checkpoint["epoch"] + 1 if checkpoint else 1
    if start > args.epochs:
        raise ValueError("Requested epochs are already complete")
    train_pairs = selected(data, cfg, "train")
    batches = loader(train_pairs, cfg, args.batch_size // world, args.workers, device, True, rank, world)
    if rank == 0:
        validation = loader(selected(data, cfg, "val"), cfg, args.batch_size // world, args.workers, device)
        args.output.mkdir(parents=True, exist_ok=True)
        json_file(args.output / "config.json", cfg)
        json_file(args.output / "data_audit.json", {"splits": audit, "train_used": len(train_pairs),
                                                    "val_used": len(validation.dataset)})
        print(f"[startup] {len(train_pairs)} train pairs, {len(validation.dataset)} validation pairs", flush=True)
    seed(args.seed)
    local_weights = args.weights
    if not local_weights and not args.random_init:
        filename = "resnet50-11ad3fa6.pth" if args.backbone == "resnet50" else "vit_b_16-c867db91.pth"
        candidate = Path("weights") / filename
        local_weights = candidate if candidate.is_file() else None
    initialize = checkpoint is None and rank == 0 and not args.random_init
    net = AssociationNetwork(architecture, pretrained=initialize,
                             weights_file=local_weights if initialize else None).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history, best = [], (-1., -1.)
    if checkpoint:
        net.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history, best = checkpoint["history"], tuple(checkpoint["best"])
        del checkpoint
    model = DistributedDataParallel(net, device_ids=[device.index] if device.type == "cuda" else None) if world > 1 else net
    if start == 1 and args.baseline:
        if rank == 0:
            baseline = assess(net, validation, device, cfg, "Before training")
            json_file(args.output / "baseline.json", baseline)
        if world > 1:
            dist.barrier()
    for epoch in range(start, args.epochs + 1):
        seed(args.seed + epoch * world + rank)
        if isinstance(batches.sampler, DistributedSampler):
            batches.sampler.set_epoch(epoch)
        model.train()
        totals = torch.zeros(5, device=device)
        iterator = track(batches, description=f"Train {epoch}/{args.epochs}") if rank == 0 else batches
        for images, targets in iterator:
            optimizer.zero_grad(set_to_none=True)
            losses = objective(model(images.to(device)), targets, weights)
            if not losses["loss"].isfinite():
                raise FloatingPointError(f"Non-finite loss on {[target['key'] for target in targets]}")
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            optimizer.step()
            totals[:4] += torch.stack(list(losses.values())).detach() * len(targets)
            totals[4] += len(targets)
        if world > 1:
            dist.all_reduce(totals)
        if rank == 0:
            metrics = assess(net, validation, device, cfg, f"Val {epoch}/{args.epochs}")
            score = (metrics["MDA"], metrics["mAP"] or 0.)
            row = {"epoch": epoch, "train": dict(zip(losses.keys(), (totals[:4] / totals[4]).tolist())),
                   "val": metrics}
            history.append(row)
            state = {"config": cfg, "epoch": epoch, "model": net.state_dict(), "optimizer": optimizer.state_dict(),
                     "history": history, "best": max(best, score)}
            if score > best:
                best = score
                atomic(args.output / "best.pt", lambda p: torch.save(state, p))
            atomic(last, lambda p: torch.save(state, p))
            json_file(args.output / "metrics.json", history)
            print(json.dumps(row), flush=True)
        if world > 1:
            dist.barrier()


def evaluate(args, device):
    save = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = save["config"]
    data, _, fingerprint = index_data(args.data, cfg["splits"])
    model = AssociationNetwork(Architecture(**cfg["architecture"]))
    model.load_state_dict(save["model"])
    epoch = save["epoch"]
    del save
    model.to(device)
    if args.predictions:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)

    def evaluate_split(split):
        batches = loader(selected(data, cfg, split, args.limit, args.full), cfg,
                         args.batch_size, args.workers, device)
        result = assess(model, batches, device, cfg, f"Evaluate {split}", args.predictions,
                        diagnostics=args.diagnostics)
        result.update(checkpoint_epoch=epoch, split=split,
                      data_changed=fingerprint != cfg["fingerprint"],
                      same_image_validation=cfg["overfit"] and split == "val")
        return result

    report = ({split: evaluate_split(split) for split in ("train", "val")}
              if args.split == "train-val" else evaluate_split(args.split))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        json_file(args.output, report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit, infer, audit = [commands.add_parser(name) for name in ("train", "evaluate", "audit")]
    for sub in (fit, infer, audit):
        sub.add_argument("--data", type=Path, default=Path("crossdrone-data"))
    for sub in (fit, infer):
        sub.add_argument("--device", default="auto")
        sub.add_argument("--workers", type=int, default=2)
        sub.add_argument("--batch-size", type=int, default=4)
    for sub in (fit, audit):
        for name, default in SPLITS.items():
            sub.add_argument(f"--{name}-sequences", nargs="+", default=default)
    fit.add_argument("--backbone", choices=["resnet50", "vit_b_16"], default="resnet50")
    fit.add_argument("--weights", type=Path)
    fit.add_argument("--random-init", action="store_true")
    fit.add_argument("--size", type=int, nargs=2, default=[576,1024], metavar=("H","W"))
    fit.add_argument("--output", type=Path, default=Path("runs/paper-resnet50"))
    fit.add_argument("--epochs", type=int, default=100)
    fit.add_argument("--lr", type=float, default=1e-4)
    fit.add_argument("--weight-decay", type=float, default=1e-5)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--baseline", action="store_true")
    fit.add_argument("--resume", action="store_true")
    fit.add_argument("--overfit", action="store_true")
    fit.add_argument("--no-augmentation", action="store_true", help="keep an independent validation split")
    fit.add_argument("--limit-train", type=int)
    fit.add_argument("--limit-val", type=int)
    fit.add_argument("--confidence-threshold", type=float, default=.5)
    fit.add_argument("--association-iou", type=float, default=.5)
    for field, default in (("box",1.), ("reg",1.), ("conf",1.), ("l1",5.), ("giou",2.)):
        fit.add_argument(f"--lambda-{field}", type=float, default=default)
    infer.add_argument("checkpoint", type=Path)
    infer.add_argument("--split", choices=(*SPLITS, "train-val"), default="test")
    selection = infer.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int)
    selection.add_argument("--full", action="store_true")
    infer.add_argument("--output", type=Path)
    infer.add_argument("--predictions", type=Path)
    infer.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    if args.command == "audit":
        _, counts, fingerprint = index_data(args.data, {s: getattr(args, f"{s}_sequences") for s in SPLITS})
        print(json.dumps({"splits": counts, "fingerprint": fingerprint}, indent=2))
        return
    if args.batch_size < 1 or args.workers < 0:
        parser.error("Invalid batch size or worker count")
    if args.command == "train":
        if args.overfit and (args.limit_train is None or args.limit_val is not None):
            parser.error("--overfit requires --limit-train and uses the same pairs for validation")
        if args.epochs < 1 or args.lr <= 0 or args.weight_decay < 0:
            parser.error("Invalid optimizer settings")
        if min(args.size) < 64 or any(x % 16 for x in args.size):
            parser.error("Image dimensions must be at least 64 and divisible by 16")
        if any(x is not None and x < 1 for x in (args.limit_train, args.limit_val)):
            parser.error("Limits must be positive")
        if args.random_init and args.weights:
            parser.error("--random-init and --weights are mutually exclusive")
        if not (0 <= args.confidence_threshold <= 1 and 0 < args.association_iou <= 1):
            parser.error("Invalid confidence or association threshold")
        if min(args.lambda_box, args.lambda_reg, args.lambda_conf, args.lambda_l1, args.lambda_giou) < 0:
            parser.error("Loss weights must be nonnegative")
    else:
        if args.limit is not None and args.limit < 1:
            parser.error("Limits must be positive")
        if args.split == "train-val" and args.predictions:
            parser.error("--predictions requires a single split")
        paths = [p.resolve() for p in (args.checkpoint, args.output, args.predictions) if p]
        if len(set(paths)) != len(paths):
            parser.error("Checkpoint, metrics and predictions require distinct paths")
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    spec = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if world > 1:
        if args.command != "train" or device.type not in {"cuda","cpu"} or device.index is not None:
            parser.error("torchrun supports training with --device cuda or cpu")
        if args.batch_size % world:
            parser.error("Global batch size must be divisible by process count")
        if device.type == "cuda":
            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
            torch.cuda.set_device(device)
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        if args.command == "train":
            train(args, device, rank, world)
        else:
            evaluate(args, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
