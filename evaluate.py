"""Evaluate a Cross-Drone checkpoint against full dual-view vehicle annotations."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import torch

from crossdrone.data import index_dataset
from crossdrone.diagnostics import Diagnostics, evaluation_scenes
from crossdrone.engine import evaluate, make_loader, resolve_device, write_json
from model import ModelConfig, build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=Path("数据集"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, help="override checkpoint scene limit")
    selection.add_argument("--full", action="store_true", help="ignore saved scene limit")
    parser.add_argument("--diagnostics", action="store_true", help="include raw box and score diagnostics")
    parser.add_argument("--output", type=Path, help="write evaluation metrics as JSON")
    parser.add_argument("--predictions", type=Path, help="export per-frame predictions as JSONL")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or (args.limit is not None and args.limit < 1):
        parser.error("invalid batch size, number of workers or frame limit")
    paths = [path.resolve() for path in (args.checkpoint, args.output, args.predictions) if path]
    if len(set(paths)) != len(paths):
        parser.error("checkpoint, metrics and predictions must use different files")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    settings = checkpoint["settings"]
    data, audit = index_dataset(args.data, settings["sequences"])
    dataset_changed = audit["annotation_and_availability_sha256"] != settings["annotation_and_availability_sha256"]
    device = resolve_device(args.device)
    loader = make_loader(
        evaluation_scenes(data, settings, args.split, args.limit, args.full),
        settings["size"], device, args.batch_size, args.workers
    )
    model_settings = {"mvoq_share_attention": True, **settings["model"], "pretrained": False}
    net = build_model(ModelConfig(**model_settings))
    net.load_state_dict(checkpoint["model"])
    epoch = checkpoint["epoch"]
    del checkpoint
    net.to(device)
    for path in (args.output, args.predictions):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    predictions = args.predictions.open("w", encoding="utf-8") if args.predictions else nullcontext()
    diagnostics = Diagnostics(settings["confidence_threshold"]) if args.diagnostics else None
    with predictions as stream:
        result = evaluate(net, loader, device, settings["confidence_threshold"], settings["association_iou"], stream,
                          progress_label=f"Eval {args.split}", diagnostics=diagnostics)
    if diagnostics is not None:
        result["diagnostics"] = diagnostics.compute()
    result = {"checkpoint_epoch": epoch, "overfit_run": settings.get("overfit", False),
              "mvoq_attention_shared": model_settings["mvoq_share_attention"], "split": args.split, "dataset_index_changed": dataset_changed, **result}
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
