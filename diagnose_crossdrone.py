"""Compare existing Cross-Drone runs without updating weights."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from statistics import mean


DEFAULT_RUNS = (
    "runs/overfit-8-position-key",
    "runs/overfit-8-content-key",
    "runs/overfit-8-position-size-prior",
)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def history_summary(path):
    if not path.is_file():
        return None
    rows = load_json(path)
    if not rows:
        return None
    recent = rows[-10:]
    peak = max(rows, key=lambda x: x["val"]["mAP50"] or 0)
    raw_iou = [x["val"]["raw_best_iou"]["mean"] for x in recent if "raw_best_iou" in x["val"]]
    ap50 = [x["val"]["mAP50"] for x in recent if x["val"]["mAP50"] is not None]
    return {
        "last_epoch": rows[-1]["epoch"],
        "first_nonzero_ap50_epoch": next((x["epoch"] for x in rows if x["val"]["mAP50"]), None),
        "peak_ap50": peak["val"]["mAP50"],
        "peak_ap50_epoch": peak["epoch"],
        "recent_epochs": len(recent),
        "recent_mean_ap50": mean(ap50) if ap50 else None,
        "recent_mean_raw_iou": mean(raw_iou) if raw_iou and None not in raw_iou else None,
        "last_regression_loss": rows[-1]["train"]["loss_reg"],
        "last_gradient_norms": rows[-1]["train"].get("last_step_gradient_norms"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="*", help="run directories (default: the three 8-pair experiments)")
    parser.add_argument("--data", type=Path, default=Path("crossdrone-data"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.data.is_dir():
        parser.error(f"dataset directory missing: {args.data}")
    if args.batch_size < 1 or args.workers < 0:
        parser.error("invalid batch size or worker count")

    runs = args.runs or list(map(Path, DEFAULT_RUNS))
    output = args.output or Path("runs") / datetime.now().strftime("diagnosis-%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    evaluate = Path(__file__).with_name("evaluate.py")
    report = {"runs": [], "errors": []}
    lines = [
        "# Cross-Drone checkpoint diagnostics", "",
        "| Run | Checkpoint | Split | Epoch | Mean raw IoU | Raw / kept coverage | AP50 | MDA | UCR |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    details = [
        "", "## Why targets were missed", "",
        "| Run / checkpoint / split | No IoU≥0.5 | Vehicle score | Visibility score | Both scores | Different candidates pass each score | Recenter oracle | Resize oracle | Same-query coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    histories = [
        "", "## Training history", "",
        "| Run | Last epoch | First nonzero AP50 | Peak AP50 (epoch) | Recent mean AP50 |",
        "|---|---:|---:|---:|---:|",
    ]

    def percent(value):
        return "—" if value is None else f"{value:.2%}"

    for i, run in enumerate(runs, 1):
        item = {"run": str(run), "checkpoints": []}
        report["runs"].append(item)
        try:
            item["history"] = history_summary(run / "metrics.json")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report["errors"].append(f"{run}/metrics.json: {exc}")
        history = item.get("history")
        if history is not None:
            histories.append(
                f"| {i}. {run.name} | {history['last_epoch']} | "
                f"{history['first_nonzero_ap50_epoch'] or '—'} | "
                f"{percent(history['peak_ap50'])} ({history['peak_ap50_epoch']}) | "
                f"{percent(history['recent_mean_ap50'])} |"
            )
        for tag in ("best", "last"):
            checkpoint = run / f"{tag}.pt"
            if not checkpoint.is_file():
                report["errors"].append(f"Missing {checkpoint}")
                continue
            splits = ["train"]
            for split in splits:
                prefix = output / f"{i:02d}-{run.name}-{tag}-{split}"
                log = Path(f"{prefix}.log")
                result_path = Path(f"{prefix}.json")
                predictions_path = Path(f"{prefix}.jsonl")
                print(f"Evaluating {checkpoint} ({split}); log: {log}", flush=True)
                with log.open("w", encoding="utf-8") as stream:
                    process = subprocess.run(
                        [sys.executable, "-u", str(evaluate), str(checkpoint), "--data", str(args.data),
                         "--split", split, "--device", args.device, "--batch-size", str(args.batch_size),
                         "--workers", str(args.workers), "--diagnostics", "--output", str(result_path),
                         "--predictions", str(predictions_path)],
                        stdout=stream, stderr=subprocess.STDOUT, check=False,
                    )
                if process.returncode:
                    report["errors"].append(f"{checkpoint} ({split}) failed; see {log}")
                    continue
                result = load_json(result_path)
                detail = result.pop("diagnostics")["summary"]
                item["checkpoints"].append({
                    "tag": tag, "report": result_path.name, "summary": detail, **result,
                })
                if result["dataset_index_changed"]:
                    report["errors"].append(f"Data index differs from {checkpoint}")
                # Same-image overfit validation duplicates train.
                if split == "train" and not result["overfit_run"]:
                    splits.append("val")
                misses = detail["coverage_failure_counts"]
                shared = detail["raw_same_query_iou50"]
                details.append(
                    f"| {i}. {run.name} / {tag} / {split} | {misses['localization']} | "
                    f"{misses['vehicle']} | {misses['visibility']} | {misses['both']} | "
                    f"{misses['disjoint_candidates']} | "
                    f"{percent(detail['oracle_center_fixed_coverage_iou50'])} | "
                    f"{percent(detail['oracle_size_fixed_coverage_iou50'])} | "
                    f"{shared['covered']}/{shared['shared_targets']} |"
                )
                raw_mean = detail["raw_best_iou"]["mean"]
                raw_iou = "—" if raw_mean is None else f"{raw_mean:.4f}"
                lines.append(
                    f"| {i}. {run.name} | {tag} | {split} | {result['checkpoint_epoch']} | "
                    f"{raw_iou} | "
                    f"{percent(detail['raw_oracle_coverage_iou50'])} / "
                    f"{percent(detail['kept_oracle_coverage_iou50'])} | "
                    f"{percent(result['mAP50'])} | {percent(result['MDA'])} | {percent(result['UCR'])} |"
                )

    lines.extend(details)
    lines.extend(histories)
    lines.extend(["", "See summary.json for center/size oracles, threshold failure counts, "
                  "same-query cross-view coverage, and training history summaries."])
    if report["errors"]:
        lines.extend(["", "## Missing, failed, or changed inputs", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {output / 'summary.md'}", flush=True)
    return bool(report["errors"])


if __name__ == "__main__":
    sys.exit(main())
