#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
mode="${1:-help}"
case "$mode" in
  diagnose)
    if [[ $# -lt 2 ]]; then
      printf 'Usage: bash debug_crossdrone.sh diagnose CHECKPOINT [OUTPUT_DIR]\n' >&2
      exit 2
    fi
    checkpoint="$2"
    output="${3:-runs/diagnostics-$(date +%Y%m%d-%H%M%S)}"
    mkdir -p "$output"
    for split in train val; do
      # One process on the first visible GPU; no distributed synchronization required.
      python -u evaluate.py "$checkpoint" --data crossdrone-data --device cuda \
        --split "$split" --diagnostics --workers 2 --batch-size 2 \
        --output "$output/$split.json" > "$output/$split.log" 2>&1
      printf 'Saved %s/%s.json\n' "$output" "$split"
    done
    ;;
  overfit)
    output="${2:-runs/overfit-$(date +%Y%m%d-%H%M%S)}"
    mkdir -p "$output"
    bash train_crossdrone.sh --overfit --limit-train 8 --epochs 200 \
      --workers 2 --validate-before-training --output "$output" \
      2>&1 | tee "$output/train.log"
    bash debug_crossdrone.sh diagnose "$output/last.pt" "$output/diagnostics"
    ;;
  *)
    printf 'Usage:\n  bash debug_crossdrone.sh diagnose CHECKPOINT [OUTPUT_DIR]\n  bash debug_crossdrone.sh overfit [OUTPUT_DIR]\n'
    [[ "$mode" == help ]]
    ;;
esac
