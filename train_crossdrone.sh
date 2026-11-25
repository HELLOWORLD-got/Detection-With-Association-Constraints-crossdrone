#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

printf '[Startup] Launching torchrun with CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES" >&2
exec python -u -m torch.distributed.run --standalone --nproc-per-node=2 --local-addr=127.0.0.1 \
    "$project_dir/train.py" \
    --device cuda \
    --data "$project_dir/crossdrone-data" \
    --output "$project_dir/runs/crossdrone-resnet50" \
    "$@"
