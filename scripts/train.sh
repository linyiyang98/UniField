#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${GPUS:-4,5}"
export TOKENIZERS_PARALLELISM=false
PYTHON="${PYTHON:-python}"
NPROC="${NPROC:-2}"
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" -m unifield.train "$@"
