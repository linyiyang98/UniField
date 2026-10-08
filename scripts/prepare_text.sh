#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${GPUS:-4}"
export TOKENIZERS_PARALLELISM=false
exec "${PYTHON:-python}" -m unifield.prepare_text "$@"
