#!/usr/bin/env bash
set -eo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 DATASET_PATH OUTPUT_PATH BETA [ITERATIONS]" >&2
  exit 2
fi

DATASET_PATH=$1
OUTPUT_PATH=$2
KERNEL_BETA=$3
ITERATIONS=${4:-20}

case "$KERNEL_BETA" in
  1|2|4) ;;
  *) echo "BETA must be 1, 2, or 4" >&2; exit 2 ;;
esac

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
GS_ROOT="$EXPERIMENT_ROOT/vendor/gaussian-splatting"

if [[ ! -d "$DATASET_PATH" ]]; then
  echo "Dataset does not exist: $DATASET_PATH" >&2
  exit 1
fi
if [[ -e "$OUTPUT_PATH" ]]; then
  echo "Refusing to overwrite existing smoke output: $OUTPUT_PATH" >&2
  exit 1
fi

cd "$GS_ROOT"
python train.py \
  -s "$DATASET_PATH" \
  -m "$OUTPUT_PATH" \
  --eval \
  --iterations "$ITERATIONS" \
  --save_iterations "$ITERATIONS" \
  --kernel_beta "$KERNEL_BETA"

echo "Smoke training completed: beta=$KERNEL_BETA iterations=$ITERATIONS output=$OUTPUT_PATH"
