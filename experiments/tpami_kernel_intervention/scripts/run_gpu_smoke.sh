#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

python "$EXPERIMENT_ROOT/tests/gpu_smoke_test.py" "$@"
python "$EXPERIMENT_ROOT/tests/gpu_depth_gradient_regression.py"
