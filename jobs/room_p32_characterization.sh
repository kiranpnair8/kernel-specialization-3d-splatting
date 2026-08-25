#!/bin/bash
#SBATCH --job-name=room-p32-char
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/room_p32_characterization_%j.out
#SBATCH --error=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/room_p32_characterization_%j.err

set -eo pipefail

PROJECT_ROOT=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting
CONFIG=$PROJECT_ROOT/configs/room_3dgs_ges_drk_p32.json
ALIGNMENT_JSON=$PROJECT_ROOT/results/room/3dgs_vs_ges_vs_drk_p32/alignment_audit.json
RESULTS_DIR=$PROJECT_ROOT/results/room/3dgs_vs_ges_vs_drk_p32

mkdir -p "$PROJECT_ROOT/jobs/logs" "$RESULTS_DIR"

source /home/usd.local/kiran.prasannannair/miniforge3/etc/profile.d/conda.sh
set +u
conda activate /home/rizk_lab/shared/kiran/envs/kernel_splat
set -u

cd "$PROJECT_ROOT"

python scripts/evaluate/audit_alignment.py \
    --reference-gt-dir outputs/3dgs/room_baseline/test/ours_30000/gt \
    --candidate-gt-dir outputs/drk/room_baseline_DRK/metric/test \
    --candidate-name-template 'gt_{stem}.png' \
    --output-json "$ALIGNMENT_JSON"

python - "$ALIGNMENT_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
if not payload.get("exact_or_unambiguous"):
    print(f"Alignment audit failed; inspect {path}", file=sys.stderr)
    raise SystemExit(2)
if payload.get("reference_count") != 39:
    print(f"Expected 39 Room test views, got {payload.get('reference_count')}; inspect {path}", file=sys.stderr)
    raise SystemExit(2)
print(f"Alignment audit passed: {path}")
PY

/usr/bin/time -v python scripts/evaluate/local_compare.py \
    --config "$CONFIG" \
    --no-maps \
    --no-predictors

/usr/bin/time -v python scripts/evaluate/characterize_specialization.py \
    --config "$CONFIG"
