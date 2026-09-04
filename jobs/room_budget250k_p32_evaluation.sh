#!/bin/bash
#SBATCH --job-name=room-budget250k-p32
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/room_budget250k_p32_evaluation_%j.out
#SBATCH --error=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/room_budget250k_p32_evaluation_%j.err

set -eo pipefail

PROJECT_ROOT=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting
CONFIG=$PROJECT_ROOT/configs/room_3dgs_ges_drk_budget250k_p32.json
BASELINE_RESULTS=$PROJECT_ROOT/results/room/3dgs_vs_ges_vs_drk_p32
BUDGET_RESULTS=$PROJECT_ROOT/results/room/3dgs_vs_ges_vs_drk_budget250k_p32
ALIGNMENT_JSON=$BUDGET_RESULTS/alignment_audit.json
COMPARISON_DIR=$BUDGET_RESULTS/baseline_comparison

mkdir -p "$PROJECT_ROOT/jobs/logs" "$BUDGET_RESULTS" "$COMPARISON_DIR"

source /home/usd.local/kiran.prasannannair/miniforge3/etc/profile.d/conda.sh
set +u
conda activate /home/rizk_lab/shared/kiran/envs/kernel_splat
set -u

cd "$PROJECT_ROOT"

printf 'Room budget250k p32 evaluation\n'
printf 'hostname: '; hostname
printf 'config: %s\n' "$CONFIG"
printf 'baseline_results: %s\n' "$BASELINE_RESULTS"
printf 'budget_results: %s\n' "$BUDGET_RESULTS"

python scripts/evaluate/audit_alignment.py \
    --reference-gt-dir outputs/3dgs/room_baseline/test/ours_30000/gt \
    --candidate-gt-dir outputs/drk/room_budget250k_DRK/metric/test \
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

python scripts/evaluate/local_compare.py \
    --config "$CONFIG" \
    --inspect-only

/usr/bin/time -v python scripts/evaluate/local_compare.py \
    --config "$CONFIG" \
    --no-maps \
    --no-predictors

python scripts/evaluate/compare_local_summaries.py \
    --baseline-results "$BASELINE_RESULTS" \
    --candidate-results "$BUDGET_RESULTS" \
    --baseline-label room_baseline_p32 \
    --candidate-label room_budget250k_p32 \
    --output-dir "$COMPARISON_DIR"

python - "$BUDGET_RESULTS/summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
summary = json.loads(path.read_text())
oracle = summary["oracle"]
methods = summary["config"]["methods"]
print("\nRoom budget250k p32 local summary")
for key in ["patch_count", "decisive_patch_count", "tie_fraction", "oracle_patch_mse", "oracle_patch_psnr"]:
    print(f"{key}: {oracle[key]}")
non_3dgs = 0.0
for method in methods:
    winner_fraction = float(oracle[f"{method}_winner_fraction"])
    if method != "3dgs":
        non_3dgs += winner_fraction
    method_mse = float(oracle[f"{method}_patch_mse"])
    improvement = float(oracle[f"oracle_improvement_mse_vs_{method}"])
    relative = 0.0 if method_mse == 0.0 else 100.0 * improvement / method_mse
    print(f"{method}_winner_fraction: {winner_fraction}")
    print(f"{method}_patch_mse: {method_mse}")
    print(f"oracle_relative_improvement_pct_vs_{method}: {relative}")
print(f"non_3dgs_winner_fraction: {non_3dgs}")
PY
