#!/bin/bash
#SBATCH --job-name=phase3b-prepare
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/phase3/setup/phase3b_prepare_%j.out
#SBATCH --error=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/phase3/setup/phase3b_prepare_%j.err

set -eo pipefail

PROJECT_ROOT=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting
DATASET_ROOT=$PROJECT_ROOT/datasets/synthetic/phase3_controlled_pilot
CONFIG=$PROJECT_ROOT/configs/synthetic/phase3_controlled_multiseed.json
SEEDS=1,2,3,4
POINT_COUNT=100000

mkdir -p "$PROJECT_ROOT/jobs/logs/phase3/setup"

module purge
source /home/usd.local/kiran.prasannannair/miniforge3/etc/profile.d/conda.sh
set +u
conda activate /home/rizk_lab/shared/kiran/envs/kernel_splat
set -u

cd "$PROJECT_ROOT"

echo "============================================================"
echo "PHASE III-B SYNTHETIC MULTI-SEED GENERATION + INPUT PREP"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Dataset root: $DATASET_ROOT"
echo "Seeds: $SEEDS"
echo "Started: $(date)"
echo "============================================================"

python scripts/synthetic/generate_phase3b_multiseed.py \
  --config "$CONFIG" \
  --seeds "$SEEDS"

mapfile -t SCENES < <(python scripts/synthetic/generate_phase3b_multiseed.py \
  --config "$CONFIG" \
  --seeds "$SEEDS" \
  --print-scene-ids)

for SCENE_ID in "${SCENES[@]}"; do
  SEED=$(python - "$SCENE_ID" <<'PY'
import re
import sys
match = re.search(r"seed(\d+)$", sys.argv[1])
if not match:
    raise SystemExit(f"Cannot parse seed from scene id: {sys.argv[1]}")
print(int(match.group(1)))
PY
)
  echo "Preparing deterministic points3d.ply for $SCENE_ID with seed $SEED"
  python scripts/synthetic/prepare_nerf_synthetic_inputs.py \
    --dataset-root "$DATASET_ROOT" \
    --scene-id "$SCENE_ID" \
    --point-count "$POINT_COUNT" \
    --seed "$SEED"
done

python scripts/synthetic/inventory_phase3_outputs.py || true

echo "Phase III-B generation/preparation complete at $(date)"
