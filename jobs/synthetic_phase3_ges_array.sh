#!/bin/bash
#SBATCH --job-name=synthetic-ges
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu003,gpu004,gpu005
#SBATCH --array=0-8%3
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/phase3/ges/synthetic_ges_%A_%a.out
#SBATCH --error=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/phase3/ges/synthetic_ges_%A_%a.err

set -eo pipefail

PROJECT_ROOT=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting
GES_ROOT=$PROJECT_ROOT/external/ges-splatting
DATASET_ROOT=$PROJECT_ROOT/datasets/synthetic/phase3_controlled_pilot
OUTPUT_ROOT=$PROJECT_ROOT/outputs/synthetic/phase3_controlled_pilot/ges
SCENES=(
  phase3_edge_sharpness_low_seed0000
  phase3_edge_sharpness_medium_seed0000
  phase3_edge_sharpness_high_seed0000
  phase3_spatial_frequency_low_seed0000
  phase3_spatial_frequency_medium_seed0000
  phase3_spatial_frequency_high_seed0000
  phase3_curvature_low_seed0000
  phase3_curvature_medium_seed0000
  phase3_curvature_high_seed0000
)
SCENE_ID=${SCENES[$SLURM_ARRAY_TASK_ID]}
DATASET=$DATASET_ROOT/$SCENE_ID
OUTPUT=$OUTPUT_ROOT/$SCENE_ID

mkdir -p "$OUTPUT" "$PROJECT_ROOT/jobs/logs/phase3/ges"

module purge
module load cuda/12.3
source /home/usd.local/kiran.prasannannair/miniforge3/etc/profile.d/conda.sh
set +u
conda activate /home/rizk_lab/shared/kiran/envs/ges_splat
set -u

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
export TORCH_CUDA_ARCH_LIST="7.0"
export WANDB_MODE=offline
export WANDB_SILENT=true
LOCAL_SCRATCH=${SLURM_TMPDIR:-/tmp}/ges_synthetic_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}
export WANDB_DIR=$LOCAL_SCRATCH/wandb
export WANDB_CACHE_DIR=$LOCAL_SCRATCH/wandb_cache
export WANDB_CONFIG_DIR=$LOCAL_SCRATCH/wandb_config
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

resolve_model_path() {
  python - "$OUTPUT" <<'PY'
from pathlib import Path
import sys
requested = Path(sys.argv[1])
def looks_like_model(path):
    return (path / "cfg_args").exists() or (path / "cameras.json").exists() or any(path.glob("point_cloud/iteration_*/point_cloud.ply"))
if looks_like_model(requested):
    print(requested)
    raise SystemExit(0)
parent = requested.parent
prefix = requested.name
candidates = [path for path in parent.glob(prefix + "*") if path.is_dir() and looks_like_model(path)]
if not candidates:
    print(requested)
    raise SystemExit(0)
def latest_mtime(path):
    mtimes = [p.stat().st_mtime for p in path.rglob("*") if p.exists()]
    mtimes.append(path.stat().st_mtime)
    return max(mtimes)
print(max(candidates, key=latest_mtime))
PY
}

cd "$PROJECT_ROOT"
python scripts/synthetic/patch_nerf_synthetic_loader_dtype.py --project-root "$PROJECT_ROOT"
python scripts/synthetic/patch_nerf_synthetic_loader_dtype.py --project-root "$PROJECT_ROOT" --verify-only
python scripts/synthetic/prepare_nerf_synthetic_inputs.py --dataset-root "$DATASET_ROOT" --scene-id "$SCENE_ID" --point-count 100000 --seed 0 --check-only

echo "============================================================"
echo "GES SYNTHETIC PHASE-III PILOT"
echo "Job ID: ${SLURM_JOB_ID:-unknown} / task ${SLURM_ARRAY_TASK_ID:-unknown}"
echo "Node: $(hostname)"
echo "Scene: $SCENE_ID"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT"
echo "W&B dir: $WANDB_DIR"
echo "Started: $(date)"
echo "============================================================"
nvidia-smi

cd "$GES_ROOT"
/usr/bin/time -v python train_ges.py -s "$DATASET" -m "$OUTPUT" --eval
ACTUAL_OUTPUT=$(resolve_model_path)
echo "Requested output: $OUTPUT"
echo "Actual output: $ACTUAL_OUTPUT"
python render.py -m "$ACTUAL_OUTPUT" --skip_train
python metrics.py -m "$ACTUAL_OUTPUT"

du -sh "$ACTUAL_OUTPUT"
python - <<PY
from pathlib import Path
from plyfile import PlyData
root = Path("$ACTUAL_OUTPUT")
plys = list(root.glob("point_cloud/iteration_*/point_cloud.ply"))
if not plys:
    print("No PLY found.")
else:
    def iteration(path): return int(path.parent.name.split("_")[-1])
    path = max(plys, key=iteration)
    ply = PlyData.read(str(path))
    print("Final PLY:", path)
    print("Final iteration:", iteration(path))
    print("Final primitive count:", len(ply["vertex"]))
PY

echo "GES synthetic task complete: $SCENE_ID at $(date)"
