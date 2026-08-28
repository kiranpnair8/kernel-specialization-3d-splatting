#!/bin/bash
#SBATCH --job-name=synthetic-drk-curv-corrected
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu003,gpu004,gpu005
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/phase3/drk/synthetic_drk_curvature_high_corrected_%j.out
#SBATCH --error=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/phase3/drk/synthetic_drk_curvature_high_corrected_%j.err

set -eo pipefail

PROJECT_ROOT=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting
DRK_ROOT=$PROJECT_ROOT/external/drk
DATASET_ROOT=$PROJECT_ROOT/datasets/synthetic/phase3_controlled_pilot
SCENE_ID=phase3_curvature_high_corrected_seed0000
DATASET=$DATASET_ROOT/$SCENE_ID
OUTPUT=$PROJECT_ROOT/outputs/synthetic/phase3_controlled_pilot/drk/$SCENE_ID

mkdir -p "$PROJECT_ROOT/jobs/logs/phase3/drk" "$(dirname "$OUTPUT")"
if [ -d "$OUTPUT" ] && [ -n "$(find "$OUTPUT" -mindepth 1 -print -quit)" ]; then
  echo "Refusing to overwrite non-empty output directory: $OUTPUT"
  exit 1
fi

module purge
module load cuda/12.3
source /home/usd.local/kiran.prasannannair/miniforge3/etc/profile.d/conda.sh
set +u
conda activate /home/rizk_lab/shared/kiran/envs/drk_splat
set -u

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
export TORCH_CUDA_ARCH_LIST="7.0"
export WANDB_MODE=offline
export WANDB_SILENT=true

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
python scripts/synthetic/promote_corrected_curvature.py --validate-only
python scripts/synthetic/prepare_nerf_synthetic_inputs.py --dataset-root "$DATASET_ROOT" --scene-id "$SCENE_ID" --point-count 100000 --seed 0 --check-only

echo "============================================================"
echo "DRK SYNTHETIC PHASE-III CORRECTED CURVATURE HIGH"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Scene: $SCENE_ID"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT"
echo "Started: $(date)"
echo "============================================================"
nvidia-smi

cd "$DRK_ROOT"
/usr/bin/time -v python train.py \
  -s "$DATASET" \
  -m "$OUTPUT" \
  --eval \
  --gs_type DRK \
  --kernel_density dense \
  --cache_sort

ACTUAL_OUTPUT=$(resolve_model_path)
echo "Requested output: $OUTPUT"
echo "Actual output: $ACTUAL_OUTPUT"
python train.py \
  -s "$DATASET" \
  -m "$OUTPUT" \
  --eval \
  --gs_type DRK \
  --kernel_density dense \
  --cache_sort \
  --metric \
  --load_iteration -1
ACTUAL_OUTPUT=$(resolve_model_path)

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

echo "DRK corrected curvature-high task complete at $(date)"
