#!/bin/bash
#SBATCH --job-name=synthetic-3dgs-curv-corrected
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu003
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/phase3/3dgs/synthetic_3dgs_curvature_high_corrected_%j.out
#SBATCH --error=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/phase3/3dgs/synthetic_3dgs_curvature_high_corrected_%j.err

set -eo pipefail

PROJECT_ROOT=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting
GS_ROOT=$PROJECT_ROOT/external/gaussian-splatting
DATASET_ROOT=$PROJECT_ROOT/datasets/synthetic/phase3_controlled_pilot
SCENE_ID=phase3_curvature_high_corrected_seed0000
DATASET=$DATASET_ROOT/$SCENE_ID
OUTPUT=$PROJECT_ROOT/outputs/synthetic/phase3_controlled_pilot/3dgs/$SCENE_ID

mkdir -p "$PROJECT_ROOT/jobs/logs/phase3/3dgs" "$(dirname "$OUTPUT")"
if [ -d "$OUTPUT" ] && [ -n "$(find "$OUTPUT" -mindepth 1 -print -quit)" ]; then
  echo "Refusing to overwrite non-empty output directory: $OUTPUT"
  exit 1
fi

module purge
module load cuda/12.3
source /home/usd.local/kiran.prasannannair/miniforge3/etc/profile.d/conda.sh
set +u
conda activate /home/rizk_lab/shared/kiran/envs/kernel_splat
set -u

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
export TORCH_CUDA_ARCH_LIST="7.0"
export WANDB_MODE=offline
export WANDB_SILENT=true

cd "$PROJECT_ROOT"
python scripts/synthetic/patch_nerf_synthetic_loader_dtype.py --project-root "$PROJECT_ROOT"
python scripts/synthetic/patch_nerf_synthetic_loader_dtype.py --project-root "$PROJECT_ROOT" --verify-only
python scripts/synthetic/promote_corrected_curvature.py --validate-only
python scripts/synthetic/prepare_nerf_synthetic_inputs.py --dataset-root "$DATASET_ROOT" --scene-id "$SCENE_ID" --point-count 100000 --seed 0 --check-only

echo "============================================================"
echo "3DGS SYNTHETIC PHASE-III CORRECTED CURVATURE HIGH"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Scene: $SCENE_ID"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT"
echo "Started: $(date)"
echo "============================================================"
nvidia-smi

cd "$GS_ROOT"
/usr/bin/time -v python train.py -s "$DATASET" -m "$OUTPUT" --eval
python render.py -m "$OUTPUT" --skip_train
python metrics.py -m "$OUTPUT"

du -sh "$OUTPUT"
python - <<PY
from pathlib import Path
from plyfile import PlyData
root = Path("$OUTPUT")
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

echo "3DGS corrected curvature-high task complete at $(date)"
