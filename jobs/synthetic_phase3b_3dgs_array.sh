#!/bin/bash
#SBATCH --job-name=phase3b-3dgs
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=gpu[003-005]
#SBATCH --gres=gpu:1
#SBATCH --array=0-35%3
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/phase3/3dgs/phase3b_3dgs_%A_%a.out
#SBATCH --error=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/phase3/3dgs/phase3b_3dgs_%A_%a.err

set -eo pipefail

PROJECT_ROOT=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting
GS_ROOT=$PROJECT_ROOT/external/gaussian-splatting
DATASET_ROOT=$PROJECT_ROOT/datasets/synthetic/phase3_controlled_pilot
OUTPUT_ROOT=$PROJECT_ROOT/outputs/synthetic/phase3_controlled_pilot/3dgs
FAMILIES=(edge_sharpness spatial_frequency curvature)
LEVELS=(low medium high)
SEEDS=(1 2 3 4)

SEED_INDEX=$((SLURM_ARRAY_TASK_ID / 9))
WITHIN_SEED=$((SLURM_ARRAY_TASK_ID % 9))
FAMILY_INDEX=$((WITHIN_SEED / 3))
LEVEL_INDEX=$((WITHIN_SEED % 3))
SEED=${SEEDS[$SEED_INDEX]}
FAMILY=${FAMILIES[$FAMILY_INDEX]}
LEVEL=${LEVELS[$LEVEL_INDEX]}
SEED_PADDED=$(printf "%04d" "$SEED")
SCENE_ID=phase3_${FAMILY}_${LEVEL}_seed${SEED_PADDED}
if [[ "$FAMILY" == "curvature" && "$LEVEL" == "high" ]]; then
  SCENE_ID=phase3_curvature_high_corrected_seed${SEED_PADDED}
fi
DATASET=$DATASET_ROOT/$SCENE_ID
OUTPUT=$OUTPUT_ROOT/$SCENE_ID

mkdir -p "$PROJECT_ROOT/jobs/logs/phase3/3dgs"

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
python scripts/synthetic/prepare_nerf_synthetic_inputs.py --dataset-root "$DATASET_ROOT" --scene-id "$SCENE_ID" --point-count 100000 --seed "$SEED" --check-only

echo "============================================================"
echo "3DGS SYNTHETIC PHASE III-B MULTI-SEED"
echo "Job ID: ${SLURM_JOB_ID:-unknown} / task ${SLURM_ARRAY_TASK_ID:-unknown}"
echo "Node: $(hostname)"
echo "Scene: $SCENE_ID"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT"
echo "Started: $(date)"
echo "============================================================"
nvidia-smi

if [[ ! -d "$DATASET" ]]; then
  echo "Missing dataset: $DATASET" >&2
  exit 1
fi
if [[ -d "$OUTPUT" && -n "$(find "$OUTPUT" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite non-empty output: $OUTPUT" >&2
  exit 1
fi
mkdir -p "$OUTPUT"

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

echo "3DGS Phase III-B task complete: $SCENE_ID at $(date)"
