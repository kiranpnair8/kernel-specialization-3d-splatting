#!/bin/bash
#SBATCH --job-name=synthetic-3dgs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-8%3
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/synthetic_3dgs_%A_%a.out
#SBATCH --error=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/synthetic_3dgs_%A_%a.err

set -eo pipefail

PROJECT_ROOT=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting
GS_ROOT=$PROJECT_ROOT/external/gaussian-splatting
DATASET_ROOT=$PROJECT_ROOT/datasets/synthetic/phase3_controlled_pilot
OUTPUT_ROOT=$PROJECT_ROOT/outputs/synthetic/phase3_controlled_pilot/3dgs
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

mkdir -p "$OUTPUT" "$PROJECT_ROOT/jobs/logs"

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
python scripts/synthetic/prepare_nerf_synthetic_inputs.py --dataset-root "$DATASET_ROOT" --scene-id "$SCENE_ID" --point-count 100000 --seed 0 --check-only

echo "============================================================"
echo "3DGS SYNTHETIC PHASE-III PILOT"
echo "Job ID: ${SLURM_JOB_ID:-unknown} / task ${SLURM_ARRAY_TASK_ID:-unknown}"
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

echo "3DGS synthetic task complete: $SCENE_ID at $(date)"
