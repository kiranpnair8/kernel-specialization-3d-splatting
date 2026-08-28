#!/bin/bash
#SBATCH --job-name=synthetic-phase3-loader-check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/phase3/setup/synthetic_phase3_loader_check_%j.out
#SBATCH --error=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/phase3/setup/synthetic_phase3_loader_check_%j.err

set -eo pipefail

PROJECT_ROOT=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting
DATASET=$PROJECT_ROOT/datasets/synthetic/phase3_controlled_pilot/phase3_edge_sharpness_low_seed0000
SCENE_ID=phase3_edge_sharpness_low_seed0000

mkdir -p "$PROJECT_ROOT/jobs/logs/phase3/setup"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.3
source /home/usd.local/kiran.prasannannair/miniforge3/etc/profile.d/conda.sh

check_loader() {
  local env_path=$1
  local method_root=$2
  local label=$3
  set +u
  conda activate "$env_path"
  set -u
  cd "$method_root"
  echo "============================================================"
  echo "LOADER CHECK: $label"
  echo "Dataset: $DATASET"
  echo "============================================================"
  python - "$DATASET" <<'PY'
import inspect
import sys
from scene.dataset_readers import sceneLoadTypeCallbacks

path = sys.argv[1]
reader = sceneLoadTypeCallbacks["Blender"]
params = inspect.signature(reader).parameters
if "depths" in params:
    scene = reader(path, False, "", True)
else:
    scene = reader(path, False, True)
assert len(scene.train_cameras) == 24, len(scene.train_cameras)
assert len(scene.test_cameras) == 8, len(scene.test_cameras)
print("train cameras:", len(scene.train_cameras))
print("test cameras:", len(scene.test_cameras))
print("ply path:", scene.ply_path)
print("first train image:", scene.train_cameras[0].image_path)
print("first test image:", scene.test_cameras[0].image_path)
PY
  cd "$PROJECT_ROOT"
}

set +u
conda activate /home/rizk_lab/shared/kiran/envs/kernel_splat
set -u
python -m compileall scripts/synthetic/prepare_nerf_synthetic_inputs.py scripts/synthetic/patch_nerf_synthetic_loader_dtype.py scripts/synthetic/inventory_phase3_outputs.py
bash -n jobs/synthetic_phase3_loader_check.sh jobs/synthetic_phase3_3dgs_array.sh jobs/synthetic_phase3_ges_array.sh jobs/synthetic_phase3_drk_array.sh
python scripts/synthetic/patch_nerf_synthetic_loader_dtype.py --project-root "$PROJECT_ROOT"
python scripts/synthetic/patch_nerf_synthetic_loader_dtype.py --project-root "$PROJECT_ROOT" --verify-only
python scripts/synthetic/prepare_nerf_synthetic_inputs.py \
  --dataset-root "$PROJECT_ROOT/datasets/synthetic/phase3_controlled_pilot" \
  --scene-id "$SCENE_ID" \
  --point-count 100000 \
  --seed 0

check_loader /home/rizk_lab/shared/kiran/envs/kernel_splat "$PROJECT_ROOT/external/gaussian-splatting" 3DGS
check_loader /home/rizk_lab/shared/kiran/envs/ges_splat "$PROJECT_ROOT/external/ges-splatting" GES
check_loader /home/rizk_lab/shared/kiran/envs/drk_splat "$PROJECT_ROOT/external/drk" DRK

echo "Synthetic loader compatibility check complete."
