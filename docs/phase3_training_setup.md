# Phase III Synthetic Training Setup

## Dataset Format

The Phase-III controlled pilot uses NeRF-synthetic style scene folders under:

```text
datasets/synthetic/phase3_controlled_pilot/
```

Each scene contains:

- `train/*.png`
- `test/*.png`
- `transforms_train.json`
- `transforms_test.json`
- `metadata.json`
- deterministic `points3d.ply` after input preparation

The generated `transforms_train.json` and `transforms_test.json` preserve the exact 24 train / 8 test split. Do not re-split these scenes with LLFF holdout rules.

## Loader Compatibility

3DGS, GES, and DRK directly consume the NeRF-synthetic transform files. The loader check job verified 24 train and 8 test cameras for all three methods on `phase3_edge_sharpness_low_seed0000`.

3DGS and GES require the tracked compatibility patch script:

```bash
python scripts/synthetic/patch_nerf_synthetic_loader_dtype.py --project-root "$PROJECT_ROOT"
```

The patch changes the NeRF-synthetic Pillow image cast in gitignored upstream checkouts from signed `np.byte` to `np.uint8`. DRK already uses a compatible dtype on this path.

## Corrected High-Curvature Scene

The original `phase3_curvature_high_seed0000` stimulus is preserved but invalid for controlled curvature analysis because its foreground occupancy collapsed. The accepted corrected high-curvature candidate uses `paraboloid_amplitude=0.30` and is promoted as a distinct scene:

```text
datasets/synthetic/phase3_controlled_pilot/phase3_curvature_high_corrected_seed0000
```

Promotion command:

```bash
python scripts/synthetic/promote_corrected_curvature.py
```

The promotion script copies `/tmp/phase3_curvature_validation_$USER/phase3_curvature_high_candidate_0p3_seed0000`, updates metadata and manifests, runs the same deterministic input-preparation logic to write `points3d.ply`, and validates:

- 24 train views
- 8 test views
- foreground fraction within `0.54 +/- 0.03`
- finite rendered image values
- deterministic `points3d.ply` exists

## Training Jobs

The original 9-scene pilot arrays are:

```bash
sbatch jobs/synthetic_phase3_3dgs_array.sh
sbatch jobs/synthetic_phase3_ges_array.sh
sbatch jobs/synthetic_phase3_drk_array.sh
```

Train only the corrected high-curvature scene with:

```bash
sbatch jobs/synthetic_phase3_3dgs_curvature_high_corrected.sh
sbatch jobs/synthetic_phase3_ges_curvature_high_corrected.sh
sbatch jobs/synthetic_phase3_drk_curvature_high_corrected.sh
```

The corrected-scene jobs use the same method protocols, budgets, environments, CUDA/compiler settings, and V100-only node restriction as the successful Phase-III arrays. They write new outputs under:

```text
outputs/synthetic/phase3_controlled_pilot/3dgs/phase3_curvature_high_corrected_seed0000
outputs/synthetic/phase3_controlled_pilot/ges/phase3_curvature_high_corrected_seed0000
outputs/synthetic/phase3_controlled_pilot/drk/phase3_curvature_high_corrected_seed0000
```

Each job refuses to start if its requested corrected output directory already exists and is non-empty.
