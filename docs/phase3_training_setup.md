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

By default, the promotion script first looks for `/tmp/phase3_curvature_validation_$USER/phase3_curvature_high_candidate_0p3_seed0000` and other matching `/tmp/phase3_curvature_validation_*/` candidates. If the temp candidate is unavailable, which can happen because `/tmp` is node-local, it regenerates the deterministic `0.30` candidate in the requested temp validation root before promotion. Use `--no-regenerate-source` if you want it to fail instead.

The promotion script copies the accepted candidate, updates metadata and manifests, runs the same deterministic input-preparation logic to write `points3d.ply`, and validates:

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

## Phase III-B Multi-Seed Replication

Phase III-B extends the controlled synthetic experiment from the canonical seed `0` to seeds `0,1,2,3,4`, giving five independent realizations per condition. Seed `0` remains untouched. New datasets are generated only for seeds `1,2,3,4` with:

```bash
sbatch jobs/synthetic_phase3b_generate_prepare.sh
```

The Phase III-B config is:

```text
configs/synthetic/phase3_controlled_multiseed.json
```

The controlled factor levels remain:

| Family | Low | Medium | High |
| --- | ---: | ---: | ---: |
| `edge_sharpness` transition width | 0.32 | 0.12 | 0.035 |
| `spatial_frequency` cycles per scene width | 2.0 | 6.0 | 14.0 |
| `curvature` paraboloid amplitude | 0.02 | 0.18 | 0.30 |

The invalid amplitude-0.42 curvature-high seed-0 scene remains excluded. All new curvature-high scenes use corrected scene IDs such as:

```text
phase3_curvature_high_corrected_seed0001
```

### What Varies Across Seeds

Seed-level nuisance variation is deterministic and is held fixed across low/medium/high levels within a sweep family:

- `edge_sharpness`: transition orientation and small transition offset.
- `spatial_frequency`: texture orientation and two sinusoidal phases.
- `curvature`: a low-contrast albedo phase/orientation nuisance shared by curvature levels within the seed.
- deterministic `points3d.ply` initialization seed passed to `prepare_nerf_synthetic_inputs.py`.

### What Remains Fixed Across Seeds

- camera protocol, camera radius, elevation, and field of view
- image resolution `256x256`
- 24 train views and 8 test views
- scene extent and coordinate convention
- background color
- lighting rule
- controlled factor level values
- method-specific training budgets and commands
- V100/sm_70 CUDA extension assumptions

### Validation

`synthetic_phase3b_generate_prepare.sh` runs the generator and deterministic input preparation. The generator writes:

```text
datasets/synthetic/phase3_controlled_pilot/phase3b_stimulus_validation.csv
datasets/synthetic/phase3_controlled_pilot/phase3b_stimulus_validation.json
```

The validation checks finite rendered images, exact train/test image counts, and curvature foreground occupancy within `0.54 +/- 0.03`. It exits nonzero on violation.

### Training Arrays

After generation/preparation succeeds, submit:

```bash
sbatch jobs/synthetic_phase3b_3dgs_array.sh
sbatch jobs/synthetic_phase3b_ges_array.sh
sbatch jobs/synthetic_phase3b_drk_array.sh
```

Each array has 36 tasks: 4 new seeds x 9 conditions. Across the three methods this covers 108 method-scene trainings. The jobs use `--nodes=1` with `--nodelist=gpu[003-005]`, which constrains each task to one known V100 node without requesting all three nodes at once.

Outputs are written under:

```text
outputs/synthetic/phase3_controlled_pilot/{3dgs,ges,drk}/{scene_id}/
```

GES and DRK may append their usual method-specific suffixes; the inventory logic resolves actual model directories as before.

## Phase III-B Evaluation And Statistical Units

After the seed 1-4 trainings finish, rerun inventory and evaluation:

```bash
python scripts/synthetic/inventory_phase3_outputs.py --fail-on-incomplete
python scripts/synthetic/evaluate_phase3_results.py
python scripts/synthetic/analyze_phase3_seed_statistics.py
python scripts/synthetic/analyze_phase3_seed_interactions.py
```

For Phase III-B inference, independent scene seeds are the primary experimental units. Use `phase3_seed_condition_summary.csv/json` for per-condition mean and SD across seeds and `phase3_seed_paired_delta_summary.csv/json` for paired method deltas across matched seeds. Use `phase3_seed_interaction_tests.csv/json` and `phase3_seed_interaction_findings.md` to test whether pairwise method gaps change across low/medium/high factor levels with matched seeds. View-level bootstrap intervals from `evaluate_phase3_results.py` remain secondary diagnostics, not independent replication.
