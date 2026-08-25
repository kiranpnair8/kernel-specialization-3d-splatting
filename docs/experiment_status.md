# Paper-1 Experiment Status Ledger

Last updated: 2026-08-25

## Project Objective

Paper-1 studies whether different splatting/kernel families exhibit different local representation behavior. Current results are observational characterization only. They do not establish causal kernel specialization because training recipes, representation capacities, primitive counts, and implementations differ across systems.

## Experimental Pipeline

1. Dataset setup: create scene directories, downsampled images, COLMAP links/files, and held-out test lists.
2. Train: run method-specific baseline training through reproducible Slurm jobs under `jobs/`.
3. Render/evaluate: use each method's saved-model evaluation path, not intermediate training logs.
4. Global metrics: record PSNR, SSIM, LPIPS, final iteration, primitive count, output location.
5. GT alignment audit: verify held-out GT/test-view identity before local comparisons.
6. Local patch comparison: compute per-patch errors, winner/tie labels, oracle summaries, and structural descriptors.
7. Sensitivity analysis: vary patch size and tie threshold; use explicit stride policy.
8. Descriptor-based specialization characterization: summarize descriptor associations with winning labels.
9. Cross-scene replication: compare completed scene-level patterns only after per-scene alignment, p32 comparison, characterization, and sensitivity are complete.

## Checklist

| Scene | 3DGS | GES | DRK | Alignment | Local p32 | Sensitivity | Characterization |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Garden | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Bicycle | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Room | ✓ | ✓ | ✓ | pending | pending | pending | pending |

## Garden Status

Overall chain: `baselines -> alignment -> local comparison -> sensitivity -> characterization` = COMPLETE.

### Baselines

| Method | Iteration | PSNR | SSIM | LPIPS | Primitive count | Canonical output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 3DGS | 30000 | 27.4781 | 0.8680687 | 0.1060733 | 4,146,866 | `outputs/3dgs/garden_baseline` |
| GES | 40000 | 27.1016693 | 0.8524919 | 0.1347301 | 2,709,992 | `outputs/ges/garden_baseline_00_2026-08-17--17-16-41` |
| DRK | 35000 | 25.43460 | 0.91927 | 0.18802 | 614,345 | `outputs/drk/garden_baseline_DRK` |

Sources: `docs/baseline_3dgs_garden.md`, `docs/baseline_ges_garden.md`, `docs/baseline_drk_garden.md`.

### Canonical Paths

- Three-family primary config: `configs/garden_3dgs_ges_drk.json`
- Three-family p32 config: `configs/garden_3dgs_ges_drk_p32.json`
- Stage-0 3DGS/GES validation config: `configs/garden_3dgs_vs_ges.json`
- 3DGS GT/renders: `outputs/3dgs/garden_baseline/test/ours_30000/gt`, `outputs/3dgs/garden_baseline/test/ours_30000/renders`
- GES renders: `outputs/ges/garden_baseline_00_2026-08-17--17-16-41/test/ours_40000/renders`
- DRK held-out outputs: `outputs/drk/garden_baseline_DRK/metric/test/`
- Three-family primary results: `results/garden/3dgs_vs_ges_vs_drk/`
- p32 analysis results: `results/garden/3dgs_vs_ges_vs_drk_p32/`
- p32 characterization results: `results/garden/3dgs_vs_ges_vs_drk_p32/characterization/`
- Sensitivity results: `results/garden/3dgs_vs_ges_vs_drk/sensitivity/`

### Completed Analyses

- Training/evaluation: COMPLETE for 3DGS, GES, DRK.
- GT alignment audit: COMPLETE before using DRK in local comparisons.
- Three-family local comparison: COMPLETE.
- p32 characterization: COMPLETE with `patch=32`, `stride=16`, `tie_threshold_mse=1e-5`.
- Sensitivity analysis: COMPLETE for patch sizes `32,64,128`, half-patch stride, tie thresholds `0,1e-5,5e-5`.

### Conservative Garden Observations

- Global metrics are not aligned across families: 3DGS has the best Garden PSNR/LPIPS among the recorded baselines, while DRK has the best Garden SSIM and far fewer primitives than 3DGS.
- The completed local comparison, sensitivity, and p32 characterization support studying spatial error complementarity across independently trained systems.
- These are observational results. They are not controlled causal evidence that one kernel family is locally specialized.

## Bicycle Status

Overall chain: `baselines -> alignment -> local comparison -> sensitivity -> characterization` = COMPLETE.

Next chain: `Garden/Bicycle cross-scene comparison`.

### Baselines

| Method | Iteration | PSNR | SSIM | LPIPS | Primitive count | Canonical output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 3DGS | 30000 | 25.2347298 | 0.7669119 | 0.2075760 | 4,925,950 | `outputs/3dgs/bicycle_baseline` |
| GES | 40000 | 24.8159904 | 0.7323636 | 0.2628537 | 2,773,438 | `outputs/ges/bicycle_baseline_00_2026-08-20--16-01-04` |
| DRK | 35000 | 23.36252 | 0.79918 | 0.36445 | 1,578,713 | `outputs/drk/bicycle_baseline_DRK` |

Bicycle global observation: 3DGS has the best PSNR and LPIPS; DRK has the best SSIM and uses substantially fewer primitives than 3DGS. Treat this only as a global representation observation, not evidence of local specialization.

### Canonical Paths

- Dataset: `datasets/mipnerf360/bicycle`
- 3DGS output: `outputs/3dgs/bicycle_baseline`
- GES output: `outputs/ges/bicycle_baseline_00_2026-08-20--16-01-04`
- DRK output: `outputs/drk/bicycle_baseline_DRK`
- p32 analysis config: `configs/bicycle_3dgs_ges_drk_p32.json`
- p32 analysis result root: `results/bicycle/3dgs_vs_ges_vs_drk_p32/`
- p32 characterization results: `results/bicycle/3dgs_vs_ges_vs_drk_p32/characterization/`
- Sensitivity results: `results/bicycle/3dgs_vs_ges_vs_drk_p32/sensitivity/`
- Baseline jobs: `jobs/3dgs_bicycle_baseline.sh`, `jobs/ges_bicycle_baseline.sh`, `jobs/drk_bicycle_baseline.sh`
- Analysis jobs: `jobs/bicycle_p32_characterization.sh`, `jobs/bicycle_sensitivity.sh`

### Completed Analyses

- GT alignment audit: COMPLETE for the 25 held-out Bicycle test views before local comparison.
- p32 local comparison: COMPLETE with `patch=32`, `stride=16`, `tie_threshold_mse=1e-5`, `generate_maps=false`, `run_predictors=false`.
- Descriptor-based characterization: COMPLETE from `results/bicycle/3dgs_vs_ges_vs_drk_p32/patches.csv`.
- Sensitivity analysis: COMPLETE for patch sizes `32,64,128`, half-patch stride, tie thresholds `0,1e-5,5e-5`.

## Room Status

Overall chain: `baselines` = COMPLETE.

Next chain: `alignment audit -> p32 local comparison -> characterization -> sensitivity`.

### Baselines

Room baselines are complete for 3DGS, GES, and DRK. Canonical metrics should be recorded from saved-model render/evaluation outputs before manuscript use.

### Canonical Paths

- Dataset: `datasets/mipnerf360/room`
- 3DGS output: `outputs/3dgs/room_baseline`
- GES output: `outputs/ges/room_baseline_00_2026-08-24--14-15-39`
- DRK output: `outputs/drk/room_baseline_DRK`
- p32 analysis config: `configs/room_3dgs_ges_drk_p32.json`
- p32 analysis result root: `results/room/3dgs_vs_ges_vs_drk_p32/`
- Baseline jobs: `jobs/3dgs_room_baseline.sh`, `jobs/ges_room_baseline.sh`, `jobs/drk_room_baseline.sh`
- Analysis jobs: `jobs/room_p32_characterization.sh`, `jobs/room_sensitivity.sh`

### Pending Analyses

- GT alignment audit: pending; configured to compare 3DGS GT against DRK GT for 39 held-out Room test views.
- p32 local comparison: pending; configured with `patch=32`, `stride=16`, `tie_threshold_mse=1e-5`, `generate_maps=false`, `run_predictors=false`.
- Descriptor-based characterization: pending; will consume `results/room/3dgs_vs_ges_vs_drk_p32/patches.csv`.
- Sensitivity analysis: pending; configured for patch sizes `32,64,128`, half-patch stride, tie thresholds `0,1e-5,5e-5`.

## Cross-Scene Analysis Status

- Reusable aggregation script: `scripts/evaluate/cross_scene_specialization.py`.
- Reproducible Slurm job: `jobs/cross_scene_specialization.sh`.
- Intended output root: `results/cross_scene/garden_vs_bicycle_p32/`.
- Status: configured and ready to run; cross-scene outputs should be inspected before recording any conclusions.
- Required per-scene inputs: each scene's `summary.json`, `characterization/descriptor_summary.csv`, `characterization/pairwise_effects.csv`, and `characterization/winner_probability_by_descriptor.csv`.

## Current Status / Next Experiment

Garden: `baselines -> alignment -> local comparison -> sensitivity -> characterization` = COMPLETE.

Bicycle: `baselines -> alignment -> local comparison -> sensitivity -> characterization` = COMPLETE.

Room: `baselines` = COMPLETE; local-specialization setup is prepared but not yet run.

Next: `Room alignment audit -> Room p32 local comparison -> Room characterization -> Room sensitivity`.

## Experimental Safeguards / Interpretation Rules

- Identical GT/test-view alignment must be verified before any local comparison.
- Local comparison jobs must stop if scene alignment is not exact/unambiguous for all held-out test views.
- Global metric superiority does not imply local superiority.
- 3DGS vs GES alone cannot establish cross-family specialization because Gaussian is nested within GES.
- Three-family comparisons remain observational because methods differ in training recipes, capacities/primitive counts, and implementations.
- Cross-scene agreement strengthens robustness of observed patterns but still does not prove causal kernel specialization.
- Patch conclusions must survive patch-size and tie-threshold sensitivity analysis.
- Descriptor associations are not causal evidence.
- Canonical metrics should come from saved-model render/evaluation outputs rather than intermediate training logs.
- Long-running training and analyses should have reproducible Slurm job scripts under `jobs/` rather than being intended primarily for interactive execution.
