# Room multi-seed and same-family rebuttal control

Status: **implemented and locally validated; no GPU training has been launched.**

This isolated experiment tests whether the Room local complementarity reported in the paper exceeds variation among independently trained instances of the same representation family. It does not modify or reinterpret the original Room outputs.

## Repository audit and reused protocol

| Item | Canonical source | Reused behavior |
|---|---|---|
| 3DGS | `external/gaussian-splatting`, graphdeco-inria commit `54c035f7834b564019656c3e3fcc3646292f727d` | `train.py ... --eval` for 30,000 iterations; `render.py --skip_train`; `metrics.py` |
| GES | `external/ges-splatting`, ajhamdi commit `c05fc7dbb22e270a5a6f490e7040adac4af02c96` | `train_ges.py ... --eval` for 40,000 iterations; timestamped output; official render/metrics scripts |
| DRK | `external/drk`, VAST-AI-Research commit `5146227784496c813e0df5d5637373f1468d28fd` | 35,000 iterations with `--gs_type DRK --kernel_density dense --cache_sort --is_unbounded`; evaluation through `train.py --metric --load_iteration -1` |
| Room dataset/split | `datasets/mipnerf360/room` | Existing COLMAP initialization (`images -> images_2`, `sparse -> sparse_2`) and 39 held-out views |
| Original jobs | `jobs/{3dgs,ges,drk}_room_baseline.sh` | Environments, CUDA 12.3, V100/sm_70 compiler settings, schedules, rendering, and metric paths |
| Original p32 analysis | `configs/room_3dgs_ges_drk_p32.json`, `scripts/evaluate/local_compare.py` | 32x32 patches, stride 16, MSE tie threshold `1e-5` |
| Image pairing/normalization | `analysis/local_error/image_io.py` | Filename/template pairing; PIL RGB conversion; float32 divided by 255 |
| Patch grid/MSE | `analysis/local_error/patches.py`, `analysis/local_error/metrics.py` | Border-anchored canonical grid and RGB MSE |
| Winner maps | `analysis/winner_maps/visualize.py` | Overlapping-patch color averaging, never sequential rectangle painting |
| Oracle | `analysis/oracle/oracle.py` | Lowest patch MSE and PSNR from mean oracle MSE |
| Primitive controls | `scripts/setup/apply_primitive_budget_control.py` | Audited but deliberately **not used**; this experiment uses native model capacity |

Original natural Room outputs, used only as methodological references, remain:

- `outputs/3dgs/room_baseline`
- `outputs/ges/room_baseline_00_2026-08-24--14-15-39`
- `outputs/drk/room_baseline_DRK`

The new run root is `outputs/rebuttal_multiseed/room/{method}/seed_{0,1,2}/`. Each execution creates an immutable `attempt_NNN/`; `run_status.json` identifies the selected attempt. Failed attempts are retained. A failed/interrupted run requires `--retry`, preventing accidental reuse or overwrite. Upstream training checkpoints do not capture every RNG state, so this launcher intentionally resumes at the **whole-run** level rather than claiming bitwise-correct mid-training continuation.

```text
outputs/rebuttal_multiseed/room/
├── 3dgs/seed_{0,1,2}/attempt_NNN/model/
├── ges/seed_{0,1,2}/attempt_NNN/model_00_<timestamp>/
└── drk/seed_{0,1,2}/attempt_NNN/model_DRK/
```

## Seed control and remaining nondeterminism

`scripts/apply_seed_control.py` is a tracked, idempotent compatibility patch for the gitignored upstream checkouts. It fails if expected source anchors are absent.

- 3DGS and DRK originally hard-code seed 0 in `safe_state` and expose no seed CLI. The patch adds `--seed`, propagates it, and seeds Python, NumPy, Torch CPU, and all CUDA generators.
- GES already exposes and propagates `--seed`; the patch adds explicit CUDA-generator seeding and deterministic cuDNN flags.
- The launcher also exports `PYTHONHASHSEED` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- These controls cover Python camera selection, scene-camera shuffling, Torch random backgrounds/operations, and applicable random initialization/densification decisions.

The implementations use custom CUDA rasterizers, parallel reductions, atomic operations, sorting, and GPU densification kernels. Those can remain nondeterministic despite complete RNG seeding. Therefore seeds are independent stochastic training conditions, not a promise of bitwise reproducibility. Before interpreting results, verify that completed runs record different seed values and inspect whether model hashes/metrics differ; this cannot be established without running training.

## Files and stages

1. `run_multiseed.py`: one seed, native training, official rendering/evaluation, atomic status log.
2. `evaluate_multiseed.py`: requires 9 complete runs; audits 39 filenames, dimensions, and pixel-identical GT; reads global PSNR/SSIM/LPIPS and final PLY counts; computes one shared p32 MSE matrix.
3. `analyze_within_family.py`: same-family winners/ties, seed-pair disagreement, all-reference oracle headroom, per-view values, view-cluster bootstrap, and 3DGS same-family winner map.
4. `analyze_cross_family.py`: all 27 combinations, dependent-combination summaries, per-view values, view-cluster bootstrap, same-reference fair oracle comparison, cross-family map, and diagnostic figures.
5. `generate_rebuttal_table.py`: writes CSV/LaTeX only from completed analysis outputs.

The 27 combinations reuse nine models. They are summarized descriptively and are never treated as 27 independent training experiments. Overlapping patches are likewise not treated as independent replicates; uncertainty resamples complete held-out views. The direct fair comparison reports the within-3DGS and cross-family oracle relative to the same selected 3DGS seed, plus absolute oracle MSE.

## Prerequisites and unresolved items

Before submission, confirm on HPC:

- all three ignored upstream repositories exist at the exact commits listed in the config;
- the Room dataset and canonical 3DGS GT directory exist;
- `kernel_splat`, `ges_splat`, and `drk_splat` environments still import their V100-compiled extensions;
- `cuda/12.3`, V100 nodes, `/usr/bin/time`, Pillow, NumPy, Matplotlib, and `plyfile`/official metric dependencies are available;
- `jobs/logs/rebuttal_multiseed/` exists before `sbatch` (Slurm opens logs before the script runs);
- final training runs genuinely differ where expected; identical results would require investigating whether stochastic code paths are active.

No original checkpoints, paper results, tables, or figures are read for writing or overwritten.

## Dry-run and patch verification

From the project root:

```bash
CONFIG=experiments/rebuttal_multiseed/configs/room_multiseed.json

python experiments/rebuttal_multiseed/scripts/apply_seed_control.py --method 3dgs --root external/gaussian-splatting
python experiments/rebuttal_multiseed/scripts/apply_seed_control.py --method ges  --root external/ges-splatting
python experiments/rebuttal_multiseed/scripts/apply_seed_control.py --method drk  --root external/drk

for method in 3dgs ges drk; do
  for seed in 0 1 2; do
    python experiments/rebuttal_multiseed/scripts/run_multiseed.py --config "$CONFIG" --method "$method" --seed "$seed" --dry-run
  done
done
```

## Training and official rendering/evaluation

Do **not** submit until explicitly approved. The arrays are serial within each method (`%1`) and each task requests one known V100 node. Rendering and official metrics are part of each task after successful training.

```bash
mkdir -p jobs/logs/rebuttal_multiseed
sbatch experiments/rebuttal_multiseed/jobs/rebuttal_multiseed_3dgs_array.sh
sbatch experiments/rebuttal_multiseed/jobs/rebuttal_multiseed_ges_array.sh
sbatch experiments/rebuttal_multiseed/jobs/rebuttal_multiseed_drk_array.sh
```

Equivalent individual launcher commands, after activating the method's environment and installing its seed patch, are:

```bash
python experiments/rebuttal_multiseed/scripts/run_multiseed.py --config "$CONFIG" --method 3dgs --seed 0
python experiments/rebuttal_multiseed/scripts/run_multiseed.py --config "$CONFIG" --method 3dgs --seed 1
python experiments/rebuttal_multiseed/scripts/run_multiseed.py --config "$CONFIG" --method 3dgs --seed 2
python experiments/rebuttal_multiseed/scripts/run_multiseed.py --config "$CONFIG" --method ges --seed 0
python experiments/rebuttal_multiseed/scripts/run_multiseed.py --config "$CONFIG" --method ges --seed 1
python experiments/rebuttal_multiseed/scripts/run_multiseed.py --config "$CONFIG" --method ges --seed 2
python experiments/rebuttal_multiseed/scripts/run_multiseed.py --config "$CONFIG" --method drk --seed 0
python experiments/rebuttal_multiseed/scripts/run_multiseed.py --config "$CONFIG" --method drk --seed 1
python experiments/rebuttal_multiseed/scripts/run_multiseed.py --config "$CONFIG" --method drk --seed 2
```

To retry a failed seed while preserving its prior attempt, add `--retry`.

For traceability, the official post-training commands invoked by the launcher are:

```bash
# 3DGS (MODEL is the attempt's model directory)
python render.py -m "$MODEL" --skip_train
python metrics.py -m "$MODEL"

# GES (ACTUAL_MODEL is the timestamped directory discovered after training)
python render.py -m "$ACTUAL_MODEL" --skip_train
python metrics.py -m "$ACTUAL_MODEL"

# DRK (MODEL_BASE is the unsuffixed -m path; DRK resolves MODEL_BASE_DRK)
python train.py -s "$DATASET" -m "$MODEL_BASE" --eval --gs_type DRK \
  --kernel_density dense --cache_sort --is_unbounded --seed "$SEED" \
  --metric --load_iteration -1
```

## Analysis after all nine runs complete

```bash
mkdir -p jobs/logs/rebuttal_multiseed
sbatch experiments/rebuttal_multiseed/jobs/rebuttal_multiseed_analysis.sh
```

The exact commands inside that CPU job are:

```bash
python experiments/rebuttal_multiseed/scripts/evaluate_multiseed.py --config "$CONFIG"
python experiments/rebuttal_multiseed/scripts/analyze_within_family.py --config "$CONFIG"
python experiments/rebuttal_multiseed/scripts/analyze_cross_family.py --config "$CONFIG"
python experiments/rebuttal_multiseed/scripts/generate_rebuttal_table.py --config "$CONFIG"
```

Generated numerical files are isolated under `experiments/rebuttal_multiseed/results/`, including the required `rebuttal_multiseed_summary.csv`, `rebuttal_multiseed_detailed.csv`, and `rebuttal_multiseed_table.tex`. Diagnostic images are isolated under `experiments/rebuttal_multiseed/figures/`.

## Workload estimate

Reliable tracked Room wall-clock measurements were not found. The proven Room jobs request 4 hours for 3DGS, 4 hours for GES, and 6 hours for DRK. Using those limits as conservative upper bounds gives 42 GPU-hours for nine runs (12 + 12 + 18), plus CPU analysis. With the three method arrays submitted concurrently and each array serialized, the conservative wall-time bound is about 18 hours; queueing and evaluation overhead are additional. These are allocation-based bounds, not measured runtimes.

## Local validation

```bash
python -m unittest discover -s experiments/rebuttal_multiseed/tests -v
python -m compileall -q experiments/rebuttal_multiseed
python -m json.tool experiments/rebuttal_multiseed/configs/room_multiseed.json >/dev/null
bash -n experiments/rebuttal_multiseed/jobs/*.sh
```
