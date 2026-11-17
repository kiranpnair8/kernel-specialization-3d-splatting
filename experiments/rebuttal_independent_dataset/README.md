# Independent-dataset validation (ICLR 2027 rebuttal)

This experiment evaluates the established 3DGS, GES, and DRK implementations on two benchmarks not used in the paper's original real-scene study: Tanks and Temples **Train** and Deep Blending **Dr Johnson**. It is observational validation, not a kernel-only causal comparison.

## Dataset audit and acquisition

The tracked checkout intentionally contains no benchmark images. The official 3DGS repository provides the preprocessed COLMAP bundle at:

`https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip`

Install/extract it so these canonical directories exist:

- `datasets/tandt/train/{images,sparse/0}`
- `datasets/deepblending/drjohnson/{images,sparse/0}`

Each `sparse/0` must contain COLMAP `cameras.bin`, `images.bin`, and `points3D.bin` (text camera/image models are also accepted by the audit). All methods consume the same undistorted `PINHOLE` or `SIMPLE_PINHOLE` cameras. Under `--eval`, the three audited loaders sort image names lexicographically and select indices `0, 8, 16, ...` as test views. They use the same NeRF++ normalization of training-camera centers. With the default `resolution=-1`, all three loaders preserve images up to width 1600 and apply the same width-1600 automatic reduction above that size.

Run the audit before submitting training:

```bash
cd /home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting
python experiments/rebuttal_independent_dataset/scripts/audit_datasets.py \
  --config experiments/rebuttal_independent_dataset/configs/independent_scenes.json \
  --strict
```

It writes `dataset_manifest.json/csv` under the ignored experiment results directory. The manifest records dataset paths, COLMAP and image resolutions, camera models, ordered train/test counts and SHA-256 split hashes, COLMAP model hashes, method commits/training budgets, and expected output paths. Training refuses to proceed if this audit fails.

## Training and official evaluation

The six array tasks use seed 0 and the project's established native protocols: 3DGS 30k, GES 40k, and DRK 35k. Each task runs the method's official training plus render/metric path. The array concurrency is `%1`, so only one V100 is used at a time. Outputs are isolated under `outputs/rebuttal_independent_dataset/<scene>/<method>/seed_0/`; retries preserve prior attempts.

```bash
mkdir -p jobs/logs/rebuttal_independent
TRAIN_JOB=$(sbatch --parsable experiments/rebuttal_independent_dataset/jobs/rebuttal_independent_training_array.sh)
echo "$TRAIN_JOB"
sbatch --dependency=afterok:${TRAIN_JOB} experiments/rebuttal_independent_dataset/jobs/rebuttal_independent_analysis.sh
```

To submit analysis only after manually confirming all six runs:

```bash
sbatch experiments/rebuttal_independent_dataset/jobs/rebuttal_independent_analysis.sh
```

The training array is pinned to `gpu005`, a single known V100 node, and requests one GPU, 4 CPUs, 64 GB RAM, and 6 hours per task. It applies the tracked seed-control compatibility patch, verifies upstream commits, uses CUDA 12.3 and `sm_70`, records commands/timestamps/elapsed training time in `run_status.json`, and keeps `/usr/bin/time -v` in Slurm logs. GES uses job-local W&B directories to avoid shared-filesystem locks.

## Analysis and outputs

The CPU analysis job performs a strict GT audit before local analysis. Candidate GT images are matched one-to-one to 3DGS GT by RGB-pixel SHA-256 and dimensions, so DRK's indexed filenames are not assumed to match camera identities. Renders inherit the matched candidate GT identity. Any missing, ambiguous, or dimension-mismatched view stops the pipeline.

It reuses the established RGB MSE, edge-anchored 32x32/stride-16 patch grid, stable best/second-best tie rule (`tau=1e-5`), overlapping-patch winner rasterization, and GT descriptor conventions. Structural complexity is the mean scene-wide percentile rank of GT mean gradient magnitude, edge density (gradient > 0.05), and local texture variance, split into empirical tertiles. View-cluster bootstrap intervals resample complete held-out views and preserve patch-count weighting.

Generated, ignored outputs include:

- `global_quality.csv/json` and `tables/global_quality.tex`
- `local_specialization.csv/json`, `local_per_view.csv/json`, and `tables/local_specialization.tex`
- `structural_complexity.csv/json` and `tables/structural_complexity.tex`
- per-scene `patch_errors.npz/json`, `alignment_audit.json`, winner/error maps, and representative crops
- `figures/independent_dataset_qualitative.{pdf,png}`
- `figures/independent_dataset_structural.{pdf,png}`

The qualitative region is selected deterministically as the patch with the largest post-hoc oracle MSE reduction relative to 3DGS in the view with the largest mean reduction; this is documented in `analysis_summary.json`. No result should be interpreted as isolating kernel functional form because optimization, representation, and candidate capacity differ across methods.

## Local validation

```bash
python -m unittest discover -s experiments/rebuttal_independent_dataset/tests -v
python -m compileall -q experiments/rebuttal_independent_dataset analysis/structure_features/descriptors.py
bash -n experiments/rebuttal_independent_dataset/jobs/rebuttal_independent_training_array.sh
bash -n experiments/rebuttal_independent_dataset/jobs/rebuttal_independent_analysis.sh
```
