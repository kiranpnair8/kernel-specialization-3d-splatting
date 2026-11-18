# TPAMI matched-size multi-seed control

This experiment asks whether two-model 3DGS+GES oracle complementarity on Deep Blending / Dr Johnson exceeds the variability of equally sized 3DGS+3DGS and GES+GES pools. The analysis does **not** assume a cross-family advantage. Every comparison contains exactly two models, and all available seed pairs are reported.

## Design

- Dataset and split: the exact Experiment-2 Dr Johnson COLMAP input and sorted every-eighth holdout (`230` train, `33` test).
- Existing models: 3DGS seed 0 and GES seed 0 from `outputs/rebuttal_independent_dataset/`.
- New models: seeds 1 and 2 for 3DGS (30k) and GES (40k), isolated under `outputs/tpami_multiseed_control/`.
- Local analysis: RGB MSE, 32x32 patches, stride 16, tie threshold `1e-5`.
- Sensitivity: patch sizes 32/64/128 with half-patch strides and tie thresholds `0`, `1e-5`, `5e-5`.
- Pair classes: 3 within-3DGS, 3 within-GES, and all 9 cross-family pairs.
- Primary pair reference: the lower scene-level MSE of the pair's two individual members. Bootstrap replicates reselect the better member after aggregating sampled views.
- Uncertainty: 2,000 held-out-view clustered bootstrap replicates; patch-count weighting is preserved by MSE sums and counts.

The 15 seed-pair combinations are dependent because individual models are reused. Combination-level standardized differences are descriptive; held-out views are the bootstrap sampling clusters.

## Reproducible seed control

The jobs reuse Experiment 2's official training, rendering, and metric paths. Before training, the tracked idempotent seed patch is installed and verified. It propagates the configured seed to Python, NumPy, PyTorch CPU/CUDA RNGs, disables cuDNN benchmarking, and enables deterministic cuDNN behavior. Each training command explicitly contains `--seed`; the evaluator rejects missing or mismatched seed records. Run status records preserve the command, upstream commit, split hash, configuration, elapsed time, and actual timestamped model path.

The upstream save schedule is unchanged from Experiment 2: official intermediate/final model snapshots and the final evaluated PLY are retained. No extra optimizer-checkpoint schedule is introduced because that would depart from the established training protocol.

## Submission

```bash
cd /home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting
git pull origin main
mkdir -p jobs/logs/tpami_multiseed

TRAIN_JOB=$(sbatch --parsable experiments/tpami_multiseed_control/jobs/tpami_multiseed_training_array.sh)
echo "training array: $TRAIN_JOB"

sbatch --dependency=afterok:${TRAIN_JOB} \
  experiments/tpami_multiseed_control/jobs/tpami_multiseed_analysis.sh
```

The array is `0-3%1`, so only one V100 is used at a time:

| Task | Method | Seed |
|---:|---|---:|
| 0 | 3DGS | 1 |
| 1 | 3DGS | 2 |
| 2 | GES | 1 |
| 3 | GES | 2 |

No job is submitted by the repository scripts themselves.

## Analysis-only commands

After all four new runs complete:

```bash
cd /home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting
CONFIG=experiments/tpami_multiseed_control/configs/tpami_multiseed_control.json

python experiments/tpami_multiseed_control/scripts/evaluate_six_models.py --config "$CONFIG"
python experiments/tpami_multiseed_control/scripts/analyze_matched_oracles.py --config "$CONFIG"
```

The evaluator first aligns every run by exact GT RGB pixels and dimensions. It then requires its `3dgs_s0` and `ges_s0` errors, held-out ordering, image dimensions, and p32 patch coordinates to reproduce Experiment 2's `patch_errors.npz` within `1e-15`.

## Recovery

Completed runs are skipped. Failed/interrupted attempts are never overwritten. Resubmit only the failed array task with retry enabled:

```bash
# Example: retry task 2 (GES seed 1)
sbatch --array=2 --export=ALL,TPAMI_RETRY=1 \
  experiments/tpami_multiseed_control/jobs/tpami_multiseed_training_array.sh
```

To inspect status without changing anything:

```bash
find outputs/tpami_multiseed_control/deepblending_drjohnson -name run_status.json -print -exec python -m json.tool {} \;
```

## Outputs

Generated files are ignored under `experiments/tpami_multiseed_control/results/`:

- `experiment_manifest.json`, `model_inventory.csv/json`, `alignment_audit.json`
- `patch_errors.npz/json` with columns `3dgs_s0,3dgs_s1,3dgs_s2,ges_s0,ges_s1,ges_s2`
- `pairwise_oracle_summary.csv/json` and `pairwise_oracle_per_view.csv/json`
- `group_summary`, `bootstrap_summary`, and `group_contrasts` CSV/JSON
- pairwise error correlations and structural-complexity summaries
- complete 3x3 patch-size/tie-threshold sensitivity summaries
- publication figures in PDF and PNG
- `matched_size_oracle_table.tex` and `matched_size_contrasts_table.tex`

## Local validation

```bash
python -m compileall -q experiments/tpami_multiseed_control
python -m unittest discover -s experiments/tpami_multiseed_control/tests -v
bash -n experiments/tpami_multiseed_control/jobs/tpami_multiseed_training_array.sh
bash -n experiments/tpami_multiseed_control/jobs/tpami_multiseed_analysis.sh
```
