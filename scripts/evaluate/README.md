# Evaluation Scripts

Run the Garden 3DGS-vs-GES local comparison from the repository root:

```bash
python scripts/evaluate/local_compare.py --config configs/garden_3dgs_vs_ges.json
```

Use `--inspect-only` first on the HPC checkout to confirm that GT images and
rendered test images have exactly matching filenames:

```bash
python scripts/evaluate/local_compare.py --config configs/garden_3dgs_vs_ges.json --inspect-only
```

The default config expects:

- `outputs/3dgs/garden_baseline/test/ours_30000/gt`
- `outputs/3dgs/garden_baseline/test/ours_30000/renders`
- `outputs/ges/garden_baseline_00_2026-08-17--17-16-41/test/ours_40000/renders`

These paths are gitignored and should be adjusted in the config if the actual
HPC render folder names differ.

Run the Stage-0 pipeline-validation sensitivity grid:

```bash
python scripts/evaluate/sensitivity.py --config configs/garden_3dgs_vs_ges.json
```

By default this evaluates patch sizes `32,64,128` and tie thresholds
`0,1e-5,5e-5`, writing per-run outputs plus aggregate
`sensitivity_summary.csv` and `sensitivity_summary.json` under the configured
results directory. Unless a config or CLI option specifies otherwise,
the sensitivity runner preserves the base config stride for backward-compatible
Stage-0 runs.

Before running the three-family Garden comparison, audit DRK GT alignment:

```bash
python scripts/evaluate/audit_alignment.py \
  --reference-gt-dir outputs/3dgs/garden_baseline/test/ours_30000/gt \
  --candidate-gt-dir outputs/drk/garden_baseline_DRK/metric/test \
  --candidate-name-template 'gt_{stem}.png'
```

If the audit reports `exact_or_unambiguous: true`, run the three-family
comparison:

```bash
python scripts/evaluate/local_compare.py --config configs/garden_3dgs_ges_drk.json
```

Run the Stage-1 three-family sensitivity grid:

```bash
python scripts/evaluate/sensitivity.py --config configs/garden_3dgs_ges_drk.json
```

The three-family config sets `sensitivity_stride_policy` to `half_patch`, so
the predefined grid uses 50% overlap: patch size `32` uses stride `16`, patch
size `64` uses stride `32`, and patch size `128` uses stride `64`. Sensitivity
runs are metrics-only: they skip visualization maps, per-run patch CSVs, and
predictor fitting, and write only aggregate `sensitivity_summary.csv` and
`sensitivity_summary.json`.

Run the Stage-1 three-family structure characterization after generating a
full patch-level `patches.csv` for patch size `32`, stride `16`, and tie
threshold `1e-5`:

```bash
python scripts/evaluate/local_compare.py \
  --config configs/garden_3dgs_ges_drk_p32.json

python scripts/evaluate/characterize_specialization.py \
  --config configs/garden_3dgs_ges_drk_p32.json
```

The p32 config enables lightweight patch-export mode by setting
`generate_maps: false` and `run_predictors: false`. Equivalently, any
comparison run can disable those expensive extras from the command line:

```bash
python scripts/evaluate/local_compare.py \
  --config configs/garden_3dgs_ges_drk_p32.json \
  --no-maps \
  --no-predictors
```

This still writes `patches.csv`, `per_image_metrics.csv`, and `summary.json`,
but skips PNG map generation and predictor fitting.

This is a post-hoc observational analysis of the existing patch rows. It writes
descriptor summaries, pairwise nonparametric tests with FDR correction,
winner-probability quantile curves, a descriptor-correlation matrix, and
decisive patch exemplars under
`results/garden/3dgs_vs_ges_vs_drk_p32/characterization/`. The characterization
script exits non-zero if the supplied `patches.csv` geometry does not match the
requested patch size and stride.
