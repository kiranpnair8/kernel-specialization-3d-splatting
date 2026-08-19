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
results directory.

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
