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
