from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_nine_model_oracle import (
    aggregate_statistics,
    bootstrap_view_clusters,
    compute_oracle_arrays,
    confidence_intervals,
    load_patch_coordinates,
    model_indices_and_labels,
    precompute_view_statistics,
    region_rows,
    run_analysis,
)


def fixture() -> tuple[np.ndarray, list[str], np.ndarray]:
    run_ids = [f"{method}_seed_{seed}" for method in ("3dgs", "ges", "drk") for seed in range(3)]
    errors = np.asarray([
        [0.50, 0.40, 0.60, 0.30, 0.35, 0.45, 0.55, 0.50, 0.48],  # GES seed 0
        [0.20, 0.25, 0.30, 0.20, 0.24, 0.28, 0.23, 0.29, 0.31],  # tie with baseline: no improvement
        [0.30, 0.35, 0.40, 0.28, 0.29, 0.27, 0.10, 0.20, 0.25],  # DRK seed 0
        [0.30, 0.32, 0.34, 0.15, 0.18, 0.20, 0.15, 0.19, 0.22],  # GES/DRK tie; stable GES
        [0.18, 0.22, 0.21, 0.19, 0.20, 0.23, 0.18, 0.24, 0.25],  # ties only
    ], dtype=np.float64)
    views = np.asarray(["view_a.png", "view_a.png", "view_b.png", "view_b.png", "view_b.png"])
    return errors, run_ids, views


class NineModelOracleTests(unittest.TestCase):
    def test_oracle_never_exceeds_three_seed_3dgs_oracle(self) -> None:
        errors, run_ids, _ = fixture()
        arrays = compute_oracle_arrays(errors, run_ids, [0, 1, 2])
        self.assertTrue(np.all(arrays["full_mse"] <= arrays["baseline_mse"]))

    def test_family_seed_assignments_and_strict_ties(self) -> None:
        errors, run_ids, _ = fixture()
        arrays = compute_oracle_arrays(errors, run_ids, [0, 1, 2])
        self.assertEqual(arrays["full_family"].tolist(), ["ges", "3dgs", "drk", "ges", "3dgs"])
        self.assertEqual(arrays["full_seed"].tolist(), [0, 0, 0, 0, 0])
        self.assertEqual(arrays["improved"].tolist(), [True, False, True, True, False])
        self.assertEqual(int(arrays["full_tie_count"][3]), 2)
        self.assertFalse(bool(arrays["improved"][1]))
        self.assertFalse(bool(arrays["improved"][4]))

    def test_view_cluster_statistics_preserve_alignment_and_weighting(self) -> None:
        errors, run_ids, views = fixture()
        arrays = compute_oracle_arrays(errors, run_ids, [0, 1, 2])
        unique_views, statistics = precompute_view_statistics(views, arrays)
        self.assertEqual(unique_views.tolist(), ["view_a.png", "view_b.png"])
        self.assertEqual(statistics["patch_count"].tolist(), [2, 3])
        summary = aggregate_statistics(statistics)
        self.assertEqual(summary["patch_count"], 5)
        self.assertEqual(summary["improved_patch_count"], 3)
        self.assertAlmostEqual(summary["improved_patch_fraction"], 3 / 5)
        self.assertAlmostEqual(summary["baseline_3dgs_oracle_mse"], arrays["baseline_mse"].mean())
        self.assertAlmostEqual(summary["nine_model_oracle_mse"], arrays["full_mse"].mean())

    def test_bootstrap_is_reproducible(self) -> None:
        errors, run_ids, views = fixture()
        arrays = compute_oracle_arrays(errors, run_ids, [0, 1, 2])
        _, statistics = precompute_view_statistics(views, arrays)
        first = bootstrap_view_clusters(statistics, replicates=100, seed=2027)
        second = bootstrap_view_clusters(statistics, replicates=100, seed=2027)
        self.assertEqual(set(first), set(second))
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
        intervals = confidence_intervals(first)
        self.assertIn("incremental_gain_fraction", intervals)
        self.assertLessEqual(intervals["incremental_gain_fraction"]["low"], intervals["incremental_gain_fraction"]["high"])

    def test_model_index_validation_includes_all_nine_runs(self) -> None:
        _, run_ids, _ = fixture()
        indices, labels = model_indices_and_labels(run_ids, [0, 1, 2])
        self.assertEqual(set(indices), {"3dgs", "ges", "drk"})
        self.assertTrue(all(len(value) == 3 for value in indices.values()))
        self.assertEqual(len(labels), 9)
        self.assertEqual(len({run_id for _, _, run_id in labels}), 9)

    def test_synthetic_end_to_end_outputs_and_schemas(self) -> None:
        errors, run_ids, views = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            result_root = project_root / "results"
            result_root.mkdir()
            x = np.asarray([0, 16, 0, 16, 32], dtype=np.int64)
            y = np.asarray([0, 0, 16, 16, 16], dtype=np.int64)
            width = np.full(len(errors), 32, dtype=np.int64)
            height = np.full(len(errors), 32, dtype=np.int64)
            np.savez_compressed(
                result_root / "patch_errors.npz",
                errors=errors,
                run_ids=np.asarray(run_ids),
                views=views,
                x=x,
                y=y,
                width=width,
                height=height,
            )
            (result_root / "patch_errors.json").write_text(json.dumps({
                "scene": "synthetic", "patch_size": 32, "stride": 16,
                "run_ids": run_ids, "patch_count": len(errors),
            }), encoding="utf-8")
            config_path = project_root / "config.json"
            config_path.write_text(json.dumps({
                "scene": "synthetic",
                "project_root": str(project_root),
                "result_root": "results",
                "reference_gt": "unused_gt",
                "seeds": [0, 1, 2],
                "patch_size": 32,
                "stride": 16,
                "bootstrap_replicates": 50,
                "bootstrap_seed": 11,
            }), encoding="utf-8")

            document = run_analysis(config_path, skip_visualizations=True)
            required = [
                "nine_model_oracle_summary.json",
                "nine_model_oracle_per_view.csv",
                "nine_model_oracle_regions.csv",
                "nine_model_oracle_table.tex",
            ]
            for filename in required:
                self.assertTrue((result_root / filename).is_file(), filename)
            self.assertEqual(document["summary"]["patch_count"], 5)
            summary_keys = {
                "baseline_3dgs_oracle_mse", "nine_model_oracle_mse",
                "incremental_mse_reduction", "incremental_gain_fraction",
                "improved_patch_count", "improved_patch_fraction",
                "ges_incremental_gain_fraction", "drk_incremental_gain_fraction",
            }
            self.assertTrue(summary_keys.issubset(document["summary"]))
            with (result_root / "nine_model_oracle_per_view.csv").open(newline="", encoding="utf-8") as handle:
                per_view_reader = csv.DictReader(handle)
                per_view = list(per_view_reader)
            with (result_root / "nine_model_oracle_regions.csv").open(newline="", encoding="utf-8") as handle:
                region_reader = csv.DictReader(handle)
                regions = list(region_reader)
            self.assertEqual(len(per_view), 2)
            self.assertEqual(len(regions), 3)
            self.assertTrue({"view", "patch_count", "incremental_gain_fraction", "improved_patch_count"}.issubset(per_view_reader.fieldnames or []))
            self.assertTrue({"view", "x", "y", "winning_family", "winning_seed", "baseline_3dgs_mse", "nine_model_oracle_mse", "improvement_mse"}.issubset(region_reader.fieldnames or []))
            self.assertEqual(regions[0]["winning_family"], "drk")
            coordinates = load_patch_coordinates(result_root / "patch_errors.npz", len(errors))
            arrays = compute_oracle_arrays(errors, run_ids, [0, 1, 2])
            self.assertEqual(len(region_rows(views, coordinates, arrays)), 3)


if __name__ == "__main__":
    unittest.main()
