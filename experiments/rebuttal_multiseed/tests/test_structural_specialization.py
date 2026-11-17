from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_nine_model_oracle import compute_oracle_arrays
from analyze_structural_specialization import (
    bootstrap_categories,
    category_metrics,
    complexity_categories,
    image_patch_descriptors,
    matched_size_oracles,
    percentile_intervals,
    precompute_category_view_statistics,
    run_analysis,
)


def run_ids() -> list[str]:
    return [f"{method}_seed_{seed}" for method in ("3dgs", "ges", "drk") for seed in range(3)]


class StructuralSpecializationTests(unittest.TestCase):
    def test_integral_descriptors_match_direct_patch_statistics(self) -> None:
        horizontal = np.linspace(0.0, 1.0, 8, dtype=np.float64)
        gray = np.tile(horizontal, (8, 1))
        rgb = np.repeat(gray[..., None], 3, axis=2)
        x = np.asarray([0, 2]); y = np.asarray([0, 1])
        width = np.asarray([4, 5]); height = np.asarray([4, 6])
        descriptors = image_patch_descriptors(rgb, x, y, width, height, edge_threshold=0.05)
        for index in range(2):
            patch = gray[y[index]:y[index] + height[index], x[index]:x[index] + width[index]]
            self.assertAlmostEqual(descriptors["local_texture_variance"][index], float(np.var(patch)))
            self.assertAlmostEqual(descriptors["mean_gradient_magnitude"][index], 1.0 / 7.0)
            self.assertAlmostEqual(descriptors["edge_density"][index], 1.0)

    def test_complexity_tertiles_are_ordered_and_complete(self) -> None:
        descriptors = {
            "mean_gradient_magnitude": np.arange(12, dtype=np.float64),
            "edge_density": np.arange(12, dtype=np.float64) ** 2,
            "local_texture_variance": np.arange(12, dtype=np.float64) ** 3,
        }
        score, categories, ranks = complexity_categories(descriptors)
        self.assertEqual(set(categories.tolist()), {0, 1, 2})
        self.assertTrue(np.all(np.diff(score) >= 0.0))
        self.assertTrue(all(values.shape == (12,) for values in ranks.values()))
        self.assertLess(score[categories == 0].max(), score[categories == 2].min())

    def test_matched_size_controls_include_27_combinations(self) -> None:
        rng = np.random.default_rng(3)
        errors = rng.uniform(1e-4, 1e-2, size=(15, 9))
        cross, ges, drk, combinations = matched_size_oracles(errors, run_ids(), [0, 1, 2])
        self.assertEqual(cross.shape, (15, 27))
        self.assertEqual(ges.shape, (15,))
        self.assertEqual(drk.shape, (15,))
        self.assertEqual(len(combinations), 27)
        self.assertEqual(len(set(combinations)), 27)

    def test_view_cluster_bootstrap_is_reproducible_with_missing_view_categories(self) -> None:
        rng = np.random.default_rng(7)
        errors = rng.uniform(1e-4, 1e-2, size=(12, 9))
        views = np.asarray(["a.png"] * 4 + ["b.png"] * 4 + ["c.png"] * 4)
        categories = np.asarray([0, 0, 1, 1, 0, 2, 2, 2, 1, 1, 2, 2])
        oracle = compute_oracle_arrays(errors, run_ids(), [0, 1, 2])
        cross, ges, drk, _ = matched_size_oracles(errors, run_ids(), [0, 1, 2])
        _, statistics = precompute_category_view_statistics(views, categories, oracle, cross, ges, drk)
        first = bootstrap_categories(statistics, replicates=100, seed=29)
        second = bootstrap_categories(statistics, replicates=100, seed=29)
        for category in first:
            for metric in first[category]:
                np.testing.assert_array_equal(first[category][metric], second[category][metric])
        intervals = percentile_intervals(first)
        self.assertGreater(intervals["medium"]["nine_model_incremental_gain_fraction"]["effective_replicates"], 0)

    def test_category_metrics_preserve_patch_weighting(self) -> None:
        errors = np.asarray([
            [0.4, 0.5, 0.6, 0.3, 0.5, 0.6, 0.35, 0.5, 0.6],
            [0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.1, 0.3, 0.4],
            [0.8, 0.9, 1.0, 0.7, 0.9, 1.0, 0.6, 0.9, 1.0],
            [0.5, 0.6, 0.7, 0.4, 0.6, 0.7, 0.3, 0.6, 0.7],
            [0.7, 0.8, 0.9, 0.6, 0.8, 0.9, 0.5, 0.8, 0.9],
        ])
        views = np.asarray(["a.png", "a.png", "b.png", "a.png", "b.png"])
        categories = np.asarray([0, 0, 0, 1, 2])
        oracle = compute_oracle_arrays(errors, run_ids(), [0, 1, 2])
        cross, ges, drk, _ = matched_size_oracles(errors, run_ids(), [0, 1, 2])
        _, statistics = precompute_category_view_statistics(views, categories, oracle, cross, ges, drk)
        metrics = category_metrics(statistics, 0)
        self.assertAlmostEqual(metrics["baseline_3dgs_three_seed_oracle_mse"], np.mean([0.4, 0.2, 0.8]))
        self.assertAlmostEqual(metrics["nine_model_oracle_mse"], np.mean([0.3, 0.1, 0.6]))

    def test_synthetic_end_to_end_smoke_and_output_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_root = root / "results"
            gt_root = root / "gt"
            figure_root = root / "figures"
            result_root.mkdir(); gt_root.mkdir(); figure_root.mkdir()
            views_list = []
            x_values = []; y_values = []
            for view_index, name in enumerate(("view_a.png", "view_b.png")):
                yy, xx = np.mgrid[0:64, 0:64]
                if view_index == 0:
                    image = np.stack((xx / 63.0, yy / 63.0, ((xx // 8) % 2)), axis=-1)
                else:
                    checker = ((xx // 4 + yy // 4) % 2).astype(np.float64)
                    image = np.stack((checker, xx / 63.0, yy / 63.0), axis=-1)
                Image.fromarray((image * 255 + 0.5).astype(np.uint8)).save(gt_root / name)
                for y in (0, 16, 32):
                    for x in (0, 16, 32):
                        views_list.append(name); x_values.append(x); y_values.append(y)
            patch_count = len(views_list)
            rng = np.random.default_rng(101)
            errors = rng.uniform(0.002, 0.012, size=(patch_count, 9))
            errors[:, 3:6] *= np.linspace(0.75, 1.10, patch_count)[:, None]
            errors[:, 6:9] *= np.linspace(1.05, 0.65, patch_count)[:, None]
            np.savez_compressed(
                result_root / "patch_errors.npz",
                errors=errors, run_ids=np.asarray(run_ids()), views=np.asarray(views_list),
                x=np.asarray(x_values), y=np.asarray(y_values),
                width=np.full(patch_count, 32), height=np.full(patch_count, 32),
            )
            (result_root / "patch_errors.json").write_text(json.dumps({
                "scene": "synthetic", "patch_size": 32, "stride": 16,
                "patch_count": patch_count, "run_ids": run_ids(),
            }), encoding="utf-8")
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "scene": "synthetic", "project_root": str(root),
                "result_root": "results", "figure_root": "figures", "reference_gt": "gt",
                "seeds": [0, 1, 2], "patch_size": 32, "stride": 16,
                "bootstrap_replicates": 50, "bootstrap_seed": 2027,
            }), encoding="utf-8")
            document = run_analysis(config_path, skip_figure=True)
            required = [
                "structural_specialization_summary.csv",
                "structural_specialization_per_view.csv",
                "structural_specialization_representatives.csv",
                "structural_specialization_summary.json",
                "structural_specialization_table.tex",
            ]
            self.assertTrue(all((result_root / name).is_file() for name in required))
            self.assertEqual(len(document["summary"]), 3)
            self.assertEqual(document["config"]["matched_cross_family_combination_count"], 27)
            with (result_root / "structural_specialization_summary.csv").open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle); rows = list(reader)
            self.assertEqual([row["complexity_category"] for row in rows], ["low", "medium", "high"])
            self.assertTrue({"nine_model_incremental_gain_fraction", "matched_cross_family_gain_mean", "positive_view_fraction"}.issubset(reader.fieldnames or []))


if __name__ == "__main__":
    unittest.main()
