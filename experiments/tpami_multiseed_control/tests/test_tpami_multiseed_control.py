from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = EXPERIMENT_ROOT / "scripts"
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
for item in (SCRIPT_ROOT, REPO_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from analyze_matched_oracles import bootstrap_groups, main as analysis_main, pair_rows
from common import enumerate_pairs, expected_run_ids, pair_metrics
from evaluate_six_models import EXP2, verify_against_experiment2, verify_seed_record


class TpamiMultiseedControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.run_ids = expected_run_ids()
        self.errors = np.asarray([
            [0.30, 0.25, 0.28, 0.22, 0.24, 0.23],
            [0.10, 0.12, 0.11, 0.14, 0.09, 0.13],
            [0.40, 0.35, 0.38, 0.31, 0.33, 0.29],
            [0.20, 0.18, 0.19, 0.21, 0.17, 0.20],
            [0.50, 0.48, 0.45, 0.42, 0.44, 0.41],
        ], dtype=np.float64)
        self.views = np.asarray(["a.png", "a.png", "b.png", "b.png", "b.png"])

    def test_pair_enumeration_is_matched_size_and_complete(self) -> None:
        pairs = enumerate_pairs(self.run_ids)
        self.assertEqual(len(pairs), 15)
        self.assertEqual(sum(pair["group"] == "within_3dgs" for pair in pairs), 3)
        self.assertEqual(sum(pair["group"] == "within_ges" for pair in pairs), 3)
        self.assertEqual(sum(pair["group"] == "cross_family" for pair in pairs), 9)
        self.assertTrue(all(pair["left"] != pair["right"] for pair in pairs))

    def test_oracle_and_strict_tie_calculation(self) -> None:
        values = np.asarray([[0.2, 0.1], [0.1, 0.100005], [0.3, 0.4]])
        metrics = pair_metrics(values, "a", "b", 1e-5)
        self.assertAlmostEqual(metrics["oracle_mse"], np.mean([0.1, 0.1, 0.3]))
        self.assertEqual(metrics["better_single_model"], "a")
        self.assertAlmostEqual(metrics["tie_fraction"], 1 / 3)
        self.assertAlmostEqual(metrics["left_winner_fraction"], 1 / 3)
        self.assertAlmostEqual(metrics["right_winner_fraction"], 1 / 3)
        exact = pair_metrics(np.asarray([[0.1, 0.1]]), "a", "b", 0.0)
        self.assertEqual(exact["left_winner_fraction"], 1.0)
        self.assertEqual(exact["tie_fraction"], 0.0)

    def test_bootstrap_reproducibility(self) -> None:
        first = bootstrap_groups(self.errors, self.run_ids, self.views, 100, 91)
        second = bootstrap_groups(self.errors, self.run_ids, self.views, 100, 91)
        self.assertEqual(first, second)

    def test_seed_record_requires_explicit_matching_seed(self) -> None:
        status = {"method": "3dgs", "seed": 2, "commands": [{"stage": "train", "command": ["python", "train.py", "--seed", "2"]}]}
        verify_seed_record(status, "3dgs", 2)
        with self.assertRaises(ValueError):
            verify_seed_record(status, "3dgs", 1)

    def test_pixel_alignment_does_not_assume_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); reference = root / "reference"; candidate = root / "candidate"; renders = root / "renders"
            reference.mkdir(); candidate.mkdir(); renders.mkdir(); lookup = {}
            for index, value in enumerate((10, 90)):
                Image.fromarray(np.full((4, 6, 3), value, dtype=np.uint8)).save(reference / f"camera_{index}.png")
            for index, value in enumerate((90, 10)):
                gt = candidate / f"gt_{index}.png"; render = renders / f"render_{index}.png"
                Image.fromarray(np.full((4, 6, 3), value, dtype=np.uint8)).save(gt)
                Image.fromarray(np.full((4, 6, 3), value + 1, dtype=np.uint8)).save(render); lookup[gt.name] = render
            mapping = EXP2.align_by_gt_pixels(sorted(reference.glob("*.png")), sorted(candidate.glob("*.png")), lookup)
            self.assertTrue(mapping["camera_0.png"]["candidate_gt"].endswith("gt_1.png"))

    def test_experiment2_bundle_alignment_and_seed0_reproduction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / "exp2"; source.mkdir()
            views = np.asarray(["a.png", "b.png"]); coordinates = {"x": np.asarray([0, 0]), "y": np.asarray([0, 0]), "width": np.asarray([32, 32]), "height": np.asarray([32, 32])}
            original = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
            np.savez_compressed(source / "patch_errors.npz", errors=original, methods=np.asarray(["3dgs", "ges", "drk"]), views=views, **coordinates)
            six = np.asarray([[0.1, 0.11, 0.12, 0.2, 0.21, 0.22], [0.4, 0.41, 0.42, 0.5, 0.51, 0.52]])
            config = {"project_root": str(root), "experiment2_result": "exp2"}
            result = verify_against_experiment2(config, six, views, coordinates)
            self.assertEqual(result["status"], "exact")
            bad_coordinates = dict(coordinates); bad_coordinates["x"] = np.asarray([1, 0])
            with self.assertRaisesRegex(ValueError, "coordinates"):
                verify_against_experiment2(config, six, views, bad_coordinates)

    def test_synthetic_analysis_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); results = root / "results"; sensitivity = results / "sensitivity"
            sensitivity.mkdir(parents=True)
            rng = np.random.default_rng(4)
            views = np.repeat(np.asarray(["v0.png", "v1.png", "v2.png"]), 4)
            coordinates = {
                "x": np.tile(np.asarray([0, 16, 0, 16]), 3), "y": np.tile(np.asarray([0, 0, 16, 16]), 3),
                "width": np.full(12, 32), "height": np.full(12, 32),
            }
            errors = rng.uniform(0.01, 0.08, size=(12, 6))
            metadata_views = []
            for index, view in enumerate(("v0.png", "v1.png", "v2.png")):
                yy, xx = np.mgrid[:64, :64]
                image = np.stack([(xx + index * 5) % 64, (yy * (index + 1)) % 64, (xx + yy) % 64], axis=-1).astype(np.uint8) * 4
                path = root / view; Image.fromarray(image).save(path)
                metadata_views.append({"view": view, "gt": str(path), "renders": {}})
            for size, destination in ((32, results / "patch_errors.npz"), (64, sensitivity / "patch_errors_p64.npz"), (128, sensitivity / "patch_errors_p128.npz")):
                np.savez_compressed(destination, errors=errors, run_ids=np.asarray(self.run_ids), views=views, **coordinates)
                destination.with_suffix(".json").write_text(json.dumps({"views": metadata_views, "patch_size": size}), encoding="utf-8")
            inventory = [{"run_id": run_id, "method": run_id.split("_")[0], "seed": int(run_id[-1]), "psnr": 25.0, "ssim": 0.8, "lpips": 0.2, "primitive_count": 1000} for run_id in self.run_ids]
            (results / "model_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
            config = {
                "project_root": str(root), "result_root": "results", "patch_size": 32, "stride": 16,
                "tie_threshold_mse": 1e-5, "bootstrap_replicates": 20, "bootstrap_seed": 5,
                "sensitivity": {"patch_sizes": [32, 64, 128], "tie_thresholds_mse": [0.0, 1e-5, 5e-5]},
            }
            config_path = root / "config.json"; config_path.write_text(json.dumps(config), encoding="utf-8")
            with patch.object(sys, "argv", ["analyze_matched_oracles.py", "--config", str(config_path), "--no-figures"]):
                self.assertEqual(analysis_main(), 0)
            self.assertTrue((results / "pairwise_oracle_summary.csv").is_file())
            self.assertTrue((results / "matched_size_oracle_table.tex").is_file())
            self.assertEqual(len(json.loads((results / "sensitivity_summary.json").read_text(encoding="utf-8"))), 27)


if __name__ == "__main__":
    unittest.main()
