from __future__ import annotations

import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

TEST_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = TEST_ROOT / "scripts"
REPO_ROOT = TEST_ROOT.parents[1]
for item in (SCRIPT_ROOT, REPO_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from audit_datasets import build_manifest, digest_text
from evaluate_independent import align_by_gt_pixels
from analyze_independent import complexity_categories, summarize, winner_labels
from analysis.local_error.metrics import mse
from analysis.structure_features.descriptors import image_patch_complexity_descriptors


def write_colmap_dataset(
    root: Path,
    count: int = 9,
    colmap_size: tuple[int, int] = (8, 6),
    image_size: tuple[int, int] | None = None,
    focal: tuple[float, float] = (7.0, 7.0),
) -> None:
    image_size = image_size or colmap_size
    image_root = root / "images"; sparse = root / "sparse" / "0"
    image_root.mkdir(parents=True); sparse.mkdir(parents=True)
    with (sparse / "cameras.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 1)); handle.write(struct.pack("<iiQQ", 1, 1, *colmap_size)); handle.write(struct.pack("<dddd", *focal, colmap_size[0] / 2, colmap_size[1] / 2))
    with (sparse / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", count))
        for index in range(count):
            name = f"image_{index:03d}.png"
            handle.write(struct.pack("<idddddddi", index + 1, 1.0, 0.0, 0.0, 0.0, float(index), 0.0, 0.0, 1))
            handle.write(name.encode() + b"\x00"); handle.write(struct.pack("<Q", 0))
            Image.fromarray(np.full((image_size[1], image_size[0], 3), index, dtype=np.uint8)).save(image_root / name)
    (sparse / "points3D.txt").write_text("# synthetic fixture\n", encoding="utf-8")


class IndependentDatasetTests(unittest.TestCase):
    @staticmethod
    def config(root: Path, dataset: str = "datasets/tandt/train") -> dict:
        return {
            "project_root": str(root), "output_root": "outputs", "split": {"holdout": 8},
            "scenes": {"train": {"label": "Train", "dataset": dataset, "official_archive": "x"}},
            "methods": {name: {} for name in ("3dgs", "ges", "drk")},
        }

    def test_exact_resolution_and_every_eighth_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); dataset = root / "datasets" / "tandt" / "train"
            write_colmap_dataset(dataset)
            row = build_manifest(self.config(root))[0]
            self.assertEqual(row["status"], "ready")
            self.assertEqual(row["train_view_count"], 7)
            self.assertEqual(row["test_view_count"], 2)
            self.assertEqual(row["test_split_sha256"], digest_text(["image_000.png", "image_008.png"]))
            self.assertFalse(row["images_are_pre_downsampled"])
            self.assertEqual(row["camera_geometry"]["loaded_resolutions"], [[8, 6]])

    def test_valid_odd_width_predownsample_and_effective_focal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); dataset = root / "datasets" / "tandt" / "train"
            write_colmap_dataset(dataset, colmap_size=(1959, 1090), image_size=(980, 545), focal=(1200.0, 1180.0))
            row = build_manifest(self.config(root))[0]
            self.assertTrue(row["colmap_image_geometry_consistent"])
            self.assertTrue(row["images_are_pre_downsampled"])
            self.assertEqual(row["camera_geometry"]["loaded_resolutions"], [[980, 545]])
            self.assertAlmostEqual(row["camera_geometry"]["downsample_factor_range"][0], 2.0, delta=0.002)
            self.assertAlmostEqual(row["camera_geometry"]["effective_focal_x_range"][0], 1200.0 * 980 / 1959)
            self.assertAlmostEqual(row["camera_geometry"]["effective_focal_y_range"][0], 1180.0 * 545 / 1090)
            self.assertEqual(len(set(row["method_test_split_sha256"].values())), 1)

    def test_invalid_nonuniform_resolution_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); dataset = root / "datasets" / "tandt" / "train"
            write_colmap_dataset(dataset, colmap_size=(1959, 1090), image_size=(980, 500))
            with self.assertRaisesRegex(ValueError, "not a uniformly scaled version"):
                build_manifest(self.config(root))

    def test_alignment_uses_pixels_not_candidate_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); reference = root / "reference"; candidate = root / "candidate"; renders = root / "renders"
            reference.mkdir(); candidate.mkdir(); renders.mkdir()
            render_lookup = {}
            for index, value in enumerate((20, 80)):
                Image.fromarray(np.full((5, 7, 3), value, dtype=np.uint8)).save(reference / f"camera_{index}.png")
            for index, value in enumerate((80, 20)):
                gt = candidate / f"gt_{index:05d}.png"; render = renders / f"render_{index:05d}.png"
                Image.fromarray(np.full((5, 7, 3), value, dtype=np.uint8)).save(gt)
                Image.fromarray(np.full((5, 7, 3), value + 1, dtype=np.uint8)).save(render)
                render_lookup[gt.name] = render
            mapping = align_by_gt_pixels(sorted(reference.glob("*.png")), sorted(candidate.glob("*.png")), render_lookup)
            self.assertTrue(mapping["camera_0.png"]["candidate_gt"].endswith("gt_00001.png"))
            self.assertTrue(mapping["camera_1.png"]["candidate_gt"].endswith("gt_00000.png"))

    def test_metrics_winners_ties_and_oracle(self) -> None:
        errors = np.asarray([[0.20, 0.10, 0.30], [0.10, 0.100005, 0.50], [0.40, 0.30, 0.20]])
        winners, margin = winner_labels(errors, 1e-5)
        self.assertEqual(winners.tolist(), ["ges", "tie", "drk"])
        summary = summarize(errors, winners, margin, np.ones(3, dtype=bool))
        self.assertEqual(summary["decisive_patch_count"], 2)
        self.assertAlmostEqual(summary["tie_fraction"], 1 / 3)
        self.assertAlmostEqual(summary["oracle_mse"], np.mean([0.10, 0.10, 0.20]))
        self.assertGreater(summary["oracle_relative_improvement_pct_vs_3dgs"], 0)
        target = np.zeros((2, 2, 3)); prediction = np.ones((2, 2, 3)) * 0.5
        self.assertAlmostEqual(mse(target, prediction), 0.25)

    def test_gt_descriptors_and_categories(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.float64); image[:, 4:] = 1.0
        descriptors = image_patch_complexity_descriptors(image, np.asarray([0, 2]), np.asarray([0, 0]), np.asarray([4, 4]), np.asarray([8, 8]))
        self.assertTrue(all(np.isfinite(value).all() for value in descriptors.values()))
        self.assertGreater(descriptors["edge_density"][1], descriptors["edge_density"][0])
        synthetic = {name: np.arange(9, dtype=np.float64) for name in descriptors}
        _, categories, _ = complexity_categories(synthetic)
        self.assertEqual(set(categories.tolist()), {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
