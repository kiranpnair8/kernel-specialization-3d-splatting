from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from analysis.local_error.patches import iter_patches
from scripts.evaluate.characterize_specialization import validate_patch_setting


def patch_rows(view: str, width: int, height: int, patch_size: int, stride: int) -> list[dict[str, object]]:
    return [
        {
            "view": view,
            "x": patch.x,
            "y": patch.y,
            "width": patch.width,
            "height": patch.height,
        }
        for patch in iter_patches(height, width, patch_size, stride)
    ]


class CharacterizationGeometryValidationTest(unittest.TestCase):
    def test_accepts_edge_anchored_final_patches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gt_dir = root / "gt"
            gt_dir.mkdir()
            Image.new("RGB", (1297, 840)).save(gt_dir / "00000.png")

            rows = patch_rows("00000.png", width=1297, height=840, patch_size=32, stride=16)
            self.assertIn((1265, 808, 32, 32), {(row["x"], row["y"], row["width"], row["height"]) for row in rows})

            validation = validate_patch_setting(
                rows,
                patch_size=32,
                stride=16,
                config_path=root / "config.json",
                config={"gt_dir": "gt"},
            )

            self.assertTrue(validation["patch_size_matches"])
            self.assertTrue(validation["coordinate_grid_matches"])
            self.assertEqual(validation["view_mismatch_count"], 0)

    def test_rejects_unexpected_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gt_dir = root / "gt"
            gt_dir.mkdir()
            Image.new("RGB", (1297, 840)).save(gt_dir / "00000.png")

            rows = patch_rows("00000.png", width=1297, height=840, patch_size=32, stride=16)
            rows[0] = {**rows[0], "x": 1}

            validation = validate_patch_setting(
                rows,
                patch_size=32,
                stride=16,
                config_path=root / "config.json",
                config={"gt_dir": "gt"},
            )

            self.assertTrue(validation["patch_size_matches"])
            self.assertFalse(validation["coordinate_grid_matches"])
            self.assertEqual(validation["view_mismatch_count"], 1)
            mismatch = validation["view_mismatches"][0]
            self.assertGreater(mismatch["missing_count"], 0)
            self.assertGreater(mismatch["unexpected_count"], 0)


if __name__ == "__main__":
    unittest.main()
