#!/usr/bin/env python3
"""Promote the accepted corrected Phase-III high-curvature candidate."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_controlled_pilot as generator
import prepare_nerf_synthetic_inputs as input_prep

DEST_SCENE_ID = "phase3_curvature_high_corrected_seed0000"
SOURCE_SCENE_ID = "phase3_curvature_high_candidate_0p3_seed0000"
INVALID_ORIGINAL_SCENE_ID = "phase3_curvature_high_seed0000"
DEFAULT_CONFIG = Path("configs/synthetic/phase3_controlled_pilot.json")
DEFAULT_DATASET_ROOT = Path("datasets/synthetic/phase3_controlled_pilot")
DEFAULT_TARGET = 0.54
DEFAULT_TOLERANCE = 0.03
DEFAULT_PARAMETER_VALUE = 0.30


@dataclass
class PromotionValidation:
    scene_id: str
    dataset_path: str
    train_view_count: int
    test_view_count: int
    foreground_fraction: float
    foreground_min: float
    foreground_max: float
    foreground_target: float
    foreground_tolerance: float
    foreground_pass: bool
    finite_images: bool
    points3d_ply_exists: bool
    points3d_ply: str


def default_source() -> Path:
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or Path.home().name
    return Path(f"/tmp/phase3_curvature_validation_{user}/{SOURCE_SCENE_ID}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", type=Path, default=default_source())
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--destination-scene-id", default=DEST_SCENE_ID)
    parser.add_argument("--parameter-value", type=float, default=DEFAULT_PARAMETER_VALUE)
    parser.add_argument("--point-count", type=int, default=input_prep.DEFAULT_POINT_COUNT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--foreground-target", type=float, default=DEFAULT_TARGET)
    parser.add_argument("--foreground-tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--foreground-threshold", type=float, default=0.03)
    parser.add_argument("--validate-only", action="store_true", help="Validate the promoted scene without copying or preparing it.")
    parser.add_argument(
        "--no-regenerate-source",
        action="store_true",
        help="Fail if the accepted temp candidate is missing instead of regenerating it deterministically.",
    )
    parser.add_argument(
        "--replace-destination",
        action="store_true",
        help="Replace only the corrected destination scene if it already exists. Never touches the old invalid high scene.",
    )
    return parser.parse_args()


def resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def find_existing_source(project_root: Path, requested_source: Path) -> Path | None:
    source = resolve_path(project_root, requested_source).resolve()
    if source.exists():
        return source
    matches = sorted(Path("/tmp").glob(f"phase3_curvature_validation_*/{SOURCE_SCENE_ID}"))
    existing = [path.resolve() for path in matches if path.is_dir()]
    return existing[0] if existing else None


def regenerate_source(project_root: Path, config_path: Path, requested_source: Path, parameter_value: float) -> Path:
    source = resolve_path(project_root, requested_source).resolve()
    output_root = source.parent
    canonical_root = resolve_path(project_root, DEFAULT_DATASET_ROOT).resolve()
    if output_root == canonical_root or canonical_root in output_root.parents:
        raise ValueError(f"Refusing to regenerate validation candidate under canonical dataset root: {output_root}")
    config = generator.load_config(resolve_path(project_root, config_path))
    config = generator.curvature_candidate_config(config, [parameter_value])
    generator.generate(config, output_root, tiny_test=False)
    if not source.exists():
        raise FileNotFoundError(f"Regenerated candidate did not appear at expected path: {source}")
    print(f"Regenerated accepted source candidate at {source}")
    return source


def update_metadata(scene_dir: Path, source: Path, scene_id: str, parameter_value: float) -> dict[str, Any]:
    metadata_path = scene_dir / "metadata.json"
    metadata = load_json(metadata_path) if metadata_path.exists() else {}
    metadata.update(
        {
            "scene_id": scene_id,
            "sweep_family": "curvature",
            "level": "high",
            "parameter_name": "paraboloid_amplitude",
            "parameter_value": parameter_value,
            "dataset_path": str(scene_dir),
            "promoted_from_candidate": str(source),
            "supersedes_invalid_scene_id": INVALID_ORIGINAL_SCENE_ID,
            "promotion_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    metadata.setdefault("fixed_factors", {})
    write_json(metadata_path, metadata)
    return metadata


def frame_count(scene_dir: Path, split: str) -> int:
    payload = load_json(scene_dir / f"transforms_{split}.json")
    return len(payload.get("frames", []))


def image_paths(scene_dir: Path) -> list[Path]:
    return sorted([*scene_dir.glob("train/*.png"), *scene_dir.glob("test/*.png")])


def test_image_paths(scene_dir: Path) -> list[Path]:
    return sorted(scene_dir.glob("test/*.png"))


def background_from_metadata(metadata: Mapping[str, Any]) -> np.ndarray:
    fixed = metadata.get("fixed_factors") if isinstance(metadata, Mapping) else None
    value = fixed.get("background_color") if isinstance(fixed, Mapping) else None
    if value is None:
        value = metadata.get("background_color", [1.0, 1.0, 1.0])
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"background_color must have three channels, got {arr}")
    return arr


def validate_finite_images(scene_dir: Path) -> bool:
    paths = image_paths(scene_dir)
    if not paths:
        raise FileNotFoundError(f"No train/test PNGs found under {scene_dir}")
    for path in paths:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        if not np.all(np.isfinite(rgb)):
            raise FloatingPointError(f"Non-finite image values found in {path}")
    return True


def foreground_fractions(scene_dir: Path, background: np.ndarray, threshold: float) -> list[float]:
    fractions: list[float] = []
    paths = test_image_paths(scene_dir)
    if not paths:
        raise FileNotFoundError(f"No test PNGs found under {scene_dir}")
    for path in paths:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        distance = np.linalg.norm(rgb - background.reshape(1, 1, 3), axis=-1)
        fractions.append(float(np.mean(distance > threshold)))
    return fractions


def manifest_row(scene_dir: Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scene_id": metadata["scene_id"],
        "sweep_family": metadata["sweep_family"],
        "level": metadata["level"],
        "parameter_value": metadata["parameter_value"],
        "seed": metadata.get("seed", 0),
        "train_view_count": metadata.get("train_view_count", frame_count(scene_dir, "train")),
        "test_view_count": metadata.get("test_view_count", frame_count(scene_dir, "test")),
        "resolution": metadata.get("resolution", ""),
        "dataset_path": str(scene_dir),
    }


def update_root_manifests(dataset_root: Path, row: dict[str, Any]) -> None:
    csv_path = dataset_root / "manifest.csv"
    json_path = dataset_root / "manifest.json"
    fieldnames = [
        "scene_id",
        "sweep_family",
        "level",
        "parameter_value",
        "seed",
        "train_view_count",
        "test_view_count",
        "resolution",
        "dataset_path",
    ]
    rows: list[dict[str, Any]] = []
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [existing for existing in rows if existing.get("scene_id") != row["scene_id"]]
    rows.append({field: row[field] for field in fieldnames})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_rows: list[dict[str, Any]] = []
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            json_rows = payload
    json_rows = [existing for existing in json_rows if existing.get("scene_id") != row["scene_id"]]
    json_rows.append(row)
    write_json(json_path, json_rows)


def validate_scene(
    scene_dir: Path,
    scene_id: str,
    expected_parameter_value: float,
    foreground_target: float,
    foreground_tolerance: float,
    foreground_threshold: float,
) -> PromotionValidation:
    metadata = load_json(scene_dir / "metadata.json")
    if metadata.get("scene_id") != scene_id:
        raise ValueError(f"metadata scene_id mismatch: {metadata.get('scene_id')} != {scene_id}")
    if metadata.get("sweep_family") != "curvature" or metadata.get("level") != "high":
        raise ValueError("Corrected scene metadata must be curvature/high")
    if abs(float(metadata.get("parameter_value")) - expected_parameter_value) > 1e-9:
        raise ValueError(f"Corrected scene parameter_value must be {expected_parameter_value}")
    train_count = frame_count(scene_dir, "train")
    test_count = frame_count(scene_dir, "test")
    if train_count != 24 or test_count != 8:
        raise ValueError(f"Expected 24 train / 8 test views, found {train_count} / {test_count}")
    finite = validate_finite_images(scene_dir)
    background = background_from_metadata(metadata)
    fractions = foreground_fractions(scene_dir, background, foreground_threshold)
    mean_foreground = float(np.mean(fractions))
    foreground_pass = abs(mean_foreground - foreground_target) <= foreground_tolerance
    ply = scene_dir / "points3d.ply"
    validation = PromotionValidation(
        scene_id=scene_id,
        dataset_path=str(scene_dir),
        train_view_count=train_count,
        test_view_count=test_count,
        foreground_fraction=mean_foreground,
        foreground_min=float(np.min(fractions)),
        foreground_max=float(np.max(fractions)),
        foreground_target=foreground_target,
        foreground_tolerance=foreground_tolerance,
        foreground_pass=foreground_pass,
        finite_images=finite,
        points3d_ply_exists=ply.exists(),
        points3d_ply=str(ply),
    )
    if not foreground_pass:
        raise ValueError(
            f"Foreground fraction {mean_foreground:.4f} is outside "
            f"{foreground_target:.4f} +/- {foreground_tolerance:.4f}"
        )
    if not ply.exists():
        raise FileNotFoundError(f"Missing deterministic point scaffold: {ply}")
    return validation


def promote(args: argparse.Namespace) -> PromotionValidation:
    project_root = args.project_root.resolve()
    source = find_existing_source(project_root, args.source)
    dataset_root = resolve_path(project_root, args.dataset_root).resolve()
    destination = dataset_root / args.destination_scene_id
    invalid_original = dataset_root / INVALID_ORIGINAL_SCENE_ID

    if source is None and not args.validate_only and not args.no_regenerate_source:
        source = regenerate_source(project_root, args.config, args.source, args.parameter_value)
    if source is None and not args.validate_only:
        requested = resolve_path(project_root, args.source).resolve()
        raise FileNotFoundError(f"Accepted source candidate does not exist: {requested}")
    if destination.exists():
        if not args.replace_destination and not args.validate_only:
            raise FileExistsError(
                f"Corrected destination already exists: {destination}. "
                "Use --replace-destination only if you intend to replace this corrected scene."
            )
    if not args.validate_only:
        assert source is not None
        if args.replace_destination and destination.exists():
            if destination.name != args.destination_scene_id:
                raise RuntimeError(f"Refusing to replace unexpected path: {destination}")
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        metadata = update_metadata(destination, source, args.destination_scene_id, args.parameter_value)
        row = manifest_row(destination, metadata)
        update_root_manifests(dataset_root, row)
        input_prep.prepare_scene(destination, args.point_count, args.seed, check_only=False)
    else:
        if not destination.exists():
            raise FileNotFoundError(f"Corrected destination is missing for --validate-only: {destination}")

    validation = validate_scene(
        destination,
        args.destination_scene_id,
        args.parameter_value,
        args.foreground_target,
        args.foreground_tolerance,
        args.foreground_threshold,
    )
    write_json(destination / "promotion_validation.json", asdict(validation))
    if invalid_original.exists():
        print(f"Preserved invalid original scene: {invalid_original}")
    return validation


def main() -> int:
    args = parse_args()
    validation = promote(args)
    print(f"Corrected curvature scene ready: {validation.dataset_path}")
    print(f"train/test views: {validation.train_view_count}/{validation.test_view_count}")
    print(f"foreground fraction: {validation.foreground_fraction:.4f}")
    print(f"points3d.ply: {validation.points3d_ply}")
    print(f"validation: {Path(validation.dataset_path) / 'promotion_validation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
