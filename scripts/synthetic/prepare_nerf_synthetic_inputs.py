#!/usr/bin/env python3
"""Validate and prepare Phase-III NeRF-synthetic pilot inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets/synthetic/phase3_controlled_pilot"
DEFAULT_POINT_COUNT = 100_000
EXPECTED_TRAIN = 24
EXPECTED_TEST = 8


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def stable_seed(scene_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{scene_id}:{seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def frame_image_path(scene_dir: Path, frame: dict[str, Any]) -> Path:
    raw = Path(str(frame["file_path"]))
    if raw.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raw = raw.with_suffix(".png")
    if raw.is_absolute():
        return raw
    return (scene_dir / raw).resolve()


def validate_transform_matrix(matrix: Any, label: str) -> None:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"{label} transform_matrix must be 4x4, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} transform_matrix contains non-finite values")
    if not np.allclose(arr[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-7):
        raise ValueError(f"{label} transform_matrix last row is not homogeneous")


def validate_split(scene_dir: Path, split: str, expected_count: int) -> dict[str, Any]:
    transform_path = scene_dir / f"transforms_{split}.json"
    if not transform_path.exists():
        raise FileNotFoundError(f"Missing {transform_path}")
    payload = load_json(transform_path)
    if "camera_angle_x" not in payload:
        raise ValueError(f"{transform_path} is missing camera_angle_x")
    if not math.isfinite(float(payload["camera_angle_x"])):
        raise ValueError(f"{transform_path} camera_angle_x is not finite")
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"{transform_path} frames must be a list")
    if len(frames) != expected_count:
        raise ValueError(f"{scene_dir.name} {split} expected {expected_count} frames, found {len(frames)}")

    image_sizes: set[tuple[int, int]] = set()
    frame_paths = []
    for index, frame in enumerate(frames):
        label = f"{scene_dir.name}:{split}:{index}"
        if "file_path" not in frame:
            raise ValueError(f"{label} is missing file_path")
        file_path = str(frame["file_path"])
        if f"/{split}/" not in file_path and not file_path.startswith(f"./{split}/"):
            raise ValueError(f"{label} file_path does not stay inside {split}/: {file_path}")
        validate_transform_matrix(frame.get("transform_matrix"), label)
        image_path = frame_image_path(scene_dir, frame)
        if not image_path.exists():
            raise FileNotFoundError(f"{label} image does not exist: {image_path}")
        with Image.open(image_path) as image:
            image_sizes.add(image.size)
        frame_paths.append(str(image_path))

    if len(image_sizes) != 1:
        raise ValueError(f"{scene_dir.name} {split} images have inconsistent sizes: {sorted(image_sizes)}")
    return {"count": len(frames), "image_size": next(iter(image_sizes)), "images": frame_paths}


def deterministic_points(scene_id: str, seed: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(stable_seed(scene_id, seed))
    xyz = rng.uniform(-1.25, 1.25, size=(count, 3)).astype("<f4")
    rgb = rng.integers(0, 256, size=(count, 3), dtype=np.uint8)
    return xyz, rgb


def write_binary_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    header = "".join(
        [
            "ply\n",
            "format binary_little_endian 1.0\n",
            f"element vertex {len(xyz)}\n",
            "property float x\n",
            "property float y\n",
            "property float z\n",
            "property float nx\n",
            "property float ny\n",
            "property float nz\n",
            "property uchar red\n",
            "property uchar green\n",
            "property uchar blue\n",
            "end_header\n",
        ]
    ).encode("ascii")
    normals = np.zeros_like(xyz, dtype="<f4")
    with path.open("wb") as handle:
        handle.write(header)
        for point, normal, color in zip(xyz, normals, rgb, strict=True):
            handle.write(struct.pack("<ffffffBBB", *point.tolist(), *normal.tolist(), *color.tolist()))


def prepare_scene(scene_dir: Path, point_count: int, seed: int, check_only: bool) -> dict[str, Any]:
    metadata_path = scene_dir / "metadata.json"
    metadata = load_json(metadata_path) if metadata_path.exists() else {}
    scene_id = str(metadata.get("scene_id", scene_dir.name))
    train_expected = int(metadata.get("train_view_count", EXPECTED_TRAIN))
    test_expected = int(metadata.get("test_view_count", EXPECTED_TEST))
    train_info = validate_split(scene_dir, "train", train_expected)
    test_info = validate_split(scene_dir, "test", test_expected)
    if train_info["image_size"] != test_info["image_size"]:
        raise ValueError(f"{scene_id} train/test image sizes differ: {train_info['image_size']} vs {test_info['image_size']}")

    ply_path = scene_dir / "points3d.ply"
    wrote_ply = False
    if not check_only:
        xyz, rgb = deterministic_points(scene_id, seed, point_count)
        tmp_path = ply_path.with_suffix(".ply.tmp")
        write_binary_ply(tmp_path, xyz, rgb)
        tmp_path.replace(ply_path)
        wrote_ply = True
    elif not ply_path.exists():
        print(f"warning: {ply_path} is missing; loaders can auto-create it, but run without --check-only for deterministic prep")

    return {
        "scene_id": scene_id,
        "dataset_path": str(scene_dir),
        "train_view_count": train_info["count"],
        "test_view_count": test_info["count"],
        "resolution": f"{train_info['image_size'][0]}x{train_info['image_size'][1]}",
        "points3d_ply": str(ply_path),
        "point_count": point_count,
        "wrote_points3d_ply": wrote_ply,
        "format": "nerf_synthetic_transforms",
    }


def discover_scenes(dataset_root: Path, scene_id: str | None) -> list[Path]:
    if scene_id:
        scene_dir = dataset_root / scene_id
        if not scene_dir.exists():
            raise FileNotFoundError(f"Scene not found: {scene_dir}")
        return [scene_dir]
    manifest = dataset_root / "manifest.csv"
    if manifest.exists():
        with manifest.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return [Path(row["dataset_path"]) if Path(row["dataset_path"]).is_absolute() else dataset_root / row["scene_id"] for row in rows]
    return sorted(path for path in dataset_root.iterdir() if path.is_dir() and (path / "transforms_train.json").exists())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--point-count", type=int, default=DEFAULT_POINT_COUNT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    dataset_root = resolve_path(args.dataset_root)
    scenes = discover_scenes(dataset_root, args.scene_id)
    if not scenes:
        raise SystemExit(f"No scenes found under {dataset_root}")
    summaries = [prepare_scene(scene.resolve(), args.point_count, args.seed, args.check_only) for scene in scenes]
    summary_path = dataset_root / "synthetic_input_preparation.json"
    if not args.check_only:
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summaries, handle, indent=2, sort_keys=True)
            handle.write("\n")
    for summary in summaries:
        print(
            f"{summary['scene_id']}: train={summary['train_view_count']} test={summary['test_view_count']} "
            f"resolution={summary['resolution']} points={summary['point_count']}"
        )
    print(f"Validated {len(summaries)} scene(s) under {dataset_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
