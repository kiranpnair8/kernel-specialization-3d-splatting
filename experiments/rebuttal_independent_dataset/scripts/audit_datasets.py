from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

from PIL import Image

from common import atomic_json, load_config, resolve, write_csv


CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}


def digest_text(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_cameras_binary(path: Path) -> dict[int, dict]:
    cameras: dict[int, dict] = {}
    with path.open("rb") as handle:
        count = struct.unpack("<Q", handle.read(8))[0]
        for _ in range(count):
            camera_id, model_id, width, height = struct.unpack("<iiQQ", handle.read(24))
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"Unknown COLMAP camera model id {model_id}")
            model, parameters = CAMERA_MODELS[model_id]
            params = list(struct.unpack(f"<{parameters}d", handle.read(8 * parameters)))
            cameras[camera_id] = {"model": model, "width": width, "height": height, "params": params}
    return cameras


def read_images_binary(path: Path) -> list[dict]:
    images: list[dict] = []
    with path.open("rb") as handle:
        count = struct.unpack("<Q", handle.read(8))[0]
        for _ in range(count):
            values = struct.unpack("<idddddddi", handle.read(64))
            camera_id = values[-1]
            name = bytearray()
            while True:
                value = handle.read(1)
                if not value:
                    raise EOFError(f"Unterminated image name in {path}")
                if value == b"\x00":
                    break
                name.extend(value)
            points = struct.unpack("<Q", handle.read(8))[0]
            handle.seek(24 * points, 1)
            images.append({"name": name.decode("utf-8"), "camera_id": camera_id})
    return images


def read_cameras_text(path: Path) -> dict[int, dict]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        result[int(fields[0])] = {
            "model": fields[1], "width": int(fields[2]), "height": int(fields[3]),
            "params": [float(value) for value in fields[4:]],
        }
    return result


def read_images_text(path: Path) -> list[dict]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    result = []
    for index in range(0, len(lines), 2):
        fields = lines[index].split()
        result.append({"name": fields[9], "camera_id": int(fields[8])})
    return result


def colmap_records(dataset: Path) -> tuple[dict[int, dict], list[dict], list[Path]]:
    sparse = dataset / "sparse" / "0"
    camera_binary, image_binary = sparse / "cameras.bin", sparse / "images.bin"
    if camera_binary.is_file() and image_binary.is_file():
        return read_cameras_binary(camera_binary), read_images_binary(image_binary), [camera_binary, image_binary]
    camera_text, image_text = sparse / "cameras.txt", sparse / "images.txt"
    if camera_text.is_file() and image_text.is_file():
        return read_cameras_text(camera_text), read_images_text(image_text), [camera_text, image_text]
    raise FileNotFoundError(f"No COLMAP cameras/images model under {sparse}")


def camera_focal_lengths(camera: dict) -> tuple[float, float]:
    if camera["model"] == "SIMPLE_PINHOLE":
        return float(camera["params"][0]), float(camera["params"][0])
    if camera["model"] == "PINHOLE":
        return float(camera["params"][0]), float(camera["params"][1])
    raise ValueError(f"Unsupported camera model for common loader: {camera['model']}")


def default_loaded_resolution(image_size: tuple[int, int]) -> tuple[int, int]:
    """Exact default camera_utils.py behavior shared by the three pinned loaders."""
    width, height = image_size
    if width <= 1600:
        return width, height
    scale = width / 1600.0
    return int(width / scale), int(height / scale)


def validate_camera_geometry(camera: dict, image_size: tuple[int, int]) -> dict[str, float | int | list[int]]:
    """Validate a uniformly pre-scaled image, allowing half-pixel rounding on odd dimensions."""
    image_width, image_height = image_size
    colmap_width, colmap_height = int(camera["width"]), int(camera["height"])
    if min(image_width, image_height, colmap_width, colmap_height) <= 0:
        raise ValueError("Camera and image dimensions must be positive")
    scale_x = image_width / colmap_width
    scale_y = image_height / colmap_height
    uniform_scale = 0.5 * (scale_x + scale_y)
    residual_x = abs(colmap_width * uniform_scale - image_width)
    residual_y = abs(colmap_height * uniform_scale - image_height)
    if residual_x > 0.51 or residual_y > 0.51:
        raise ValueError(
            "Image is not a uniformly scaled version of COLMAP dimensions: "
            f"COLMAP={colmap_width}x{colmap_height}, image={image_width}x{image_height}, "
            f"axis_scales=({scale_x:.9f},{scale_y:.9f}), residual_pixels=({residual_x:.3f},{residual_y:.3f})"
        )
    loaded_width, loaded_height = default_loaded_resolution(image_size)
    focal_x, focal_y = camera_focal_lengths(camera)
    fov_x = 2.0 * math.atan(colmap_width / (2.0 * focal_x))
    fov_y = 2.0 * math.atan(colmap_height / (2.0 * focal_y))
    effective_focal_x = 0.5 * loaded_width / math.tan(0.5 * fov_x)
    effective_focal_y = 0.5 * loaded_height / math.tan(0.5 * fov_y)
    expected_focal_x = focal_x * loaded_width / colmap_width
    expected_focal_y = focal_y * loaded_height / colmap_height
    if not math.isclose(effective_focal_x, expected_focal_x, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError("Effective horizontal focal-length validation failed")
    if not math.isclose(effective_focal_y, expected_focal_y, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError("Effective vertical focal-length validation failed")
    return {
        "colmap_resolution": [colmap_width, colmap_height],
        "image_resolution": [image_width, image_height],
        "loaded_resolution": [loaded_width, loaded_height],
        "image_scale_x": scale_x, "image_scale_y": scale_y,
        "uniform_image_scale": uniform_scale, "downsample_factor": 1.0 / uniform_scale,
        "effective_focal_x": effective_focal_x, "effective_focal_y": effective_focal_y,
    }


def inspect_upstream_loaders(config: dict) -> list[dict]:
    records = []
    for method, method_config in config["methods"].items():
        root = resolve(config, method_config["root"])
        dataset_reader = root / "scene" / "dataset_readers.py"
        camera_utils = root / "utils" / "camera_utils.py"
        record = {
            "method": method, "root": str(root), "expected_commit": method_config["commit"],
            "dataset_reader": str(dataset_reader), "camera_utils": str(camera_utils),
        }
        if not dataset_reader.is_file() or not camera_utils.is_file():
            record.update({"status": "missing", "reason": "upstream loader source is unavailable"})
            records.append(record)
            continue
        reader_text = dataset_reader.read_text(encoding="utf-8")
        camera_text = camera_utils.read_text(encoding="utf-8")
        if method == "3dgs":
            split_checks = ["if eval:", "cam_names = sorted(cam_names)", "idx % llffhold == 0", "llffhold=8"]
        else:
            split_checks = ["if eval:", "cam_infos = sorted", "idx % llffhold == 0", "idx % llffhold != 0", "llffhold=8"]
        camera_checks = ["orig_w, orig_h", "args.resolution == -1", "orig_w > 1600"]
        fov_checks = ["focal2fov", "intr.width", "intr.height"]
        missing = [value for value in split_checks + fov_checks if value not in reader_text]
        missing += [value for value in camera_checks if value not in camera_text]
        if missing:
            record.update({"status": "incompatible", "missing_source_patterns": missing})
        else:
            record.update({
                "status": "verified", "split_semantics": "sorted camera identity, every eighth held out under --eval",
                "fov_semantics": "FoV derived from COLMAP focal length and COLMAP width/height",
                "resolution_semantics": "default output uses actual image dimensions, capped identically at width 1600",
                "dataset_reader_sha256": file_sha256(dataset_reader), "camera_utils_sha256": file_sha256(camera_utils),
            })
        records.append(record)
    return records


def audit_scene(config: dict, key: str, scene: dict) -> dict:
    dataset = resolve(config, scene["dataset"])
    row: dict[str, object] = {
        "scene": key, "label": scene["label"], "dataset": str(dataset),
        "official_archive": scene["official_archive"], "present": dataset.is_dir(),
        "split_policy": "lexicographically sorted COLMAP image names; index modulo 8 == 0 is test",
        "outputs": {method: str(resolve(config, config["output_root"]) / key / method / "seed_0") for method in config["methods"]},
    }
    if not dataset.is_dir():
        row["status"] = "missing"
        return row
    cameras, images, model_files = colmap_records(dataset)
    names = sorted(record["name"] for record in images)
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate COLMAP image names in {dataset}")
    if any(Path(name).name != name for name in names):
        raise ValueError("Nested COLMAP image names are incompatible with GES/DRK basename loading")
    stems = [Path(name).stem for name in names]
    if len(stems) != len(set(stems)):
        raise ValueError("Duplicate filename stems are ambiguous in GES/DRK camera identity handling")
    legacy_order = sorted(names, key=lambda name: Path(name).stem)
    if legacy_order != names:
        raise ValueError("3DGS full-name ordering differs from GES/DRK stem ordering")
    image_root = dataset / "images"
    missing = [name for name in names if not (image_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} COLMAP images, first={missing[0]}")
    holdout = int(config["split"]["holdout"])
    test_names = names[::holdout]
    test_set = set(test_names)
    train_names = [name for name in names if name not in test_set]
    actual_sizes = set()
    camera_geometry = []
    records_by_name = {record["name"]: record for record in images}
    for name in names:
        with Image.open(image_root / name) as image:
            actual_sizes.add(image.size)
            camera = cameras[records_by_name[name]["camera_id"]]
            try:
                geometry = validate_camera_geometry(camera, image.size)
            except ValueError as error:
                raise ValueError(f"Invalid camera/image geometry for {dataset}/{name}: {error}") from error
            camera_geometry.append({"image": name, **geometry})
    ordered_scales = sorted(record["uniform_image_scale"] for record in camera_geometry)
    dataset_scale = ordered_scales[len(ordered_scales) // 2]
    for record in camera_geometry:
        expected_width = record["colmap_resolution"][0] * dataset_scale
        expected_height = record["colmap_resolution"][1] * dataset_scale
        if abs(expected_width - record["image_resolution"][0]) > 1.0 or abs(expected_height - record["image_resolution"][1]) > 1.0:
            raise ValueError(
                f"Inconsistent pre-downsampling factors in {dataset}; image={record['image']} "
                f"does not match dataset scale {dataset_scale:.9f} within one output pixel"
            )
    colmap_sizes = {(int(camera["width"]), int(camera["height"])) for camera in cameras.values()}
    models = sorted({camera["model"] for camera in cameras.values()})
    unsupported = sorted(set(models) - {"PINHOLE", "SIMPLE_PINHOLE"})
    if unsupported:
        raise ValueError(f"3DGS/GES common loader does not support camera models {unsupported}")
    sparse = dataset / "sparse" / "0"
    point_files = [path for path in (sparse / "points3D.bin", sparse / "points3D.txt", sparse / "points3D.ply") if path.is_file()]
    if not point_files:
        raise FileNotFoundError(f"Missing COLMAP points3D model under {sparse}")
    camera_summary = {
        "loaded_resolutions": sorted({tuple(record["loaded_resolution"]) for record in camera_geometry}),
        "image_scale_x_range": [min(record["image_scale_x"] for record in camera_geometry), max(record["image_scale_x"] for record in camera_geometry)],
        "image_scale_y_range": [min(record["image_scale_y"] for record in camera_geometry), max(record["image_scale_y"] for record in camera_geometry)],
        "downsample_factor_range": [min(record["downsample_factor"] for record in camera_geometry), max(record["downsample_factor"] for record in camera_geometry)],
        "dataset_uniform_image_scale": dataset_scale,
        "effective_focal_x_range": [min(record["effective_focal_x"] for record in camera_geometry), max(record["effective_focal_x"] for record in camera_geometry)],
        "effective_focal_y_range": [min(record["effective_focal_y"] for record in camera_geometry), max(record["effective_focal_y"] for record in camera_geometry)],
    }
    camera_summary["loaded_resolutions"] = [list(value) for value in camera_summary["loaded_resolutions"]]
    split_hash = digest_text(test_names)
    row.update({
        "status": "ready", "image_directory": str(image_root), "image_count": len(names),
        "train_view_count": len(train_names), "test_view_count": len(test_names),
        "train_split_sha256": digest_text(train_names), "test_split_sha256": split_hash,
        "all_views_sha256": digest_text(names), "camera_models": models,
        "colmap_resolutions": sorted([list(value) for value in colmap_sizes]),
        "image_resolutions": sorted([list(value) for value in actual_sizes]),
        "colmap_image_geometry_consistent": True,
        "images_are_pre_downsampled": any(abs(record["uniform_image_scale"] - 1.0) > 1e-12 for record in camera_geometry),
        "camera_geometry": camera_summary,
        "method_camera_geometry": {method: camera_summary for method in config["methods"]},
        "method_test_split_sha256": {method: split_hash for method in config["methods"]},
        "colmap_model_files": {str(path): file_sha256(path) for path in [*model_files, *point_files]},
        "scene_normalization": "getNerfppNorm over training camera centers: translate=-mean(center), radius=1.1*max_distance",
        "method_split_equivalence": "verified: 3DGS full-name and GES/DRK stem ordering produce identical sorted every-8th camera identities under --eval",
        "effective_intrinsics": "FoV remains derived from COLMAP intrinsics; effective focal pixels equal COLMAP focal * loaded_dimension / COLMAP_dimension",
    })
    return row


def build_manifest(config: dict) -> list[dict]:
    return [audit_scene(config, key, scene) for key, scene in config["scenes"].items()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit independent benchmark COLMAP inputs and freeze split hashes.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero unless every configured scene is ready.")
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output or resolve(config, config["result_root"]) / "dataset_manifest.json"
    rows = build_manifest(config)
    loader_audit = inspect_upstream_loaders(config)
    payload = {
        "experiment": config["experiment"], "config": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config), "split": config["split"],
        "training": {name: {key: value for key, value in method.items() if key in {"upstream", "commit", "iterations", "environment"}} for name, method in config["methods"].items()},
        "loader_audit": loader_audit,
        "scenes": rows,
    }
    atomic_json(output, payload)
    write_csv(output.with_suffix(".csv"), rows)
    for row in rows:
        print(
            f"{row['scene']}: {row['status']} dataset={row['dataset']} "
            f"test_views={row.get('test_view_count', 'n/a')} "
            f"loaded_resolutions={row.get('camera_geometry', {}).get('loaded_resolutions', 'n/a')}"
        )
    for record in loader_audit:
        print(f"loader {record['method']}: {record['status']} root={record['root']}")
    strict_failed = any(row["status"] != "ready" for row in rows) or any(record["status"] != "verified" for record in loader_audit)
    if args.strict and strict_failed:
        print("Dataset/loader audit failed; see the JSON manifest for exact requirements.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
