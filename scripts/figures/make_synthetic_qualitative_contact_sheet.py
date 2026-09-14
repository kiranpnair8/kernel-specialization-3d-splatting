#!/usr/bin/env python3
"""Create an appendix qualitative contact sheet for Phase-III synthetic scenes.

The script reads existing synthetic datasets and trained render outputs only. It
uses seed 0, a fixed held-out test-view index, and the canonical corrected
curvature-high scene.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from PIL import Image, ImageDraw, ImageFont


FACTORS = ("edge_sharpness", "spatial_frequency", "curvature")
LEVELS = ("low", "high")
METHODS = ("3dgs", "ges", "drk")
COLUMNS = (("GT", "gt"), ("3DGS", "3dgs"), ("GES", "ges"), ("DRK", "drk"))
FACTOR_LABELS = {
    "edge_sharpness": "Edge sharp.\nLow",
    "edge_sharpness_high": "Edge sharp.\nHigh",
    "spatial_frequency": "Spatial freq.\nLow",
    "spatial_frequency_high": "Spatial freq.\nHigh",
    "curvature": "Curvature\nLow",
    "curvature_high": "Curvature\nHigh",
}
SCENE_LABELS = {
    ("edge_sharpness", "low"): "Edge sharp.\nLow",
    ("edge_sharpness", "high"): "Edge sharp.\nHigh",
    ("spatial_frequency", "low"): "Spatial freq.\nLow",
    ("spatial_frequency", "high"): "Spatial freq.\nHigh",
    ("curvature", "low"): "Curvature\nLow",
    ("curvature", "high"): "Curvature\nHigh",
}


@dataclass(frozen=True)
class PanelPaths:
    scene_id: str
    dataset_scene_dir: Path
    gt_path: Path
    output_dirs: dict[str, Path]
    render_paths: dict[str, Path]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=root / "datasets" / "synthetic" / "phase3_controlled_pilot",
        help="Synthetic dataset root containing generated Phase-III scenes.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "outputs" / "synthetic" / "phase3_controlled_pilot",
        help="Synthetic method-output root containing 3dgs/ges/drk subdirectories.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=root / "results" / "synthetic" / "figures" / "synthetic_qualitative_contact_sheet.png",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=root / "results" / "synthetic" / "figures" / "synthetic_qualitative_contact_sheet.pdf",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed to visualize. Default: 0.")
    parser.add_argument("--test-index", type=int, default=0, help="Held-out test view index to visualize. Default: 0.")
    parser.add_argument("--spacing", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=300, help="Resolution metadata for saved PNG/PDF.")
    parser.add_argument("--no-pdf", action="store_true", help="Only write PNG output.")
    return parser


def seed_suffix(seed: int) -> str:
    return f"seed{seed:04d}"


def scene_id(factor: str, level: str, seed: int) -> str:
    suffix = seed_suffix(seed)
    if factor == "curvature" and level == "high":
        return f"phase3_curvature_high_corrected_{suffix}"
    return f"phase3_{factor}_{level}_{suffix}"


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def natural_sort_key(path: Path) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def choose_highest_iteration_dir(test_root: Path) -> Path:
    candidates = [path for path in test_root.glob("ours_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No ours_* test render directory found under {test_root}")

    def iteration(path: Path) -> int:
        match = re.search(r"ours_(\d+)", path.name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=iteration)


def find_method_output(output_root: Path, method: str, scene: str) -> Path:
    method_root = output_root / method
    if method == "3dgs":
        candidates = [method_root / scene]
    elif method == "ges":
        candidates = sorted(method_root.glob(scene + "*"), key=lambda path: (path.name != scene, path.name))
    elif method == "drk":
        candidates = [method_root / f"{scene}_DRK", method_root / scene]
    else:
        raise ValueError(f"Unknown method: {method}")
    existing = [path for path in candidates if path.is_dir()]
    if not existing:
        formatted = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"No output directory found for {method}/{scene}; checked: {formatted}")
    if method == "ges":
        # Prefer timestamped/completed GES outputs when present; sorted() keeps this deterministic.
        return existing[-1]
    return existing[0]


def indexed_png(directory: Path, index: int, *, prefixes: Iterable[str] = ("",), width_options: Iterable[int] = (3, 5)) -> Path:
    names: list[str] = []
    for prefix in prefixes:
        for width in width_options:
            names.append(f"{prefix}{index:0{width}d}.png")
        names.append(f"{prefix}{index}.png")
    for name in names:
        path = directory / name
        if path.exists():
            return path
    pngs = sorted(directory.glob("*.png"), key=natural_sort_key)
    if index < len(pngs):
        return pngs[index]
    raise FileNotFoundError(f"No PNG for index {index} found in {directory}; tried {names} and sorted fallback")


def gt_path_for_scene(dataset_scene_dir: Path, test_index: int) -> Path:
    test_dir = dataset_scene_dir / "test"
    path = test_dir / f"r_{test_index:03d}.png"
    if path.exists():
        return path
    return indexed_png(test_dir, test_index, prefixes=("r_", ""), width_options=(3, 5))


def render_path_for_method(method_output: Path, method: str, test_index: int) -> Path:
    if method in {"3dgs", "ges"}:
        test_dir = choose_highest_iteration_dir(method_output / "test")
        render_dir = test_dir / "renders"
        return indexed_png(render_dir, test_index, prefixes=("r_", ""), width_options=(3, 5))
    metric_dir = method_output / "metric" / "test"
    return indexed_png(metric_dir, test_index, prefixes=("render_", ""), width_options=(5, 3))


def collect_panel_paths(dataset_root: Path, output_root: Path, seed: int, test_index: int) -> list[tuple[str, PanelPaths]]:
    rows: list[tuple[str, PanelPaths]] = []
    for factor in FACTORS:
        for level in LEVELS:
            scene = scene_id(factor, level, seed)
            dataset_scene_dir = dataset_root / scene
            if not dataset_scene_dir.is_dir():
                raise FileNotFoundError(f"Synthetic dataset scene directory missing: {dataset_scene_dir}")
            gt_path = gt_path_for_scene(dataset_scene_dir, test_index)
            output_dirs = {method: find_method_output(output_root, method, scene) for method in METHODS}
            render_paths = {
                method: render_path_for_method(output_dirs[method], method, test_index)
                for method in METHODS
            }
            label = SCENE_LABELS[(factor, level)]
            rows.append(
                (
                    label,
                    PanelPaths(
                        scene_id=scene,
                        dataset_scene_dir=dataset_scene_dir,
                        gt_path=gt_path,
                        output_dirs=output_dirs,
                        render_paths=render_paths,
                    ),
                )
            )
    return rows


def open_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def validate_available(rows: list[tuple[str, PanelPaths]]) -> None:
    missing: list[str] = []
    for _, paths in rows:
        checks = [paths.gt_path, *paths.render_paths.values()]
        for path in checks:
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing required synthetic qualitative image(s):\n" + "\n".join(missing))


def text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4, align="center")
    return int(round(bbox[2] - bbox[0])), int(round(bbox[3] - bbox[1]))


def make_contact_sheet(rows: list[tuple[str, PanelPaths]], output_png: Path, output_pdf: Path | None, spacing: int, dpi: int) -> None:
    validate_available(rows)
    images: list[tuple[str, dict[str, Image.Image]]] = []
    for label, paths in rows:
        panels = {"gt": open_rgb(paths.gt_path)}
        for method in METHODS:
            panels[method] = open_rgb(paths.render_paths[method])
        sizes = {key: image.size for key, image in panels.items()}
        if len(set(sizes.values())) != 1:
            raise ValueError(f"Panel sizes differ for {paths.scene_id}: {sizes}")
        images.append((label, panels))

    panel_width, panel_height = next(iter(images[0][1].values())).size
    for label, panels in images:
        for key, image in panels.items():
            if image.size != (panel_width, panel_height):
                raise ValueError(f"Image size for {label}/{key} is {image.size}; expected {(panel_width, panel_height)}")

    header_font = load_font(28)
    row_font = load_font(22)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1), (255, 255, 255)))
    row_label_width = max(text_bbox(measure, label, row_font)[0] for label, _ in images) + 24
    header_height = max(text_bbox(measure, label, header_font)[1] for label, _ in COLUMNS) + 22

    canvas_width = row_label_width + len(COLUMNS) * panel_width + (len(COLUMNS) - 1) * spacing
    canvas_height = header_height + len(images) * panel_height + (len(images) - 1) * spacing
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for col_idx, (column_label, _) in enumerate(COLUMNS):
        x = row_label_width + col_idx * (panel_width + spacing)
        label_w, label_h = text_bbox(draw, column_label, header_font)
        draw.text((x + (panel_width - label_w) / 2, (header_height - label_h) / 2), column_label, fill=(20, 20, 20), font=header_font)

    for row_idx, (row_label, panels) in enumerate(images):
        y = header_height + row_idx * (panel_height + spacing)
        label_w, label_h = text_bbox(draw, row_label, row_font)
        draw.multiline_text(((row_label_width - label_w) / 2, y + (panel_height - label_h) / 2), row_label, fill=(20, 20, 20), font=row_font, spacing=4, align="center")
        for col_idx, (_, key) in enumerate(COLUMNS):
            x = row_label_width + col_idx * (panel_width + spacing)
            canvas.paste(panels[key], (x, y))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png, dpi=(dpi, dpi))
    print(f"wrote: {output_png}")
    if output_pdf is not None:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_pdf, "PDF", resolution=float(dpi))
        print(f"wrote: {output_pdf}")


def print_reproducibility(rows: list[tuple[str, PanelPaths]], seed: int, test_index: int) -> None:
    payload = {
        "seed": seed,
        "test_view_index": test_index,
        "rows": [
            {
                "label": label.replace("\n", " "),
                "scene_id": paths.scene_id,
                "dataset_scene_dir": str(paths.dataset_scene_dir),
                "gt_path": str(paths.gt_path),
                "output_dirs": {method: str(path) for method, path in paths.output_dirs.items()},
                "render_paths": {method: str(path) for method, path in paths.render_paths.items()},
            }
            for label, paths in rows
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    args = build_parser().parse_args()
    rows = collect_panel_paths(args.dataset_root, args.output_root, args.seed, args.test_index)
    print_reproducibility(rows, args.seed, args.test_index)
    make_contact_sheet(
        rows,
        args.output_png,
        None if args.no_pdf else args.output_pdf,
        spacing=args.spacing,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
