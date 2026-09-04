from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.local_error.image_io import MethodSpec, ensure_same_shape, load_rgb, pair_views
from analysis.local_error.patches import Patch
from analysis.winner_maps.visualize import save_winner_map
from scripts.evaluate.local_compare import load_config, method_colors, resolve_path

DEFAULT_CONFIG = Path("configs/room_3dgs_ges_drk_budget250k_p32.json")
DEFAULT_PATCHES_CSV = Path("results/room/3dgs_vs_ges_vs_drk_budget250k_p32/patches.csv")
DEFAULT_OUTPUT_DIR = Path("results/room/3dgs_vs_ges_vs_drk_budget250k_p32/figure1_candidates")
DEFAULT_VIEW = "00038.png"
METHODS = ("3dgs", "ges", "drk")
TIE_COLOR = (160, 160, 160)
BOX_COLORS = {
    "3dgs": (45, 114, 210),
    "ges": (218, 87, 67),
    "drk": (62, 150, 91),
}


class Candidate(Tuple[object, ...]):
    pass


def require_columns(fieldnames: Iterable[str] | None, required: Iterable[str]) -> List[str]:
    if fieldnames is None:
        raise ValueError("patches.csv has no header row")
    fields = list(fieldnames)
    missing = [column for column in required if column not in fields]
    if missing:
        raise ValueError(
            "patches.csv is missing required columns "
            f"{missing}; observed columns: {fields}"
        )
    return fields


def load_view_rows(patches_csv: Path, view: str) -> tuple[List[Dict[str, object]], List[str]]:
    rows: List[Dict[str, object]] = []
    with patches_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = require_columns(reader.fieldnames, ["view", "x", "y", "width", "height", "winner"])
        for row_index, row in enumerate(reader, start=2):
            if (row.get("view") or "").strip() != view:
                continue
            winner = (row.get("winner") or "").strip().lower()
            if winner not in {*METHODS, "tie"}:
                raise ValueError(
                    f"row {row_index} has unexpected winner={winner!r}; "
                    f"expected one of {(*METHODS, 'tie')}"
                )
            x = int(float(row["x"]))
            y = int(float(row["y"]))
            width = int(float(row["width"]))
            height = int(float(row["height"]))
            rows.append(
                {
                    "view": view,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "center_x": x + width / 2.0,
                    "center_y": y + height / 2.0,
                    "winner": winner,
                }
            )
    if not rows:
        raise ValueError(f"No patch rows found for view {view} in {patches_csv}")
    return rows, fields


def load_paired_view(config_path: Path, config: Mapping[str, object], view: str) -> object:
    gt_dir = resolve_path(config_path, str(config["gt_dir"]))
    methods = [
        MethodSpec(
            name=str(item["name"]),
            render_dir=resolve_path(config_path, str(item["render_dir"])),
            render_name_template=item.get("render_name_template"),
        )
        for item in config["methods"]
    ]
    paired = {item.view: item for item in pair_views(gt_dir, methods)}
    if view not in paired:
        raise ValueError(f"View {view} is not present in paired GT/render views")
    return paired[view]


def patches_from_rows(rows: Sequence[Mapping[str, object]], patch_size: int) -> tuple[List[Patch], List[str]]:
    patches: List[Patch] = []
    winners: List[str] = []
    for row in rows:
        patch = Patch(x=int(row["x"]), y=int(row["y"]), width=int(row["width"]), height=int(row["height"]))
        if patch.width != patch_size or patch.height != patch_size:
            raise ValueError(
                f"Patch geometry mismatch for {row['view']}: observed {patch.width}x{patch.height}, "
                f"expected {patch_size}x{patch_size}"
            )
        patches.append(patch)
        winners.append(str(row["winner"]))
    return patches, winners


def patch_rows_in_box(rows: Sequence[Mapping[str, object]], x0: int, y0: int, x1: int, y1: int) -> List[Mapping[str, object]]:
    return [
        row
        for row in rows
        if x0 <= float(row["center_x"]) < x1 and y0 <= float(row["center_y"]) < y1
    ]


def fraction(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total


def largest_component_fraction(rows: Sequence[Mapping[str, object]], target: str) -> float:
    target_centers = {
        (round(float(row["center_x"]), 4), round(float(row["center_y"]), 4))
        for row in rows
        if row["winner"] == target
    }
    if not target_centers:
        return 0.0
    xs = sorted({point[0] for point in target_centers})
    ys = sorted({point[1] for point in target_centers})
    dx = min((b - a for a, b in zip(xs, xs[1:]) if b > a), default=1.0)
    dy = min((b - a for a, b in zip(ys, ys[1:]) if b > a), default=1.0)
    unvisited = set(target_centers)
    largest = 0
    while unvisited:
        start = unvisited.pop()
        queue = deque([start])
        size = 1
        while queue:
            x, y = queue.popleft()
            for neighbor in ((x - dx, y), (x + dx, y), (x, y - dy), (x, y + dy)):
                key = (round(neighbor[0], 4), round(neighbor[1], 4))
                if key in unvisited:
                    unvisited.remove(key)
                    queue.append(key)
                    size += 1
        largest = max(largest, size)
    return largest / len(target_centers)


def score_candidate(rows: Sequence[Mapping[str, object]], target: str) -> Optional[Dict[str, object]]:
    total = len(rows)
    if total <= 0:
        return None
    counts = {label: sum(1 for row in rows if row["winner"] == label) for label in (*METHODS, "tie")}
    decisive = total - counts["tie"]
    target_fraction = fraction(counts[target], total)
    tie_fraction = fraction(counts["tie"], total)
    decisive_fraction = fraction(decisive, total)
    coherence = largest_component_fraction(rows, target)
    strongest_other = max(fraction(counts[method], total) for method in METHODS if method != target)
    dominance_margin = target_fraction - strongest_other
    dominance_score = target_fraction * decisive_fraction * coherence + max(0.0, dominance_margin) - 0.5 * tie_fraction
    return {
        "patch_count": total,
        "decisive_patch_count": decisive,
        "3dgs_winner_fraction": fraction(counts["3dgs"], total),
        "ges_winner_fraction": fraction(counts["ges"], total),
        "drk_winner_fraction": fraction(counts["drk"], total),
        "tie_fraction": tie_fraction,
        "coherence_fraction": coherence,
        "dominance_margin": dominance_margin,
        "dominance_score": dominance_score,
    }


def candidate_windows(width: int, height: int, sizes: Sequence[int], step: int) -> Iterable[tuple[int, int, int, int]]:
    for size in sizes:
        if size > width or size > height:
            continue
        x_positions = list(range(0, width - size + 1, step))
        y_positions = list(range(0, height - size + 1, step))
        if x_positions[-1] != width - size:
            x_positions.append(width - size)
        if y_positions[-1] != height - size:
            y_positions.append(height - size)
        for y0 in y_positions:
            for x0 in x_positions:
                yield x0, y0, x0 + size, y0 + size


def iou(left: Mapping[str, object], right: Mapping[str, object]) -> float:
    x0 = max(int(left["x0"]), int(right["x0"]))
    y0 = max(int(left["y0"]), int(right["y0"]))
    x1 = min(int(left["x1"]), int(right["x1"]))
    y1 = min(int(left["y1"]), int(right["y1"]))
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    if inter == 0:
        return 0.0
    left_area = (int(left["x1"]) - int(left["x0"])) * (int(left["y1"]) - int(left["y0"]))
    right_area = (int(right["x1"]) - int(right["x0"])) * (int(right["y1"]) - int(right["y0"]))
    return inter / (left_area + right_area - inter)


def select_top(candidates: Sequence[Mapping[str, object]], top_k: int, max_iou: float) -> List[Mapping[str, object]]:
    selected: List[Mapping[str, object]] = []
    for candidate in sorted(candidates, key=lambda item: float(item["dominance_score"]), reverse=True):
        if all(iou(candidate, existing) <= max_iou for existing in selected):
            selected.append(candidate)
        if len(selected) == top_k:
            break
    return selected


def find_candidates(
    rows: Sequence[Mapping[str, object]],
    width: int,
    height: int,
    sizes: Sequence[int],
    step: int,
    top_k: int,
    max_tie_fraction: float,
    min_patches: int,
    max_iou: float,
) -> List[Dict[str, object]]:
    selected_all: List[Dict[str, object]] = []
    for family in METHODS:
        raw: List[Dict[str, object]] = []
        for x0, y0, x1, y1 in candidate_windows(width, height, sizes, step):
            box_rows = patch_rows_in_box(rows, x0, y0, x1, y1)
            if len(box_rows) < min_patches:
                continue
            summary = score_candidate(box_rows, family)
            if summary is None or float(summary["tie_fraction"]) > max_tie_fraction:
                continue
            raw.append(
                {
                    "family": family,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "crop_width": x1 - x0,
                    "crop_height": y1 - y0,
                    **summary,
                }
            )
        family_top = select_top(raw, top_k, max_iou)
        if len(family_top) < top_k:
            print(
                f"warning: selected {len(family_top)} candidates for {family}; "
                f"consider adjusting sizes/tie threshold only if visual inspection needs more options",
                file=sys.stderr,
            )
        for rank, candidate in enumerate(family_top, start=1):
            record = dict(candidate)
            record["family_rank"] = rank
            selected_all.append(record)
    return selected_all


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = [
        "family",
        "family_rank",
        "x0",
        "y0",
        "x1",
        "y1",
        "crop_width",
        "crop_height",
        "patch_count",
        "decisive_patch_count",
        "3dgs_winner_fraction",
        "ges_winner_fraction",
        "drk_winner_fraction",
        "tie_fraction",
        "coherence_fraction",
        "dominance_margin",
        "dominance_score",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def draw_label_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: tuple[int, int, int], font: ImageFont.ImageFont) -> None:
    bbox = draw.textbbox(xy, text, font=font)
    pad = 6
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=(255, 255, 255), outline=color, width=3)
    draw.text(xy, text, fill=color, font=font)


def draw_candidates(panel: Image.Image, candidates: Sequence[Mapping[str, object]], x_offset: int = 0) -> None:
    draw = ImageDraw.Draw(panel)
    font = load_font(30)
    for index, candidate in enumerate(candidates, start=1):
        family = str(candidate["family"])
        color = BOX_COLORS[family]
        x0 = int(candidate["x0"]) + x_offset
        y0 = int(candidate["y0"])
        x1 = int(candidate["x1"]) + x_offset
        y1 = int(candidate["y1"])
        for inset in range(4):
            draw.rectangle((x0 + inset, y0 + inset, x1 - inset, y1 - inset), outline=color, width=2)
        draw_label_box(draw, (x0 + 10, y0 + 10), f"{index}:{family.upper()}", color, font)


def draw_legend(draw: ImageDraw.ImageDraw, xy: tuple[int, int], colors: Mapping[str, tuple[int, int, int]]) -> None:
    font = load_font(28)
    x, y = xy
    for label in ("3dgs", "ges", "drk", "tie"):
        color = colors[label]
        draw.rectangle((x, y + 4, x + 28, y + 32), fill=color, outline=(30, 30, 30), width=2)
        draw.text((x + 40, y), label.upper() if label != "tie" else "tie", fill=(20, 20, 20), font=font)
        x += 155 if label != "3dgs" else 175


def build_overlay(
    gt_path: Path,
    winner_map_path: Path,
    output_path: Path,
    candidates: Sequence[Mapping[str, object]],
    colors: Mapping[str, tuple[int, int, int]],
) -> None:
    gt = Image.open(gt_path).convert("RGB")
    winner = Image.open(winner_map_path).convert("RGB")
    if gt.size != winner.size:
        raise ValueError(f"GT/winner map shape mismatch: {gt.size} vs {winner.size}")
    gutter = 16
    header = 76
    canvas = Image.new("RGB", (gt.width * 2 + gutter, gt.height + header), (255, 255, 255))
    canvas.paste(gt, (0, header))
    canvas.paste(winner, (gt.width + gutter, header))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(34)
    draw.text((12, 18), "GT", fill=(20, 20, 20), font=title_font)
    draw.text((gt.width + gutter + 12, 18), "Local Winner Map", fill=(20, 20, 20), font=title_font)
    draw_legend(draw, (gt.width + gutter + 340, 18), colors)
    shifted_candidates = []
    for candidate in candidates:
        shifted_candidates.append({**candidate, "y0": int(candidate["y0"]) + header, "y1": int(candidate["y1"]) + header})
    draw_candidates(canvas, shifted_candidates, x_offset=0)
    draw_candidates(canvas, shifted_candidates, x_offset=gt.width + gutter)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Find coherent local winner regions for Figure 1(b) crop selection.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--patches-csv", type=Path, default=DEFAULT_PATCHES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--view", default=DEFAULT_VIEW)
    parser.add_argument("--crop-sizes", nargs="+", type=int, default=[128, 160, 192, 224, 256])
    parser.add_argument("--search-step", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-tie-fraction", type=float, default=0.40)
    parser.add_argument("--min-patches", type=int, default=16)
    parser.add_argument("--max-overlap-iou", type=float, default=0.35)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    patch_size = int(config.get("patch_size", 32))
    view_paths = load_paired_view(config_path, config, args.view)
    gt = load_rgb(view_paths.gt_path)
    renders = {name: load_rgb(path) for name, path in view_paths.render_paths.items()}
    ensure_same_shape([gt, *renders.values()], args.view)
    height, width = gt.shape[:2]

    rows, fields = load_view_rows(args.patches_csv, args.view)
    print(f"inspected patches schema: {fields}")
    patches, winners = patches_from_rows(rows, patch_size)
    colors = method_colors([str(item["name"]) for item in config["methods"]])
    colors["tie"] = TIE_COLOR

    winner_map_path = args.output_dir / f"{Path(args.view).stem}_candidate_regions_winner_map.png"
    save_winner_map(winner_map_path, (height, width), patches, winners, colors, tie_color=TIE_COLOR)

    candidates = find_candidates(
        rows,
        width=width,
        height=height,
        sizes=args.crop_sizes,
        step=args.search_step,
        top_k=args.top_k,
        max_tie_fraction=args.max_tie_fraction,
        min_patches=args.min_patches,
        max_iou=args.max_overlap_iou,
    )
    csv_path = args.output_dir / f"{Path(args.view).stem}_candidate_regions.csv"
    overlay_path = args.output_dir / f"{Path(args.view).stem}_candidate_regions_overlay.png"
    json_path = args.output_dir / f"{Path(args.view).stem}_candidate_regions.json"
    write_csv(csv_path, candidates)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(candidates, handle, indent=2, sort_keys=True)
    build_overlay(view_paths.gt_path, winner_map_path, overlay_path, candidates, colors)

    print("source_paths:")
    print(json.dumps({
        "gt": str(view_paths.gt_path),
        "3dgs": str(view_paths.render_paths["3dgs"]),
        "ges": str(view_paths.render_paths["ges"]),
        "drk": str(view_paths.render_paths["drk"]),
    }, indent=2, sort_keys=True))
    print(f"wrote: {csv_path}")
    print(f"wrote: {json_path}")
    print(f"wrote: {overlay_path}")
    print("top candidates:")
    for index, candidate in enumerate(candidates, start=1):
        print(json.dumps({"overlay_label": index, **candidate}, sort_keys=True))


if __name__ == "__main__":
    main()
