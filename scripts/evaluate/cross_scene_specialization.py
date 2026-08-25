#!/usr/bin/env python3
"""Cross-scene aggregation for observational local specialization results."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
METHODS = ["3dgs", "ges", "drk"]
DESCRIPTORS = [
    "mean_gradient_magnitude",
    "edge_strength",
    "laplacian_energy",
    "local_variance",
    "high_frequency_energy",
    "entropy",
]
HIGH_COMPLEXITY_DESCRIPTORS = ["edge_strength", "high_frequency_energy", "local_variance"]
COLORS = {
    "3dgs": (45, 92, 160),
    "ges": (26, 126, 97),
    "drk": (170, 76, 54),
    "tie": (120, 120, 120),
    "non_3dgs": (104, 78, 150),
}


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _as_float(value: Any) -> float | None:
    if value in (None, "", "nan", "NaN"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


def _sign(value: float | None, eps: float = 1e-12) -> str:
    if value is None:
        return "missing"
    if abs(value) <= eps:
        return "zero"
    return "positive" if value > 0 else "negative"


def _status_from_signs(signs: list[str]) -> str:
    if not signs or any(sign == "missing" for sign in signs):
        return "missing"
    nonzero = [sign for sign in signs if sign != "zero"]
    if not nonzero:
        return "disappears_or_weakens"
    if len(set(nonzero)) == 1 and len(nonzero) == len(signs):
        return "same_direction"
    if len(set(nonzero)) == 1:
        return "disappears_or_weakens"
    return "mixed_directions"


def _slope(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return None
    x = np.array([p[0] for p in pairs], dtype=float)
    y = np.array([p[1] for p in pairs], dtype=float)
    if float(np.ptp(x)) == 0.0:
        return None
    return float(np.polyfit(x, y, deg=1)[0])


def _parse_scene(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Scene inputs must use name=path syntax")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Scene name is empty")
    return name, _resolve(path.strip())


def load_scene(name: str, root: Path) -> dict[str, Any]:
    char_dir = root / "characterization"
    files = {
        "summary": root / "summary.json",
        "descriptor_summary": char_dir / "descriptor_summary.csv",
        "pairwise_effects": char_dir / "pairwise_effects.csv",
        "winner_probability": char_dir / "winner_probability_by_descriptor.csv",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files for scene {name}: {missing}")
    return {
        "name": name,
        "root": str(root),
        "summary": _read_json(files["summary"]),
        "descriptor_summary": _read_csv(files["descriptor_summary"]),
        "pairwise_effects": _read_csv(files["pairwise_effects"]),
        "winner_probability": _read_csv(files["winner_probability"]),
    }


def scene_summary_row(scene: dict[str, Any]) -> dict[str, Any]:
    oracle = scene["summary"].get("oracle", {})
    row: dict[str, Any] = {
        "scene": scene["name"],
        "result_dir": scene["root"],
        "patch_count": oracle.get("patch_count"),
        "decisive_patch_count": oracle.get("decisive_patch_count"),
        "tie_fraction": oracle.get("tie_fraction"),
        "oracle_patch_mse": oracle.get("oracle_patch_mse"),
        "oracle_patch_psnr": oracle.get("oracle_patch_psnr"),
    }
    non_3dgs = 0.0
    for method in METHODS:
        frac = _as_float(oracle.get(f"{method}_winner_fraction"))
        row[f"{method}_winner_fraction"] = frac
        row[f"{method}_patch_mse"] = oracle.get(f"{method}_patch_mse")
        row[f"{method}_patch_psnr"] = oracle.get(f"{method}_patch_psnr")
        if method != "3dgs" and frac is not None:
            non_3dgs += frac
    row["non_3dgs_winner_fraction"] = non_3dgs
    base = _as_float(oracle.get("3dgs_patch_mse"))
    gain = _as_float(oracle.get("oracle_improvement_mse_vs_3dgs"))
    row["oracle_improvement_mse_vs_3dgs"] = gain
    row["oracle_relative_improvement_pct_vs_3dgs"] = None if not base or gain is None else 100.0 * gain / base
    return row


def descriptor_summary_by_scene(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        for row in scene["descriptor_summary"]:
            out = {"scene": scene["name"]}
            out.update(row)
            rows.append(out)
    return rows


def winner_probability_by_scene(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        for row in scene["winner_probability"]:
            out = {"scene": scene["name"]}
            out.update(row)
            rows.append(out)
    return rows


def probability_slopes(scene: dict[str, Any]) -> dict[tuple[str, str], float | None]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in scene["winner_probability"]:
        grouped[row["descriptor"]].append(row)
    slopes: dict[tuple[str, str], float | None] = {}
    for descriptor, rows in grouped.items():
        ordered = sorted(rows, key=lambda r: int(float(r.get("bin_index", 0))))
        xs = [_as_float(r.get("bin_mid")) for r in ordered]
        for method in METHODS:
            ys = [_as_float(r.get(f"p_{method}_wins")) for r in ordered]
            clean_x = [x for x, y in zip(xs, ys) if x is not None and y is not None]
            clean_y = [y for x, y in zip(xs, ys) if x is not None and y is not None]
            slopes[(descriptor, method)] = _slope(clean_x, clean_y)
    return slopes


def descriptor_medians(scene: dict[str, Any]) -> dict[tuple[str, str], float | None]:
    medians: dict[tuple[str, str], float | None] = {}
    for row in scene["descriptor_summary"]:
        medians[(row.get("descriptor", ""), row.get("winner", ""))] = _as_float(row.get("median"))
    return medians


def _attach_scene_values(row: dict[str, Any], scene_values: dict[str, tuple[float | None, str]], scene_names: list[str]) -> None:
    for scene_name in scene_names:
        value, direction = scene_values.get(scene_name, (None, "missing"))
        row[f"{scene_name}_value"] = value
        row[f"{scene_name}_direction"] = direction
    signs = [row[f"{scene_name}_direction"] for scene_name in scene_names]
    row["status"] = _status_from_signs(signs)
    row["direction_consistent_across_scenes"] = row["status"] == "same_direction"


def build_effect_consistency(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scene_names = [scene["name"] for scene in scenes]
    rows: list[dict[str, Any]] = []
    pairs: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for scene in scenes:
        for row in scene["pairwise_effects"]:
            key = (row.get("descriptor", ""), row.get("left_winner", ""), row.get("right_winner", ""))
            pairs[key][scene["name"]] = row
    for (descriptor, left, right), by_scene in sorted(pairs.items()):
        out: dict[str, Any] = {
            "analysis": "descriptor_median_pairwise",
            "descriptor": descriptor,
            "method": "",
            "left_winner": left,
            "right_winner": right,
            "interpretation_note": "Observational descriptor distribution difference; not causal evidence.",
        }
        scene_values = {}
        for scene_name in scene_names:
            scene_row = by_scene.get(scene_name, {})
            value = _as_float(scene_row.get("median_difference_left_minus_right"))
            scene_values[scene_name] = (value, _sign(value))
            out[f"{scene_name}_p_value_bh_fdr"] = scene_row.get("p_value_bh_fdr")
        _attach_scene_values(out, scene_values, scene_names)
        rows.append(out)
    slopes = {scene["name"]: probability_slopes(scene) for scene in scenes}
    slope_keys = sorted(set().union(*(scene_slopes.keys() for scene_slopes in slopes.values())))
    for descriptor, method in slope_keys:
        out = {
            "analysis": "winner_probability_slope",
            "descriptor": descriptor,
            "method": method,
            "left_winner": "",
            "right_winner": "",
            "interpretation_note": "Slope of binned winner probability vs descriptor value.",
        }
        scene_values = {}
        for scene_name in scene_names:
            value = slopes[scene_name].get((descriptor, method))
            scene_values[scene_name] = (value, _sign(value))
        _attach_scene_values(out, scene_values, scene_names)
        rows.append(out)
    medians = {scene["name"]: descriptor_medians(scene) for scene in scenes}
    for descriptor in DESCRIPTORS:
        out = {
            "analysis": "drk_median_minus_other_methods",
            "descriptor": descriptor,
            "method": "drk",
            "left_winner": "drk",
            "right_winner": "3dgs_ges_mean",
            "interpretation_note": "Negative values mean DRK-winning patches have lower descriptor median than the 3DGS/GES average.",
        }
        scene_values = {}
        for scene_name in scene_names:
            scene_medians = medians[scene_name]
            drk = scene_medians.get((descriptor, "drk"))
            others = [scene_medians.get((descriptor, method)) for method in ("3dgs", "ges")]
            others = [value for value in others if value is not None]
            value = None if drk is None or not others else drk - statistics.fmean(others)
            scene_values[scene_name] = (value, _sign(value))
        _attach_scene_values(out, scene_values, scene_names)
        rows.append(out)
    return rows


def build_question_summary(scene_rows: list[dict[str, Any]], effects: list[dict[str, Any]], scene_names: list[str]) -> dict[str, Any]:
    slope_rows = [row for row in effects if row["analysis"] == "winner_probability_slope"]
    ges_high = [row for row in slope_rows if row["method"] == "ges" and row["descriptor"] in HIGH_COMPLEXITY_DESCRIPTORS]
    drk_low = [row for row in effects if row["analysis"] == "drk_median_minus_other_methods" and row["descriptor"] in HIGH_COMPLEXITY_DESCRIPTORS]

    def all_direction(row: dict[str, Any], direction: str) -> bool:
        return all(row.get(f"{scene_name}_direction") == direction for scene_name in scene_names)

    comparisons = []
    for metric in ["3dgs_winner_fraction", "ges_winner_fraction", "drk_winner_fraction", "tie_fraction", "non_3dgs_winner_fraction", "oracle_improvement_mse_vs_3dgs", "oracle_relative_improvement_pct_vs_3dgs"]:
        values = [_as_float(row.get(metric)) for row in scene_rows]
        finite = [value for value in values if value is not None]
        comparisons.append({
            "metric": metric,
            **{scene_names[i]: values[i] for i in range(len(scene_names))},
            "min": None if not finite else min(finite),
            "max": None if not finite else max(finite),
            "range": None if len(finite) < 2 else max(finite) - min(finite),
        })
    return {
        "observational_warning": "Cross-scene patterns summarize independently trained systems and must not be interpreted as controlled causal kernel-family effects.",
        "same_direction_descriptor_winner_trends": [row for row in slope_rows if row["status"] == "same_direction"],
        "effects_that_reverse_or_disappear": [row for row in effects if row["status"] != "same_direction"],
        "ges_high_complexity_assessment": {
            "descriptors": ges_high,
            "all_high_complexity_slopes_positive_in_all_scenes": bool(ges_high) and all(all_direction(row, "positive") for row in ges_high),
        },
        "drk_lower_complexity_assessment": {
            "descriptors": drk_low,
            "all_high_complexity_median_differences_negative_in_all_scenes": bool(drk_low) and all(all_direction(row, "negative") for row in drk_low),
        },
        "winner_fraction_and_oracle_gain_similarity": comparisons,
    }


def _font(size: int = 14):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, y_label: str = "P(win)") -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(215, 215, 215), width=1)
    draw.line((x0, y1, x1, y1), fill=(40, 40, 40), width=2)
    draw.line((x0, y0, x0, y1), fill=(40, 40, 40), width=2)
    draw.text((x0, y0 - 24), title, fill=(30, 30, 30), font=_font(14))
    draw.text((x0 - 46, y0), y_label, fill=(80, 80, 80), font=_font(11))


def draw_probability_plots(out_dir: Path, scenes: list[dict[str, Any]]) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for scene in scenes:
        for row in scene["winner_probability"]:
            grouped[row["descriptor"]][scene["name"]].append(row)
    for descriptor in sorted(grouped):
        width = max(1200, 410 * len(scenes))
        height = 520
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        draw.text((28, 16), f"Winner probability by {descriptor}", fill=(20, 20, 20), font=_font(18))
        panel_w = (width - 110) // max(1, len(scenes))
        for i, scene in enumerate(scenes):
            rows = sorted(grouped[descriptor].get(scene["name"], []), key=lambda r: int(float(r.get("bin_index", 0))))
            x0 = 60 + i * panel_w
            box = (x0, 90, x0 + panel_w - 45, height - 80)
            _draw_axes(draw, box, scene["name"])
            xs_raw = [_as_float(row.get("bin_mid")) for row in rows]
            finite = [x for x in xs_raw if x is not None]
            if len(finite) < 2:
                continue
            xmin, xmax = min(finite), max(finite)
            if xmin == xmax:
                xmax = xmin + 1.0
            bx0, by0, bx1, by1 = box
            for method in METHODS:
                points = []
                for row in rows:
                    x = _as_float(row.get("bin_mid"))
                    y = _as_float(row.get(f"p_{method}_wins"))
                    if x is None or y is None:
                        continue
                    px = bx0 + int((x - xmin) / (xmax - xmin) * (bx1 - bx0))
                    py = by1 - int(max(0.0, min(1.0, y)) * (by1 - by0))
                    points.append((px, py))
                if len(points) >= 2:
                    draw.line(points, fill=COLORS[method], width=3)
                for point in points:
                    draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=COLORS[method])
            draw.text((bx0, by1 + 14), "descriptor quantile bins", fill=(80, 80, 80), font=_font(11))
        lx, ly = width - 145, 58
        for method in METHODS:
            draw.line((lx, ly, lx + 30, ly), fill=COLORS[method], width=4)
            draw.text((lx + 38, ly - 8), method, fill=(35, 35, 35), font=_font(12))
            ly += 24
        img.save(plot_dir / f"winner_probability_{descriptor}.png")


def draw_bar_plot(path: Path, title: str, rows: list[dict[str, Any]], metrics: list[str]) -> None:
    width, height = 1200, 540
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((28, 18), title, fill=(20, 20, 20), font=_font(18))
    x0, y0, x1, y1 = 70, 85, width - 40, height - 92
    draw.line((x0, y1, x1, y1), fill=(40, 40, 40), width=2)
    draw.line((x0, y0, x0, y1), fill=(40, 40, 40), width=2)
    values = [_as_float(row.get(metric)) or 0.0 for row in rows for metric in metrics]
    ymax = max(values + [1.0])
    ymax = 1.0 if ymax <= 1.0 else ymax * 1.1
    group_w = (x1 - x0) / max(1, len(metrics))
    bar_w = min(30, group_w / (len(rows) + 1.5))
    scene_colors = [(72, 116, 180), (238, 137, 73), (82, 156, 86), (170, 90, 150)]
    for mi, metric in enumerate(metrics):
        center = x0 + group_w * (mi + 0.5)
        draw.text((center - 55, y1 + 14), metric.replace("_", " ")[:22], fill=(50, 50, 50), font=_font(10))
        for si, row in enumerate(rows):
            value = _as_float(row.get(metric)) or 0.0
            bx0 = center - (len(rows) * bar_w) / 2 + si * bar_w
            bx1 = bx0 + bar_w * 0.8
            by0 = y1 - (value / ymax) * (y1 - y0)
            draw.rectangle((bx0, by0, bx1, y1), fill=scene_colors[si % len(scene_colors)])
    lx, ly = x1 - 155, y0 + 8
    for si, row in enumerate(rows):
        draw.rectangle((lx, ly, lx + 16, ly + 16), fill=scene_colors[si % len(scene_colors)])
        draw.text((lx + 24, ly - 2), str(row.get("scene", "scene")), fill=(35, 35, 35), font=_font(12))
        ly += 24
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", action="append", type=_parse_scene, default=None, help="Scene input as name=result_dir. Repeat for each scene.")
    parser.add_argument("--output-dir", default="results/cross_scene/garden_bicycle_room_p32", help="Directory for cross-scene outputs.")
    args = parser.parse_args()
    scene_args = args.scene or [
        ("garden", _resolve("results/garden/3dgs_vs_ges_vs_drk_p32")),
        ("bicycle", _resolve("results/bicycle/3dgs_vs_ges_vs_drk_p32")),
        ("room", _resolve("results/room/3dgs_vs_ges_vs_drk_p32")),
    ]
    if len(scene_args) < 2:
        parser.error("At least two scenes are required for cross-scene consistency comparisons")
    out_dir = _resolve(args.output_dir)
    scenes = [load_scene(name, root) for name, root in scene_args]
    scene_rows = [scene_summary_row(scene) for scene in scenes]
    effect_rows = build_effect_consistency(scenes)
    scene_names = [scene["name"] for scene in scenes]
    question_summary = build_question_summary(scene_rows, effect_rows, scene_names)

    _write_csv(out_dir / "cross_scene_summary.csv", scene_rows)
    _write_json(out_dir / "cross_scene_summary.json", {
        "scenes": scene_rows,
        "questions": question_summary,
        "inputs": {scene["name"]: scene["root"] for scene in scenes},
    })
    _write_csv(out_dir / "effect_consistency.csv", effect_rows)
    _write_json(out_dir / "effect_consistency.json", effect_rows)
    _write_csv(out_dir / "descriptor_summary_by_scene.csv", descriptor_summary_by_scene(scenes))
    _write_csv(out_dir / "winner_probability_by_descriptor_by_scene.csv", winner_probability_by_scene(scenes))
    draw_probability_plots(out_dir, scenes)
    draw_bar_plot(out_dir / "figures" / "winner_fraction_comparison.png", "Winner fractions by scene", scene_rows, ["3dgs_winner_fraction", "ges_winner_fraction", "drk_winner_fraction", "tie_fraction", "non_3dgs_winner_fraction"])
    draw_bar_plot(out_dir / "figures" / "oracle_mse_improvement_vs_3dgs.png", "Oracle MSE improvement vs 3DGS", scene_rows, ["oracle_improvement_mse_vs_3dgs"])
    draw_bar_plot(out_dir / "figures" / "oracle_relative_gain_vs_3dgs.png", "Oracle relative MSE improvement vs 3DGS", scene_rows, ["oracle_relative_improvement_pct_vs_3dgs"])
    print(f"Wrote cross-scene outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
