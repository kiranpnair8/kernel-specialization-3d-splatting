from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for item in (SCRIPT_DIR, REPO_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from analysis.local_error.image_io import load_rgb
from analysis.local_error.patches import Patch
from analysis.structure_features.descriptors import image_patch_complexity_descriptors
from analysis.winner_maps.visualize import rasterize_patch_values, save_float_map, save_winner_map
from common import atomic_json, load_config, resolve, write_csv


METHODS = ("3dgs", "ges", "drk")
COLORS = {"3dgs": (45, 114, 210), "ges": (218, 87, 67), "drk": (62, 150, 91)}
CATEGORIES = ("low", "medium", "high")


def winner_labels(errors: np.ndarray, tie: float) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(errors, axis=1, kind="stable")
    margin = errors[np.arange(len(errors)), order[:, 1]] - errors[np.arange(len(errors)), order[:, 0]]
    winners = np.asarray(METHODS, dtype=object)[order[:, 0]]
    winners[margin < tie] = "tie"
    return winners, margin


def psnr(value: float) -> float:
    return float("inf") if value <= 1e-12 else float(-10.0 * math.log10(value))


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks / max(len(values) - 1, 1)


def complexity_categories(descriptors: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[float]]:
    score = np.mean(np.column_stack([percentile_ranks(values) for values in descriptors.values()]), axis=1)
    thresholds = np.quantile(score, [1 / 3, 2 / 3])
    categories = np.digitize(score, thresholds, right=True)
    if set(categories.tolist()) != {0, 1, 2}:
        raise ValueError("Structural tertiles failed to produce all categories")
    return score, categories, [float(value) for value in thresholds]


def bootstrap_summary(errors: np.ndarray, winners: np.ndarray, views: np.ndarray, mask: np.ndarray, replicates: int, seed: int) -> dict[str, float]:
    unique = np.asarray(sorted(set(views[mask])), dtype=str)
    counts = np.asarray([np.count_nonzero(mask & (views == view)) for view in unique], dtype=np.float64)
    method_sums = np.asarray([[errors[mask & (views == view), index].sum() for view in unique] for index in range(3)])
    oracle = errors.min(axis=1)
    oracle_sums = np.asarray([oracle[mask & (views == view)].sum() for view in unique])
    winner_counts = np.asarray([[np.count_nonzero((winners == label) & mask & (views == view)) for view in unique] for label in (*METHODS, "tie")])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(unique), size=(replicates, len(unique)))
    total = counts[samples].sum(axis=1)
    sampled_oracle = oracle_sums[samples].sum(axis=1)
    result = {}
    for index, method in enumerate(METHODS):
        sampled_method = method_sums[index][samples].sum(axis=1)
        gains = (sampled_method - sampled_oracle) / sampled_method
        result[f"oracle_gain_vs_{method}_ci_low"], result[f"oracle_gain_vs_{method}_ci_high"] = [float(value) for value in np.quantile(gains, [0.025, 0.975])]
    for index, label in enumerate((*METHODS, "tie")):
        fractions = winner_counts[index][samples].sum(axis=1) / total
        result[f"{label}_fraction_ci_low"], result[f"{label}_fraction_ci_high"] = [float(value) for value in np.quantile(fractions, [0.025, 0.975])]
    return result


def summarize(errors: np.ndarray, winners: np.ndarray, margin: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    values = errors[mask]
    selected_winners = winners[mask]
    oracle = values.min(axis=1)
    row: dict[str, float | int] = {
        "patch_count": int(len(values)), "decisive_patch_count": int(np.count_nonzero(selected_winners != "tie")),
        "tie_fraction": float(np.mean(selected_winners == "tie")), "mean_winner_margin_mse": float(margin[mask].mean()),
        "oracle_mse": float(oracle.mean()), "oracle_psnr": psnr(float(oracle.mean())),
    }
    for index, method in enumerate(METHODS):
        method_mse = float(values[:, index].mean())
        row[f"{method}_winner_fraction"] = float(np.mean(selected_winners == method))
        row[f"{method}_patch_mse"] = method_mse
        row[f"{method}_patch_psnr"] = psnr(method_mse)
        row[f"oracle_relative_improvement_pct_vs_{method}"] = float(100 * (method_mse - row["oracle_mse"]) / method_mse)
    row["non_3dgs_winner_fraction"] = float(row["ges_winner_fraction"] + row["drk_winner_fraction"])
    return row


def descriptors_for_scene(metadata: dict, views: np.ndarray, coordinates: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    output = {name: np.zeros(len(views), dtype=np.float64) for name in ("mean_gradient_magnitude", "edge_density", "local_texture_variance")}
    lookup = {record["view"]: record for record in metadata["views"]}
    for index, view in enumerate(sorted(set(views))):
        mask = views == view
        values = image_patch_complexity_descriptors(load_rgb(Path(lookup[view]["gt"])), *(coordinates[name][mask] for name in ("x", "y", "width", "height")), edge_threshold=0.05)
        for name in output:
            output[name][mask] = values[name]
        print(f"[structural] {index + 1}/{len(lookup)} {view}")
    if not all(np.isfinite(value).all() for value in output.values()):
        raise ValueError("Non-finite GT structural descriptor")
    return output


def load_bundle(root: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict]:
    with np.load(root / "patch_errors.npz", allow_pickle=False) as bundle:
        methods = tuple(str(value) for value in bundle["methods"])
        if methods != METHODS:
            raise ValueError(f"Expected method order {METHODS}, found {methods}")
        errors = np.asarray(bundle["errors"], dtype=np.float64)
        views = np.asarray(bundle["views"]).astype(str)
        coordinates = {name: np.asarray(bundle[name], dtype=np.int64) for name in ("x", "y", "width", "height")}
    metadata = json.loads((root / "patch_errors.json").read_text(encoding="utf-8"))
    return errors, views, coordinates, metadata


def render_scene_diagnostics(scene: str, root: Path, errors: np.ndarray, views: np.ndarray, coordinates: dict[str, np.ndarray], metadata: dict, winners: np.ndarray) -> dict:
    oracle = errors.min(axis=1)
    improvement = errors[:, 0] - oracle
    view_scores = {view: float(improvement[views == view].mean()) for view in sorted(set(views))}
    view = max(view_scores, key=view_scores.get)
    mask = views == view
    record = next(item for item in metadata["views"] if item["view"] == view)
    gt = load_rgb(Path(record["gt"]))
    patches = [Patch(int(x), int(y), int(w), int(h)) for x, y, w, h in zip(*(coordinates[name][mask] for name in ("x", "y", "width", "height")))]
    map_root = root / "maps"
    save_winner_map(map_root / f"{Path(view).stem}_winner.png", gt.shape[:2], patches, winners[mask], COLORS)
    for method_index, method in enumerate(METHODS):
        save_float_map(map_root / f"{Path(view).stem}_{method}_patch_mse.png", rasterize_patch_values(gt.shape[:2], patches, errors[mask, method_index]))
    gain_map = map_root / f"{Path(view).stem}_oracle_gain_vs_3dgs.png"
    save_float_map(gain_map, rasterize_patch_values(gt.shape[:2], patches, improvement[mask]))
    local_indices = np.flatnonzero(mask)
    best = local_indices[np.argmax(improvement[mask])]
    center_x = int(coordinates["x"][best] + coordinates["width"][best] // 2)
    center_y = int(coordinates["y"][best] + coordinates["height"][best] // 2)
    size = min(256, gt.shape[0], gt.shape[1])
    x0 = min(max(center_x - size // 2, 0), gt.shape[1] - size)
    y0 = min(max(center_y - size // 2, 0), gt.shape[0] - size)
    crop_paths = {}
    sources = {"gt": Path(record["gt"]), **{method: Path(record["renders"][method]) for method in METHODS}}
    crop_root = root / "representative_crops"
    crop_root.mkdir(parents=True, exist_ok=True)
    for label, source in sources.items():
        image = Image.open(source).convert("RGB")
        output = crop_root / f"{scene}_{label}.png"
        image.crop((x0, y0, x0 + size, y0 + size)).save(output)
        crop_paths[label] = str(output)
    winner_map = map_root / f"{Path(view).stem}_winner.png"
    winner_crop = crop_root / f"{scene}_winner.png"
    Image.open(winner_map).crop((x0, y0, x0 + size, y0 + size)).save(winner_crop)
    gain_crop = crop_root / f"{scene}_oracle_gain_vs_3dgs.png"
    Image.open(gain_map).crop((x0, y0, x0 + size, y0 + size)).save(gain_crop)
    return {"scene": scene, "view": view, "x0": x0, "y0": y0, "x1": x0 + size, "y1": y0 + size, "selection": "view and patch with largest oracle MSE reduction relative to 3DGS", "crops": crop_paths, "winner_map": str(winner_map), "winner_crop": str(winner_crop), "oracle_gain_map": str(gain_map), "oracle_gain_crop": str(gain_crop)}


def write_latex(result_root: Path, global_rows: list[dict], local_rows: list[dict], structural_rows: list[dict]) -> None:
    table_root = result_root / "tables"; table_root.mkdir(parents=True, exist_ok=True)
    labels = {"tandt_train": r"T\&T Train", "deepblending_drjohnson": "Dr Johnson", "3dgs": "3DGS", "ges": "GES", "drk": "DRK"}
    global_lines = [r"\begin{tabular}{llrrrr}", r"\toprule", r"Scene & Method & PSNR & SSIM & LPIPS & Primitives \\", r"\midrule"]
    for row in global_rows:
        global_lines.append(f"{labels[row['scene']]} & {labels[row['method']]} & {row['psnr']:.3f} & {row['ssim']:.4f} & {row['lpips']:.4f} & {int(row['primitive_count']):,} \\\\")
    global_lines += [r"\bottomrule", r"\end{tabular}"]
    (table_root / "global_quality.tex").write_text("\n".join(global_lines) + "\n", encoding="utf-8")
    local_lines = [r"\begin{tabular}{lrrrrr}", r"\toprule", r"Scene & 3DGS win & GES win & DRK win & Tie & Oracle gain vs. 3DGS \\", r"\midrule"]
    for row in local_rows:
        local_lines.append(f"{labels[row['scene']]} & {100*row['3dgs_winner_fraction']:.2f} & {100*row['ges_winner_fraction']:.2f} & {100*row['drk_winner_fraction']:.2f} & {100*row['tie_fraction']:.2f} & {row['oracle_relative_improvement_pct_vs_3dgs']:.2f} \\\\")
    local_lines += [r"\bottomrule", r"\end{tabular}"]
    (table_root / "local_specialization.tex").write_text("\n".join(local_lines) + "\n", encoding="utf-8")
    structural_lines = [r"\begin{tabular}{llrrrr}", r"\toprule", r"Scene & Complexity & Non-3DGS win & Gain vs. 3DGS & Gain vs. GES & Gain vs. DRK \\", r"\midrule"]
    for row in structural_rows:
        structural_lines.append(f"{labels[row['scene']]} & {row['category'].title()} & {100*row['non_3dgs_winner_fraction']:.2f} & {row['oracle_relative_improvement_pct_vs_3dgs']:.2f} & {row['oracle_relative_improvement_pct_vs_ges']:.2f} & {row['oracle_relative_improvement_pct_vs_drk']:.2f} \\\\")
    structural_lines += [r"\bottomrule", r"\end{tabular}"]
    (table_root / "structural_complexity.tex").write_text("\n".join(structural_lines) + "\n", encoding="utf-8")


def make_figures(config: dict, diagnostics: list[dict], structural_rows: list[dict]) -> None:
    import matplotlib.pyplot as plt
    figure_root = resolve(config, config["figure_root"]); figure_root.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(diagnostics), 6, figsize=(10.5, 3.8), squeeze=False)
    for row_index, item in enumerate(diagnostics):
        images = [item["crops"][key] for key in ("gt", *METHODS)] + [item["winner_crop"], item["oracle_gain_crop"]]
        for column, path in enumerate(images):
            axes[row_index, column].imshow(Image.open(path))
            axes[row_index, column].axis("off")
        axes[row_index, 0].set_ylabel(item["scene"].replace("_", "\n"), fontsize=8)
    for axis, title in zip(axes[0], ("GT", "3DGS", "GES", "DRK", "Winner map", "Oracle gain")):
        axis.set_title(title, fontsize=9)
    fig.tight_layout(pad=0.35)
    fig.savefig(figure_root / "independent_dataset_qualitative.pdf", bbox_inches="tight")
    fig.savefig(figure_root / "independent_dataset_qualitative.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, len(config["scenes"]), figsize=(7.0, 2.7), sharey=True, squeeze=False)
    for axis, scene in zip(axes[0], config["scenes"]):
        rows = [row for row in structural_rows if row["scene"] == scene]
        x = np.arange(3); width = 0.24
        for offset, method in enumerate(METHODS):
            axis.bar(x + (offset - 1) * width, [row[f"oracle_relative_improvement_pct_vs_{method}"] for row in rows], width, label=method.upper(), color=np.asarray(COLORS[method]) / 255)
        axis.set_xticks(x, [value.title() for value in CATEGORIES]); axis.set_title(config["scenes"][scene]["label"], fontsize=9); axis.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("Oracle MSE improvement (%)")
    fig.legend(loc="lower center", ncol=3, frameon=False); fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(figure_root / "independent_dataset_structural.pdf", bbox_inches="tight")
    fig.savefig(figure_root / "independent_dataset_structural.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize independent-scene local and GT-structural specialization.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config); result_root = resolve(config, config["result_root"])
    global_rows = json.loads((result_root / "global_quality.json").read_text(encoding="utf-8"))
    local_rows = []; structural_rows = []; per_view_rows = []; diagnostics = []; structural_metadata = {}
    for scene in config["scenes"]:
        scene_root = result_root / scene
        errors, views, coordinates, metadata = load_bundle(scene_root)
        winners, margin = winner_labels(errors, float(config["tie_threshold_mse"]))
        all_mask = np.ones(len(errors), dtype=bool)
        local = {"scene": scene, **summarize(errors, winners, margin, all_mask), **bootstrap_summary(errors, winners, views, all_mask, int(config["bootstrap_replicates"]), int(config["bootstrap_seed"]))}
        local_rows.append(local)
        for view in sorted(set(views)):
            per_view_rows.append({"scene": scene, "view": view, **summarize(errors, winners, margin, views == view)})
        descriptors = descriptors_for_scene(metadata, views, coordinates)
        score, categories, thresholds = complexity_categories(descriptors)
        structural_metadata[scene] = {"thresholds": thresholds, "edge_gradient_threshold": 0.05}
        for category_index, category in enumerate(CATEGORIES):
            mask = categories == category_index
            row = {"scene": scene, "category": category, "complexity_score_mean": float(score[mask].mean()), **{f"{name}_median": float(np.median(value[mask])) for name, value in descriptors.items()}, **summarize(errors, winners, margin, mask), **bootstrap_summary(errors, winners, views, mask, int(config["bootstrap_replicates"]), int(config["bootstrap_seed"]) + category_index + 1)}
            structural_rows.append(row)
        diagnostics.append(render_scene_diagnostics(scene, scene_root, errors, views, coordinates, metadata, winners))
    for name, rows in (("local_specialization", local_rows), ("local_per_view", per_view_rows), ("structural_complexity", structural_rows)):
        write_csv(result_root / f"{name}.csv", rows); atomic_json(result_root / f"{name}.json", rows)
    atomic_json(result_root / "analysis_summary.json", {"interpretation": "Observational independent-dataset validation; differences do not isolate kernel family causally.", "descriptor_definitions": {"mean_gradient_magnitude": "Mean GT grayscale gradient magnitude per patch.", "edge_density": "Fraction of GT patch pixels with gradient magnitude > 0.05.", "local_texture_variance": "GT grayscale variance from patch first and second moments.", "complexity_category": "Tertiles of the mean scene-wide percentile rank of the three descriptors."}, "structural_metadata": structural_metadata, "representatives": diagnostics})
    write_latex(result_root, global_rows, local_rows, structural_rows)
    if not args.no_figures:
        make_figures(config, diagnostics, structural_rows)
    print(f"Wrote independent-dataset summaries to {result_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
