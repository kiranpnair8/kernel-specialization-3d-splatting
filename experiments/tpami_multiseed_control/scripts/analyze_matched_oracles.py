from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for item in (SCRIPT_DIR, REPO_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from analysis.local_error.image_io import load_rgb
from analysis.structure_features.descriptors import image_patch_complexity_descriptors
from common import atomic_json, enumerate_pairs, load_bundle, load_config, pair_metrics, resolve, write_csv


GROUPS = ("within_3dgs", "within_ges", "cross_family")
CATEGORIES = ("low", "medium", "high")
COLORS = {"within_3dgs": "#2d72d2", "within_ges": "#da5743", "cross_family": "#3e965b"}


def pair_rows(errors: np.ndarray, run_ids: list[str], views: np.ndarray, tie: float, mask: np.ndarray | None = None) -> tuple[list[dict], list[dict]]:
    selected = np.ones(len(errors), dtype=bool) if mask is None else mask
    pairs = enumerate_pairs(run_ids)
    rows, per_view = [], []
    for pair_index, pair in enumerate(pairs):
        left_index, right_index = run_ids.index(pair["left"]), run_ids.index(pair["right"])
        row = {"pair_index": pair_index, **pair, **pair_metrics(errors[selected][:, [left_index, right_index]], str(pair["left"]), str(pair["right"]), tie)}
        rows.append(row)
        for view in sorted(set(views[selected])):
            view_mask = selected & (views == view)
            per_view.append({"pair_index": pair_index, **pair, "view": view, **pair_metrics(errors[view_mask][:, [left_index, right_index]], str(pair["left"]), str(pair["right"]), tie)})
    return rows, per_view


def group_summary(rows: list[dict]) -> list[dict]:
    output = []
    for group in GROUPS:
        selected = [row for row in rows if row["group"] == group]
        output.append({
            "group": group, "pair_count": len(selected),
            "oracle_mse_mean": float(np.mean([row["oracle_mse"] for row in selected])),
            "oracle_mse_sd": float(np.std([row["oracle_mse"] for row in selected], ddof=1)),
            "oracle_psnr_mean": float(np.mean([row["oracle_psnr"] for row in selected])),
            "gain_vs_better_single_mean": float(np.mean([row["oracle_gain_vs_better_single_fraction"] for row in selected])),
            "gain_vs_better_single_sd": float(np.std([row["oracle_gain_vs_better_single_fraction"] for row in selected], ddof=1)),
            "gain_vs_mean_single_mean": float(np.mean([row["oracle_gain_vs_mean_single_fraction"] for row in selected])),
            "left_winner_fraction_mean": float(np.mean([row["left_winner_fraction"] for row in selected])),
            "right_winner_fraction_mean": float(np.mean([row["right_winner_fraction"] for row in selected])),
            "tie_fraction_mean": float(np.mean([row["tie_fraction"] for row in selected])),
        })
    return output


def view_sufficient_statistics(errors: np.ndarray, run_ids: list[str], views: np.ndarray) -> tuple[np.ndarray, list[dict], dict[str, np.ndarray]]:
    unique_views = np.asarray(sorted(set(views)), dtype=str)
    pairs = enumerate_pairs(run_ids)
    statistics = {
        "count": np.zeros(len(unique_views), dtype=np.int64),
        "left_sum": np.zeros((len(pairs), len(unique_views))),
        "right_sum": np.zeros((len(pairs), len(unique_views))),
        "oracle_sum": np.zeros((len(pairs), len(unique_views))),
    }
    for view_index, view in enumerate(unique_views):
        mask = views == view
        statistics["count"][view_index] = mask.sum()
        for pair_index, pair in enumerate(pairs):
            left = errors[mask, run_ids.index(pair["left"])]
            right = errors[mask, run_ids.index(pair["right"])]
            statistics["left_sum"][pair_index, view_index] = left.sum()
            statistics["right_sum"][pair_index, view_index] = right.sum()
            statistics["oracle_sum"][pair_index, view_index] = np.minimum(left, right).sum()
    return unique_views, pairs, statistics


def bootstrap_groups(errors: np.ndarray, run_ids: list[str], views: np.ndarray, replicates: int, seed: int) -> tuple[list[dict], list[dict]]:
    unique_views, pairs, stats = view_sufficient_statistics(errors, run_ids, views)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(unique_views), size=(replicates, len(unique_views)))
    counts = stats["count"][samples].sum(axis=1)
    pair_gain = np.zeros((replicates, len(pairs))); pair_oracle = np.zeros_like(pair_gain)
    for pair_index in range(len(pairs)):
        left = stats["left_sum"][pair_index][samples].sum(axis=1)
        right = stats["right_sum"][pair_index][samples].sum(axis=1)
        oracle = stats["oracle_sum"][pair_index][samples].sum(axis=1)
        pair_gain[:, pair_index] = (np.minimum(left, right) - oracle) / np.minimum(left, right)
        pair_oracle[:, pair_index] = oracle / counts
    group_rows = []
    group_values = {}
    for group in GROUPS:
        indices = [index for index, pair in enumerate(pairs) if pair["group"] == group]
        gains = pair_gain[:, indices].mean(axis=1); oracle = pair_oracle[:, indices].mean(axis=1)
        group_values[group] = gains
        gain_ci = np.quantile(gains, [0.025, 0.975]); oracle_ci = np.quantile(oracle, [0.025, 0.975])
        group_rows.append({"group": group, "bootstrap_replicates": replicates, "gain_vs_better_single_ci_low": float(gain_ci[0]), "gain_vs_better_single_ci_high": float(gain_ci[1]), "oracle_mse_ci_low": float(oracle_ci[0]), "oracle_mse_ci_high": float(oracle_ci[1])})
    contrast_rows = []
    for baseline in ("within_3dgs", "within_ges"):
        difference = group_values["cross_family"] - group_values[baseline]
        ci = np.quantile(difference, [0.025, 0.975])
        contrast_rows.append({"contrast": f"cross_family_minus_{baseline}", "metric": "mean oracle gain vs better member", "difference_ci_low": float(ci[0]), "difference_ci_high": float(ci[1]), "bootstrap_replicates": replicates})
    return group_rows, contrast_rows


def descriptive_contrasts(rows: list[dict]) -> list[dict]:
    output = []
    cross = np.asarray([row["oracle_gain_vs_better_single_fraction"] for row in rows if row["group"] == "cross_family"])
    for baseline in ("within_3dgs", "within_ges"):
        within = np.asarray([row["oracle_gain_vs_better_single_fraction"] for row in rows if row["group"] == baseline])
        pooled_variance = ((len(cross) - 1) * cross.var(ddof=1) + (len(within) - 1) * within.var(ddof=1)) / (len(cross) + len(within) - 2)
        effect = float((cross.mean() - within.mean()) / math.sqrt(pooled_variance)) if pooled_variance > 0 else 0.0
        output.append({
            "contrast": f"cross_family_minus_{baseline}", "cross_mean": float(cross.mean()), "within_mean": float(within.mean()),
            "difference": float(cross.mean() - within.mean()), "standardized_mean_difference": effect,
            "effect_size_note": "descriptive pooled-SD standardization over dependent seed-pair combinations",
        })
    return output


def correlations(errors: np.ndarray, run_ids: list[str], views: np.ndarray) -> list[dict]:
    rows = []
    for pair in enumerate_pairs(run_ids):
        left, right = errors[:, run_ids.index(pair["left"])], errors[:, run_ids.index(pair["right"])]
        per_view = [float(np.corrcoef(left[views == view], right[views == view])[0, 1]) for view in sorted(set(views))]
        rows.append({**pair, "pearson_patch_error_correlation": float(np.corrcoef(left, right)[0, 1]), "per_view_correlation_mean": float(np.nanmean(per_view)), "per_view_correlation_sd": float(np.nanstd(per_view, ddof=1))})
    return rows


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable"); ranks = np.empty(len(values), dtype=np.float64); start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1); start = end
    return ranks / max(len(values) - 1, 1)


def structural_categories(metadata: dict, views: np.ndarray, coordinates: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray, list[float]]:
    descriptors = {name: np.zeros(len(views)) for name in ("mean_gradient_magnitude", "edge_density", "local_texture_variance")}
    lookup = {record["view"]: record for record in metadata["views"]}
    for view in sorted(set(views)):
        mask = views == view
        values = image_patch_complexity_descriptors(load_rgb(Path(lookup[view]["gt"])), *(coordinates[name][mask] for name in ("x", "y", "width", "height")), edge_threshold=0.05)
        for name in descriptors:
            descriptors[name][mask] = values[name]
    score = np.mean(np.column_stack([percentile_ranks(value) for value in descriptors.values()]), axis=1)
    thresholds = np.quantile(score, [1 / 3, 2 / 3]); categories = np.digitize(score, thresholds, right=True)
    return descriptors, categories, [float(value) for value in thresholds]


def sensitivity_analysis(config: dict, result_root: Path) -> list[dict]:
    rows = []
    for patch_size in config["sensitivity"]["patch_sizes"]:
        path = result_root / "patch_errors.npz" if int(patch_size) == int(config["patch_size"]) else result_root / "sensitivity" / f"patch_errors_p{patch_size}.npz"
        errors, run_ids, views, _, _ = load_bundle(path)
        for tie in config["sensitivity"]["tie_thresholds_mse"]:
            pairs, _ = pair_rows(errors, run_ids, views, float(tie))
            for group in group_summary(pairs):
                rows.append({"patch_size": patch_size, "stride": int(patch_size) // 2, "tie_threshold_mse": tie, **group})
    return rows


def write_latex(result_root: Path, groups: list[dict], bootstrap: list[dict], contrasts: list[dict], inventory: list[dict], structural_groups: list[dict]) -> None:
    bootstrap_by_group = {row["group"]: row for row in bootstrap}
    names = {"within_3dgs": "3DGS+3DGS", "within_ges": "GES+GES", "cross_family": "3DGS+GES"}
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{lrrr}", r"\toprule", r"Pair class & Pairs & Oracle MSE & Gain vs. better member (\%) \\", r"\midrule"]
    for row in groups:
        ci = bootstrap_by_group[row["group"]]
        lines.append(f"{names[row['group']]} & {row['pair_count']} & {row['oracle_mse_mean']:.6f} & {100*row['gain_vs_better_single_mean']:.2f} [{100*ci['gain_vs_better_single_ci_low']:.2f}, {100*ci['gain_vs_better_single_ci_high']:.2f}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\caption{Matched-size two-model oracle controls on Dr Johnson. Values summarize all seed pairs; intervals resample complete held-out views. Cross-family and within-family pools always contain two models.}", r"\label{tab:tpami_matched_size_multiseed}", r"\end{table}"]
    (result_root / "matched_size_oracle_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    contrast_lines = [r"\begin{tabular}{lrr}", r"\toprule", r"Contrast & Gain difference & Descriptive effect size \\", r"\midrule"]
    for row in contrasts:
        label = row["contrast"].replace("_", r"\_")
        contrast_lines.append(f"{label} & {100*row['difference']:.2f} pp & {row['standardized_mean_difference']:.2f} \\\\")
    contrast_lines += [r"\bottomrule", r"\end{tabular}"]
    (result_root / "matched_size_contrasts_table.tex").write_text("\n".join(contrast_lines) + "\n", encoding="utf-8")
    quality = [r"\begin{tabular}{llrrrr}", r"\toprule", r"Family & Seed & PSNR & SSIM & LPIPS & Primitives \\", r"\midrule"]
    for row in inventory:
        quality.append(f"{str(row['method']).upper()} & {row['seed']} & {row['psnr']:.3f} & {row['ssim']:.4f} & {row['lpips']:.4f} & {int(row['primitive_count']):,} \\\\")
    quality += [r"\bottomrule", r"\end{tabular}"]
    (result_root / "model_quality_table.tex").write_text("\n".join(quality) + "\n", encoding="utf-8")
    structural = [r"\begin{tabular}{llrr}", r"\toprule", r"Complexity & Pair class & Oracle MSE & Gain vs. better member (\%) \\", r"\midrule"]
    names = {"within_3dgs": "3DGS+3DGS", "within_ges": "GES+GES", "cross_family": "3DGS+GES"}
    for row in structural_groups:
        structural.append(f"{row['category'].title()} & {names[row['group']]} & {row['oracle_mse_mean']:.6f} & {100*row['gain_vs_better_single_mean']:.2f} \\\\")
    structural += [r"\bottomrule", r"\end{tabular}"]
    (result_root / "structural_complexity_table.tex").write_text("\n".join(structural) + "\n", encoding="utf-8")


def make_figures(result_root: Path, pair_records: list[dict], per_view: list[dict], structural_group_rows: list[dict]) -> None:
    import matplotlib.pyplot as plt
    figure_root = result_root / "figures"; figure_root.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5.8, 3.2))
    for x, group in enumerate(GROUPS):
        values = [100 * row["oracle_gain_vs_better_single_fraction"] for row in pair_records if row["group"] == group]
        jitter = np.linspace(-0.08, 0.08, len(values)); axis.scatter(x + jitter, values, color=COLORS[group], s=28, zorder=3)
        axis.hlines(np.mean(values), x - 0.22, x + 0.22, color="black", linewidth=1.5)
    axis.set_xticks(range(3), ["3DGS+3DGS", "GES+GES", "3DGS+GES"]); axis.set_ylabel("Oracle gain vs. better member (%)"); axis.grid(axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(figure_root / "matched_size_pair_distributions.pdf"); fig.savefig(figure_root / "matched_size_pair_distributions.png", dpi=300); plt.close(fig)
    fig, axis = plt.subplots(figsize=(6.5, 3.1))
    views = sorted({row["view"] for row in per_view})
    for group in GROUPS:
        values = [100 * np.mean([row["oracle_gain_vs_better_single_fraction"] for row in per_view if row["group"] == group and row["view"] == view]) for view in views]
        axis.plot(range(len(views)), values, label=group.replace("_", " "), color=COLORS[group], linewidth=1.2)
    axis.axhline(0, color="black", linewidth=0.8); axis.set_xlabel("Held-out view (sorted)"); axis.set_ylabel("Mean pair oracle gain (%)"); axis.legend(frameon=False, ncol=3, fontsize=7); axis.grid(alpha=0.15)
    fig.tight_layout(); fig.savefig(figure_root / "per_view_oracle_gains.pdf"); fig.savefig(figure_root / "per_view_oracle_gains.png", dpi=300); plt.close(fig)
    fig, axis = plt.subplots(figsize=(5.8, 3.2)); x = np.arange(3)
    for group in GROUPS:
        rows = [row for row in structural_group_rows if row["group"] == group]
        axis.plot(x, [100 * row["gain_vs_better_single_mean"] for row in rows], marker="o", label=group.replace("_", " "), color=COLORS[group])
    axis.set_xticks(x, [value.title() for value in CATEGORIES]); axis.set_xlabel("GT structural complexity"); axis.set_ylabel("Mean oracle gain (%)"); axis.legend(frameon=False, fontsize=7); axis.grid(alpha=0.15)
    fig.tight_layout(); fig.savefig(figure_root / "structural_complexity_controls.pdf"); fig.savefig(figure_root / "structural_complexity_controls.png", dpi=300); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze matched-size within- and cross-family Dr Johnson seed pairs.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(); config = load_config(args.config); result_root = resolve(config, config["result_root"])
    errors, run_ids, views, coordinates, metadata = load_bundle(result_root / "patch_errors.npz")
    inventory = json.loads((result_root / "model_inventory.json").read_text(encoding="utf-8"))
    tie = float(config["tie_threshold_mse"])
    pair_records, per_view = pair_rows(errors, run_ids, views, tie)
    groups = group_summary(pair_records)
    bootstrap, bootstrap_contrasts = bootstrap_groups(errors, run_ids, views, int(config["bootstrap_replicates"]), int(config["bootstrap_seed"]))
    contrasts = descriptive_contrasts(pair_records)
    for row in contrasts:
        row.update(next(item for item in bootstrap_contrasts if item["contrast"] == row["contrast"]))
    correlation_rows = correlations(errors, run_ids, views)
    descriptors, category_index, thresholds = structural_categories(metadata, views, coordinates)
    structural_pairs, structural_groups = [], []
    for category, label in enumerate(CATEGORIES):
        rows, _ = pair_rows(errors, run_ids, views, tie, category_index == category)
        for row in rows:
            structural_pairs.append({"category": label, **row})
        for row in group_summary(rows):
            structural_groups.append({"category": label, **row})
    sensitivity = sensitivity_analysis(config, result_root)
    outputs = {
        "pairwise_oracle_summary": pair_records, "pairwise_oracle_per_view": per_view,
        "group_summary": groups, "bootstrap_summary": bootstrap, "group_contrasts": contrasts,
        "pairwise_error_correlations": correlation_rows, "structural_pair_summary": structural_pairs,
        "structural_group_summary": structural_groups, "sensitivity_summary": sensitivity,
    }
    for name, rows in outputs.items():
        write_csv(result_root / f"{name}.csv", rows); atomic_json(result_root / f"{name}.json", rows)
    atomic_json(result_root / "analysis_summary.json", {
        "scientific_constraint": "All oracle comparisons use exactly two models. Cross-family superiority is not assumed; combination rows are dependent because seeds are reused.",
        "primary_reference": "For each pair, the lower scene-level MSE of its two individual members; bootstrap replicates reselect the better member after aggregating sampled views.",
        "pair_counts": {"within_3dgs": 3, "within_ges": 3, "cross_family": 9},
        "bootstrap": {"unit": "held-out view", "replicates": config["bootstrap_replicates"], "seed": config["bootstrap_seed"], "patch_weighting": "preserved by summed MSE/count sufficient statistics"},
        "structural_descriptors": {"names": list(descriptors), "edge_threshold": 0.05, "category_thresholds": thresholds, "categories": "tertiles of mean scene-wide descriptor percentile rank"},
        "combination_dependence": "The 15 pair records are descriptive dependent combinations, not independent experimental replicates.",
    })
    write_latex(result_root, groups, bootstrap, contrasts, inventory, structural_groups)
    if not args.no_figures:
        make_figures(result_root, pair_records, per_view, structural_groups)
    print(f"Wrote matched-size multi-seed analysis to {result_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
