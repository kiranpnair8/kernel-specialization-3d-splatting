from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.local_error.image_io import load_rgb
from analysis.local_error.patches import Patch
from analysis.winner_maps.visualize import save_winner_map
from common import atomic_json, fractions, load_config, load_error_bundle, oracle_metrics, resolve_project_path, summarize, winner_labels, write_csv


COLORS = {"3dgs": (45, 114, 210), "ges": (218, 87, 67), "drk": (62, 150, 91)}


def combo_record(values: np.ndarray, labels: list[str], tie: float) -> tuple[dict[str, float], np.ndarray]:
    winners, _ = winner_labels(values, labels, tie)
    stats = fractions(winners, labels)
    oracle = oracle_metrics(values, [0])
    result = {
        **stats,
        "non_3dgs_winner_fraction": stats["ges_winner_fraction"] + stats["drk_winner_fraction"],
        "oracle_mse": float(oracle["oracle_mse"]),
        "oracle_psnr": float(oracle["oracle_psnr"]),
        "oracle_gain_vs_3dgs_fraction": float(oracle["references"][0]["headroom_fraction"]),
        "ges_minus_3dgs_mse": float(np.mean(values[:, 1] - values[:, 0])),
        "drk_minus_3dgs_mse": float(np.mean(values[:, 2] - values[:, 0])),
        "ges_minus_drk_mse": float(np.mean(values[:, 1] - values[:, 2])),
    }
    return result, winners


def save_figures(result_root: Path, figure_root: Path, combination_rows: list[dict[str, object]], per_view_rows: list[dict[str, object]], errors: np.ndarray, run_ids: list[str], fair_rows: list[dict[str, object]]) -> None:
    figure_root.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    seed_values = sorted({int(row["3dgs_seed"]) for row in fair_rows})
    within_gain = []
    cross_gain = []
    for seed in seed_values:
        selected = [row for row in fair_rows if int(row["3dgs_seed"]) == seed]
        within_gain.append(100 * float(selected[0]["within_3dgs_gain_fraction_same_reference"]))
        cross_gain.append(100 * float(np.mean([row["cross_family_gain_fraction_same_reference"] for row in selected])))
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    x = np.arange(len(seed_values)); width = 0.36
    ax.bar(x - width / 2, within_gain, width, label="3DGS three-seed oracle", color="#2d72d2")
    ax.bar(x + width / 2, cross_gain, width, label="Cross-family oracle (mean)", color="#7a5195")
    ax.set_xticks(x, [f"3DGS seed {seed}" for seed in seed_values])
    ax.set_ylabel("Oracle headroom vs. same 3DGS reference (%)")
    ax.set_title("Room p32 same-reference control")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(figure_root / "oracle_gain_comparison.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    by_combo = {}
    for row in per_view_rows:
        by_combo.setdefault(row["combination"], []).append(100 * float(row["oracle_gain_vs_3dgs_fraction"]))
    ax.boxplot(list(by_combo.values()), showfliers=False)
    ax.set_xlabel("Cross-family seed combination")
    ax.set_ylabel("Per-view oracle gain vs. selected 3DGS (%)")
    fig.tight_layout(); fig.savefig(figure_root / "per_view_oracle_gain_distribution.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    positions = np.arange(3)
    for method, color in COLORS.items():
        means = [errors[:, run_ids.index(f"{method}_seed_{seed}")].mean() for seed in range(3)]
        ax.plot(positions, means, marker="o", label=method.upper(), color=np.asarray(color) / 255)
    ax.set_xticks(positions, ["Seed 0", "Seed 1", "Seed 2"])
    ax.set_ylabel("Mean patch MSE")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout(); fig.savefig(figure_root / "seed_local_error_variation.png", dpi=300); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze all 27 dependent cross-family seed combinations.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    result_root = resolve_project_path(config, str(config["result_root"]))
    figure_root = resolve_project_path(config, str(config["figure_root"]))
    errors, run_ids, views, metadata = load_error_bundle(result_root / "patch_errors.npz")
    seeds = [int(seed) for seed in config["seeds"]]
    combinations = list(product(seeds, repeat=3))
    if len(combinations) != 27 or len(set(combinations)) != 27:
        raise AssertionError("Cross-family combination enumeration is not exactly 3^3")
    tie = float(config["tie_threshold_mse"])
    labels = ["3dgs", "ges", "drk"]
    combination_rows: list[dict[str, object]] = []
    per_view_rows: list[dict[str, object]] = []
    canonical_winners = None
    for three_seed, ges_seed, drk_seed in combinations:
        combination = f"3dgs{three_seed}_ges{ges_seed}_drk{drk_seed}"
        indices = [run_ids.index(f"3dgs_seed_{three_seed}"), run_ids.index(f"ges_seed_{ges_seed}"), run_ids.index(f"drk_seed_{drk_seed}")]
        values = errors[:, indices]
        stats, winners = combo_record(values, labels, tie)
        row = {"combination": combination, "3dgs_seed": three_seed, "ges_seed": ges_seed, "drk_seed": drk_seed, **stats}
        combination_rows.append(row)
        if (three_seed, ges_seed, drk_seed) == (0, 0, 0):
            canonical_winners = winners
        for view in sorted(set(views)):
            mask = views == view
            view_stats, _ = combo_record(values[mask], labels, tie)
            per_view_rows.append({"combination": combination, "view": view, **view_stats})

    metric_names = [
        "3dgs_winner_fraction", "ges_winner_fraction", "drk_winner_fraction", "tie_fraction",
        "non_3dgs_winner_fraction", "oracle_mse", "oracle_psnr", "oracle_gain_vs_3dgs_fraction",
        "ges_minus_3dgs_mse", "drk_minus_3dgs_mse", "ges_minus_drk_mse",
    ]
    summary = {metric: summarize([float(row[metric]) for row in combination_rows]) for metric in metric_names}

    rng = np.random.default_rng(int(config["bootstrap_seed"]))
    unique_views = sorted(set(views))
    bootstrap = []
    for _ in range(int(config["bootstrap_replicates"])):
        sampled_views = rng.choice(unique_views, size=len(unique_views), replace=True)
        sampled_indices = np.concatenate([np.flatnonzero(views == view) for view in sampled_views])
        gains = []
        for three_seed, ges_seed, drk_seed in combinations:
            indices = [run_ids.index(f"3dgs_seed_{three_seed}"), run_ids.index(f"ges_seed_{ges_seed}"), run_ids.index(f"drk_seed_{drk_seed}")]
            gain = combo_record(errors[sampled_indices][:, indices], labels, tie)[0]["oracle_gain_vs_3dgs_fraction"]
            gains.append(gain)
        bootstrap.append(float(np.mean(gains)))
    summary["mean_oracle_gain_view_cluster_bootstrap_ci"] = {
        "low": float(np.quantile(bootstrap, 0.025)), "high": float(np.quantile(bootstrap, 0.975))
    }

    within_3dgs_indices = [run_ids.index(f"3dgs_seed_{seed}") for seed in seeds]
    within_3dgs_oracle = errors[:, within_3dgs_indices].min(axis=1)
    fair_rows: list[dict[str, object]] = []
    for three_seed in seeds:
        reference = errors[:, run_ids.index(f"3dgs_seed_{three_seed}")]
        within_mse = float(within_3dgs_oracle.mean())
        for ges_seed, drk_seed in product(seeds, repeat=2):
            cross = np.column_stack((reference, errors[:, run_ids.index(f"ges_seed_{ges_seed}")], errors[:, run_ids.index(f"drk_seed_{drk_seed}")])).min(axis=1)
            cross_mse = float(cross.mean())
            if cross_mse > float(reference.mean()) + 1e-15:
                raise AssertionError("Cross-family oracle is worse than its 3DGS reference")
            fair_rows.append({
                "3dgs_seed": three_seed, "ges_seed": ges_seed, "drk_seed": drk_seed,
                "reference_3dgs_mse": float(reference.mean()), "within_3dgs_oracle_mse": within_mse,
                "cross_family_oracle_mse": cross_mse,
                "within_3dgs_gain_fraction_same_reference": float((reference.mean() - within_mse) / reference.mean()),
                "cross_family_gain_fraction_same_reference": float((reference.mean() - cross_mse) / reference.mean()),
                "incremental_cross_family_gain_fraction_of_same_reference": float((within_mse - cross_mse) / reference.mean()),
            })

    write_csv(result_root / "cross_family_combinations.csv", combination_rows)
    write_csv(result_root / "cross_family_per_view.csv", per_view_rows)
    write_csv(result_root / "fair_oracle_comparison.csv", fair_rows)
    atomic_json(result_root / "cross_family_summary.json", {
        "combination_count": 27, "summary": summary,
        "statistical_note": "The 27 combinations reuse nine trained models and are not independent experiments. The reported CI resamples complete held-out views and recomputes the mean across all dependent combinations.",
    })

    diagnostic = str(config["diagnostic_view"])
    if diagnostic not in set(views): diagnostic = sorted(set(views))[0]
    mask = views == diagnostic
    with np.load(result_root / "patch_errors.npz", allow_pickle=False) as bundle:
        patches = [Patch(int(x), int(y), int(w), int(h)) for x, y, w, h in zip(bundle["x"][mask], bundle["y"][mask], bundle["width"][mask], bundle["height"][mask])]
    gt = load_rgb(resolve_project_path(config, str(config["reference_gt"])) / diagnostic)
    figure_root.mkdir(parents=True, exist_ok=True)
    save_winner_map(figure_root / f"cross_family_seed000_{Path(diagnostic).stem}_winner.png", gt.shape[:2], patches, canonical_winners[mask], COLORS)
    save_figures(result_root, figure_root, combination_rows, per_view_rows, errors, run_ids, fair_rows)
    print(f"Wrote 27-combination analysis to {result_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
