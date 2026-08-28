#!/usr/bin/env python3
"""Inventory Phase-III synthetic pilot outputs without modifying them."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

METHODS = ("3dgs", "ges", "drk")
EXPECTED_ITERATIONS = {"3dgs": 30000, "ges": 40000, "drk": 35000}
FINAL_SUCCESSFUL_ARRAYS = {"3dgs": "1254695", "ges": "1254696", "drk": "1254698"}
SUPERSEDED_ARRAYS = ["1254668", "1254669", "1254675"]
LOADER_CHECK_JOB = "1254667"


@dataclass
class CandidateStatus:
    path: str
    status: str
    final_iteration: int | None
    render_count: int
    gt_count: int
    metrics_found: bool
    final_ply: str | None
    primitive_count: int | None
    notes: list[str]


@dataclass
class InventoryRecord:
    scene_id: str
    sweep_family: str
    level: str
    method: str
    expected_iteration: int
    expected_test_views: int
    selected_output: str | None
    status: str
    final_iteration: int | None
    render_count: int
    gt_count: int
    metrics_found: bool
    primitive_count: int | None
    notes: str
    stale_or_partial_candidate_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/synthetic/phase3_controlled_pilot"),
        help="Phase-III synthetic dataset root containing manifest.csv/json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/synthetic/phase3_controlled_pilot"),
        help="Root containing method output directories.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/synthetic/phase3_controlled_pilot"),
        help="Directory where inventory_status.{md,csv,json} will be written.",
    )
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser.parse_args()


def resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def rel(path: str | Path | None, root: Path) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def load_manifest(dataset_root: Path) -> list[dict[str, Any]]:
    csv_path = dataset_root / "manifest.csv"
    json_path = dataset_root / "manifest.json"
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("scenes", "records", "manifest"):
                if key in data and isinstance(data[key], list):
                    return data[key]
        if isinstance(data, list):
            return data
    raise FileNotFoundError(f"No manifest.csv or manifest.json found under {dataset_root}")


def as_int(row: dict[str, Any], key: str, default: int) -> int:
    value = row.get(key, default)
    if value in (None, ""):
        return default
    return int(value)


def iteration_from_path(path: Path | None) -> int | None:
    if path is None:
        return None
    match = re.search(r"iteration_(\d+)", str(path))
    return int(match.group(1)) if match else None


def read_ply_vertex_count(path: Path | None) -> int | None:
    if path is None:
        return None
    try:
        with path.open("rb") as f:
            for raw in f:
                line = raw.decode("ascii", errors="ignore").strip()
                if line.startswith("element vertex "):
                    return int(line.split()[-1])
                if line == "end_header":
                    break
    except OSError:
        return None
    return None


def choose_highest_iteration_dir(paths: list[Path]) -> Path | None:
    if not paths:
        return None

    def key(path: Path) -> tuple[int, float]:
        match = re.search(r"ours_(\d+)", path.name)
        iteration = int(match.group(1)) if match else -1
        return (iteration, path.stat().st_mtime if path.exists() else 0.0)

    return max(paths, key=key)


def method_render_counts(path: Path, method: str) -> tuple[int, int]:
    if method in {"3dgs", "ges"}:
        test_root = path / "test"
        test_dir = choose_highest_iteration_dir([p for p in test_root.glob("ours_*") if p.is_dir()])
        if test_dir is None:
            return 0, 0
        renders = len(list((test_dir / "renders").glob("*.png")))
        gt = len(list((test_dir / "gt").glob("*.png")))
        return renders, gt
    metric_dir = path / "metric" / "test"
    return len(list(metric_dir.glob("render_*.png"))), len(list(metric_dir.glob("gt_*.png")))


def metrics_present(path: Path, method: str) -> bool:
    if method in {"3dgs", "ges"}:
        return (path / "results.json").exists() and (path / "per_view.json").exists()
    return any((path / "metric").glob("test_*.txt"))


def candidate_status(path: Path, method: str, expected_iteration: int, expected_test_views: int) -> CandidateStatus:
    plys = list(path.glob("point_cloud/iteration_*/point_cloud.ply"))
    final_ply = max(plys, key=lambda p: iteration_from_path(p) or -1) if plys else None
    final_iteration = iteration_from_path(final_ply)
    render_count, gt_count = method_render_counts(path, method)
    found_metrics = metrics_present(path, method)
    primitive_count = read_ply_vertex_count(final_ply)

    notes: list[str] = []
    if final_iteration is None:
        notes.append("missing final PLY")
    elif final_iteration < expected_iteration:
        notes.append(f"final iteration {final_iteration} < expected {expected_iteration}")
    if render_count < expected_test_views:
        notes.append(f"renders {render_count}/{expected_test_views}")
    if gt_count < expected_test_views:
        notes.append(f"gt {gt_count}/{expected_test_views}")
    if not found_metrics:
        notes.append("missing metrics")

    complete = (
        final_iteration is not None
        and final_iteration >= expected_iteration
        and render_count >= expected_test_views
        and gt_count >= expected_test_views
        and found_metrics
    )
    status = "complete" if complete else "partial"
    return CandidateStatus(
        path=str(path),
        status=status,
        final_iteration=final_iteration,
        render_count=render_count,
        gt_count=gt_count,
        metrics_found=found_metrics,
        final_ply=str(final_ply) if final_ply else None,
        primitive_count=primitive_count,
        notes=notes,
    )


def find_candidates(method_root: Path, scene_id: str) -> list[Path]:
    if not method_root.exists():
        return []
    candidates: dict[str, Path] = {}
    for path in [method_root / scene_id, method_root / f"{scene_id}_DRK"]:
        if path.is_dir():
            candidates[str(path)] = path
    for path in method_root.glob(scene_id + "*"):
        if path.is_dir():
            candidates[str(path)] = path
    return list(candidates.values())


def candidate_sort_key(candidate: CandidateStatus) -> tuple[int, int, int, int, float]:
    path = Path(candidate.path)
    complete = 1 if candidate.status == "complete" else 0
    final_iteration = candidate.final_iteration if candidate.final_iteration is not None else -1
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (complete, final_iteration, candidate.render_count, candidate.gt_count, mtime)


def unknown_outputs(method_root: Path, scene_ids: set[str], root: Path) -> list[str]:
    if not method_root.exists():
        return []
    unknown: list[str] = []
    for path in method_root.iterdir():
        if not path.is_dir():
            continue
        if not any(path.name.startswith(scene_id) for scene_id in scene_ids):
            unknown.append(rel(path, root) or str(path))
    return sorted(unknown)


def write_markdown(path: Path, payload: dict[str, Any], records: list[InventoryRecord]) -> None:
    lines = [
        "# Phase-III Synthetic Pilot Inventory",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "This report is read-only with respect to training outputs. Stale or partial candidates are listed for inspection only.",
        "",
        "## Summary",
        "",
        f"- Expected method-scene outputs: {payload['summary']['total_expected']}",
        f"- Complete: {payload['summary']['complete_count']}",
        f"- Partial: {payload['summary']['partial_count']}",
        f"- Missing: {payload['summary']['missing_count']}",
        f"- Stale/partial extra candidates: {payload['summary']['stale_candidate_count']}",
        f"- Unknown output directories: {payload['summary']['unknown_output_count']}",
        "",
        "## Outputs",
        "",
        "| Scene | Method | Status | Iter | Renders | GT | Metrics | Primitives | Selected output | Notes |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for record in records:
        lines.append(
            "| {scene} | {method} | {status} | {iteration} | {renders} | {gt} | {metrics} | {primitives} | `{output}` | {notes} |".format(
                scene=record.scene_id,
                method=record.method.upper(),
                status=record.status.upper(),
                iteration=record.final_iteration if record.final_iteration is not None else "",
                renders=record.render_count,
                gt=record.gt_count,
                metrics="yes" if record.metrics_found else "no",
                primitives=record.primitive_count if record.primitive_count is not None else "",
                output=record.selected_output or "",
                notes=record.notes or "",
            )
        )

    stale = payload.get("stale_or_partial_outputs", [])
    if stale:
        lines.extend(["", "## Stale Or Partial Candidates", ""])
        for item in stale:
            lines.append(
                f"- `{item['path']}` ({item['method'].upper()}, {item['scene_id']}): {item['status']}; "
                f"iter={item.get('final_iteration')}, renders={item.get('render_count')}, gt={item.get('gt_count')}; "
                f"notes={'; '.join(item.get('notes', [])) or 'not selected'}"
            )

    unknown = payload.get("unknown_outputs", {})
    if any(unknown.values()):
        lines.extend(["", "## Unknown Output Directories", ""])
        for method, paths in unknown.items():
            for output in paths:
                lines.append(f"- {method.upper()}: `{output}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[InventoryRecord]) -> None:
    fieldnames = list(asdict(records[0]).keys()) if records else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    dataset_root = resolve_path(project_root, args.dataset_root)
    output_root = resolve_path(project_root, args.output_root)
    results_dir = resolve_path(project_root, args.results_dir)
    manifest = load_manifest(dataset_root)
    scene_ids = {str(row["scene_id"]) for row in manifest}

    records: list[InventoryRecord] = []
    stale_or_partial: list[dict[str, Any]] = []

    for row in manifest:
        scene_id = str(row["scene_id"])
        expected_test_views = as_int(row, "test_view_count", 8)
        for method in METHODS:
            expected_iteration = EXPECTED_ITERATIONS[method]
            method_root = output_root / method
            candidate_paths = find_candidates(method_root, scene_id)
            candidate_statuses = [candidate_status(p, method, expected_iteration, expected_test_views) for p in candidate_paths]
            selected = max(candidate_statuses, key=candidate_sort_key) if candidate_statuses else None

            status = selected.status if selected else "missing"
            notes = list(selected.notes) if selected else ["missing output directory"]
            if selected and selected.status != "complete":
                notes.append("best candidate is incomplete")

            unselected = [c for c in candidate_statuses if selected is None or c.path != selected.path]
            for candidate in unselected:
                item = asdict(candidate)
                item.update({"scene_id": scene_id, "method": method, "path": rel(candidate.path, project_root)})
                stale_or_partial.append(item)
            if selected and selected.status != "complete":
                item = asdict(selected)
                item.update({"scene_id": scene_id, "method": method, "path": rel(selected.path, project_root)})
                stale_or_partial.append(item)

            records.append(
                InventoryRecord(
                    scene_id=scene_id,
                    sweep_family=str(row.get("sweep_family", "")),
                    level=str(row.get("level", "")),
                    method=method,
                    expected_iteration=expected_iteration,
                    expected_test_views=expected_test_views,
                    selected_output=rel(selected.path, project_root) if selected else None,
                    status=status,
                    final_iteration=selected.final_iteration if selected else None,
                    render_count=selected.render_count if selected else 0,
                    gt_count=selected.gt_count if selected else 0,
                    metrics_found=selected.metrics_found if selected else False,
                    primitive_count=selected.primitive_count if selected else None,
                    notes="; ".join(notes),
                    stale_or_partial_candidate_count=len(unselected),
                )
            )

    unknown = {method: unknown_outputs(output_root / method, scene_ids, project_root) for method in METHODS}
    complete_count = sum(1 for r in records if r.status == "complete")
    partial_count = sum(1 for r in records if r.status == "partial")
    missing_count = sum(1 for r in records if r.status == "missing")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "dataset_root": rel(dataset_root, project_root),
        "manifest_path": rel((dataset_root / "manifest.csv") if (dataset_root / "manifest.csv").exists() else (dataset_root / "manifest.json"), project_root),
        "output_root": rel(output_root, project_root),
        "results_dir": rel(results_dir, project_root),
        "expected": {
            "scene_count": len(manifest),
            "method_count": len(METHODS),
            "methods": list(METHODS),
            "iterations": EXPECTED_ITERATIONS,
            "loader_check_job": LOADER_CHECK_JOB,
            "successful_array_jobs": FINAL_SUCCESSFUL_ARRAYS,
            "superseded_array_jobs": SUPERSEDED_ARRAYS,
        },
        "summary": {
            "total_expected": len(records),
            "complete_count": complete_count,
            "partial_count": partial_count,
            "missing_count": missing_count,
            "stale_candidate_count": len(stale_or_partial),
            "unknown_output_count": sum(len(paths) for paths in unknown.values()),
            "all_complete": complete_count == len(records),
        },
        "records": [asdict(record) for record in records],
        "stale_or_partial_outputs": stale_or_partial,
        "unknown_outputs": unknown,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "inventory_status.json"
    csv_path = results_dir / "inventory_status.csv"
    md_path = results_dir / "inventory_status.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, records)
    write_markdown(md_path, payload, records)

    print(f"Phase-III inventory complete: {complete_count}/{len(records)} complete")
    print(f"Wrote {rel(md_path, project_root)}")
    print(f"Wrote {rel(csv_path, project_root)}")
    print(f"Wrote {rel(json_path, project_root)}")
    if args.fail_on_incomplete and complete_count != len(records):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
