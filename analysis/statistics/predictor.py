from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Mapping

import numpy as np


FEATURE_COLUMNS = [
    "mean_gradient_magnitude",
    "edge_strength",
    "laplacian_energy",
    "local_variance",
    "high_frequency_energy",
    "entropy",
]


def evaluate_predictors(rows: Iterable[Mapping[str, object]], methods: List[str]) -> Dict[str, object]:
    decisive = [row for row in rows if row["winner"] in methods]
    if len(decisive) < 2:
        return {"status": "skipped", "reason": "not enough decisive patches"}

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - depends on local environment
        return {"status": "skipped", "reason": f"scikit-learn unavailable: {exc}"}

    views = sorted({str(row["view"]) for row in decisive})
    if len(views) < 2:
        return {"status": "skipped", "reason": "need at least two held-out views for view-level CV"}

    x = np.asarray([[float(row[col]) for col in FEATURE_COLUMNS] for row in decisive], dtype=np.float64)
    y = np.asarray([methods.index(str(row["winner"])) for row in decisive], dtype=np.int64)
    row_views = np.asarray([str(row["view"]) for row in decisive])

    class_counts = Counter(y.tolist())
    majority = max(class_counts.values()) / len(y)
    results: Dict[str, object] = {
        "status": "ok",
        "patch_count": int(len(y)),
        "view_count": int(len(views)),
        "majority_class_accuracy": float(majority),
    }

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="auto"),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=0,
            class_weight="balanced_subsample",
            min_samples_leaf=5,
        ),
    }

    for name, model in models.items():
        accuracies = []
        balanced = []
        for view in views:
            train = row_views != view
            test = row_views == view
            if len(set(y[train].tolist())) < 2:
                continue
            model.fit(x[train], y[train])
            pred = model.predict(x[test])
            accuracies.append(float(accuracy_score(y[test], pred)))
            balanced.append(float(balanced_accuracy_score(y[test], pred)))
        if accuracies:
            results[f"{name}_view_cv_accuracy_mean"] = float(np.mean(accuracies))
            results[f"{name}_view_cv_accuracy_std"] = float(np.std(accuracies))
            results[f"{name}_view_cv_balanced_accuracy_mean"] = float(np.mean(balanced))
            results[f"{name}_view_cv_folds"] = int(len(accuracies))
        else:
            results[f"{name}_status"] = "skipped: no fold had at least two train classes"

    return results
