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
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            confusion_matrix,
            precision_recall_fscore_support,
            roc_auc_score,
        )
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
            LogisticRegression(max_iter=1000, class_weight="balanced"),
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
        all_true = []
        all_pred = []
        all_proba = []
        for view in views:
            train = row_views != view
            test = row_views == view
            if len(set(y[train].tolist())) < 2:
                continue
            model.fit(x[train], y[train])
            pred = model.predict(x[test])
            accuracies.append(float(accuracy_score(y[test], pred)))
            balanced.append(float(balanced_accuracy_score(y[test], pred)))
            all_true.extend(y[test].tolist())
            all_pred.extend(pred.tolist())
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(x[test])
                fold_proba = np.zeros((len(pred), len(methods)), dtype=np.float64)
                for class_index, class_label in enumerate(model.classes_):
                    fold_proba[:, int(class_label)] = proba[:, class_index]
                all_proba.append(fold_proba)
        if accuracies:
            y_true = np.asarray(all_true, dtype=np.int64)
            y_pred = np.asarray(all_pred, dtype=np.int64)
            labels = list(range(len(methods)))
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, labels=labels, zero_division=0
            )
            matrix = confusion_matrix(y_true, y_pred, labels=labels)

            results[f"{name}_view_cv_accuracy_mean"] = float(np.mean(accuracies))
            results[f"{name}_view_cv_accuracy_std"] = float(np.std(accuracies))
            results[f"{name}_view_cv_balanced_accuracy_mean"] = float(np.mean(balanced))
            results[f"{name}_view_cv_folds"] = int(len(accuracies))
            results[f"{name}_confusion_matrix"] = matrix.tolist()
            results[f"{name}_classes"] = methods
            results[f"{name}_per_class"] = {
                method: {
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                    "support": int(support[index]),
                }
                for index, method in enumerate(methods)
            }
            if all_proba and len(set(y_true.tolist())) > 1:
                proba = np.vstack(all_proba)
                try:
                    if len(methods) == 2:
                        auroc = roc_auc_score(y_true, proba[:, 1])
                    else:
                        auroc = roc_auc_score(y_true, proba, labels=labels, multi_class="ovr")
                    results[f"{name}_view_cv_auroc"] = float(auroc)
                except ValueError as exc:
                    results[f"{name}_view_cv_auroc_status"] = f"skipped: {exc}"
        else:
            results[f"{name}_status"] = "skipped: no fold had at least two train classes"

    return results
