"""
Utility functions for Modeling (Baseline & Advanced).
Used in Phase 3: 04_model_baseline.ipynb & 05_model_advanced.ipynb
"""

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold


def evaluate_model_cv(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
) -> dict:
    """
    Evaluate one classifier using the provided CV splitter.

    Parameters
    ----------
    estimator : sklearn-compatible estimator
        Any classifier implementing fit/predict.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target vector.
    cv : StratifiedKFold
        Cross-validation splitter.

    Returns
    -------
    dict
        Accuracy, F1, precision, recall per fold and aggregated mean/std.
    """
    metrics_per_fold = {
        "accuracy": [],
        "f1": [],
        "precision": [],
        "recall": [],
    }

    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        model = clone(estimator)
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_valid)

        metrics_per_fold["accuracy"].append(accuracy_score(y_valid, preds))
        metrics_per_fold["f1"].append(f1_score(y_valid, preds, zero_division=0))
        metrics_per_fold["precision"].append(
            precision_score(y_valid, preds, zero_division=0)
        )
        metrics_per_fold["recall"].append(recall_score(y_valid, preds, zero_division=0))

    results = {
        "fold_metrics": {
            metric: [float(v) for v in values]
            for metric, values in metrics_per_fold.items()
        }
    }

    for metric, values in metrics_per_fold.items():
        arr = np.asarray(values, dtype=float)
        results[f"{metric}_mean"] = float(arr.mean())
        results[f"{metric}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    return results


def manual_model_search(
    model_factory: Callable[..., object],
    param_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    sort_metric: str = "accuracy_mean",
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Run a small manual hyperparameter search and keep a tidy results table.

    Parameters
    ----------
    model_factory : Callable[..., object]
        Callable that receives keyword parameters and returns an estimator.
    param_grid : dict
        Hyperparameter grid compatible with sklearn's ParameterGrid.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target vector.
    cv : StratifiedKFold
        Cross-validation splitter.
    sort_metric : str
        Metric used to rank candidate models.

    Returns
    -------
    tuple[pd.DataFrame, dict, dict]
        Results table, best params, best metrics.
    """
    rows = []
    best_params = None
    best_metrics = None
    best_score = -np.inf

    for trial_idx, params in enumerate(ParameterGrid(param_grid), start=1):
        estimator = model_factory(**params)
        metrics = evaluate_model_cv(estimator=estimator, X=X, y=y, cv=cv)

        row = {"trial": trial_idx, **params}
        for key, value in metrics.items():
            if key == "fold_metrics":
                continue
            row[key] = value
        row["accuracy_folds"] = metrics["fold_metrics"]["accuracy"]
        row["f1_folds"] = metrics["fold_metrics"]["f1"]
        rows.append(row)

        score = metrics[sort_metric]
        if score > best_score:
            best_score = score
            best_params = dict(params)
            best_metrics = metrics

    results_df = pd.DataFrame(rows).sort_values(
        by=[sort_metric, "f1_mean"], ascending=False
    )
    results_df = results_df.reset_index(drop=True)
    return results_df, best_params, best_metrics


def format_cv_summary(model_name: str, metrics: dict) -> pd.DataFrame:
    """
    Convert the aggregated CV metrics of one model into a one-row DataFrame.
    """
    return pd.DataFrame(
        [
            {
                "Model": model_name,
                "Accuracy (CV)": (
                    f"{metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}"
                ),
                "F1": f"{metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}",
                "Precision": (
                    f"{metrics['precision_mean']:.4f} ± {metrics['precision_std']:.4f}"
                ),
                "Recall": (
                    f"{metrics['recall_mean']:.4f} ± {metrics['recall_std']:.4f}"
                ),
            }
        ]
    )

# ==========================================
# ZONE: THÀNH
# ==========================================




# ==========================================
# ZONE: TUẤN
# ==========================================
# Ghi chú: Các class / hàm train Logistic Regression, Decision Tree




# ==========================================
# ZONE: KIỆT
# ==========================================
# Ghi chú: Các class / hàm train Random Forest, SVM




# ==========================================
# ZONE: ĐÀM ĐẠT
# ==========================================
# Ghi chú: Các class / hàm train XGBoost, Tuning params




# ==========================================
# ZONE: TÂM
# ==========================================
# Ghi chú: Các class / hàm train LightGBM, Tuning params

