"""
Shared evaluation utilities: with fraud at 0.58% of transactions, accuracy
is meaningless (predicting "never fraud" scores >99%), so every model here
is scored on precision/recall/F1/ROC-AUC/PR-AUC instead, at both the
default 0.5 threshold and a threshold chosen to maximize F1 on validation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_best_threshold(y_true, y_proba, metric="f1"):
    """
    Sweep thresholds from the precision-recall curve and return the one
    maximizing F1. Must be called on VALIDATION data only -- choosing a
    threshold using test data would leak test information into your
    deployed operating point.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    precisions, recalls = precisions[:-1], recalls[:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = np.where(
            (precisions + recalls) > 0,
            2 * precisions * recalls / (precisions + recalls),
            0,
        )

    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


def evaluate_at_threshold(y_true, y_proba, threshold) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def full_evaluation(model_name, y_true, y_proba) -> dict:
    """
    Compute the standard metric set for one model's predictions on one
    split: threshold-independent (ROC-AUC, PR-AUC) plus both a default-
    threshold and a tuned-threshold view of precision/recall/F1.
    """
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    default = evaluate_at_threshold(y_true, y_proba, 0.5)
    best_threshold, _ = find_best_threshold(y_true, y_proba)
    tuned = evaluate_at_threshold(y_true, y_proba, best_threshold)

    return {
        "model": model_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "default_threshold_precision": default["precision"],
        "default_threshold_recall": default["recall"],
        "default_threshold_f1": default["f1"],
        "tuned_threshold": tuned["threshold"],
        "tuned_precision": tuned["precision"],
        "tuned_recall": tuned["recall"],
        "tuned_f1": tuned["f1"],
        "tuned_false_positives": tuned["false_positives"],
        "tuned_false_negatives": tuned["false_negatives"],
    }


def results_to_dataframe(results: list) -> pd.DataFrame:
    df = pd.DataFrame(results)
    cols = [
        "model", "roc_auc", "pr_auc",
        "default_threshold_precision", "default_threshold_recall", "default_threshold_f1",
        "tuned_threshold", "tuned_precision", "tuned_recall", "tuned_f1",
        "tuned_false_positives", "tuned_false_negatives",
    ]
    return df[cols].sort_values("pr_auc", ascending=False).reset_index(drop=True)


def save_single_result(metrics: dict, path):
    """Save one model's metrics dict as a one-row CSV -- see build_leaderboard."""
    pd.DataFrame([metrics]).to_csv(path, index=False)


# Every training script's own result file, in one place -- add a new
# filename here (rather than re-merging CSVs inside each training script)
# whenever a new model type is added, so the leaderboard always reflects
# every model that's been trained so far, regardless of run order.
LEADERBOARD_SOURCE_FILES = [
    "milestone3_model_comparison.csv",   # logreg, logreg+smote, rf, xgboost baselines
    "milestone3b_tuning_comparison.csv",  # xgboost_tuned, random_forest_tuned
    "milestone4_fnn_result.csv",
    "milestone4_lstm_result.csv",
    "milestone4_hybrid_result.csv",
]


def build_leaderboard(reports_dir) -> pd.DataFrame:
    """
    Aggregate every model's results, from whichever of LEADERBOARD_SOURCE_FILES
    exist, into one sorted table. Safe to call after any training script --
    it only reads what's on disk, never assumes what's been run, and dedupes
    by model name (keeping the last-seen row) so re-running a script updates
    its entry without duplicating it.
    """
    reports_dir = Path(reports_dir)
    frames = []
    for filename in LEADERBOARD_SOURCE_FILES:
        path = reports_dir / filename
        if path.exists():
            frames.append(pd.read_csv(path))

    if not frames:
        return pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.drop_duplicates(subset="model", keep="last")
    return all_df.sort_values("pr_auc", ascending=False).reset_index(drop=True)
