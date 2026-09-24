"""
Milestone 3b: hyperparameter tuning for the two leading baseline models
(XGBoost, Random Forest) from train_baselines.py.

Search strategy:

- **Search on a stratified subsample of train, score on the full val set**,
  via a single-fold `PredefinedSplit` -- consistent with how
  train_baselines.py evaluates (fit train / score val), so tuned and
  baseline numbers are directly comparable. A 25-candidate k-fold search
  directly on 1.1M rows would mean hundreds of full-scale fits.
- **Scored on PR-AUC**, matching the ranking metric from Milestone 3.
- **Final refit on the FULL training set** with the winning
  hyperparameters, then evaluated on val. Test stays untouched.

Usage:
    python src/tune_models.py
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier

from config import MODELS_DIR, RANDOM_SEED, REPORTS_DIR
from data_loader import load_engineered_train, load_engineered_val
from evaluation import full_evaluation, results_to_dataframe
from preprocessing import build_pipeline, get_X_y

SEARCH_SUBSAMPLE_SIZE = 200_000
N_ITER = 25


def make_search_split(X_train, y_train, X_val, y_val):
    if len(X_train) > SEARCH_SUBSAMPLE_SIZE:
        X_sub, _, y_sub, _ = train_test_split(
            X_train, y_train,
            train_size=SEARCH_SUBSAMPLE_SIZE,
            stratify=y_train,
            random_state=RANDOM_SEED,
        )
    else:
        X_sub, y_sub = X_train, y_train

    X_search = pd.concat([X_sub, X_val], ignore_index=True)
    y_search = pd.concat([y_sub, y_val], ignore_index=True)
    test_fold = np.array([-1] * len(X_sub) + [0] * len(X_val))
    ps = PredefinedSplit(test_fold)

    print(f"  search fits on {len(X_sub):,} train rows "
          f"(fraud rate {y_sub.mean():.4%}), scores on {len(X_val):,} val rows")
    return X_search, y_search, ps


def tune_xgboost(X_train, y_train, X_val, y_val, scale_pos_weight):
    print("\nTuning XGBoost...")
    X_search, y_search, ps = make_search_split(X_train, y_train, X_val, y_val)

    param_distributions = {
        "classifier__n_estimators": randint(150, 600),
        "classifier__max_depth": randint(3, 9),
        "classifier__learning_rate": loguniform(0.01, 0.3),
        "classifier__subsample": uniform(0.6, 0.4),
        "classifier__colsample_bytree": uniform(0.6, 0.4),
        "classifier__min_child_weight": randint(1, 8),
        "classifier__gamma": uniform(0, 0.5),
        "classifier__reg_alpha": loguniform(1e-3, 1.0),
        "classifier__reg_lambda": loguniform(0.5, 3.0),
        "classifier__scale_pos_weight": [
            scale_pos_weight * 0.5, scale_pos_weight, scale_pos_weight * 1.5,
        ],
    }

    base_pipeline = build_pipeline(
        XGBClassifier(
            tree_method="hist",
            eval_metric="aucpr",
            n_jobs=1,
            random_state=RANDOM_SEED,
        )
    )

    search = RandomizedSearchCV(
        base_pipeline,
        param_distributions=param_distributions,
        n_iter=N_ITER,
        scoring="average_precision",
        cv=ps,
        refit=False,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=1,
    )
    start = time.time()
    search.fit(X_search, y_search)
    print(f"  search time: {time.time() - start:.1f}s")
    print(f"  best val PR-AUC during search: {search.best_score_:.4f}")
    print(f"  best params: {search.best_params_}")

    best_classifier_params = {
        k.replace("classifier__", ""): v for k, v in search.best_params_.items()
    }
    final_pipeline = build_pipeline(
        XGBClassifier(
            tree_method="hist",
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=RANDOM_SEED,
            **best_classifier_params,
        )
    )
    print("  refitting best params on FULL training set...")
    start = time.time()
    final_pipeline.fit(X_train, y_train)
    print(f"  full refit time: {time.time() - start:.1f}s")

    return final_pipeline, search.best_params_


def tune_random_forest(X_train, y_train, X_val, y_val):
    print("\nTuning Random Forest...")
    X_search, y_search, ps = make_search_split(X_train, y_train, X_val, y_val)

    param_distributions = {
        "classifier__n_estimators": randint(150, 500),
        "classifier__max_depth": [None, 10, 15, 20, 30, 40],
        "classifier__min_samples_split": randint(2, 15),
        "classifier__min_samples_leaf": randint(1, 8),
        "classifier__max_features": ["sqrt", "log2", 0.3, 0.5],
        "classifier__class_weight": ["balanced", "balanced_subsample"],
    }

    base_pipeline = build_pipeline(
        RandomForestClassifier(n_jobs=1, random_state=RANDOM_SEED)
    )

    search = RandomizedSearchCV(
        base_pipeline,
        param_distributions=param_distributions,
        n_iter=N_ITER,
        scoring="average_precision",
        cv=ps,
        refit=False,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=1,
    )
    start = time.time()
    search.fit(X_search, y_search)
    print(f"  search time: {time.time() - start:.1f}s")
    print(f"  best val PR-AUC during search: {search.best_score_:.4f}")
    print(f"  best params: {search.best_params_}")

    best_classifier_params = {
        k.replace("classifier__", ""): v for k, v in search.best_params_.items()
    }
    final_pipeline = build_pipeline(
        RandomForestClassifier(n_jobs=-1, random_state=RANDOM_SEED, **best_classifier_params)
    )
    print("  refitting best params on FULL training set...")
    start = time.time()
    final_pipeline.fit(X_train, y_train)
    print(f"  full refit time: {time.time() - start:.1f}s")

    return final_pipeline, search.best_params_


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading engineered train/val splits...")
    train_df = load_engineered_train()
    val_df = load_engineered_val()
    X_train, y_train = get_X_y(train_df)
    X_val, y_val = get_X_y(val_df)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos

    results = []
    best_params_log = {}

    xgb_pipeline, xgb_params = tune_xgboost(X_train, y_train, X_val, y_val, scale_pos_weight)
    y_proba = xgb_pipeline.predict_proba(X_val)[:, 1]
    metrics = full_evaluation("xgboost_tuned", y_val, y_proba)
    results.append(metrics)
    best_params_log["xgboost_tuned"] = xgb_params
    joblib.dump(xgb_pipeline, Path(MODELS_DIR) / "xgboost_tuned.joblib")

    rf_pipeline, rf_params = tune_random_forest(X_train, y_train, X_val, y_val)
    y_proba = rf_pipeline.predict_proba(X_val)[:, 1]
    metrics = full_evaluation("random_forest_tuned", y_val, y_proba)
    results.append(metrics)
    best_params_log["random_forest_tuned"] = rf_params
    joblib.dump(rf_pipeline, Path(MODELS_DIR) / "random_forest_tuned.joblib")

    results_df = results_to_dataframe(results)
    out_path = Path(REPORTS_DIR) / "milestone3b_tuning_comparison.csv"
    results_df.to_csv(out_path, index=False)

    params_df = pd.DataFrame([
        {"model": name, **params} for name, params in best_params_log.items()
    ])
    params_path = Path(REPORTS_DIR) / "milestone3b_best_params.csv"
    params_df.to_csv(params_path, index=False)

    print("\n" + "=" * 80)
    print("Tuned model results (val set):")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print(f"\nSaved -> {out_path}")
    print(f"Best params saved -> {params_path}")


if __name__ == "__main__":
    main()
