"""
Milestone 3: baseline supervised models.

Trains Logistic Regression, Random Forest, and XGBoost on the engineered
training set, each with class-weight-based imbalance handling, plus one
SMOTE + Logistic Regression variant for direct comparison. All models are
evaluated on the VALIDATION set only -- fraudTest.csv stays untouched
until a final model/approach is picked.

Usage:
    python src/build_features.py     # once, or after changing features.py
    python src/train_baselines.py
"""

import time
from pathlib import Path

import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from config import MODELS_DIR, RANDOM_SEED, REPORTS_DIR
from data_loader import load_engineered_train, load_engineered_val
from evaluation import full_evaluation, results_to_dataframe
from preprocessing import build_pipeline, build_preprocessor, get_X_y


def make_models(scale_pos_weight: float) -> dict:
    return {
        "logistic_regression": build_pipeline(
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_SEED,
            )
        ),
        "random_forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_SEED,
            )
        ),
        "xgboost": build_pipeline(
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                tree_method="hist",
                scale_pos_weight=scale_pos_weight,
                eval_metric="aucpr",
                n_jobs=-1,
                random_state=RANDOM_SEED,
            )
        ),
    }


def make_smote_logreg_pipeline(sampling_strategy: float = 0.1) -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("smote", SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_SEED)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
        ]
    )


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
    print(f"Train: {len(X_train):,} rows, {n_pos:,} fraud "
          f"({y_train.mean():.4%}). scale_pos_weight={scale_pos_weight:.1f}")
    print(f"Val:   {len(X_val):,} rows, {y_val.sum():,} fraud "
          f"({y_val.mean():.4%})")

    models = make_models(scale_pos_weight)
    models["logistic_regression_smote"] = make_smote_logreg_pipeline()

    results = []
    for name, pipeline in models.items():
        print(f"\nTraining {name}...")
        start = time.time()
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - start
        print(f"  fit time: {elapsed:.1f}s")

        y_proba = pipeline.predict_proba(X_val)[:, 1]
        metrics = full_evaluation(name, y_val, y_proba)
        metrics["fit_time_sec"] = round(elapsed, 1)
        results.append(metrics)

        model_path = Path(MODELS_DIR) / f"{name}.joblib"
        joblib.dump(pipeline, model_path)
        print(f"  saved -> {model_path}")
        print(
            f"  ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  "
            f"tuned F1={metrics['tuned_f1']:.4f} "
            f"(precision={metrics['tuned_precision']:.4f}, "
            f"recall={metrics['tuned_recall']:.4f}, "
            f"threshold={metrics['tuned_threshold']:.3f})"
        )

    results_df = results_to_dataframe(results)
    out_path = Path(REPORTS_DIR) / "milestone3_model_comparison.csv"
    results_df.to_csv(out_path, index=False)

    print("\n" + "=" * 80)
    print("Model comparison (sorted by PR-AUC):")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
