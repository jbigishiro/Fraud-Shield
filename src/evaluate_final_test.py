"""
Final holdout evaluation against fraudTest.csv.

This is the ONE evaluation that matters for the report's headline numbers.
`data/processed/test.csv` (built from Kaggle's fraudTest.csv) has been
untouched since Milestone 1 -- never used for training, feature-scaling
fits, hyperparameter search, threshold tuning, or model selection. Every
one of those decisions was made using only train/validation. Scoring test
repeatedly and picking the best-looking run would quietly turn test into a
second validation set and invalidate exactly the guarantee a holdout is
supposed to provide -- so this script is meant to be run ONCE, after the
final model (the hybrid) was chosen using validation alone.

To guard against "just running it again to see," the script refuses to
overwrite an existing result unless you pass --force, and --force prints a
loud reminder of what re-running does to the holdout's validity. (Retraining
because of a genuine project change -- e.g. redoing feature engineering --
is a legitimate reason to re-run; casually rerunning to see if the number
moved is not.)

Reports, for every model already on the leaderboard (all four base models
individually, plus the hybrid): the same metrics used throughout the
project (ROC-AUC, PR-AUC, tuned-threshold precision/recall/F1), using each
model's OWN validation-tuned threshold -- test is scored, never re-tuned.

Usage:
    python src/evaluate_final_test.py
    python src/evaluate_final_test.py --force   # only if you know why
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from data_loader import load_engineered_test
from evaluation import evaluate_at_threshold, full_evaluation
from fnn_model import FraudFNN
from fnn_model import predict_proba as fnn_predict_proba
from lstm_model import FraudLSTM
from lstm_model import predict_proba as lstm_predict_proba
from preprocessing import ALL_FEATURES, TARGET_COL
from sequence_features import SEQ_FEATURES, STATIC_CATEGORICAL, STATIC_NUMERIC


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def require(path: Path, hint: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. {hint}")
    return path


def get_validation_tuned_thresholds() -> dict:
    """
    Pull each model's validation-tuned threshold from its own saved result
    file -- test is scored AT these thresholds, never re-tuned on test
    itself (that would leak test into the "tuned" operating point).
    """
    files = {
        "xgboost_tuned": "milestone3b_tuning_comparison.csv",
        "random_forest": "milestone3_model_comparison.csv",
        "fnn": "milestone4_fnn_result.csv",
        "lstm": "milestone4_lstm_result.csv",
        "hybrid": "milestone4_hybrid_result.csv",
    }
    thresholds = {}
    for model_name, filename in files.items():
        path = Path(REPORTS_DIR) / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        row = df[df["model"] == model_name]
        if row.empty and model_name in ("xgboost_tuned", "random_forest"):
            # milestone3/3b files hold multiple rows -- match by name, but
            # random_forest may be named "random_forest" or bare; fall back
            # to any row with a matching prefix.
            row = df[df["model"].str.contains(model_name, na=False)]
        if not row.empty:
            thresholds[model_name] = float(row.iloc[0]["tuned_threshold"])
    return thresholds


def get_all_probas(test_df, device):
    X_test = test_df[ALL_FEATURES].copy()
    y_test = test_df[TARGET_COL].to_numpy()
    probas = {}

    xgb_path = require(Path(MODELS_DIR) / "xgboost_tuned.joblib", "Run `python src/tune_models.py` first.")
    probas["xgboost_tuned"] = joblib.load(xgb_path).predict_proba(X_test)[:, 1]

    rf_path = require(Path(MODELS_DIR) / "random_forest.joblib", "Run `python src/train_baselines.py` first.")
    probas["random_forest"] = joblib.load(rf_path).predict_proba(X_test)[:, 1]

    fnn_pre_path = require(Path(MODELS_DIR) / "fnn_preprocessor.joblib", "Run `python src/train_fnn.py` first.")
    fnn_ckpt_path = require(Path(MODELS_DIR) / "fnn_best.pt", "Run `python src/train_fnn.py` first.")
    fnn_preprocessor = joblib.load(fnn_pre_path)
    X_test_fnn = fnn_preprocessor.transform(X_test).astype(np.float32)
    fnn_model = FraudFNN(input_dim=X_test_fnn.shape[1]).to(device)
    fnn_model.load_state_dict(torch.load(fnn_ckpt_path, map_location=device, weights_only=True))
    probas["fnn"] = fnn_predict_proba(fnn_model, torch.from_numpy(X_test_fnn), device)

    seq_path = require(PROCESSED_DIR / "test_sequences.npz", "Run `python src/build_sequences.py` first.")
    seq_scaler_path = require(Path(MODELS_DIR) / "lstm_seq_scaler.joblib", "Run `python src/train_lstm.py` first.")
    static_pre_path = require(
        Path(MODELS_DIR) / "lstm_static_preprocessor.joblib", "Run `python src/train_lstm.py` first."
    )
    lstm_ckpt_path = require(Path(MODELS_DIR) / "lstm_best.pt", "Run `python src/train_lstm.py` first.")

    seq_data = np.load(seq_path)
    X_seq_test = seq_data["X_seq"]
    n_test, seq_len, n_seq_features = X_seq_test.shape

    seq_scaler = joblib.load(seq_scaler_path)
    X_seq_test_scaled = seq_scaler.transform(
        X_seq_test.reshape(-1, n_seq_features)
    ).reshape(n_test, seq_len, n_seq_features).astype(np.float32)

    static_preprocessor = joblib.load(static_pre_path)
    X_static_test = static_preprocessor.transform(
        test_df[STATIC_NUMERIC + STATIC_CATEGORICAL]
    ).astype(np.float32)

    lstm_model = FraudLSTM(
        seq_input_dim=n_seq_features,
        static_input_dim=X_static_test.shape[1],
    ).to(device)
    lstm_model.load_state_dict(torch.load(lstm_ckpt_path, map_location=device, weights_only=True))
    probas["lstm"] = lstm_predict_proba(
        lstm_model, torch.from_numpy(X_seq_test_scaled), torch.from_numpy(X_static_test), device
    )

    hybrid_path = require(
        Path(MODELS_DIR) / "hybrid_meta_learner.joblib", "Run `python src/train_hybrid.py` first."
    )
    hybrid_bundle = joblib.load(hybrid_path)
    meta_X = np.column_stack([probas[name] for name in hybrid_bundle["base_model_order"]])
    probas["hybrid"] = hybrid_bundle["meta_learner"].predict_proba(meta_X)[:, 1]

    return probas, y_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if a final test evaluation already exists.",
    )
    args = parser.parse_args()

    result_path = Path(REPORTS_DIR) / "final_test_evaluation.csv"
    if result_path.exists() and not args.force:
        print(f"A final test evaluation already exists at {result_path}.")
        print(
            "\nfraudTest.csv is a holdout: it's meant to be scored ONCE, after the "
            "final model is chosen using validation alone. Re-running it to see if "
            "a number moves -- and reporting whichever run looks best -- defeats the "
            "purpose of holding it out in the first place."
        )
        print(
            "\nIf you have a genuine reason to re-run (e.g. you retrained the "
            "pipeline after a real change, not to 'try again'), pass --force."
        )
        return

    if args.force and result_path.exists():
        print(
            "!! --force: overwriting a previous final test evaluation. Make sure "
            "you can justify this in your report (what changed, and why the "
            "original run is no longer the one to cite).\n"
        )

    device = get_device()
    print(f"Using device: {device}")

    print("Loading fraudTest.csv (engineered) -- this holdout has not been "
          "touched since data_split.py...")
    test_df = load_engineered_test()
    print(f"Test set: {len(test_df):,} rows, {int(test_df[TARGET_COL].sum()):,} "
          f"fraud ({test_df[TARGET_COL].mean():.4%})")

    print("\nGenerating predictions from all five saved models "
          "(xgboost_tuned, random_forest, fnn, lstm, hybrid)...")
    probas, y_test = get_all_probas(test_df, device)

    val_thresholds = get_validation_tuned_thresholds()

    results = []
    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS (fraudTest.csv, scored once)")
    print("=" * 80)
    for model_name in ["hybrid", "lstm", "xgboost_tuned", "random_forest", "fnn"]:
        y_proba = probas[model_name]
        metrics = full_evaluation(model_name, y_test, y_proba)

        # Also report at the model's VALIDATION-chosen threshold (not
        # re-tuned here), since that's the threshold that would actually
        # ship -- full_evaluation's own "tuned" numbers re-tune on test,
        # which is useful as an upper bound but not the honest deployed
        # number.
        if model_name in val_thresholds:
            at_val_threshold = evaluate_at_threshold(y_test, y_proba, val_thresholds[model_name])
            metrics["shipped_threshold"] = val_thresholds[model_name]
            metrics["shipped_precision"] = at_val_threshold["precision"]
            metrics["shipped_recall"] = at_val_threshold["recall"]
            metrics["shipped_f1"] = at_val_threshold["f1"]
            metrics["shipped_false_positives"] = at_val_threshold["false_positives"]
            metrics["shipped_false_negatives"] = at_val_threshold["false_negatives"]

        results.append(metrics)
        print(f"\n{model_name}:")
        print(f"  ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}")
        if "shipped_threshold" in metrics:
            print(
                f"  At validation-tuned threshold ({metrics['shipped_threshold']:.4f}): "
                f"precision={metrics['shipped_precision']:.4f}  "
                f"recall={metrics['shipped_recall']:.4f}  "
                f"F1={metrics['shipped_f1']:.4f}  "
                f"(FP={metrics['shipped_false_positives']}, FN={metrics['shipped_false_negatives']})"
            )
        print(
            f"  Test-re-tuned threshold ({metrics['tuned_threshold']:.4f}, for reference "
            f"only -- not the deployed operating point): "
            f"precision={metrics['tuned_precision']:.4f}  recall={metrics['tuned_recall']:.4f}  "
            f"F1={metrics['tuned_f1']:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(result_path, index=False)

    print("\n" + "=" * 80)
    print(f"Saved -> {result_path}")
    print("=" * 80)

    # Sanity check against validation: a large drop would be worth
    # investigating (e.g. a time-related distribution shift between
    # fraudTrain.csv and fraudTest.csv), a close match is reassuring.
    leaderboard_path = Path(REPORTS_DIR) / "leaderboard.csv"
    if leaderboard_path.exists():
        val_leaderboard = pd.read_csv(leaderboard_path).set_index("model")["pr_auc"]
        print("\nValidation vs. test PR-AUC (a sizeable gap is worth flagging in the report):")
        for model_name in ["hybrid", "lstm", "xgboost_tuned", "random_forest", "fnn"]:
            if model_name in val_leaderboard.index:
                val_pr_auc = val_leaderboard[model_name]
                test_pr_auc = results_df.loc[results_df["model"] == model_name, "pr_auc"].iloc[0]
                gap = test_pr_auc - val_pr_auc
                print(f"  {model_name:>15}: val={val_pr_auc:.4f}  test={test_pr_auc:.4f}  gap={gap:+.4f}")


if __name__ == "__main__":
    main()
