"""
Milestone 4c: hybrid framework -- stacking ensemble combining the leading
supervised model (XGBoost, tuned), Random Forest, the FNN, and the LSTM.

This is the project brief's "hybrid detection framework combining
supervised learning and deep learning methods into a unified system."
A meta-learner (Logistic Regression) is trained on the four base models'
predicted probabilities -- letting it learn how much to trust each model,
rather than averaging them blindly.

Evaluation design, and why: the base models were already selected/tuned
using validation, so fitting AND scoring the meta-learner on validation
too would let it overfit trivially (4 features, ~194k rows -- the
meta-learner could memorize validation-specific noise). Instead:

  - 5-fold cross-validation WITHIN validation generates out-of-fold (OOF)
    meta-predictions: each validation row's hybrid prediction comes from a
    meta-learner that never saw that row during its own fit. This gives an
    honest hybrid score on the FULL validation set -- directly comparable
    to every other row in the leaderboard, unlike carving off yet another
    holdout slice (which would only be comparable to itself).
  - A separate FINAL meta-learner is then fit on all of validation's
    meta-features (for actual downstream use -- e.g. scoring test later,
    or the web app) and saved, but its own predictions on validation are
    NOT reported as a metric (that would be circular).

Base models are used as already trained/saved by the earlier scripts --
this script doesn't retrain them, only combines their predictions.

Usage (run after train_baselines.py, tune_models.py, train_fnn.py,
train_lstm.py have all been run at least once):
    python src/train_hybrid.py
"""

from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from config import MODELS_DIR, PROCESSED_DIR, RANDOM_SEED, REPORTS_DIR
from data_loader import load_engineered_val
from evaluation import build_leaderboard, full_evaluation, save_single_result
from fnn_model import FraudFNN
from fnn_model import predict_proba as fnn_predict_proba
from lstm_model import FraudLSTM
from lstm_model import predict_proba as lstm_predict_proba
from preprocessing import get_X_y
from sequence_features import SEQ_FEATURES, STATIC_CATEGORICAL, STATIC_NUMERIC

N_FOLDS = 5


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def require(path: Path, hint: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. {hint}")
    return path


def get_base_model_probas(val_df, device):
    """
    Returns a dict {model_name: proba_array} for all four base models,
    each evaluated on the full validation set, in the same row order as
    val_df (and as the cached val_sequences.npz, which was built from the
    same file).
    """
    X_val_raw, y_val = get_X_y(val_df)
    probas = {}

    # --- XGBoost (tuned) ---
    xgb_path = require(
        Path(MODELS_DIR) / "xgboost_tuned.joblib",
        "Run `python src/tune_models.py` first.",
    )
    xgb_pipeline = joblib.load(xgb_path)
    probas["xgboost_tuned"] = xgb_pipeline.predict_proba(X_val_raw)[:, 1]

    # --- Random Forest (baseline -- marginally beat the tuned version) ---
    rf_path = require(
        Path(MODELS_DIR) / "random_forest.joblib",
        "Run `python src/train_baselines.py` first.",
    )
    rf_pipeline = joblib.load(rf_path)
    probas["random_forest"] = rf_pipeline.predict_proba(X_val_raw)[:, 1]

    # --- FNN ---
    fnn_preprocessor_path = require(
        Path(MODELS_DIR) / "fnn_preprocessor.joblib",
        "Run `python src/train_fnn.py` first.",
    )
    fnn_ckpt_path = require(Path(MODELS_DIR) / "fnn_best.pt", "Run `python src/train_fnn.py` first.")
    fnn_preprocessor = joblib.load(fnn_preprocessor_path)
    X_val_fnn = fnn_preprocessor.transform(X_val_raw).astype(np.float32)
    fnn_model = FraudFNN(input_dim=X_val_fnn.shape[1]).to(device)
    fnn_model.load_state_dict(torch.load(fnn_ckpt_path, map_location=device, weights_only=True))
    probas["fnn"] = fnn_predict_proba(fnn_model, torch.from_numpy(X_val_fnn), device)

    # --- LSTM ---
    seq_path = require(
        PROCESSED_DIR / "val_sequences.npz",
        "Run `python src/build_sequences.py` first.",
    )
    seq_scaler_path = require(Path(MODELS_DIR) / "lstm_seq_scaler.joblib", "Run `python src/train_lstm.py` first.")
    static_pre_path = require(
        Path(MODELS_DIR) / "lstm_static_preprocessor.joblib", "Run `python src/train_lstm.py` first."
    )
    lstm_ckpt_path = require(Path(MODELS_DIR) / "lstm_best.pt", "Run `python src/train_lstm.py` first.")

    seq_data = np.load(seq_path)
    X_seq_val = seq_data["X_seq"]
    n_val, seq_len, n_seq_features = X_seq_val.shape

    seq_scaler = joblib.load(seq_scaler_path)
    X_seq_val_scaled = seq_scaler.transform(
        X_seq_val.reshape(-1, n_seq_features)
    ).reshape(n_val, seq_len, n_seq_features).astype(np.float32)

    static_preprocessor = joblib.load(static_pre_path)
    X_static_val = static_preprocessor.transform(
        val_df[STATIC_NUMERIC + STATIC_CATEGORICAL]
    ).astype(np.float32)

    lstm_model = FraudLSTM(
        seq_input_dim=n_seq_features,
        static_input_dim=X_static_val.shape[1],
    ).to(device)
    lstm_model.load_state_dict(torch.load(lstm_ckpt_path, map_location=device, weights_only=True))
    probas["lstm"] = lstm_predict_proba(
        lstm_model, torch.from_numpy(X_seq_val_scaled), torch.from_numpy(X_static_val), device
    )

    return probas, y_val.to_numpy()


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    print("Loading validation set and generating base-model predictions "
          "(XGBoost, Random Forest, FNN, LSTM)...")
    val_df = load_engineered_val()
    probas, y_val = get_base_model_probas(val_df, device)

    base_model_names = ["xgboost_tuned", "random_forest", "fnn", "lstm"]
    meta_X = np.column_stack([probas[name] for name in base_model_names])
    print(f"Meta-feature matrix: {meta_X.shape} (rows x [{', '.join(base_model_names)}])")

    # --- Out-of-fold hybrid predictions for an honest, leaderboard-comparable score ---
    print(f"\nRunning {N_FOLDS}-fold CV within validation for out-of-fold "
          "meta-learner predictions...")
    oof_proba = np.zeros(len(y_val))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_coefs = []

    for fold_idx, (train_idx, holdout_idx) in enumerate(skf.split(meta_X, y_val), start=1):
        meta_learner = LogisticRegression(class_weight="balanced", random_state=RANDOM_SEED)
        meta_learner.fit(meta_X[train_idx], y_val[train_idx])
        oof_proba[holdout_idx] = meta_learner.predict_proba(meta_X[holdout_idx])[:, 1]
        fold_coefs.append(meta_learner.coef_[0])
        print(f"  fold {fold_idx}/{N_FOLDS} done")

    hybrid_metrics = full_evaluation("hybrid", y_val, oof_proba)
    print(f"\nHybrid (out-of-fold) results on FULL validation set:")
    print(
        f"  ROC-AUC={hybrid_metrics['roc_auc']:.4f}  PR-AUC={hybrid_metrics['pr_auc']:.4f}  "
        f"tuned F1={hybrid_metrics['tuned_f1']:.4f} "
        f"(precision={hybrid_metrics['tuned_precision']:.4f}, "
        f"recall={hybrid_metrics['tuned_recall']:.4f}, "
        f"threshold={hybrid_metrics['tuned_threshold']:.3f})"
    )

    avg_coefs = np.mean(fold_coefs, axis=0)
    print("\nAverage meta-learner weight per base model (higher = more "
          "trusted by the blend; not directly comparable across models "
          "with different probability calibration, but informative "
          "directionally):")
    for name, coef in zip(base_model_names, avg_coefs):
        print(f"  {name:>15}: {coef:+.3f}")

    # --- Final meta-learner, fit on ALL validation meta-features, for downstream use ---
    final_meta_learner = LogisticRegression(class_weight="balanced", random_state=RANDOM_SEED)
    final_meta_learner.fit(meta_X, y_val)
    meta_path = Path(MODELS_DIR) / "hybrid_meta_learner.joblib"
    joblib.dump(
        {"meta_learner": final_meta_learner, "base_model_order": base_model_names},
        meta_path,
    )
    print(f"\nFinal meta-learner (fit on all of validation, for downstream "
          f"scoring -- its own val predictions are NOT reported as a "
          f"metric, since that would be circular) saved -> {meta_path}")

    # --- Save + merge into leaderboard ---
    save_single_result(hybrid_metrics, Path(REPORTS_DIR) / "milestone4_hybrid_result.csv")
    leaderboard_df = build_leaderboard(REPORTS_DIR)
    leaderboard_path = Path(REPORTS_DIR) / "leaderboard.csv"
    leaderboard_df.to_csv(leaderboard_path, index=False)

    print("\n" + "=" * 80)
    print("All models so far (sorted by PR-AUC):")
    print("=" * 80)
    print(leaderboard_df.to_string(index=False))
    print(f"\nSaved -> {leaderboard_path}")


if __name__ == "__main__":
    main()
