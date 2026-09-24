"""
Milestone 4 (part 1): Feedforward Neural Network baseline, in PyTorch.

Uses the exact same engineered tabular features as the XGBoost/Random
Forest baselines (src/preprocessing.py), so its results sit in the same
comparison table and the difference in performance reflects the model
architecture, not a different feature set.

Imbalance handling: rather than resampling, this uses
`BCEWithLogitsLoss(pos_weight=...)` -- the neural-net equivalent of
`class_weight="balanced"` in the sklearn baselines, keeping the imbalance
strategy consistent across model families for a fair comparison.

Usage:
    python src/build_features.py   # if not already run
    python src/train_fnn.py
"""

import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from config import MODELS_DIR, RANDOM_SEED, REPORTS_DIR
from data_loader import load_engineered_train, load_engineered_val
from evaluation import build_leaderboard, full_evaluation, save_single_result
from fnn_model import FraudFNN, predict_proba
from preprocessing import build_preprocessor, get_X_y

MAX_EPOCHS = 40
BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 6


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    set_seed(RANDOM_SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    print("Loading engineered train/val splits...")
    train_df = load_engineered_train()
    val_df = load_engineered_val()

    X_train_raw, y_train = get_X_y(train_df)
    X_val_raw, y_val = get_X_y(val_df)

    # Fit the same scaling/encoding used by the sklearn baselines, on train
    # only, then transform both splits to dense float32 arrays for PyTorch.
    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw).astype(np.float32)
    X_val = preprocessor.transform(X_val_raw).astype(np.float32)
    joblib.dump(preprocessor, Path(MODELS_DIR) / "fnn_preprocessor.joblib")

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    print(f"Train: {len(X_train):,} rows, {X_train.shape[1]} features after "
          f"encoding, {n_pos:,} fraud ({y_train.mean():.4%}). "
          f"pos_weight={pos_weight.item():.1f}")
    print(f"Val:   {len(X_val):,} rows, {y_val.sum():,} fraud ({y_val.mean():.4%})")

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train.values.astype(np.float32))
    X_val_t = torch.from_numpy(X_val)

    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )

    model = FraudFNN(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_val_pr_auc = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    best_model_path = Path(MODELS_DIR) / "fnn_best.pt"
    history = []

    print("\nTraining...")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        start = time.time()
        running_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / n_batches
        val_proba = predict_proba(model, X_val_t, device)
        val_pr_auc = average_precision_score(y_val, val_proba)
        scheduler.step(val_pr_auc)

        elapsed = time.time() - start
        improved = val_pr_auc > best_val_pr_auc
        marker = " *" if improved else ""
        print(
            f"  epoch {epoch:>3}/{MAX_EPOCHS}  train_loss={train_loss:.4f}  "
            f"val_PR-AUC={val_pr_auc:.4f}  ({elapsed:.1f}s){marker}"
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_pr_auc": val_pr_auc})

        if improved:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"  no improvement for {EARLY_STOP_PATIENCE} epochs, "
                      f"stopping early (best epoch: {best_epoch})")
                break

    pd.DataFrame(history).to_csv(
        Path(REPORTS_DIR) / "milestone4_fnn_training_history.csv", index=False
    )

    # Reload the best checkpoint (not necessarily the last epoch) for final evaluation.
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    val_proba = predict_proba(model, X_val_t, device)
    metrics = full_evaluation("fnn", y_val, val_proba)

    print(f"\nBest epoch: {best_epoch}, val PR-AUC: {best_val_pr_auc:.4f}")
    print(
        f"ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  "
        f"tuned F1={metrics['tuned_f1']:.4f} "
        f"(precision={metrics['tuned_precision']:.4f}, "
        f"recall={metrics['tuned_recall']:.4f}, "
        f"threshold={metrics['tuned_threshold']:.3f})"
    )

    # Save this model's own result, then rebuild the full leaderboard from
    # every training script's result file on disk (see evaluation.py) --
    # robust to run order, unlike re-merging a couple of specific CSVs here.
    save_single_result(metrics, Path(REPORTS_DIR) / "milestone4_fnn_result.csv")
    leaderboard_df = build_leaderboard(REPORTS_DIR)
    leaderboard_path = Path(REPORTS_DIR) / "leaderboard.csv"
    leaderboard_df.to_csv(leaderboard_path, index=False)

    print("\n" + "=" * 80)
    print("All models so far (sorted by PR-AUC):")
    print("=" * 80)
    print(leaderboard_df.to_string(index=False))
    print(f"\nSaved -> {leaderboard_path}")
    print(f"Model + preprocessor saved -> {best_model_path}, "
          f"{Path(MODELS_DIR) / 'fnn_preprocessor.joblib'}")


if __name__ == "__main__":
    main()
