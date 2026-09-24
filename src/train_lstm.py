"""
Milestone 4 (part 2): LSTM over per-card transaction sequences.

Unlike every other model so far (one row = one independent transaction),
the LSTM sees each transaction as the end of a short sequence of that
card's recent activity -- see sequence_features.py for how those sequences
are built, and its module docstring for why a simple zero-padding scheme
was used rather than packed sequences/masking.

Usage:
    python src/build_features.py    # if not already run
    python src/build_sequences.py   # once, or after changing sequence_features.py
    python src/train_lstm.py
"""

import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import MODELS_DIR, PROCESSED_DIR, RANDOM_SEED, REPORTS_DIR
from data_loader import load_engineered_train, load_engineered_val
from evaluation import build_leaderboard, full_evaluation, save_single_result
from lstm_model import FraudLSTM, predict_proba
from sequence_features import SEQ_FEATURES, STATIC_CATEGORICAL, STATIC_NUMERIC

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
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_static_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), STATIC_NUMERIC),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                STATIC_CATEGORICAL,
            ),
        ],
    )


def load_sequences(split_name):
    path = PROCESSED_DIR / f"{split_name}_sequences.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python src/build_sequences.py` first."
        )
    data = np.load(path)
    return data["X_seq"], data["y"]


def main():
    set_seed(RANDOM_SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    print("Loading sequences and static features...")
    X_seq_train, y_train = load_sequences("train")
    X_seq_val, y_val = load_sequences("val")

    train_df = load_engineered_train()
    val_df = load_engineered_val()

    # Scale sequence features using TRAIN statistics only. Flatten across
    # timesteps to fit (all timesteps share the same feature meaning), then
    # reshape back.
    n_train, seq_len, n_seq_features = X_seq_train.shape
    seq_scaler = StandardScaler()
    X_seq_train_scaled = seq_scaler.fit_transform(
        X_seq_train.reshape(-1, n_seq_features)
    ).reshape(n_train, seq_len, n_seq_features).astype(np.float32)
    X_seq_val_scaled = seq_scaler.transform(
        X_seq_val.reshape(-1, n_seq_features)
    ).reshape(X_seq_val.shape[0], seq_len, n_seq_features).astype(np.float32)
    joblib.dump(seq_scaler, Path(MODELS_DIR) / "lstm_seq_scaler.joblib")

    static_preprocessor = build_static_preprocessor()
    X_static_train = static_preprocessor.fit_transform(
        train_df[STATIC_NUMERIC + STATIC_CATEGORICAL]
    ).astype(np.float32)
    X_static_val = static_preprocessor.transform(
        val_df[STATIC_NUMERIC + STATIC_CATEGORICAL]
    ).astype(np.float32)
    joblib.dump(static_preprocessor, Path(MODELS_DIR) / "lstm_static_preprocessor.joblib")

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    print(f"Train: {n_train:,} sequences, seq_len={seq_len}, "
          f"{n_seq_features} seq features, {X_static_train.shape[1]} static "
          f"features after encoding. {int(n_pos):,} fraud "
          f"({y_train.mean():.4%}). pos_weight={pos_weight.item():.1f}")
    print(f"Val:   {len(X_seq_val):,} sequences, "
          f"{int(y_val.sum()):,} fraud ({y_val.mean():.4%})")

    X_seq_train_t = torch.from_numpy(X_seq_train_scaled)
    X_static_train_t = torch.from_numpy(X_static_train)
    y_train_t = torch.from_numpy(y_train)
    X_seq_val_t = torch.from_numpy(X_seq_val_scaled)
    X_static_val_t = torch.from_numpy(X_static_val)

    train_dataset = torch.utils.data.TensorDataset(X_seq_train_t, X_static_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )

    model = FraudLSTM(
        seq_input_dim=n_seq_features,
        static_input_dim=X_static_train.shape[1],
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_val_pr_auc = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    best_model_path = Path(MODELS_DIR) / "lstm_best.pt"
    history = []

    print("\nTraining...")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        start = time.time()
        running_loss = 0.0
        n_batches = 0

        for X_seq_batch, X_static_batch, y_batch in train_loader:
            X_seq_batch = X_seq_batch.to(device)
            X_static_batch = X_static_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_seq_batch, X_static_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / n_batches
        val_proba = predict_proba(model, X_seq_val_t, X_static_val_t, device)
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
        Path(REPORTS_DIR) / "milestone4_lstm_training_history.csv", index=False
    )

    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    val_proba = predict_proba(model, X_seq_val_t, X_static_val_t, device)
    metrics = full_evaluation("lstm", y_val, val_proba)

    print(f"\nBest epoch: {best_epoch}, val PR-AUC: {best_val_pr_auc:.4f}")
    print(
        f"ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  "
        f"tuned F1={metrics['tuned_f1']:.4f} "
        f"(precision={metrics['tuned_precision']:.4f}, "
        f"recall={metrics['tuned_recall']:.4f}, "
        f"threshold={metrics['tuned_threshold']:.3f})"
    )

    save_single_result(metrics, Path(REPORTS_DIR) / "milestone4_lstm_result.csv")
    leaderboard_df = build_leaderboard(REPORTS_DIR)
    leaderboard_path = Path(REPORTS_DIR) / "leaderboard.csv"
    leaderboard_df.to_csv(leaderboard_path, index=False)

    print("\n" + "=" * 80)
    print("All models so far (sorted by PR-AUC):")
    print("=" * 80)
    print(leaderboard_df.to_string(index=False))
    print(f"\nSaved -> {leaderboard_path}")
    print(f"Model + preprocessors saved -> {best_model_path}, "
          f"{Path(MODELS_DIR) / 'lstm_seq_scaler.joblib'}, "
          f"{Path(MODELS_DIR) / 'lstm_static_preprocessor.joblib'}")


if __name__ == "__main__":
    main()
