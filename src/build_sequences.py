"""
Precompute per-card transaction sequences for the LSTM and cache them to
data/processed/{train,val,test}_sequences.npz -- this is the expensive
step (a groupby-shift over the full combined timeline) you don't want to
redo on every training run.

Uses the already-engineered feature CSVs (data/processed/*_features.csv),
so run build_features.py first if you haven't.

Usage:
    python src/build_sequences.py
"""

import numpy as np
import pandas as pd

from config import DATETIME_COL, PROCESSED_DIR, TARGET_COL
from data_loader import load_engineered_test, load_engineered_train, load_engineered_val
from sequence_features import SEQ_LEN, build_lagged_sequences


def main():
    print("Loading engineered train/val/test...")
    train_df = load_engineered_train().assign(split="train")
    val_df = load_engineered_val().assign(split="val")
    test_df = load_engineered_test().assign(split="test")

    combined = (
        pd.concat([train_df, val_df, test_df], ignore_index=True)
        .sort_values(DATETIME_COL)
        .reset_index(drop=True)
    )
    print(f"Combined: {combined.shape}")

    print(f"Building lagged sequences (seq_len={SEQ_LEN})... this involves "
          "a groupby-shift over the full timeline and can take a minute.")
    seq_array = build_lagged_sequences(combined)
    print(f"Sequence array: {seq_array.shape} "
          f"(n_rows, seq_len, n_features), {seq_array.nbytes / 1e6:.0f} MB")

    for split_name in ["train", "val", "test"]:
        mask = (combined["split"] == split_name).to_numpy()
        X_seq = seq_array[mask]
        y = combined.loc[mask, TARGET_COL].to_numpy(dtype=np.float32)

        out_path = PROCESSED_DIR / f"{split_name}_sequences.npz"
        np.savez_compressed(out_path, X_seq=X_seq, y=y)
        print(f"  {split_name:>5}: X_seq={X_seq.shape}, "
              f"fraud rate {y.mean():.4%} -> {out_path}")

    print("\nDone. Use np.load(path) with keys 'X_seq', 'y' from here on.")


if __name__ == "__main__":
    main()
