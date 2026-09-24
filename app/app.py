"""
Milestone 5: Streamlit web app for Fraud Shield AI.

Loads the final hybrid pipeline -- four base models (tuned XGBoost, Random
Forest, FNN, LSTM) plus the Logistic Regression meta-learner that combines
them -- and scores transactions you provide, either as a batch CSV upload
or one manually-entered transaction.

IMPORTANT LIMITATION (read before uploading): several features --
transaction velocity (txn_count_1h/24h), the spending-deviation z-score,
and the LSTM's 5-step sequence input -- are computed from a card's *own
transaction history*. Training/validation used the full historical
timeline, but here they're computed only from what you upload in a given
batch. Upload a single transaction (or a card's very first ones) and that
history is empty, so those features fall back to neutral defaults (0 prior
transactions, 0 deviation, a zero-padded sequence) -- the model still
scores it, just with less signal than it had during training. For the most
realistic demo, upload a time-ordered slice of one card's recent
transactions (a few dozen rows) and look at the prediction for the last
row, which has real in-batch history behind it. This is a known,
documented simplification of a production system, which would look up
each card's real history from a live transaction store.
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from config import CARD_ID_COL, DATETIME_COL, MODELS_DIR, REPORTS_DIR, TARGET_COL
from features import engineer_all_features
from fnn_model import FraudFNN
from fnn_model import predict_proba as fnn_predict_proba
from lstm_model import FraudLSTM
from lstm_model import predict_proba as lstm_predict_proba
from preprocessing import ALL_FEATURES
from sequence_features import SEQ_FEATURES, SEQ_LEN, STATIC_CATEGORICAL, STATIC_NUMERIC, build_lagged_sequences

RAW_REQUIRED_COLUMNS = [
    DATETIME_COL, CARD_ID_COL, "amt", "category", "gender",
    "lat", "long", "merch_lat", "merch_long", "city_pop", "dob",
]

KNOWN_CATEGORIES = [
    "entertainment", "food_dining", "gas_transport", "grocery_net",
    "grocery_pos", "health_fitness", "home", "kids_pets", "misc_net",
    "misc_pos", "personal_care", "shopping_net", "shopping_pos", "travel",
]

st.set_page_config(page_title="Fraud Shield AI", page_icon="\U0001F6E1️", layout="wide")


# ---------------------------------------------------------------------------
# Model / artifact loading (cached -- only runs once per app session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models...")
def load_artifacts():
    missing_files = [
        name for name in [
            "xgboost_tuned.joblib", "random_forest.joblib",
            "fnn_preprocessor.joblib", "fnn_best.pt",
            "lstm_seq_scaler.joblib", "lstm_static_preprocessor.joblib", "lstm_best.pt",
            "hybrid_meta_learner.joblib",
        ]
        if not (MODELS_DIR / name).exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Missing model artifact(s): " + ", ".join(missing_files) + ". "
            "Run the training pipeline in src/ first (see README) -- "
            "train_baselines.py, tune_models.py, train_fnn.py, "
            "build_sequences.py, train_lstm.py, train_hybrid.py, in that order."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xgb_pipeline = joblib.load(MODELS_DIR / "xgboost_tuned.joblib")
    rf_pipeline = joblib.load(MODELS_DIR / "random_forest.joblib")

    fnn_preprocessor = joblib.load(MODELS_DIR / "fnn_preprocessor.joblib")
    fnn_model = FraudFNN(input_dim=len(fnn_preprocessor.get_feature_names_out()))
    fnn_model.load_state_dict(
        torch.load(MODELS_DIR / "fnn_best.pt", map_location=device, weights_only=True)
    )
    fnn_model.to(device).eval()

    lstm_seq_scaler = joblib.load(MODELS_DIR / "lstm_seq_scaler.joblib")
    lstm_static_preprocessor = joblib.load(MODELS_DIR / "lstm_static_preprocessor.joblib")
    lstm_model = FraudLSTM(
        seq_input_dim=len(SEQ_FEATURES),
        static_input_dim=len(lstm_static_preprocessor.get_feature_names_out()),
    )
    lstm_model.load_state_dict(
        torch.load(MODELS_DIR / "lstm_best.pt", map_location=device, weights_only=True)
    )
    lstm_model.to(device).eval()

    hybrid_bundle = joblib.load(MODELS_DIR / "hybrid_meta_learner.joblib")

    threshold = 0.5
    hybrid_result_path = REPORTS_DIR / "milestone4_hybrid_result.csv"
    if hybrid_result_path.exists():
        threshold = float(pd.read_csv(hybrid_result_path)["tuned_threshold"].iloc[0])

    return {
        "device": device,
        "xgb_pipeline": xgb_pipeline,
        "rf_pipeline": rf_pipeline,
        "fnn_preprocessor": fnn_preprocessor,
        "fnn_model": fnn_model,
        "lstm_seq_scaler": lstm_seq_scaler,
        "lstm_static_preprocessor": lstm_static_preprocessor,
        "lstm_model": lstm_model,
        "hybrid_meta_learner": hybrid_bundle["meta_learner"],
        "base_model_order": hybrid_bundle["base_model_order"],
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_dataframe(raw_df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """
    Takes a DataFrame of raw transactions (Kaggle schema), runs the full
    feature-engineering + 4-base-model + hybrid-meta-learner pipeline, and
    returns the original rows (in their original order) with prediction
    columns appended.
    """
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    df = raw_df.copy().reset_index(drop=True)
    df["_orig_order"] = df.index
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
    # Velocity/deviation/sequence features need time-ordered-per-card rows.
    df = df.sort_values(DATETIME_COL).reset_index(drop=True)

    df = engineer_all_features(df)
    X = df[ALL_FEATURES].copy()
    device = artifacts["device"]

    xgb_proba = artifacts["xgb_pipeline"].predict_proba(X)[:, 1]
    rf_proba = artifacts["rf_pipeline"].predict_proba(X)[:, 1]

    X_fnn = artifacts["fnn_preprocessor"].transform(X).astype(np.float32)
    fnn_proba = fnn_predict_proba(artifacts["fnn_model"], torch.from_numpy(X_fnn), device)

    X_seq = build_lagged_sequences(df, seq_features=SEQ_FEATURES, seq_len=SEQ_LEN)
    n_rows, seq_len, n_seq_features = X_seq.shape
    X_seq_scaled = (
        artifacts["lstm_seq_scaler"]
        .transform(X_seq.reshape(-1, n_seq_features))
        .reshape(n_rows, seq_len, n_seq_features)
        .astype(np.float32)
    )
    X_static = artifacts["lstm_static_preprocessor"].transform(
        df[STATIC_NUMERIC + STATIC_CATEGORICAL]
    ).astype(np.float32)
    lstm_proba = lstm_predict_proba(
        artifacts["lstm_model"], torch.from_numpy(X_seq_scaled), torch.from_numpy(X_static), device
    )

    base_probas = {
        "xgboost_tuned": xgb_proba,
        "random_forest": rf_proba,
        "fnn": fnn_proba,
        "lstm": lstm_proba,
    }
    meta_X = np.column_stack([base_probas[name] for name in artifacts["base_model_order"]])
    hybrid_proba = artifacts["hybrid_meta_learner"].predict_proba(meta_X)[:, 1]

    df["fraud_probability"] = hybrid_proba
    df["predicted_fraud"] = (hybrid_proba >= artifacts["threshold"]).astype(int)
    df["xgboost_tuned_proba"] = xgb_proba
    df["random_forest_proba"] = rf_proba
    df["fnn_proba"] = fnn_proba
    df["lstm_proba"] = lstm_proba

    # Restore the order the transactions were uploaded in.
    df = df.sort_values("_orig_order").drop(columns="_orig_order").reset_index(drop=True)
    return df


def highlight_fraud(row):
    if row["predicted_fraud"] == 1:
        return ["background-color: #ffe3e3"] * len(row)
    return [""] * len(row)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("\U0001F6E1️ Fraud Shield AI")
st.caption(
    "Hybrid stacking ensemble (tuned XGBoost + Random Forest + FNN + LSTM, "
    "combined by a Logistic Regression meta-learner) -- validation PR-AUC 0.983."
)

try:
    artifacts = load_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

with st.expander("Model performance (validation set)"):
    leaderboard_path = REPORTS_DIR / "leaderboard.csv"
    if leaderboard_path.exists():
        st.dataframe(pd.read_csv(leaderboard_path), width='stretch', hide_index=True)
    else:
        st.write("No leaderboard.csv found in reports/ yet.")
    st.caption(
        f"Flagging threshold in use: {artifacts['threshold']:.4f} "
        "(chosen on validation to maximize F1 for the hybrid model)."
    )

tab_batch, tab_manual = st.tabs(["\U0001F4C4 Upload transactions", "\U0000270D️ Enter one transaction"])

with tab_batch:
    st.write(
        "Upload a CSV with the same columns as the Kaggle dataset "
        f"(`{DATETIME_COL}`, `{CARD_ID_COL}`, `amt`, `category`, `gender`, "
        "`lat`, `long`, `merch_lat`, `merch_long`, `city_pop`, `dob`, ...). "
        "A slice of `data/raw/fraudTest.csv` works well as a demo."
    )
    uploaded = st.file_uploader("Transactions CSV", type=["csv"])

    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Couldn't read that CSV: {e}")
            raw_df = None

        if raw_df is not None:
            st.write(f"Loaded {len(raw_df):,} rows. Preview:")
            st.dataframe(raw_df.head(10), width='stretch', hide_index=True)

            if st.button("Score transactions", type="primary"):
                try:
                    with st.spinner("Engineering features and scoring..."):
                        scored = score_dataframe(raw_df, artifacts)
                except ValueError as e:
                    st.error(str(e))
                    scored = None

                if scored is not None:
                    n_flagged = int(scored["predicted_fraud"].sum())
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Transactions scored", f"{len(scored):,}")
                    col2.metric("Flagged as fraud", f"{n_flagged:,}")
                    col3.metric("Flagged rate", f"{n_flagged / len(scored):.2%}")

                    if TARGET_COL in scored.columns:
                        n_actual = int(scored[TARGET_COL].sum())
                        n_correct_flags = int(
                            ((scored["predicted_fraud"] == 1) & (scored[TARGET_COL] == 1)).sum()
                        )
                        st.info(
                            f"This upload includes ground truth (`{TARGET_COL}`): "
                            f"{n_actual:,} actually fraudulent, {n_correct_flags:,} of those "
                            "correctly flagged. (For a report-quality final metric, run the "
                            "one-time evaluation script against the full `fraudTest.csv` "
                            "holdout instead of relying on ad hoc uploads here.)"
                        )

                    display_cols = [
                        c for c in scored.columns
                        if c not in ["xgboost_tuned_proba", "random_forest_proba", "fnn_proba", "lstm_proba"]
                    ] + ["xgboost_tuned_proba", "random_forest_proba", "fnn_proba", "lstm_proba"]
                    st.dataframe(
                        scored[display_cols].style.apply(highlight_fraud, axis=1),
                        width='stretch',
                        hide_index=True,
                    )

                    st.download_button(
                        "Download scored results as CSV",
                        data=scored.to_csv(index=False).encode("utf-8"),
                        file_name="fraud_shield_predictions.csv",
                        mime="text/csv",
                    )

with tab_manual:
    st.write(
        "Score a single transaction. Since there's no prior history for a "
        "one-off entry, velocity/deviation/sequence features default to "
        "neutral values -- see the note at the top of this page."
    )
    with st.form("manual_transaction"):
        c1, c2, c3 = st.columns(3)
        with c1:
            trans_date = st.date_input("Transaction date")
            trans_time = st.time_input("Transaction time")
            amt = st.number_input("Amount ($)", min_value=0.0, value=50.0, step=1.0)
            category = st.selectbox("Merchant category", KNOWN_CATEGORIES)
        with c2:
            gender = st.selectbox("Cardholder gender", ["F", "M"])
            dob = st.date_input("Cardholder date of birth", value=pd.Timestamp("1985-01-01"))
            city_pop = st.number_input("Cardholder city population", min_value=1, value=50000, step=1000)
            cc_num = st.text_input("Card number (any placeholder value)", value="4000000000000000")
        with c3:
            lat = st.number_input("Cardholder latitude", value=40.0, format="%.4f")
            long = st.number_input("Cardholder longitude", value=-90.0, format="%.4f")
            merch_lat = st.number_input("Merchant latitude", value=40.0, format="%.4f")
            merch_long = st.number_input("Merchant longitude", value=-90.0, format="%.4f")

        submitted = st.form_submit_button("Score this transaction", type="primary")

    if submitted:
        row = pd.DataFrame([{
            DATETIME_COL: pd.Timestamp.combine(trans_date, trans_time),
            CARD_ID_COL: cc_num,
            "amt": amt,
            "category": category,
            "gender": gender,
            "lat": lat,
            "long": long,
            "merch_lat": merch_lat,
            "merch_long": merch_long,
            "city_pop": city_pop,
            "dob": dob,
        }])
        try:
            scored = score_dataframe(row, artifacts)
        except ValueError as e:
            st.error(str(e))
            scored = None

        if scored is not None:
            proba = float(scored["fraud_probability"].iloc[0])
            is_fraud = bool(scored["predicted_fraud"].iloc[0])
            st.metric("Fraud probability", f"{proba:.2%}")
            if is_fraud:
                st.error(f"\U0001F6A8 Flagged as likely fraud (threshold {artifacts['threshold']:.2%}).")
            else:
                st.success(f"✅ Not flagged (threshold {artifacts['threshold']:.2%}).")
            with st.expander("Base model breakdown"):
                st.write({
                    "xgboost_tuned": float(scored["xgboost_tuned_proba"].iloc[0]),
                    "random_forest": float(scored["random_forest_proba"].iloc[0]),
                    "fnn": float(scored["fnn_proba"].iloc[0]),
                    "lstm": float(scored["lstm_proba"].iloc[0]),
                })
