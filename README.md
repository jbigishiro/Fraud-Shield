# Fraud Shield AI

Real-time credit card fraud detection: supervised ML + deep learning models,
combined into a hybrid framework, served through a Streamlit web app.

Dataset: Kaggle's [Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
(Sparkov-simulated, ~1.85M transactions, Jan 2019 – Dec 2020, ~0.58% fraud rate).

## Project structure

```
fraud-shield-ai/
├── data/
│   ├── raw/          # untouched Kaggle CSVs (gitignored)
│   └── processed/    # time-based train/val/test splits + engineered features (gitignored)
├── notebooks/         # exploratory work (EDA)
├── src/               # reusable pipeline code (config, data, features, models)
├── app/               # Streamlit web app (Milestone 5, not yet built)
├── models/            # saved/trained model artifacts (gitignored)
└── reports/           # metrics CSVs, leaderboard, training histories
```

## Setup

1. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Get Kaggle API credentials** (optional -- only needed if you use `download_data.py`)
   - Create a Kaggle account, then go to
     https://www.kaggle.com/settings/account -> "Create New Token".
   - Save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`
     (`chmod 600 ~/.kaggle/kaggle.json` on Mac/Linux).

3. **Place the dataset**

   Put Kaggle's `fraudTrain.csv` and `fraudTest.csv` in `data/raw/`.

## Pipeline: how to run everything, in order

```bash
cd src
python data_split.py       # Milestone 1: time-based train/val/test split
python build_features.py   # Milestone 2: feature engineering (cached to data/processed/*_features.csv)
python train_baselines.py  # Milestone 3: LogReg, LogReg+SMOTE, Random Forest, XGBoost
python tune_models.py      # Milestone 3b: RandomizedSearchCV for XGBoost + Random Forest
python train_fnn.py        # Milestone 4a: PyTorch feedforward network
python build_sequences.py  # Milestone 4b: per-card lagged sequences for the LSTM
python train_lstm.py       # Milestone 4b: PyTorch LSTM
python train_hybrid.py     # Milestone 4c: stacking ensemble over all four models above
```

Every training script writes its own result row to `reports/`, and also
regenerates `reports/leaderboard.csv` (via `evaluation.build_leaderboard()`),
which aggregates every result file that exists on disk so far, deduped by
model name. That means scripts can be re-run individually, in any order,
and the leaderboard always reflects the latest run of each.

### `data_split.py` -- what it does

`fraudTest.csv` is kept exactly as Kaggle gave it and used only as the
final holdout -- never touched during model development. `fraudTrain.csv`
is sorted by timestamp and its most recent 15% is carved off as a
validation set (for tuning/model selection), with the earlier 85% as
training data. Carving val from the *tail* chronologically -- rather than
randomly -- avoids letting a card's later transactions leak into training
while an earlier transaction from the same card sits in validation.

The script also prints the actual date ranges of `fraudTrain.csv` and
`fraudTest.csv` and flags whether they overlap in time -- worth noting in
the report either way.

### `build_features.py` -- feature engineering

All features are computed backward-only (no look-ahead), so they're safe
to use on train, val, and test alike:

- **Time features**: hour, day-of-week, weekend flag, month, an
  `is_night` flag (10pm-3am, based on an EDA finding that fraud clusters
  in that window), plus cyclical (sin/cos) encodings of hour and
  day-of-week.
- **Geographical distance**: haversine distance (mi) between cardholder
  and merchant location.
- **Customer context**: age from DOB, log-scaled city population.
- **Transaction velocity**: rolling count and dollar-sum of each card's
  transactions in the trailing 1h and 24h windows (`closed="left"`, so
  the current transaction never counts itself).
- **Spending deviation**: each transaction's dollar amount as a z-score
  against that card's own historical mean/std (both computed on strictly
  prior transactions via `shift(1).expanding()`).

## Results so far

All metrics below are on the **validation set** (never on `fraudTest.csv`,
which stays untouched until a final model is chosen -- see "Final test
evaluation" below). PR-AUC (average precision) is the primary ranking
metric, since with ~0.58% fraud, ROC-AUC is misleadingly high for every
model and doesn't reflect precision/recall trade-offs the way PR-AUC does.

| model | roc_auc | pr_auc | tuned precision | tuned recall | tuned F1 | tuned threshold |
|---|---|---|---|---|---|---|
| **hybrid (stacking ensemble)** | 0.9967 | **0.9829** | 0.9719 | 0.9462 | 0.9589 | 1.000 |
| lstm | 0.9981 | 0.9804 | 0.9522 | 0.9312 | 0.9415 | 0.989 |
| xgboost_tuned | 0.9992 | 0.9760 | 0.9490 | 0.9356 | 0.9422 | 0.924 |
| xgboost (baseline) | 0.9992 | 0.9754 | 0.9479 | 0.9312 | 0.9394 | 0.946 |
| random_forest (baseline) | 0.9971 | 0.9617 | 0.9413 | 0.8923 | 0.9162 | 0.485 |
| random_forest_tuned | 0.9979 | 0.9613 | 0.9078 | 0.9126 | 0.9102 | 0.531 |
| fnn | 0.9989 | 0.9561 | 0.9313 | 0.8853 | 0.9077 | 0.993 |
| logistic_regression_smote | 0.9839 | 0.4791 | 0.5771 | 0.6408 | 0.6073 | 0.818 |
| logistic_regression | 0.9862 | 0.4306 | 0.5127 | 0.6408 | 0.5696 | 0.991 |

**Key findings:**
- **The hybrid stacking ensemble wins outright**: PR-AUC 0.9829, ahead of
  every individual model, including the LSTM alone (0.9804). It also has
  the best tuned F1 (0.9589) and the fewest combined errors at its tuned
  threshold (31 false positives + 61 false negatives, vs. 53+78 for the
  LSTM and 57+73 for tuned XGBoost). This is the real payoff of the
  "hybrid framework" the brief asks for: the meta-learner isn't just
  picking the best model, it's extracting complementary signal from
  models that disagree on different transactions. The averaged
  meta-learner coefficients back this up -- LSTM (+6.6) and XGBoost
  (+4.5) are trusted most, but FNN (+3.0) and Random Forest (+1.2) still
  pull real, non-trivial weight rather than being ignored.
- **The LSTM is the best single model** on PR-AUC (0.9804), narrowly
  ahead of tuned XGBoost (0.9760) -- the sequential/velocity signal from
  a card's recent history adds real value beyond the flat engineered
  features alone.
- **Hyperparameter tuning did not beat the baselines** for either
  XGBoost or Random Forest (tuned XGBoost: 0.9760 vs. baseline 0.9754;
  tuned RF: 0.9613 vs. baseline 0.9617) -- both are within noise of each
  other. This is a legitimate, reportable finding: the baselines'
  defaults were already close to a local optimum for this data, and the
  randomized search (25 iterations, single train/val split rather than
  k-fold, for tractability on ~1.1M rows) likely found a comparably good
  but not clearly superior region of the hyperparameter space. The
  hybrid framework uses `random_forest` (baseline) rather than
  `random_forest_tuned` for this reason.
- **SMOTE hurt logistic regression's ranking ability** relative to plain
  class-weighting on PR-AUC's precision/recall trade-off, though it
  modestly improved tuned-F1 -- worth a sentence in the report on why
  class-weighting and resampling aren't interchangeable.
- All four strong models (RF, XGBoost, FNN, LSTM) comfortably outperform
  the linear baselines, expected given the amount of nonlinear
  interaction in the engineered features (velocity x deviation x
  category, etc.), and the hybrid improves further still on top of them.

## Milestone 4c: hybrid framework (stacking ensemble)

`train_hybrid.py` combines all four base models (`xgboost_tuned`,
`random_forest`, `fnn`, `lstm`) via a Logistic Regression meta-learner
trained on their predicted probabilities, rather than averaging them
blindly -- and, per the results above, it works: it's the best model on
the leaderboard.

**Evaluation design:** the base models were already selected/tuned using
validation, so fitting *and* scoring the meta-learner on validation too
would let it overfit trivially. Instead:
- A **5-fold cross-validation within validation** generates out-of-fold
  (OOF) meta-predictions -- each validation row's hybrid prediction comes
  from a meta-learner that never saw that row during its own fit. This
  gives an honest hybrid score on the *full* validation set, directly
  comparable to every other row on the leaderboard.
- A separate **final meta-learner** is then fit on all of validation's
  meta-features (for actual downstream use -- scoring `fraudTest.csv`
  later, or the web app) and saved to
  `models/hybrid_meta_learner.joblib`, but its own predictions on
  validation are *not* reported as a metric, since that would be
  circular. This is the meta-learner the web app (Milestone 5) and the
  final test-set evaluation will use.

This is now the project's **final chosen model** for Milestone 5 and the
final test evaluation, unless later work changes that.

## Where things stand

- [x] Milestone 1: project scaffold, train/val/test split
- [x] Milestone 2: EDA and data preparation
- [x] Milestone 3: baseline supervised models (LogReg, LogReg+SMOTE, Random Forest, XGBoost)
- [x] Milestone 3b: hyperparameter tuning (XGBoost, Random Forest)
- [x] Milestone 4a: FNN (deep learning)
- [x] Milestone 4b: LSTM (deep learning, per-card sequences) -- best single model (PR-AUC 0.9804)
- [x] Milestone 4c: hybrid framework (stacking ensemble) -- **best model overall (PR-AUC 0.9829)**
- [ ] Milestone 5: Streamlit web interface (scores new transactions with the saved hybrid pipeline)
- [ ] Milestone 6: documentation and final submission
- [ ] Final evaluation against `fraudTest.csv` (run once, with the hybrid, once Milestone 5 is working)

## Notes on the dataset

The Kaggle dataset ships as two files, `fraudTrain.csv` and `fraudTest.csv`,
spanning Jan 2019 – Dec 2020 in total (simulated data via the Sparkov
generator). We use them as the given train/test split and only carve a
validation set out of `fraudTrain.csv` (see `src/data_split.py`) — see
that script's printed output for whether the two files overlap in time.

Key raw columns: `trans_date_trans_time`, `cc_num`, `merchant`, `category`,
`amt`, `lat`/`long` (cardholder), `merch_lat`/`merch_long` (merchant),
`city_pop`, `job`, `dob`, `is_fraud`.

## Final test evaluation

`data/processed/test.csv` (built from `fraudTest.csv`, untouched since
Milestone 1) stays a true holdout: run it through a chosen model exactly
once, after a final approach is picked (once the hybrid's real results are
in and Milestone 5's app is working), not before and not repeatedly --
otherwise test stops being an honest measure of generalization.
