# Fraud Shield AI

Real-time credit card fraud detection: supervised ML + deep learning models,
combined into a hybrid framework, served through a Streamlit web app.
The app is live on https://fraud-shield-gcxh.onrender.com.

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
├── app/               # Streamlit web app 
├── models/            # saved/trained model artifacts (gitignored)
└── reports/           # metrics CSVs, leaderboard, training histories
```

## Setup

1. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      
   pip install -r requirements.txt
   ```

2. **Place the dataset**

   Put Kaggle's `fraudTrain.csv` and `fraudTest.csv` in `data/raw/`.

## Pipeline: how to run everything, in order

```bash
cd src
python data_split.py       
python build_features.py   
python train_baselines.py  
python tune_models.py      
python train_fnn.py        
python build_sequences.py  
python train_lstm.py       
python train_hybrid.py     
cd ../app
streamlit run app.py      
cd ../src
python evaluate_final_test.py  # 
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

## Milestone 5: Streamlit web app

`app/app.py` loads all five saved artifacts (the four base models plus the
hybrid meta-learner) and lets you score transactions two ways:

- **Batch upload**: a CSV with the same raw columns as the Kaggle dataset
  (`data/raw/fraudTest.csv` is a ready-made example to slice from). Shows a
  summary (transactions scored / flagged / flagged rate), a table with each
  row's hybrid fraud probability plus all four base-model probabilities for
  transparency, and a CSV download of the results. If the upload includes
  `is_fraud` (ground truth), the app also reports how many actual frauds
  were caught -- useful for a quick sanity check, but not a substitute for
  the one-time, untouched-until-now `fraudTest.csv` evaluation described
  below.
- **Manual single-transaction form**: enter one transaction's fields by
  hand and get an immediate fraud probability and flag/no-flag verdict.

**A real limitation, stated plainly rather than hidden:** three of the
engineered features -- transaction velocity (`txn_count_1h/24h`), the
spending-deviation z-score, and the LSTM's 5-step sequence -- depend on a
card's own transaction history. During training that history was the full
dataset timeline; in the app, it's only whatever's in the uploaded batch
(or nothing, for the single-transaction form). A single uploaded row or a
card's first-ever transactions will score with those features at their
neutral defaults, not because the app is broken, but because there's no
history to compute them from -- exactly the situation a production system
would solve with a live transaction-history lookup, which is out of scope
here. Uploading a time-ordered slice of one card's recent transactions
(the last row will have real in-batch history) demonstrates the model at
its best.

Run it with:
```bash
cd app
streamlit run app.py
```

## Deploying to Render

The app is a standard Streamlit web service(), which Render runs as a
Python web service. Two things make this deployment different from most
"push a repo, done" cases, and both matter:

1. **The trained model artifacts have to be in the repo.** `models/` is
   gitignored during development (regenerable, no reason to bloat local
   git history while iterating), but `app/app.py` loads those files at
   *runtime* -- there's no training step in the deployment itself. Before
   your first push for deployment:
   ```bash
   du -sh models/*
   ```
   If everything's comfortably under 100MB, just commit them (the
   `.gitignore` in this delivery already stopped excluding `models/` --
   see the comment there). **If `random_forest.joblib` (or any file) is
   large** -- plausible, since it's 200 fully-grown trees over 1.1M rows
   with no `max_depth` cap -- you have a few options, roughly in order of
   effort: (a) retrain it with a `max_depth` cap in `train_baselines.py`
   to shrink the pickle (small accuracy trade-off, worth re-checking
   against the leaderboard), (b) use [Git LFS](https://git-lfs.com/) for
   that one file, or (c) host it externally (e.g. a Hugging Face Hub
   model repo, or cloud storage) and download it once at container
   startup instead of committing it. `data/` stays gitignored either way
   -- the deployed app never needs the raw dataset, only the trained
   models.

2. **`torch`'s default pip install is enormous** (recent versions pull in
   several GB of NVIDIA CUDA packages as dependencies, even though
   Render's web services have no GPU to use them). `requirements-render.txt`
   (not the main `requirements.txt`, which is for local development and
   also needs Jupyter/Kaggle/plotting libs) pins the CPU-only build via
   `--extra-index-url`, and is what both deployment paths below use.

**Option A -- Blueprint (`render.yaml`), one click:**
1. Push this repo (with `models/` committed) to GitHub.
2. In the Render dashboard: **New -> Blueprint**, point it at the repo.
   Render reads `render.yaml` and proposes the service already configured
   (build/start commands, Python version).
3. Click **Apply**.

**Option B -- manual dashboard setup** (if you'd rather not use a
Blueprint, or want to tweak settings as you go):
1. Push the repo (with `models/` committed) to GitHub.
2. Render dashboard: **New -> Web Service**, connect the repo.
3. **Build Command**: `pip install -r requirements-render.txt`
4. **Start Command**:
   `streamlit run app/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
5. **Environment**: Python 3 (matches `render.yaml`'s `PYTHON_VERSION: 3.12.2` though any recent Python 3.11/3.12 works).
6. Pick a plan -- see the RAM note below -- and deploy.

**Plan sizing:** loading five models at once (XGBoost, Random Forest, FNN,
LSTM, meta-learner) plus Streamlit's own overhead can exceed the free/
starter tier's 512MB RAM, especially if `random_forest.joblib` is large.
If the deploy builds fine but the service repeatedly restarts or the logs
show it being killed (look for "out of memory" / a sudden exit with no
Python traceback), that's the symptom -- bump to Render's "standard" plan
(2GB) rather than debugging the app code, which isn't the problem.

**Cold starts:** Render's free tier spins a service down after a period
of inactivity and takes 30-60s to wake back up on the next request.

**After it's live:** open the Render-provided URL(https://fraud-shield-gcxh.onrender.com/) and repeat the same
smoke test from Milestone 5 -- upload a time-ordered slice of one card's
transactions from `data/raw/fraudTest.csv` (kept locally; it was never
committed) and confirm you get sane fraud probabilities back.


## Notes on the dataset

The Kaggle dataset ships as two files, `fraudTrain.csv` and `fraudTest.csv`,
spanning Jan 2019 – Dec 2020 in total (simulated data via the Sparkov
generator). We use them as the given train/test split and only carve a
validation set out of `fraudTrain.csv` (see `src/data_split.py`).

Key raw columns: `trans_date_trans_time`, `cc_num`, `merchant`, `category`,
`amt`, `lat`/`long` (cardholder), `merch_lat`/`merch_long` (merchant),
`city_pop`, `job`, `dob`, `is_fraud`.

## Final test evaluation

`src/evaluate_final_test.py` scores all five models (the four base models
and the hybrid) against `data/processed/test.csv` -- untouched since
Milestone 1 -- exactly once. It reports each model's standard metrics
(ROC-AUC, PR-AUC) plus performance at the threshold **chosen on validation**
(the one that would actually ship), re-tuning a threshold on test only as a
separate, clearly-labeled reference number -- never the reported "real"
result, since tuning on test would leak it into the decision it's supposed
to check. It also prints each model's validation PR-AUC next to its test
PR-AUC so a meaningful generalization gap is easy to spot.

Run it once, after the final model is chosen:
```bash
python src/evaluate_final_test.py
```


### Final results (run once, against `fraudTest.csv`)

| Model | Test PR-AUC | Val PR-AUC | Gap | Shipped precision | Shipped recall | Shipped F1 |
| --- | --- | --- | --- | --- | --- | --- |
| **Hybrid** | **0.9731** | 0.9829 | -0.0098 | 0.9541 | 0.9203 | **0.9369** |
| XGBoost (tuned) | 0.9680 | 0.9760 | -0.0080 | 0.9230 | 0.9161 | 0.9195 |
| LSTM | 0.9470 | 0.9804 | -0.0334 | 0.9278 | 0.8984 | 0.9128 |
| Random Forest | 0.9444 | 0.9617 | -0.0173 | 0.9183 | 0.8704 | 0.8937 |
| FNN | 0.9300 | 0.9561 | -0.0261 | 0.8948 | 0.8527 | 0.8732 |

"Shipped" precision/recall/F1 are at each model's **validation-chosen**
threshold -- the one that would actually deploy -- never re-tuned on test.

**The hybrid holds its lead on genuinely unseen data**: best test PR-AUC
(0.9731) and best F1 (0.9369) of any model, confirming it as the right
final choice. Its validation-to-test gap (-0.0098) is also among the
smallest of the five, meaning its validation score was a reliable
predictor of real-world performance rather than an artifact of
overfitting to validation specifically.

**Worth flagging explicitly: the LSTM generalized noticeably worse than
its validation score suggested** -- a -0.0334 PR-AUC gap, more than 3x
the hybrid's, dropping it from the #2 model on validation to #4 on test
(behind XGBoost, just ahead of Random Forest). This is a legitimate,
reportable result, not a discrepancy to explain away: the LSTM's
sequence-based features may be more sensitive to the specific transaction
patterns present in `fraudTrain.csv`'s validation tail than the flatter,
more general features the tree ensembles and hybrid rely on. It
reinforces the value of the hybrid over deploying the LSTM alone -- the
ensemble's blend proved more robust exactly where its strongest single
component proved less so.
