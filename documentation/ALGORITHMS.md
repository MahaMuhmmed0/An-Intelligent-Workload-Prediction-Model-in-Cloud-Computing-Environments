# Actual algorithm implementations

Every algorithm below is the **exact** implementation used to produce
`results/`. **Authoritative source of truth: the per-experiment modules under
`src/`.** The `notebooks/` versions (self-contained, 19 numbered steps) contain
the same logic copied verbatim from `src/`, for inspection and manual runs.
`common/pipeline.py`, `common/sequence.py` and the root `run_tabular.py` /
`run_sequence.py` are configuration-matched batch runners kept for convenience —
reproduce with `src/`.

Shared constants: `RANDOM_STATE = 42`, `EPS = 1e-9`.

---

## PART A — Cross-sectional regression (Google 2019 Cluster **and** Alibaba 2017)

All seven regression algorithms (the six principal algorithms plus the
additional HistGradientBoosting baseline) share the same data path, split,
preprocessing and evaluation; only the estimator, grid, polynomial expansion
and scaler differ.

### A.0 Common pipeline (identical for both datasets)

| Item | Value |
|---|---|
| **Task** | predict a workload entity's observed usage from its resource request |
| **Input features — Google 2019 Cluster** | `cpu_request`, `mem_request`, `cpu_mem_ratio`, `priority`, `scheduling_class`, `collection_type` |
| **Input features — Alibaba 2017** | `cpu_request`, `mem_request`, `cpu_mem_ratio`, `plan_disk`, `cpuset_width` |
| **Targets** (one model per target) | `cpu_usage_mean`, `cpu_usage_peak`, `mem_usage_mean`, `mem_usage_peak` |
| **Target never in X** | enforced by `assert TARGET not in FEATURES` |
| **`cpu_mem_ratio`** | `cpu_request / (mem_request + EPS)` — from **requests only**, never usage |
| **Missing values** | `SimpleImputer(strategy="median")` — first step of the sklearn `Pipeline`, so fit on training folds only |
| **Skew / outlier handling** | `RobustPrep`: winsorise every feature to the **training** 1st/99th percentile, then `log1p` on non-negative features. Bounds learned from `X_train` only. |
| **Train/test split** | `train_test_split(test_size=0.20, random_state=42, shuffle=True)` — random, because the task is cross-sectional (each row is an independent workload entity, not a time step) |
| **Cross-validation** | `GridSearchCV(cv=5, scoring="neg_mean_absolute_error", n_jobs=4)` fit on the **training partition only** (where a grid exists) |
| **Evaluation** | `MAE`, `MAPE` (flagged unreliable), `RMSE`, `R²`, `sMAPE` — reported for train and test |
| **Result row** | `results/regression/<stem>.csv`; aggregated into `results/tabular_results.csv` |

`RobustPrep` (verbatim):
```python
class RobustPrep:
    def fit(self, X):
        X = np.asarray(X, float)
        self.lo_ = np.nanpercentile(X, 1, axis=0)
        self.hi_ = np.nanpercentile(X, 99, axis=0)
        self.logmask_ = np.nanmin(X, axis=0) >= 0
        return self
    def transform(self, X):
        X = np.array(X, float)
        X = np.clip(X, self.lo_, self.hi_)
        X[:, self.logmask_] = np.log1p(X[:, self.logmask_])
        return X
```

Metrics (verbatim):
```python
def smape(y, p):
    return float(np.mean(2*np.abs(p - y) / (np.abs(y) + np.abs(p) + EPS)) * 100)

MAE   = mean_absolute_error(y, p)
MAPE  = mean_absolute_percentage_error(y, p) * 100      # unreliable: usage is often ~0
RMSE  = np.sqrt(np.mean((y - p) ** 2))
R2    = r2_score(y, p)
sMAPE = smape(y, p)
```

---

### A.1 Linear Regression

| | |
|---|---|
| Estimator | `sklearn.linear_model.LinearRegression()` |
| Polynomial features | none |
| Scaler | `MinMaxScaler` |
| Hyperparameters | none (OLS) |
| GridSearchCV | not used |
| Pipeline | `SimpleImputer(median) → MinMaxScaler → LinearRegression` |
| Training | `pipe.fit(X_train, y_train)` |
| Prediction | `pipe.predict(X_train)`, `pipe.predict(X_test)` |
| Source | `src/google_2019/linear_regression.py`, `src/alibaba_2017/linear_regression.py` |
| Notebooks | `notebooks/{google_2019,alibaba_2017}/*_Linear_Regression_{CPU,Memory}_{mean,peak}.ipynb` |

### A.2 Ridge Regression (L2)

| | |
|---|---|
| Estimator | `Ridge(random_state=42)` |
| Polynomial features | none |
| Scaler | `StandardScaler` |
| Hyperparameter | `alpha` |
| GridSearchCV | `alpha ∈ {0.001, 0.01, 0.1, 1, 10, 100}`, `cv=5`, `scoring="neg_mean_absolute_error"` |
| Pipeline | `SimpleImputer(median) → StandardScaler → Ridge` |
| Training | `GridSearchCV.fit(X_train, y_train)` → `best_estimator_` |
| Source | `src/{google_2019,alibaba_2017}/ridge_regression.py` |

### A.3 Lasso Regression (L1)

| | |
|---|---|
| Estimator | `Lasso(max_iter=3000, tol=1e-3, random_state=42)` |
| Polynomial features | `PolynomialFeatures(degree=2, include_bias=False)` |
| Scaler | `StandardScaler` |
| Hyperparameter | `alpha` |
| GridSearchCV | `alpha ∈ {1e-4, 1e-3, 1e-2, 0.1}`, `cv=5`, `scoring="neg_mean_absolute_error"` |
| Pipeline | `SimpleImputer(median) → PolynomialFeatures(2) → StandardScaler → Lasso` |
| Source | `src/{google_2019,alibaba_2017}/lasso_regression.py` |

### A.4 Elastic Net (L1 + L2)

| | |
|---|---|
| Estimator | `ElasticNet(max_iter=3000, tol=1e-3, random_state=42)` |
| Polynomial features | `PolynomialFeatures(degree=2, include_bias=False)` |
| Scaler | `StandardScaler` |
| Hyperparameters | `alpha`, `l1_ratio` |
| GridSearchCV | `alpha ∈ {1e-3, 1e-2, 0.1}`, `l1_ratio ∈ {0.3, 0.6}`, `cv=5`, `scoring="neg_mean_absolute_error"` |
| Pipeline | `SimpleImputer(median) → PolynomialFeatures(2) → StandardScaler → ElasticNet` |
| Source | `src/{google_2019,alibaba_2017}/elastic_net.py` |

### A.5 Stepwise Regression

Embedded selection (not classic forward/backward). No GridSearchCV — `LassoCV`
performs its own internal 5-fold CV. Every fitted transform and the selector see
**training data only**.

```python
imp  = SimpleImputer(strategy="median").fit(X_train)
poly = PolynomialFeatures(degree=2, include_bias=False).fit(imp.transform(X_train))
sc   = StandardScaler().fit(poly.transform(imp.transform(X_train)))
Ztr  = sc.transform(poly.transform(imp.transform(X_train)))
Zte  = sc.transform(poly.transform(imp.transform(X_test)))

sel  = LassoCV(cv=5, max_iter=10000, random_state=42).fit(Ztr, y_train)   # picks alpha + selects
mask = sel.coef_ != 0                                                     # retained terms
ols  = LinearRegression().fit(Ztr[:, mask], y_train)                      # refit on retained terms
train_pred, test_pred = ols.predict(Ztr[:, mask]), ols.predict(Zte[:, mask])
```
Reported hyperparameters: `LassoCV.alpha_` and `n_selected` (= `mask.sum()`).
Source: `src/{google_2019,alibaba_2017}/stepwise_regression.py`.

### A.6 Random Forest Regression

| | |
|---|---|
| Estimator | `RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=42, max_samples=0.5, min_samples_leaf=10)` |
| Polynomial features | none (trees model interactions directly) |
| Scaler | `StandardScaler` (harmless for trees; kept so the pipeline is uniform) |
| Hyperparameter | `max_depth` |
| GridSearchCV | `max_depth ∈ {6, 12}`, `cv=5`, `scoring="neg_mean_absolute_error"` |
| Pipeline | `SimpleImputer(median) → StandardScaler → RandomForestRegressor` |
| Source | `src/{google_2019,alibaba_2017}/random_forest.py` |

### A.7 HistGradientBoosting Regression *(additional baseline)*

An established, additional non-linear ensemble baseline evaluated under the
identical pipeline as the six principal regression algorithms above, to check
whether behaviour attributed to Random Forest is specific to that model. It
is **not** a novel algorithm and is **not** a research contribution of this
work.

| | |
|---|---|
| Estimator | `HistGradientBoostingRegressor(random_state=42, early_stopping=False)`; all other params = scikit-learn defaults (`loss='squared_error'`, `max_iter=100`, `max_leaf_nodes=31`, `l2_regularization=0.0`). `early_stopping=False` so the full training fold is used, matching Random Forest and keeping runs deterministic. |
| Polynomial features | none (trees model interactions directly) |
| Scaler | `StandardScaler` |
| Hyperparameters | `learning_rate`, `max_depth` |
| GridSearchCV | `learning_rate ∈ {0.05, 0.1}`, `max_depth ∈ {6, 12}`, `cv=5`, `scoring="neg_mean_absolute_error"` |
| Pipeline | `SimpleImputer(median) → StandardScaler → HistGradientBoostingRegressor` |
| Source | `src/{google_2019,alibaba_2017}/hist_gradient_boosting.py` |
| Output | `results/regression/histgbr/{dataset}__hist_gradient_boosting.csv` (additive; does not modify `results/tabular_results.csv`) |

---

## PART B — Time-series forecasting (Alibaba 2017 only)

LSTM, BiLSTM and RNN share **one** data pipeline, compilation, training schedule
and evaluation; only the recurrent layer differs.

### B.0 Common sequence pipeline

| Item | Value |
|---|---|
| **Task** | one-step-ahead forecast of machine utilisation |
| **Input file** | `server_usage.csv` — machine-level telemetry, 5-minute cadence, 12 h (144 intervals/machine) |
| **Series construction** | one series per `machine_id`, sorted by `ts` (chronological); values rescaled percent → `[0,1]`; machines with `< 40` of the 144 intervals dropped → **1 310 machine series** |
| **Target** | `cpu_util` or `mem_util` (next interval) |
| **Window / horizon** | previous `WINDOW = 24` intervals → `HORIZON = 1` |
| **Train/test split** | **per-machine chronological**: for each machine, first 80 % of its timeline → training points, last 20 % → test points (with `WINDOW+HORIZON-1` context so the first test window is complete). No window straddles the split. |
| **Scaler** | one `MinMaxScaler`, `fit` on the concatenation of **all training points only**, then applied to test points |
| **n_train / n_test windows** | ≈ 118 500 / ≈ 38 000 |
| **Architecture** | 2 recurrent layers (64 → 32 units), `Dropout(0.2)` after each, `Dense(1)` |
| **Optimizer** | `Adam(learning_rate=1e-3)` |
| **Loss** | `Huber(delta=1.0)` |
| **Callbacks** | `EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)`, `ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)` |
| **Training** | `epochs=30`, `batch_size=128`, `validation_split=0.1` (tail of the training windows — never the test set), `shuffle=False` |
| **Seed** | `tf.random.set_seed(42)`, `np.random.seed(42)` — TF on CPU is still **not** bit-for-bit deterministic; R² varies ≈ ±0.05 between runs |
| **Prediction** | `model.predict` on train and test, then inverse-scaled with the same `MinMaxScaler` |
| **Evaluation** | `MAE`, `MAPE`, `RMSE`, `R²`, `sMAPE` on inverse-scaled values, train and test |
| **Result row** | `results/sequence/Alibaba_2017_<Model>_<CPU\|Memory>.csv`; aggregated into `results/sequence_results.csv` |

Windowing (verbatim):
```python
def make_windows(a, w=24, h=1):
    X, y = [], []
    for i in range(len(a) - w - h + 1):
        X.append(a[i:i + w]); y.append(a[i + w + h - 1])
    return np.array(X), np.array(y)
```

### B.1 LSTM
```python
Sequential([
    Input((24, 1)),
    LSTM(64, return_sequences=True), Dropout(0.2),
    LSTM(32),                        Dropout(0.2),
    Dense(1),
])
```
Source: `src/alibaba_2017/lstm.py` · Notebooks: `Alibaba_2017_LSTM_{CPU,Memory}.ipynb`

### B.2 BiLSTM
```python
Sequential([
    Input((24, 1)),
    Bidirectional(LSTM(64, return_sequences=True)), Dropout(0.2),
    Bidirectional(LSTM(32)),                        Dropout(0.2),
    Dense(1),
])
```
Source: `src/alibaba_2017/bilstm.py` · Notebooks: `Alibaba_2017_BiLSTM_{CPU,Memory}.ipynb`

### B.3 RNN (vanilla / SimpleRNN)
```python
Sequential([
    Input((24, 1)),
    SimpleRNN(64, return_sequences=True), Dropout(0.2),
    SimpleRNN(32),                        Dropout(0.2),
    Dense(1),
])
```
Source: `src/alibaba_2017/rnn.py` · Notebooks: `Alibaba_2017_RNN_{CPU,Memory}.ipynb`

---

## PART C — Google 2019 Cluster temporal diagnostic (NOT a model)

`src/google_2019/temporal_diagnostic.py` · `notebooks/google_2019/Google_2019_Temporal_Diagnostic.ipynb`

1. Load `borg_traces_data.csv` (`start_time`, `average_usage` columns).
2. Bucket `start_time` into 1-hour bins (`BUCKET_US = 3_600_000_000` µs).
3. Parse `average_usage` → `cpus` / `memory`.
4. Aggregate per hour — **both** `mean` (typical per-instance intensity) and
   `sum` (total offered load).
5. Reindex onto a regular hourly grid; `interpolate("linear").bfill().ffill()`
   the missing buckets.
6. `acf(x, lag) = corrcoef(x[:-lag], x[lag:])` at lags 1, 2, 3, 6, 12, 24, 48.
7. Noise band = `1.96 / sqrt(n)` (95 % white-noise CI half-width).
8. Comparison series: Alibaba cell-level = `server_usage` mean across machines per
   5-minute timestamp.
9. Write `results/temporal_diagnostic.json`.

Not a forecasting experiment; produces no R². See `TEMPORAL_DIAGNOSTIC.md`.
