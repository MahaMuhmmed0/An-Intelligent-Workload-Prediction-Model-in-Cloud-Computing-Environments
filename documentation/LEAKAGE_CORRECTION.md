# Leakage prevention and correction

## The original problem

The pre-rebuild notebooks (an earlier implementation that predated this
repository) reported test scores of
**R² ≈ 1.0000 / MAE ≈ 0** ("100 % / 99.9 % accuracy"). These were **not results**;
they were data leakage. Three distinct leaks were present.

### L1 — target inside the feature matrix (Alibaba notebooks)
```python
features = ['cpu_usage_server', 'mem_usage_server', 'disk_io', ...]
target   = 'cpu_usage_server'          # <-- also in `features`
X = merged_df[features]; y = merged_df[target]
```
The model received the answer as an input and learned the identity map.

### L2 — target-derived features (Google notebooks)
`mean_cpu_usage` was predicted from `std_cpu_usage`, `min_cpu_usage`,
`max_cpu_usage` — all computed from the **same** 11-value `cpu_usage_distribution`
vector as the target. This is not prediction; it is re-describing one sample from
its own order statistics. It inflated R² to 0.95–0.98.
The Google memory notebooks additionally used `cpu_memory_ratio =
mean_cpu_usage / (assigned_memory + 1e-5)` — a feature that is a function of the
target — while predicting `memory_request`.

### L3 — preprocessing / selection / validation touching the test set
* `MinMaxScaler().fit_transform(all_data)` before the split.
* `SimpleImputer` and `PolynomialFeatures` fit on the full frame.
* Stepwise selection scored each candidate feature on `y_test`.
* LSTM/BiLSTM used `validation_data=(X_test, y_test)` with
  `restore_best_weights=True` — the epoch was chosen on the test set.
* Shuffled `train_test_split` of 60-step sequence windows — adjacent
  near-identical windows landed on both sides of the split.

## How the rebuild fixes each

| Leak | Fix in `src/` and `notebooks/` | Where |
|---|---|---|
| **L1** | `FEATURES` is an explicit request/metadata list; `assert TARGET not in FEATURES` runs before every experiment | `src/*/preprocessing.py`, notebook Step 6 |
| **L2** | targets are `average_usage` / aggregated `container_usage` **usage** values; no feature is derived from any target; `cpu_mem_ratio` is `cpu_request / mem_request` (requests only) | `src/*/preprocessing.py` |
| **L3 — scaling** | `SimpleImputer`, `PolynomialFeatures`, `Scaler` are steps of an sklearn `Pipeline`; `RobustPrep` bounds come from `np.nanpercentile(X_train, …)` | `src/evaluation.py` `fit_pipeline_model`, `RobustPrep` |
| **L3 — selection** | Stepwise = `LassoCV(cv=5)` fit on `Z_train`; OLS refit on `Z_train[:, mask]` | `src/evaluation.py` `fit_stepwise` |
| **L3 — CV** | `GridSearchCV(cv=5)` receives `X_train, y_train` only | `src/evaluation.py` `fit_pipeline_model` |
| **L3 — sequence split** | per-machine **chronological** split; the `MinMaxScaler` is `fit` on training points only; `validation_split=0.1` takes the **tail of the training windows**; `shuffle=False`; no window crosses a machine's split boundary | `src/alibaba_2017/preprocessing.py` `build_forecasting_windows` |

## Evidence the correction holds

Leakage is assessed **independently of the train–test performance gap**, through
the following:

1. **Feature construction.** The feature set is an explicit list of resource-request
   and static-metadata columns (`src/*/preprocessing.py`); no feature is a target
   or a function of a target; `cpu_mem_ratio` uses requests only. An
   `assert TARGET not in FEATURES` runs in Step 6 of every regression notebook.
2. **Pipeline discipline.** Imputation, winsorisation, polynomial expansion,
   scaling, `GridSearchCV` and the `LassoCV` stepwise selector are all fit on the
   training partition only (verified by code inspection — see the table above).
3. **Plausible ceilings.** The strongest models (the non-linear ensembles,
   Random Forest and the additional HistGradientBoosting baseline) top out at
   test R² ≈ 0.83–0.84 (Google CPU) and ≈ 0.34 (Alibaba CPU) — not 1.0.
4. **Independent re-execution.** Multiple modules
   (`alibaba_2017/linear_regression`, `alibaba_2017/elastic_net`,
   `google_2019/random_forest`, the Ridge notebook, `temporal_diagnostic`) were
   run from a clean process and reproduced their recorded numbers exactly.

**On over-fitting (separate question).** There is no evidence of *substantial*
over-fitting: the mean absolute train–test R² gap across the 56 regression
experiments (including the additional HistGradientBoosting baseline) is
approximately 0.02, with a maximum of approximately 0.07 on the
noisiest target (Alibaba CPU peak). A small train-side optimism on the Alibaba
CPU targets is expected given GridSearchCV tuning and a weak signal. Similar
train and test scores are **not** taken as proof of the absence of leakage.

## Statement for the paper

> The models presented here do **not** achieve near-perfect accuracy. Earlier
> internal experiments that appeared to do so contained target leakage (the
> prediction target, or statistics derived from it, were present among the input
> features) and are not reported as results. In the corrected design the feature
> set is restricted to the resource request and static scheduling metadata, all
> preprocessing and model selection are fit on the training partition only, and
> the test partition is held out entirely.
