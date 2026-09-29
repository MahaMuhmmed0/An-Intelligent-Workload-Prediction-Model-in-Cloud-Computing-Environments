# Results summary (final, leakage-free)

Full tables: `documentation/RESULTS_TABLES.md`. Master map:
`documentation/MASTER_EXPERIMENT_TABLE.csv`. Split: regression = random 80/20
(`random_state=42`); forecasting = per-machine chronological 80/20. Metrics on the
held-out **test** set. **MAPE is not used as a principal metric**: it is undefined
for zero actuals and unstable near zero (~12 % of the Google CPU targets are
exactly zero), so it diverges to 10¹³–10¹⁴ %. It is retained in the result CSVs
for completeness only. Principal metrics: **MAE, RMSE, R², sMAPE**.

## 1. Google 2019 Cluster — regression (test R²)

| Model | CPU mean | CPU peak | Mem mean | Mem peak |
|---|---|---|---|---|
| Linear / Ridge | 0.337 | 0.337 | 0.509 | 0.503 |
| Lasso | 0.394 | 0.481 | 0.624 | 0.620 |
| Elastic Net | 0.406 | 0.444 | 0.596 | 0.593 |
| Stepwise | 0.467 | 0.488 | 0.644 | 0.644 |
| Random Forest | 0.832 | 0.810 | 0.936 | 0.935 |
| **HistGradientBoosting** *(additional baseline)* | **0.843** | **0.828** | **0.943** | **0.937** |

## 2. Alibaba 2017 — regression (test R²)

| Model | CPU mean | CPU peak | Mem mean | Mem peak |
|---|---|---|---|---|
| Linear / Ridge | 0.262 | 0.194 | 0.277 | 0.273 |
| Lasso | 0.281 | 0.206 | 0.291 | 0.290 |
| Elastic Net | 0.271 | 0.200 | 0.279 | 0.277 |
| Stepwise | 0.300 | 0.222 | 0.319 | 0.328 |
| Random Forest | 0.334 | 0.308 | 0.321 | 0.325 |
| **HistGradientBoosting** *(additional baseline)* | **0.335** | **0.339** | **0.325** | **0.329** |

HistGradientBoosting is an established, additional non-linear ensemble
baseline evaluated under the identical pipeline as the six principal
regression algorithms above — not a novel algorithm and not a research
contribution of this work. It agrees with Random Forest to within 0.03 R² on
every target (within the ±0.02 cross-version reproducibility tolerance on
seven of eight); the two non-linear ensembles are treated as equivalent, with
no universal winner.

There is no evidence of substantial over-fitting: the mean absolute train–test R²
gap across the 56 regression experiments is approximately 0.02, with a maximum
of approximately 0.07, observed on the noisiest target (Alibaba CPU peak).
Leakage is assessed independently of the train–test gap, through the feature
construction and the training pipeline (see `documentation/LEAKAGE_CORRECTION.md`).

## 3. Alibaba 2017 — forecasting (test R²; ±0.05 run-to-run)

| Model | CPU util | Mem util |
|---|---|---|
| LSTM | **0.796** | 0.930 |
| BiLSTM | 0.768 | **0.933** |
| RNN | 0.783 | 0.900 |

n_train ≈ 118,534 / n_test ≈ 37,988 windows. Forecasting >> regression on the
same resource (memory 0.93 vs 0.32; CPU 0.79 vs 0.33).

## 4. Google 2019 Cluster — forecasting: NOT PERFORMED

Temporal diagnostic (`temporal_diagnostic.json`): the hourly cell series (n=745)
has autocorrelation inside the ±0.072 noise band at every lag; the Alibaba cell
series has ACF 0.87 at lag 1, 0.55 at lag 24. `sequence_attempt.py` scores test
R² between −0.11 and −0.00 — a property of the data, not the models. Google is
used for cross-sectional regression only.

## 5. What the original notebooks reported (invalid — for contrast only)

Linear / Stepwise / Random Forest → R² 1.0000, MAE 0 on Alibaba (target was in
the feature list); Google memory models predicted the request not the usage. See
`documentation/LEAKAGE_CORRECTION.md`. Those numbers are **not** reproduced here.
