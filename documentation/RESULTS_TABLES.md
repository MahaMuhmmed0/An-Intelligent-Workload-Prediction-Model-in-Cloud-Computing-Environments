# Final verified results tables

Every value below is copied from a `results/` file. Held-out **test** partition
unless a column says "train". `n_train` / `n_test` are experiment counts (rows for
regression, windows for forecasting).

* Regression source: `results/tabular_results.csv` (+ `results/regression/*.csv`).
  Google: train 116,586 / test 29,147 entities. Alibaba: train 8,148 / test 2,037.
* Forecasting source: `results/sequence_results.csv` (+ `results/sequence/*.csv`).
  Alibaba: ≈ 118,500 train / ≈ 38,000 test windows.
* Diagnostic source: `results/temporal_diagnostic.json`.

**Metric note.** MAPE is undefined when actual values are zero and becomes
numerically unstable as actual values approach zero. In the Google 2019 Cluster
trace, approximately 12 % of the CPU mean and peak targets are exactly zero,
producing extremely large MAPE values due to division by values approaching zero;
the Alibaba utilisation targets are positive but can be very small, which also
causes MAPE to be dominated by small observations. MAPE is therefore retained for
completeness in the computational results (`results/*.csv`) but is **not** used as
a principal comparison metric. Reported here and in the paper: **MAE, RMSE, R²,
sMAPE**. **Cross-dataset comparison uses R² only** (the Google and Alibaba targets
are different physical quantities — see `DATASET_METHODOLOGY.md`).

**Over-fitting note.** There is no evidence of substantial over-fitting: the mean
absolute train–test R² gap across the 56 regression experiments (including the
additional HistGradientBoosting baseline) is approximately 0.02, with a maximum
of approximately 0.07, observed on the noisiest target (Alibaba CPU peak).
Leakage is assessed independently of the train–test gap, through the feature
construction and the training pipeline (`LEAKAGE_CORRECTION.md`).

**HistGradientBoosting note.** The tables below include a seventh regression
algorithm, HistGradientBoostingRegressor — an established, additional
non-linear ensemble baseline evaluated under the identical pipeline as the six
principal regression algorithms, added to test whether the non-linear-ensemble
behaviour seen with Random Forest is reproduced by an independent boosting
ensemble. It is not a novel algorithm and is not a contribution of this work.
It agrees with Random Forest to within 0.03 R² on every target (within the
±0.02 cross-version reproducibility tolerance on seven of eight); the two
non-linear ensembles are treated as equivalent, and neither is identified as a
universal winner.

---

## A. Google 2019 Cluster — cross-sectional regression

### A.1 CPU mean usage (`cpu_usage_mean`)

| Model | Train MAE | Train R² | Test MAE | Test R² | Test RMSE | Test sMAPE |
|---|---|---|---|---|---|---|
| Linear Regression | 0.0067 | 0.323 | 0.0066 | 0.337 | 0.0145 | 123.4 |
| Ridge Regression | 0.0067 | 0.323 | 0.0066 | 0.337 | 0.0145 | 123.4 |
| Lasso Regression | 0.0064 | 0.387 | 0.0064 | 0.394 | 0.0138 | 118.4 |
| Elastic Net | 0.0064 | 0.402 | 0.0063 | 0.406 | 0.0137 | 117.2 |
| Stepwise Regression | 0.0066 | 0.468 | 0.0066 | 0.467 | 0.0130 | 120.8 |
| Random Forest | 0.0031 | 0.868 | 0.0032 | 0.832 | 0.0073 | 77.9 |
| **HistGradientBoosting** | 0.0035 | 0.862 | 0.0036 | **0.843** | 0.0070 | 93.0 |

### A.2 CPU peak usage (`cpu_usage_peak`)

| Model | Train MAE | Train R² | Test MAE | Test R² | Test RMSE | Test sMAPE |
|---|---|---|---|---|---|---|
| Linear Regression | 0.0216 | 0.325 | 0.0215 | 0.337 | 0.0412 | 109.6 |
| Ridge Regression | 0.0216 | 0.325 | 0.0215 | 0.337 | 0.0412 | 109.6 |
| Lasso Regression | 0.0207 | 0.476 | 0.0204 | 0.481 | 0.0364 | 113.5 |
| Elastic Net | 0.0208 | 0.443 | 0.0206 | 0.444 | 0.0377 | 111.1 |
| Stepwise Regression | 0.0206 | 0.483 | 0.0204 | 0.488 | 0.0361 | 111.8 |
| Random Forest | 0.0103 | 0.832 | 0.0104 | 0.810 | 0.0220 | 74.3 |
| **HistGradientBoosting** | 0.0117 | 0.828 | 0.0116 | **0.828** | 0.0209 | 86.7 |

### A.3 Memory mean usage (`mem_usage_mean`)

| Model | Train MAE | Train R² | Test MAE | Test R² | Test RMSE | Test sMAPE |
|---|---|---|---|---|---|---|
| Linear Regression | 0.0049 | 0.504 | 0.0048 | 0.509 | 0.0114 | 114.6 |
| Ridge Regression | 0.0049 | 0.504 | 0.0048 | 0.509 | 0.0114 | 114.6 |
| Lasso Regression | 0.0043 | 0.628 | 0.0042 | 0.624 | 0.0100 | 105.5 |
| Elastic Net | 0.0042 | 0.598 | 0.0041 | 0.596 | 0.0103 | 92.3 |
| Stepwise Regression | 0.0044 | 0.648 | 0.0043 | 0.644 | 0.0097 | 107.7 |
| Random Forest | 0.0015 | 0.944 | 0.0015 | 0.936 | 0.0041 | 54.5 |
| **HistGradientBoosting** | 0.0017 | 0.945 | 0.0016 | **0.943** | 0.0039 | 65.2 |

### A.4 Memory peak usage (`mem_usage_peak`)

| Model | Train MAE | Train R² | Test MAE | Test R² | Test RMSE | Test sMAPE |
|---|---|---|---|---|---|---|
| Linear Regression | 0.0053 | 0.498 | 0.0052 | 0.503 | 0.0116 | 114.3 |
| Ridge Regression | 0.0053 | 0.498 | 0.0052 | 0.503 | 0.0116 | 114.3 |
| Lasso Regression | 0.0046 | 0.625 | 0.0045 | 0.620 | 0.0102 | 106.0 |
| Elastic Net | 0.0046 | 0.595 | 0.0045 | 0.593 | 0.0105 | 93.4 |
| Stepwise Regression | 0.0047 | 0.648 | 0.0046 | 0.644 | 0.0098 | 108.5 |
| Random Forest | 0.0016 | 0.941 | 0.0016 | 0.935 | 0.0042 | 52.8 |
| **HistGradientBoosting** | 0.0018 | 0.941 | 0.0018 | **0.937** | 0.0041 | 63.8 |

---

## B. Alibaba 2017 — cross-sectional regression

### B.1 CPU mean usage (`cpu_usage_mean`)

| Model | Train MAE | Train R² | Test MAE | Test R² | Test RMSE | Test sMAPE |
|---|---|---|---|---|---|---|
| Linear Regression | 0.0524 | 0.299 | 0.0547 | 0.262 | 0.0736 | 56.0 |
| Ridge Regression | 0.0524 | 0.299 | 0.0547 | 0.262 | 0.0736 | 56.0 |
| Lasso Regression | 0.0528 | 0.309 | 0.0544 | 0.281 | 0.0727 | 55.8 |
| Elastic Net | 0.0529 | 0.304 | 0.0547 | 0.271 | 0.0732 | 56.3 |
| Stepwise Regression | 0.0528 | 0.323 | 0.0542 | 0.300 | 0.0718 | 57.6 |
| Random Forest | 0.0506 | 0.351 | 0.0517 | 0.334 | 0.0700 | 54.2 |
| **HistGradientBoosting** | 0.0506 | 0.352 | 0.0516 | **0.335** | 0.0699 | 54.2 |

### B.2 CPU peak usage (`cpu_usage_peak`)

| Model | Train MAE | Train R² | Test MAE | Test R² | Test RMSE | Test sMAPE |
|---|---|---|---|---|---|---|
| Linear Regression | 0.1017 | 0.261 | 0.1072 | 0.194 | 0.1529 | 60.4 |
| Ridge Regression | 0.1017 | 0.261 | 0.1072 | 0.194 | 0.1529 | 60.4 |
| Lasso Regression | 0.1015 | 0.267 | 0.1066 | 0.206 | 0.1518 | 60.4 |
| Elastic Net | 0.1018 | 0.263 | 0.1072 | 0.200 | 0.1524 | 60.6 |
| Stepwise Regression | 0.1003 | 0.279 | 0.1049 | 0.222 | 0.1503 | 59.9 |
| Random Forest | 0.0959 | 0.362 | 0.0998 | 0.308 | 0.1417 | 57.9 |
| **HistGradientBoosting** | 0.0945 | 0.380 | 0.0976 | **0.339** | 0.1386 | 57.5 |

### B.3 Memory mean usage (`mem_usage_mean`)

| Model | Train MAE | Train R² | Test MAE | Test R² | Test RMSE | Test sMAPE |
|---|---|---|---|---|---|---|
| Linear Regression | 0.1045 | 0.266 | 0.1052 | 0.277 | 0.1284 | 24.4 |
| Ridge Regression | 0.1042 | 0.265 | 0.1047 | 0.277 | 0.1284 | 24.0 |
| Lasso Regression | 0.1034 | 0.277 | 0.1039 | 0.291 | 0.1271 | 24.1 |
| Elastic Net | 0.1043 | 0.267 | 0.1050 | 0.279 | 0.1282 | 24.3 |
| Stepwise Regression | 0.1012 | 0.297 | 0.1010 | 0.319 | 0.1246 | 23.6 |
| Random Forest | 0.1003 | 0.301 | 0.1000 | 0.321 | 0.1244 | 22.9 |
| **HistGradientBoosting** | 0.1000 | 0.303 | 0.0996 | **0.325** | 0.1241 | 22.8 |

### B.4 Memory peak usage (`mem_usage_peak`)

| Model | Train MAE | Train R² | Test MAE | Test R² | Test RMSE | Test sMAPE |
|---|---|---|---|---|---|---|
| Linear Regression | 0.1150 | 0.267 | 0.1151 | 0.273 | 0.1380 | 25.0 |
| Ridge Regression | 0.1149 | 0.267 | 0.1150 | 0.273 | 0.1380 | 24.9 |
| Lasso Regression | 0.1133 | 0.281 | 0.1131 | 0.290 | 0.1363 | 24.6 |
| Elastic Net | 0.1147 | 0.270 | 0.1147 | 0.277 | 0.1376 | 24.9 |
| Stepwise Regression | 0.1094 | 0.307 | 0.1078 | 0.328 | 0.1326 | 23.2 |
| Random Forest | 0.1095 | 0.306 | 0.1081 | 0.325 | 0.1330 | 23.3 |
| **HistGradientBoosting** | 0.1091 | 0.308 | 0.1075 | **0.329** | 0.1326 | 23.1 |

*(For memory-peak, Stepwise Regression, Random Forest, and HistGradientBoosting are all within 0.004 R² of each other.)*

---

## C. Alibaba 2017 — time-series forecasting

Next 5-minute interval predicted from the previous 24. `n_train = 118,534`,
`n_test = 37,988` windows (`cpu_util`); `mem_util` differs by 1. Single
representative run — R² varies ≈ ±0.05 between runs (TensorFlow CPU
nondeterminism). Source: `results/sequence_results.csv`.

### C.1 CPU utilisation (`cpu_util`)

| Model | Train MAE | Train R² | Test MAE | Test R² | Test RMSE | Test sMAPE |
|---|---|---|---|---|---|---|
| **LSTM** | 0.0317 | 0.831 | 0.0307 | **0.796** | 0.0387 | 17.6 |
| RNN | 0.0333 | 0.817 | 0.0311 | 0.783 | 0.0399 | 18.2 |
| BiLSTM | 0.0314 | 0.833 | 0.0325 | 0.768 | 0.0413 | 17.6 |

### C.2 Memory utilisation (`mem_util`)

| Model | Train MAE | Train R² | Test MAE | Test R² | Test RMSE | Test sMAPE |
|---|---|---|---|---|---|---|
| **BiLSTM** | 0.0318 | 0.896 | 0.0248 | **0.933** | 0.0309 | 6.4 |
| LSTM | 0.0326 | 0.892 | 0.0252 | 0.930 | 0.0316 | 6.5 |
| RNN | 0.0347 | 0.885 | 0.0314 | 0.900 | 0.0376 | 8.6 |

The three architectures are within the ≈ ±0.05 run-to-run band of each other on
both targets; no model is consistently superior. Test R² here is far above the
Alibaba regression test R² for the corresponding resource (memory ≈ 0.93 vs
≈ 0.32; CPU ≈ 0.79 vs ≈ 0.33).

### C.3 Google 2019 Cluster forecasting — NOT PERFORMED

`src/google_2019/sequence_attempt.py` (correct construction: chronological split,
scaler on train only) scores **test R² between −0.11 and −0.00** for LSTM/BiLSTM/RNN
on the hourly cell series (CPU −0.009 / −0.001 / −0.016; memory −0.052 / −0.071 /
−0.108). Both train and test R² are ≈ 0 — the models cannot learn, exactly as the
absent temporal autocorrelation predicts (Section D / `TEMPORAL_DIAGNOSTIC.md`).
This is **not** a result of the study; recorded only in
`results/sequence/google_2019__attempt.csv` with `not_used=True`.

---

## D. Google 2019 Cluster — temporal diagnostic (not a model)

Autocorrelation of the hourly cell-level series vs the Alibaba cell-level series
(`results/temporal_diagnostic.json`):

| Series | n | Noise band (±1.96/√n) | ACF lag 1 | ACF lag 6 | ACF lag 12 | ACF lag 24 |
|---|---|---|---|---|---|---|
| Google — CPU, mean agg | 745 | 0.072 | +0.011 | +0.033 | −0.006 | −0.013 |
| Google — CPU, sum agg | 745 | 0.072 | +0.057 | −0.020 | −0.001 | +0.003 |
| Google — memory, mean agg | 745 | 0.072 | +0.023 | +0.014 | −0.034 | −0.022 |
| Alibaba — CPU utilisation | 144 | 0.163 | **+0.872** | +0.698 | +0.539 | +0.549 |
| Alibaba — memory utilisation | 144 | 0.163 | +0.879 | +0.556 | +0.440 | +0.474 |

Google: every lag inside the noise band → white noise. Alibaba: strong decaying
structure. **Conclusion:** Google 2019 Cluster is used for regression only.

---

## E. Highest test R² per (dataset, target)

The highest value in each column, not a "best model" in a statistical sense:
HistGradientBoosting and Random Forest agree to within 0.03 R² on every
regression target (within the ±0.02 cross-version reproducibility tolerance
on seven of eight) and are treated as equivalent, established non-linear
ensembles — neither is a universal winner, and HistGradientBoosting is not a
novel algorithm. The three forecasting architectures are within their
±0.05 run-to-run variation of each other; none is identified as superior.

| Dataset | Module | Target | Highest test R² | Model | Runner-up (within tolerance) |
|---|---|---|---|---|---|
| Google 2019 Cluster | Regression | CPU mean | 0.843 | HistGradientBoosting | Random Forest 0.832 |
| Google 2019 Cluster | Regression | CPU peak | 0.828 | HistGradientBoosting | Random Forest 0.810 |
| Google 2019 Cluster | Regression | Memory mean | 0.943 | HistGradientBoosting | Random Forest 0.936 |
| Google 2019 Cluster | Regression | Memory peak | 0.937 | HistGradientBoosting | Random Forest 0.935 |
| Alibaba 2017 | Regression | CPU mean | 0.335 | HistGradientBoosting | Random Forest 0.334 |
| Alibaba 2017 | Regression | CPU peak | 0.339 | HistGradientBoosting | Random Forest 0.308 |
| Alibaba 2017 | Regression | Memory mean | 0.325 | HistGradientBoosting | Random Forest 0.321 |
| Alibaba 2017 | Regression | Memory peak | 0.329 | HistGradientBoosting | Stepwise Regression 0.328 |
| Alibaba 2017 | Forecasting | CPU utilisation | 0.796 | LSTM | RNN 0.783, BiLSTM 0.768 |
| Alibaba 2017 | Forecasting | Memory utilisation | 0.933 | BiLSTM | LSTM 0.930, RNN 0.900 |
