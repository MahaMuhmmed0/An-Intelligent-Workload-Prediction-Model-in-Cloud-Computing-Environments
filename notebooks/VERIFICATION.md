# Code → Experiment → Target → Result verification

Every notebook is generated from `build_notebooks.py`, whose code fragments are
**identical** to the `src/` package that produced `results/tabular_results.csv`
and `results/sequence_results.csv` (same `random_state=42`, same 150 000-row
Google sample, same 80/20 split, same Pipeline, same GridSearchCV grids, same
architectures and callbacks).

## Directly executed & checked against the recorded results

| Notebook | Target | Recorded result | Re-run result | Match |
|---|---|---|---|---|
| `alibaba_2017/Alibaba_2017_Linear_Regression_CPU_mean.ipynb` | `cpu_usage_mean` | test R² **0.2624**, MAE 0.0547 | test R² 0.2624, MAE 0.0547 | ✅ exact |
| `google_2019/Google_2019_RandomForest_CPU_mean.ipynb` | `cpu_usage_mean` | test R² **0.832**, train R² 0.868 | test R² 0.8322, train R² 0.8684 | ✅ exact |
| `src/google_2019/linear_regression.py` (same code path) | all 4 | test R² 0.337 / 0.509 / 0.337 / 0.503 | identical | ✅ exact |
| `google_2019/Google_2019_Temporal_Diagnostic.ipynb` | — | Google ACF≈0 all lags; Alibaba lag1≈0.89 | Google lag1 +0.011, lag24 −0.013; Alibaba lag1 +0.872, lag24 +0.549 | ✅ matches `temporal_diagnostic.json` |
| `alibaba_2017/Alibaba_2017_LSTM_CPU.ipynb` | `cpu_util` | test R² 0.73–0.80 (varies), n_test 37 988 | test R² 0.739, MAE 0.0354, RMSE 0.0438, n_test 37 988 | ✅ within variance |

## Verified by construction (identical code path, not separately executed)

The remaining **46 six-algorithm regression notebooks** call the same three
fragments that the two executed regression notebooks use — only the estimator
+ grid + target change, and each is copied verbatim from the corresponding
`src/` file. The **8 additional HistGradientBoosting regression notebooks**
(56 regression notebooks in total) follow the identical pipeline, copied
verbatim from `src/*/hist_gradient_boosting.py`; HistGradientBoosting is an
established, additional non-linear ensemble baseline, not a novel algorithm.
The **6 forecasting notebooks** share one pipeline (`build_forecasting_windows`
logic) and differ only in the recurrent layer.

## Explicit reproducibility statements

**Google 2019 Cluster · Regression** (`results/tabular_results.csv`)

| Algorithm | CPU mean R² | Mem mean R² | CPU peak R² | Mem peak R² |
|---|---|---|---|---|
| Linear / Ridge | 0.34 | 0.51 | 0.34 | 0.50 |
| Lasso | 0.39 | 0.62 | 0.48 | 0.62 |
| Elastic Net | 0.41 | 0.60 | 0.44 | 0.59 |
| Stepwise | 0.47 | 0.64 | 0.49 | 0.64 |
| Random Forest | **0.83** | **0.94** | **0.81** | **0.94** |

(no evidence of substantial over-fitting: mean |train−test R²| ≈ 0.02, max ≈ 0.07
on Alibaba CPU peak; leakage assessed from feature construction + pipeline, not
from the train–test gap)

**Alibaba 2017 · Regression** (`results/tabular_results.csv`)

| Algorithm | CPU mean R² | Mem mean R² | CPU peak R² | Mem peak R² |
|---|---|---|---|---|
| Linear / Ridge | 0.26 | 0.28 | 0.19 | 0.27 |
| Lasso | 0.28 | 0.29 | 0.21 | 0.29 |
| Elastic Net | 0.27 | 0.28 | 0.20 | 0.28 |
| Stepwise | 0.30 | 0.32 | 0.22 | 0.33 |
| Random Forest | **0.33** | **0.32** | **0.31** | **0.32** |

**Alibaba 2017 · Forecasting** (`results/sequence_results.csv`) — representative run,
±0.05 between runs:

| Model | CPU R² | Memory R² |
|---|---|---|
| LSTM | 0.73–0.80 | 0.91–0.93 |
| BiLSTM | 0.76–0.77 | 0.93 |
| RNN | 0.78–0.79 | 0.91 |

**Google 2019 Cluster · Forecasting** — **NOT a final experiment.**
`src/google_2019/sequence_attempt.py` scores R² ≈ 0 (to slightly negative) for
LSTM/BiLSTM/RNN. The reason is in the diagnostic notebook: the extract has no
temporal autocorrelation. Final design: Google → regression only.

## Unverified / could-not-verify items

1. **`results/tabular_results.csv` (combined file) is currently open in another
   program** (Excel) and locked — the individual per-notebook CSVs in
   `results/regression/` are the writable source of truth; run
   `python -m src.run_all regression` after closing it to rebuild the combined file.
2. **Forecasting exact numbers** cannot be reproduced bit-for-bit — TensorFlow on
   CPU is nondeterministic. The recorded `sequence_results.csv` is one valid run;
   any re-run lands within ±0.05 R².
3. **No original Google/Alibaba source code was found** for the *pre-rebuild*
   experiments beyond the notebooks that predated this repository; those
   contained the data-leakage bugs and their "100% / 99.9%" numbers are **not**
   reproduced here by design.
   Everything in `src/` and `notebooks/` is the corrected rebuild, clearly labelled
   as such — nothing is presented as "the original implementation".
