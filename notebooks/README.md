# Notebooks — one per experiment

63 self-contained Jupyter notebooks that reproduce the leakage-controlled
experiments behind `../results/`: 55 six-algorithm notebooks (48 regression +
6 forecasting + 1 diagnostic) plus 8 additional HistGradientBoosting
regression notebooks (an established, additional non-linear ensemble
baseline, not a novel algorithm). Open any one in Jupyter/Anaconda and run
top to bottom.

```
notebooks/
├── build_notebooks.py          generator — all notebooks come from this single source
├── _run_nb.py                  helper: run a notebook's code cells without Jupyter (for CI/verification)
├── MASTER_TABLE.md / .csv       Dataset | Module | Target | Algorithm | Source File | Notebook | Result File
├── google_2019/                24 regression notebooks + 1 temporal-diagnostic notebook
└── alibaba_2017/               24 regression notebooks + 6 forecasting notebooks
```

## What each notebook contains

**Regression** (`*_CPU_mean.ipynb`, `*_CPU_peak.ipynb`, `*_Memory_mean.ipynb`, `*_Memory_peak.ipynb`)
19 numbered steps: imports → load → preprocess → parse columns → feature engineering →
define X/y → missing values → **train/test split (random 80/20, random_state=42)** →
**RobustPrep + Pipeline preprocessing (fit on train only)** → build model →
**GridSearchCV(cv=5) on train only** → train → train preds → test preds →
train MAE/MAPE/R² → test MAE/MAPE/R² → RMSE + sMAPE → actual-vs-predicted plot →
save row to `../results/regression/<stem>.csv`.

* Target is set explicitly at the top of Step 6 (`TARGET = "cpu_usage_mean"` etc.)
  and an `assert TARGET not in FEATURES` guards against the old "100%" leakage bug.
* Features are the resource **request + static metadata only** — never a usage value.

**Forecasting** (`Alibaba_2017_{LSTM,BiLSTM,RNN}_{CPU,Memory}.ipynb`)
Full sequence pipeline in one notebook: load `server_usage.csv` → one series per
machine, chronologically ordered → per-machine **chronological** 80/20 split →
`MinMaxScaler` **fit on training points only** → transform test → previous-24-interval
windows (no window crosses the split) → architecture (64→32 recurrent, Dropout 0.2,
Dense 1) → Adam(1e-3) + Huber(δ=1.0) → EarlyStopping + ReduceLROnPlateau, 30 epochs,
`validation_split=0.1`, `shuffle=False` → predict → inverse-scale →
train & test MAE/RMSE/R²/sMAPE → plot → save row to `../results/sequence/<stem>.csv`.

**Temporal diagnostic** (`Google_2019_Temporal_Diagnostic.ipynb`)
Not a forecasting experiment. Builds the hourly cell-level aggregate (mean **and**
sum), interpolates missing buckets, computes ACF and the ±1.96/√n noise band,
compares with an Alibaba cell-level series, writes `../results/temporal_diagnostic.json`,
and prints the decision: **Google 2019 Cluster → regression only; forecasting → Alibaba only.**

## Reproducibility notes

* The `src/` package is the same code in importable form; `python -m src.run_all`
  runs everything and rebuilds `results/tabular_results.csv` and
  `results/sequence_results.csv` from the per-experiment CSVs.
* Google notebooks read a **150 000-row sample** of `borg_traces_data.csv`
  (`random_state=42`) — the exact sample behind the final results (~145 700 rows
  after filtering; train 116 586 / test 29 147).
* TensorFlow CPU training is **not bit-for-bit deterministic**; forecasting R²
  varies by ~±0.05 between runs even with a fixed seed. Regression is exactly
  reproducible.
