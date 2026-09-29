# Workload-Prediction-Rebuild

Code and reproducibility package for the paper **"An Intelligent Workload
Prediction Model in Cloud Computing Environments"** — a leakage-controlled
re-implementation of a cloud workload-prediction study on two public
cluster traces, packaged for reproduction.

* **Google 2019 Cluster** (the public extract of Google's Borg cluster-manager
  trace, 2019 — trace v3) — cross-sectional regression only
* **Alibaba 2017** (`cluster-trace-v2017`) — cross-sectional regression **and**
  time-series forecasting

63 principal experiments: 28 Google regression + 28 Alibaba regression (seven
regression algorithms × four targets per trace, including an additional
HistGradientBoosting baseline — see `results/regression/histgbr/`) + 6 Alibaba
forecasting + 1 Google temporal diagnostic.

> **Dataset label.** The dataset is referred to throughout as **"Google 2019
> Cluster"**. "Borg" is used only where the underlying trace-collection system is
> technically relevant; it is not used as the dataset name.

> **Authoritative code path.** The per-experiment modules under **`src/`** (run
> via `python -m src.<dataset>.<algorithm>` or `python -m src.run_all`) are the
> source of truth for every number in `results/`. The Jupyter notebooks under
> `notebooks/` contain the same logic, copied verbatim from `src/`, for
> inspection and manual runs. `common/pipeline.py`, `common/sequence.py`,
> `run_tabular.py` and `run_sequence.py` are configuration-matched batch runners
> kept for convenience — **use `src/` for reproduction** to avoid any risk of
> code drift. `diagnose_borg_temporal.py` is a superseded development script,
> replaced by `src/google_2019/temporal_diagnostic.py`.

---

## 1. Project layout

```
Workload-Prediction-Rebuild/
├── src/                         importable per-algorithm modules
│   ├── evaluation.py            shared metrics, RobustPrep, split, GridSearchCV wrapper, stepwise, windowing
│   ├── run_all.py               run everything; rebuild results/tabular_results.csv + sequence_results.csv
│   ├── google_2019/
│   │   ├── preprocessing.py            load_entity_table()  (150k sample, dict parsing)
│   │   ├── linear_regression.py … random_forest.py, hist_gradient_boosting.py    (7 regressors)
│   │   ├── temporal_diagnostic.py      hourly cell aggregate + ACF  -> temporal_diagnostic.json
│   │   └── sequence_attempt.py         LSTM/BiLSTM/RNN on Google -> R2≈0, flagged NOT USED
│   └── alibaba_2017/
│       ├── preprocessing.py            load_entity_table() + build_forecasting_windows()
│       ├── linear_regression.py … random_forest.py, hist_gradient_boosting.py    (7 regressors)
│       ├── _forecasting.py             shared train/eval loop
│       └── lstm.py / bilstm.py / rnn.py               (architectures)
├── notebooks/                   63 self-contained .ipynb (55 six-algorithm + 8
│                                 additional HistGradientBoosting notebooks; 19
│                                 numbered steps each)
│   ├── build_notebooks.py       generator (single source for all notebooks)
│   ├── MASTER_TABLE.md / .csv    Dataset | Module | Target | Algorithm | Source | Notebook | Result
│   ├── VERIFICATION.md          code -> result checks
│   ├── google_2019/  (25)   alibaba_2017/  (30)
├── common/                      batch runners (pipeline.py, sequence.py, run_tabular.py, run_sequence.py)
├── results/
│   ├── tabular_results.csv / .json          48 regression rows (six-algorithm set)
│   ├── regression/histgbr/*.csv             8 additional HistGradientBoosting rows
│   │                                        (56 regression rows combined)
│   ├── sequence_results.csv / .json         6 Alibaba forecasting rows
│   ├── temporal_diagnostic.json             ACF evidence
│   ├── RESULTS_SUMMARY.md
│   ├── regression/*.csv                     one row per regression experiment
│   └── sequence/*.csv                       one row per forecasting experiment
├── documentation/
│   ├── AUDIT_FINDINGS.md          ALGORITHMS.md           LEAKAGE_CORRECTION.md
│   ├── DATASET_METHODOLOGY.md     TEMPORAL_DIAGNOSTIC.md  METHODOLOGY_DRAFT.md
│   ├── RESULTS_TABLES.md          RESULTS_DISCUSSION_DRAFT.md   LIMITATIONS.md
│   └── MASTER_EXPERIMENT_TABLE.csv / .md
├── FINAL_RESEARCH_CHECKLIST.md
└── WORKFLOW.html                 methodology / corrections record (also a shared artifact)
```

## 2. Environment requirements

* Python 3.10–3.12 (developed on 3.12).
* Regression: `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`.
* Forecasting: additionally `tensorflow` (2.x; CPU is fine).

```
pip install numpy pandas scikit-learn scipy matplotlib tensorflow
```

The regression notebooks run on scikit-learn alone; only the six forecasting
notebooks and `src/google_2019/sequence_attempt.py` need TensorFlow.

## 3. Dataset locations

Neither dataset is bundled in this repository (they are large, public traces
— see the download links below). Dataset locations are **configurable**, not
hardcoded: `src/evaluation.py` resolves them from environment variables, with
a repository-relative default so a fresh clone works out of the box once the
files are placed at the default location.

| Dataset | Env var | Default (repo-relative) | Files needed |
|---|---|---|---|
| Google 2019 Cluster | `GOOGLE_2019_CSV` | `data/google_2019/borg_traces_data.csv` | `borg_traces_data.csv` |
| Alibaba 2017 | `ALIBABA_2017_DIR` | `data/alibaba_2017/` | `container_event.csv`, `container_usage.csv`, `server_usage.csv` |

Either place your copies at the default paths above, or set the environment
variables to point at your own location, e.g. (PowerShell):

```powershell
$env:GOOGLE_2019_CSV = "D:\my-data\borg_traces_data.csv"
$env:ALIBABA_2017_DIR = "D:\my-data\alibaba-2017"
```

or (bash):

```bash
export GOOGLE_2019_CSV=/data/borg_traces_data.csv
export ALIBABA_2017_DIR=/data/alibaba-2017
```

See `.env.example` for a template. The 63 self-contained notebooks under
`notebooks/` embed the same repository-relative default
(`../../data/<dataset>/...`) directly, since each notebook is meant to run
standalone without importing `src/`; edit the `BORG_CSV` / `ALIBABA_DIR` line
near the top of a notebook if your data lives elsewhere and you are not
setting the environment variables.

**Where to get the data (official public sources, not mirrored here):**
* Google 2019 Cluster (Borg trace v3) — https://github.com/google/cluster-data
  (Wilkes, J.: *Google cluster-usage traces v3*, Technical report, Google Inc., 2020).
* Alibaba cluster-trace-v2017 — https://github.com/alibaba/clusterdata.

## 4. How to run the regression experiments

**One notebook** (open in Jupyter/Anaconda, run top to bottom):
`notebooks/google_2019/Google_2019_RandomForest_CPU_mean.ipynb` →
`results/regression/Google_2019_RandomForest_CPU_mean.csv`.

**One algorithm, all four targets** (module):
```
python -m src.google_2019.random_forest
python -m src.alibaba_2017.linear_regression
```

**All 48 six-algorithm regression experiments** and rebuild the aggregate table (authoritative):
```
python -m src.run_all regression        # -> results/tabular_results.csv
```
(Legacy convenience path, configuration-matched: `python run_tabular.py --peak
--borg-sample 150000` — the `--borg-sample` flag name predates the rename to
"Google 2019 Cluster"; it refers to the same `borg_traces_data.csv` file.)

**The additional 8 HistGradientBoosting experiments** (opt-in, additive — not
folded into `tabular_results.csv`):
```
python -m src.run_all histgbr           # -> results/tabular_results_histgbr.csv
```

## 5. How to run the forecasting experiments

**One notebook:** `notebooks/alibaba_2017/Alibaba_2017_LSTM_CPU.ipynb`.

**All six** and rebuild the aggregate table (authoritative):
```
python -m src.run_all forecasting       # -> results/sequence_results.csv
```
(Legacy convenience path, configuration-matched: `python run_sequence.py --only
alibaba --epochs 30`.)

## 6. Google temporal diagnostic

```
python -m src.run_all diagnostic
# or  notebooks/google_2019/Google_2019_Temporal_Diagnostic.ipynb
```
writes `results/temporal_diagnostic.json`. Data-suitability analysis, not a
forecasting model — see `documentation/TEMPORAL_DIAGNOSTIC.md`.

## 7. How to reproduce the result tables

```
python -m src.run_all               # regression + forecasting + diagnostic
python documentation/build_master_table.py     # -> documentation/MASTER_EXPERIMENT_TABLE.*
```
`documentation/RESULTS_TABLES.md` holds the paper-ready tables.

## 8. Where each output is saved

| Experiment | Per-experiment file | Aggregated into |
|---|---|---|
| Regression (any of 48) | `results/regression/<Dataset>_<Algo>_<Target>.csv` | `results/tabular_results.csv` |
| Forecasting (any of 6) | `results/sequence/Alibaba_2017_<Model>_<CPU\|Memory>.csv` | `results/sequence_results.csv` |
| Temporal diagnostic | — | `results/temporal_diagnostic.json` |
| Google forecasting attempt (not used) | `results/sequence/google_2019__attempt.csv` | — (excluded by design) |

## 9. Reproducibility notes

* `random_state = 42` everywhere; the Google 150,000-row sample is fixed.
* Regression numbers are exactly reproducible **within one scikit-learn version**;
  CV-tuned models can differ by ~±0.02 in R² between scikit-learn versions.
  `results/tabular_results.csv` is the canonical set.
* Forecasting R² varies by ~±0.05 between runs (TensorFlow CPU nondeterminism).
* The old "≈100% / 99.9%" numbers were target leakage and are **not** reproduced —
  see `documentation/LEAKAGE_CORRECTION.md`.
* The split for **regression is random 80/20** (cross-sectional task); the split
  for **forecasting is chronological per machine**.
