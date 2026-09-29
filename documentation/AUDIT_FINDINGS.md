# Audit findings — Workload-Prediction-Rebuild

> **Addendum (2026-09-29).** This audit (dated 2026-08-31, below) predates the
> addition of a seventh regression baseline, HistGradientBoosting (an
> established, additional non-linear ensemble baseline, not a novel
> algorithm). Its "55 notebooks / experiments" total is accurate for the date
> it was written; the final principal-experiment count is **63** (56
> regression across seven algorithms + 6 Alibaba forecasting + 1 Google
> temporal diagnostic). The current authoritative inventory is
> `documentation/MASTER_EXPERIMENT_TABLE.md`. The findings below are retained
> as the historical audit record and are not rewritten.

Audit date: 2026-08-31. Scope: `src/`, `notebooks/`, `common/`, `results/`,
`WORKFLOW.html`, `MASTER_TABLE.*`, `VERIFICATION.md`, `temporal_diagnostic.json`.
Method: read every source file and every result file directly; re-executed a
sample of notebooks and compared to the recorded numbers. Previous summaries were
**not** trusted.

## 1. Experiment coverage — COMPLETE

| Expected | Found | Status |
|---|---|---|
| Google 2019 Cluster regression: 6 algorithms × 4 targets | 24 notebooks + 6 `src/google_2019/*.py` | ✅ |
| Alibaba 2017 regression: 6 algorithms × 4 targets | 24 notebooks + 6 `src/alibaba_2017/*.py` | ✅ |
| Alibaba 2017 forecasting: LSTM/BiLSTM/RNN × {CPU, memory} | 6 notebooks + `src/alibaba_2017/{lstm,bilstm,rnn}.py` | ✅ |
| Google 2019 Cluster temporal diagnostic | 1 notebook + `src/google_2019/temporal_diagnostic.py` | ✅ |
| **Total** | **55 notebooks / experiments** | ✅ nothing missing |

Google forecasting (LSTM/BiLSTM/RNN) exists only as
`src/google_2019/sequence_attempt.py`, explicitly flagged `not_used=True`, and is
**excluded** from the final design — consistent with the locked design.

## 2. Consistency check: dataset → preprocessing → algorithm → target → result

Verified consistent for the paths executed:

| Path | Recorded | Re-executed | |
|---|---|---|---|
| Alibaba 2017 → per-instance join → Linear → `cpu_usage_mean` → `tabular_results.csv` | test R² 0.262 | 0.262 | ✅ exact |
| Google 2019 Cluster → 150k sample + dict-parse → Random Forest → `cpu_usage_mean` → `tabular_results.csv` | test R² 0.832 | 0.832 | ✅ exact |
| Alibaba 2017 → per-machine `server_usage` windows → LSTM → `cpu_util` → `sequence_results.csv` | test R² 0.796 (recorded); re-runs 0.74–0.80 | 0.739 on one re-run | ✅ within TF variance |
| Google 2019 Cluster → hourly cell aggregate → ACF → `temporal_diagnostic.json` | Google ACF ≈ 0; Alibaba lag-1 ≈ 0.88 | reproduced | ✅ |

**Authoritative code path:** the per-experiment modules under **`src/`** are the
source of truth for the final results. The notebook code fragments are copied
verbatim from `src/`. `common/pipeline.py` / `common/sequence.py` and the root
`run_tabular.py` / `run_sequence.py` are configuration-matched batch runners kept
for convenience; reproduction should use `src/` (see `README.md §1, §4–7`).

## 3. Discrepancies found (reported, not silently changed)

### D1 — `results/tabular_results.csv` had a stale dataset label and lacked MAPE columns  — **RESOLVED (regeneration complete)**
* At audit time the file labelled the Google dataset **`Borg 2019`** (the former,
  incorrect label) instead of `Google 2019 Cluster`, and had **no `train_MAPE` /
  `test_MAPE` columns**.
* **The numeric values were correct and current.** Re-executing the final
  notebooks reproduces them exactly, e.g. Alibaba Ridge `cpu_usage_mean` test
  R² = **0.2624** (`GridSearchCV` selects `alpha = 0.001`, so Ridge collapses to
  OLS and equals Linear — this is expected, not a bug), and Google Random Forest
  `cpu_usage_mean` test R² = **0.832**.
* **Cause of the earlier apparent mismatch:** four per-notebook CSVs in
  `results/regression/` (`Alibaba_2017_{RandomForest_*,Ridge_Regression_Memory_mean}.csv`)
  were left over from a Jupyter run of an *earlier* notebook revision (before the
  grid / `RobustPrep` alignment) and disagreed by ~0.02–0.03 R².
* **Resolution — COMPLETE.** `python -m src.run_all regression` was run; it
  regenerated the 12 per-algorithm CSVs (`results/regression/<dataset>__<algo>.csv`)
  and reassembled `results/tabular_results.csv` (48 rows, dataset labels
  `Google 2019 Cluster` / `Alibaba 2017`, full metric columns MAE/MAPE/RMSE/R²/sMAPE).
  The nine superseded per-notebook CSVs were moved to
  `results/regression/_notebook_runs/`. `results/tabular_results.json` was rebuilt
  from the CSV.

### D2 — `results/sequence_results.csv` was stale — **RESOLVED (regeneration complete)**
* At audit time the file contained **6 rows labelled `Borg 2019`** (the *excluded*
  Google cell-aggregate forecasting attempt, `n_train = 572`, R² ≈ 0 to −0.45) and
  **zero Alibaba rows** — a prior `run_sequence.py --only borg` run had overwritten
  the Alibaba forecasting results.
* Per the locked design, Google sequence modelling is **not** a
  `sequence_results.csv` experiment; it lives in
  `results/sequence/google_2019__attempt.csv` (flagged `not_used=True`) and in the
  diagnostic.
* **Resolution — COMPLETE.** `python -m src.run_all forecasting` was run; it
  regenerated `results/sequence/alibaba_2017__{lstm,bilstm,rnn}.csv` and reassembled
  `results/sequence_results.csv` as **Alibaba-only, 6 rows** (LSTM/BiLSTM/RNN ×
  cpu_util/mem_util), `n_train ≈ 118,534` / `n_test ≈ 37,988` windows. The stale
  file was moved to `documentation/dev_logs/sequence_results_STALE_borg.csv`.
  `results/sequence_results.json` was rebuilt from the CSV.

### D3 — `alibaba_cpu_util_cell` observation count
* `temporal_diagnostic.json` records `n = 144` for the Alibaba cell series
  (grouping `server_usage` by timestamp → 144 five-minute buckets over 12 h).
* An earlier helper (`diagnose_borg_temporal.py`) reported `n ≈ 138` using a
  different construction (truncating to the shortest per-machine series).
* The `n = 144` value (one point per 5-minute bucket) is the correct, current one.
  **Not a defect** — noted for transparency. The ACF conclusion is unchanged
  (lag-1 ≈ 0.87, lag-24 ≈ 0.55).

### D4 — 4 notebooks carry saved cell outputs (~70 KB vs ~12 KB)
* `Alibaba_2017_RandomForest_{CPU_mean,CPU_peak,Memory_mean}.ipynb` and
  `Alibaba_2017_Ridge_Regression_Memory_mean.ipynb` were executed and saved with
  outputs (by a Jupyter run). Harmless — confirms they run. Can be cleared with
  `jupyter nbconvert --clear-output --inplace` if a clean repo is wanted.

### D5 — housekeeping
* `bash.exe.stackdump` (shell crash artefact) — deleted during audit.
* `results/*.out` / `*.log` are raw run logs from development iterations. Kept for
  provenance; not part of the deliverable. They can be moved to
  `documentation/dev_logs/`.

## 4. No fabrication

* No result is invented. Every number traces to a `results/*.csv` / `.json` row.
* The old "≈100% / 99.9%" numbers are **deliberately not reproduced** — they were
  the target-in-features leak (see `LEAKAGE_CORRECTION.md`).
* Metrics that are not meaningful (raw MAPE on near-zero usage) are reported but
  explicitly flagged; nothing is back-filled to make tables look uniform.

## 5. Post-regeneration verification

- [x] `results/tabular_results.csv` — **48 rows** (12 algorithms × 4 targets),
      17 columns including `train_MAPE` / `test_MAPE`, dataset labels
      `Google 2019 Cluster` / `Alibaba 2017`. Regenerated by
      `python -m src.run_all regression`; the 12 per-algorithm CSVs are
      `results/regression/<dataset>__<algo>.csv`. Re-running the current
      `Alibaba_2017_Ridge_Regression_CPU_mean.ipynb` reproduces its row exactly
      (test R² 0.2624). Nine stale notebook-run CSVs were moved to
      `results/regression/_notebook_runs/`.
- [x] `results/temporal_diagnostic.json` — regenerated; Google CPU hourly-cell
      ACF lag 1 = +0.011, lag 24 = −0.013 (band ±0.072); Alibaba CPU cell ACF
      lag 1 = +0.872, lag 24 = +0.549 (band ±0.163). A cosmetic `KeyError` in the
      module's final `print` (int vs str dict key) was fixed; it never affected
      the written JSON.
- [x] `results/sequence_results.csv` — **6 rows, `Alibaba 2017` only** (LSTM /
      BiLSTM / RNN × cpu_util / mem_util), regenerated by
      `python -m src.run_all forecasting`. `n_train ≈ 118,534`, `n_test ≈ 37,988`
      windows. Test R²: CPU 0.768–0.796, memory 0.900–0.933 — consistent with
      earlier runs (±0.05). Stale file archived at
      `documentation/dev_logs/sequence_results_STALE_borg.csv`.
- [x] `documentation/MASTER_EXPERIMENT_TABLE.csv` / `.md` — **55 rows** (48
      regression + 6 forecasting + 1 diagnostic), rebuilt from the regenerated
      aggregate files by `python documentation/build_master_table.py`.
- [x] `results/sequence/google_2019__attempt.csv` — the Google forecasting
      *attempt* (LSTM/BiLSTM/RNN on the hourly cell series), R² ≈ 0, `not_used=True`.
      Kept only as evidence for the exclusion decision.
