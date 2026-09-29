# Final research checklist

Status: **experiments, result regeneration, technical audit and documentation
cleanup are complete; the research package is frozen** (audit + regeneration
2026-08-31, documentation cleanup 2026-09-01; extended 2026-09-29 with the
additional HistGradientBoosting baseline and a repository-wide reproducibility
pass — see the new items below and `documentation/MASTER_EXPERIMENT_TABLE.md`
for the current 63-row authoritative inventory). `[x]` = verified in this
project; `[~]` = done with a caveat noted; `[ ]` = out of scope by design (not
blocking).

## Datasets & documentation

- [x] **Dataset descriptions verified** — `documentation/DATASET_METHODOLOGY.md`,
      cross-checked against the official Google v3 PDF and the Alibaba `schema.csv`.
- [x] **Google documentation verified** — normalisation (NCU / normalised bytes,
      both ÷ largest machine), usage semantics (`average_usage`, `maximum_usage`,
      `cpu_usage_distribution`), request = limit, request 0 legitimate. Quoted in
      `DATASET_METHODOLOGY.md` §2–4.
- [x] **Alibaba structure verified** — `container_event` (8 fields), `container_usage`
      (12 fields), `server_usage` (8 fields) read against `schema.csv`;
      `cpu_util`/`mem_util` = % of request; no network column.
- [x] **Datasets not assumed identical** — mapping and its limits stated explicitly
      (`DATASET_METHODOLOGY.md` §7); "only R² is comparable across datasets".

## Leakage & train/test discipline

- [x] **No target leakage** — `FEATURES` is an explicit request/metadata list;
      `assert TARGET not in FEATURES` runs in Step 6 of every regression notebook;
      no feature is derived from a target; `cpu_mem_ratio` uses requests only.
      Evidence: feature construction + pipeline discipline (not the train–test
      gap). Over-fitting assessed separately — no evidence of substantial
      over-fitting (mean |train−test R²| ≈ 0.02, max ≈ 0.07 on Alibaba CPU peak).
- [x] **Train/test separation correct** — regression: random 80/20,
      `random_state=42`, test evaluated once. Forecasting: per-machine chronological
      80/20, no window crossing the boundary.
- [x] **Scaling fitted correctly** — `RobustPrep` bounds from `X_train` percentiles;
      `SimpleImputer`/`PolynomialFeatures`/`Scaler` are `Pipeline` steps;
      forecasting `MinMaxScaler` fit on training points only.
- [x] **GridSearch uses training data only** — `GridSearchCV(cv=5)` and the
      `LassoCV` stepwise selector receive `X_train, y_train`.
- [x] **Correction documented** — `documentation/LEAKAGE_CORRECTION.md`
      (L1 target-in-features, L2 target-derived features, L3 preprocessing/selection
      /validation on test), each with its fix and the code location.

## Experiment coverage

- [x] **All 6 principal regression algorithms present** — Linear, Ridge, Lasso,
      Elastic Net, Stepwise, Random Forest — in `src/{google_2019,alibaba_2017}/`
      and notebooks.
- [x] **HistGradientBoosting (7th, additional baseline) present** — an
      established, additional non-linear ensemble baseline, not a novel
      algorithm — in `src/{google_2019,alibaba_2017}/hist_gradient_boosting.py`
      and notebooks; results in `results/regression/histgbr/`.
- [x] **All 4 regression targets present** — `cpu_usage_mean`, `cpu_usage_peak`,
      `mem_usage_mean`, `mem_usage_peak` — for both datasets (56 regression
      experiments across seven algorithms; 48 of them, the six principal
      algorithms, are frozen in `results/tabular_results.csv`).
- [x] **Alibaba LSTM present** — `src/alibaba_2017/lstm.py`,
      `Alibaba_2017_LSTM_{CPU,Memory}.ipynb`.
- [x] **Alibaba BiLSTM present** — `src/alibaba_2017/bilstm.py`,
      `Alibaba_2017_BiLSTM_{CPU,Memory}.ipynb`.
- [x] **Alibaba RNN present** — `src/alibaba_2017/rnn.py`,
      `Alibaba_2017_RNN_{CPU,Memory}.ipynb`.
- [x] **CPU forecasting present** — LSTM/BiLSTM/RNN on `cpu_util`.
- [x] **Memory forecasting present** — LSTM/BiLSTM/RNN on `mem_util`.
- [x] **Google temporal diagnostic documented** —
      `documentation/TEMPORAL_DIAGNOSTIC.md`, `results/temporal_diagnostic.json`,
      `Google_2019_Temporal_Diagnostic.ipynb`.
- [x] **Google forecasting correctly excluded** — only exists as
      `src/google_2019/sequence_attempt.py` flagged `not_used=True`; not in
      `sequence_results.csv`; excluded by the locked design.

## Traceability

- [x] **All result files mapped** — `documentation/MASTER_EXPERIMENT_TABLE.csv/.md`
      and `notebooks/MASTER_TABLE.csv`.
- [x] **All notebooks mapped** — 63 notebooks (55 six-algorithm + 8 additional
      HistGradientBoosting), one per experiment; `notebooks/MASTER_TABLE.md`.
- [x] **No invented results** — every number traces to a `results/*.csv` / `.json`
      row; missing/meaningless metrics are flagged, not back-filled.
- [x] **No "100% accuracy" claim** — best model R² ≈ 0.83 (Google CPU) / ≈ 0.94
      (Google memory) / ≈ 0.33 (Alibaba). Old "≈100%" numbers explicitly disowned
      as leakage.

## Write-up

- [x] **Limitations documented** — `documentation/LIMITATIONS.md`.
- [x] **Reproducibility instructions complete** — root `README.md` §2–9.
- [x] **Methodology draft ready** — `documentation/METHODOLOGY_DRAFT.md` (17 subsections).
- [x] **Results + discussion drafts ready** — `documentation/RESULTS_TABLES.md`,
      `documentation/RESULTS_DISCUSSION_DRAFT.md`.

## Regeneration (done during the audit)

- [x] `results/tabular_results.csv` regenerated — 48 rows (six principal
      algorithms), `Google 2019 Cluster` label, MAE/MAPE/RMSE/R²/sMAPE columns.
      `results/tabular_results.json` rebuilt. (Later, additive: 8 further rows
      for the HistGradientBoosting baseline in
      `results/regression/histgbr/` / `results/tabular_results_histgbr.csv`,
      which does not modify this 48-row file — 56 regression rows combined.)
- [x] `results/sequence_results.csv` regenerated — 6 Alibaba rows (LSTM/BiLSTM/RNN
      × cpu_util/mem_util). Stale Borg file moved to `documentation/dev_logs/`.
- [x] `results/temporal_diagnostic.json` regenerated; module print-bug fixed.
- [x] `results/sequence/google_2019__attempt.csv` — Google forecasting attempt,
      test R² −0.11 … −0.00, `not_used=True`.
- [x] `documentation/MASTER_EXPERIMENT_TABLE.csv/.md` — 55 rows, `Status` column,
      at the time of this audit. Rebuilt 2026-09-29 to 63 rows (adding the
      HistGradientBoosting baseline); the six-algorithm-only subset (48 rows) is
      still available directly from `results/tabular_results.csv`.
- [x] `WORKFLOW.html` artifact refreshed with the final forecasting numbers and a
      completed-status Section 10.
- [x] Documentation wording cleanup (2026-09-01): over-fitting claim corrected in
      all locations; MAPE documented as non-principal; stale "in progress" text
      removed; `src/` documented as the authoritative code path; forecasting
      docstrings corrected to 30 epochs; Alibaba `cpu_request` normalisation
      limitation recorded (`LIMITATIONS.md` L13).

## Caveats recorded

- [~] CV-tuned regression models can vary ~±0.02 in R² between scikit-learn
      versions; forecasting varies ~±0.05 between runs (TF CPU nondeterminism).
- [~] Google forecasting *attempt* numbers (R² ≈ 0) are retained only as evidence
      for the exclusion decision, never presented as a model result.
- [~] Nine per-notebook regression CSVs from an earlier Jupyter run are archived
      under `results/regression/_notebook_runs/` (superseded, not deleted).

## Not done (by design / out of scope)

- [ ] Statistical significance tests — not performed; the study establishes
      predictive relationships, not hypothesis tests (see `LIMITATIONS.md`).
- [ ] Batch-workload usage prediction on Alibaba — `batch_instance.csv` (actual
      batch usage) is not in the provided data; only online containers are modelled.
- [ ] The final paper — to be written next, against the user's template and
      reference list.
