"""
src/run_all.py
==============
Run every experiment and rebuild the canonical result files.

  python -m src.run_all                 # everything (the frozen 48+6+1 experiments)
  python -m src.run_all regression      # 12 regression files only
  python -m src.run_all forecasting     # 3 Alibaba forecasters only
  python -m src.run_all diagnostic      # Google temporal diagnostic + attempt
  python -m src.run_all histgbr         # ADDITIVE: HistGradientBoosting regressors only

Assembles:
  results/regression/*__*.csv           ->  results/tabular_results.csv    (frozen, 48 rows)
  results/sequence/alibaba_2017__*.csv  ->  results/sequence_results.csv
  results/temporal_diagnostic.json
  results/regression/histgbr/*__*.csv   ->  results/tabular_results_histgbr.csv  (additive, 8 rows)

NOTE: the additive HistGradientBoosting experiment writes to the results/regression/histgbr/
subdirectory, so `run_all` / `run_all regression` never fold it into the frozen
results/tabular_results.csv.  It is aggregated only by `run_all histgbr`.
"""
import glob
import os
import runpy
import sys

import pandas as pd

from src.evaluation import PROJECT_ROOT

REGRESSION = [
    "src.google_2019.linear_regression", "src.google_2019.ridge_regression",
    "src.google_2019.lasso_regression", "src.google_2019.elastic_net",
    "src.google_2019.stepwise_regression", "src.google_2019.random_forest",
    "src.alibaba_2017.linear_regression", "src.alibaba_2017.ridge_regression",
    "src.alibaba_2017.lasso_regression", "src.alibaba_2017.elastic_net",
    "src.alibaba_2017.stepwise_regression", "src.alibaba_2017.random_forest",
]
FORECASTING = ["src.alibaba_2017.lstm", "src.alibaba_2017.bilstm", "src.alibaba_2017.rnn"]
DIAGNOSTIC = ["src.google_2019.machine_coverage",
              "src.google_2019.temporal_diagnostic", "src.google_2019.sequence_attempt"]
# ADDITIVE — not part of the frozen 48-experiment regression set behind tabular_results.csv
REGRESSION_HISTGBR = ["src.google_2019.hist_gradient_boosting",
                      "src.alibaba_2017.hist_gradient_boosting"]


def _combine(pattern, out_name):
    files = sorted(glob.glob(os.path.join(PROJECT_ROOT, "results", pattern)))
    if not files:
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out = os.path.join(PROJECT_ROOT, "results", out_name)
    try:
        df.to_csv(out, index=False)
        print(f"\nassembled {out}  ({len(df)} rows from {len(files)} files)")
    except PermissionError:
        print(f"\n{out} is locked/open — close it and re-run `python -m src.run_all`")


def main(which="all"):
    if which in ("all", "regression"):
        for m in REGRESSION:
            runpy.run_module(m, run_name="__main__")
        _combine("regression/*__*.csv", "tabular_results.csv")
    if which in ("all", "forecasting"):
        for m in FORECASTING:
            runpy.run_module(m, run_name="__main__")
        _combine("sequence/alibaba_2017__*.csv", "sequence_results.csv")
    if which in ("all", "diagnostic"):
        for m in DIAGNOSTIC:
            runpy.run_module(m, run_name="__main__")
    if which == "histgbr":                       # ADDITIVE; opt-in only, never part of "all"
        for m in REGRESSION_HISTGBR:
            runpy.run_module(m, run_name="__main__")
        _combine("regression/histgbr/*__*.csv", "tabular_results_histgbr.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
