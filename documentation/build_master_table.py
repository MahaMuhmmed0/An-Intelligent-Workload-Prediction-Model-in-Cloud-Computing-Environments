"""
documentation/build_master_table.py
===================================
Assemble MASTER_EXPERIMENT_TABLE.csv / .md -- the authoritative 63-row
principal-experiment inventory -- from the canonical, frozen result files:
results/tabular_results.csv (48 six-algorithm regression rows),
results/regression/histgbr/*.csv (8 additional HistGradientBoosting
regression rows -- an established, additional non-linear ensemble baseline,
not a novel algorithm), results/sequence_results.csv (6 Alibaba forecasting
rows), and results/temporal_diagnostic.json (1 Google temporal diagnostic).
56 regression + 6 forecasting + 1 diagnostic = 63 principal experiments.

This script only reads already-computed, frozen result files and re-indexes
them; it does not run or alter any experiment.

Columns: Dataset | Module | Target | Algorithm | Source File | Notebook |
         Result File | Status | Train MAE | Train MAPE | Train R2 |
         Test MAE | Test MAPE | Test R2 | Test RMSE | Test sMAPE

Run:  python documentation/build_master_table.py
"""
import glob
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R = os.path.join(ROOT, "results")

ALGO_SRC = {
    "Linear Regression": "linear_regression", "Ridge Regression": "ridge_regression",
    "Lasso Regression": "lasso_regression", "Elastic Net": "elastic_net",
    "Stepwise Regression": "stepwise_regression", "Random Forest": "random_forest",
    "HistGradientBoostingRegressor": "hist_gradient_boosting",
}
ALGO_NB = {
    "Linear Regression": "Linear_Regression", "Ridge Regression": "Ridge_Regression",
    "Lasso Regression": "Lasso_Regression", "Elastic Net": "ElasticNet",
    "Stepwise Regression": "Stepwise", "Random Forest": "RandomForest",
    "HistGradientBoostingRegressor": "HistGradientBoosting",
}
ALGO_FILE = {  # per-algorithm csv stem under results/regression/
    "Linear Regression": "linear", "Ridge Regression": "ridge",
    "Lasso Regression": "lasso", "Elastic Net": "elastic_net",
    "Stepwise Regression": "stepwise", "Random Forest": "random_forest",
}
TGT_NB = {"cpu_usage_mean": "CPU_mean", "cpu_usage_peak": "CPU_peak",
          "mem_usage_mean": "Memory_mean", "mem_usage_peak": "Memory_peak"}
DS_DIR = {"Google 2019 Cluster": "google_2019", "Alibaba 2017": "alibaba_2017"}
DS_NB = {"Google 2019 Cluster": "Google_2019", "Alibaba 2017": "Alibaba_2017"}


def n(v, nd=4):
    try:
        return round(float(v), nd)
    except Exception:
        return ""


rows = []

# ---- regression: results/tabular_results.csv ---------------------------- #
t = pd.read_csv(os.path.join(R, "tabular_results.csv"))
for _, r in t.iterrows():
    ds, tgt, model = r["dataset"], r["target"], r["model"]
    d = DS_DIR[ds]
    rows.append({
        "Dataset": ds, "Module": "Regression", "Target": tgt, "Algorithm": model,
        "Source File": f"src/{d}/{ALGO_SRC[model]}.py",
        "Notebook": f"notebooks/{d}/{DS_NB[ds]}_{ALGO_NB[model]}_{TGT_NB[tgt]}.ipynb",
        "Result File": f"results/regression/{d}__{ALGO_FILE[model]}.csv  (+ results/tabular_results.csv)",
        "Status": "verified - exact reproduction of results/tabular_results.csv",
        "Train MAE": n(r.get("train_MAE")), "Train MAPE": n(r.get("train_MAPE"), 1),
        "Train R2": n(r.get("train_R2")),
        "Test MAE": n(r.get("test_MAE")), "Test MAPE": n(r.get("test_MAPE"), 1),
        "Test R2": n(r.get("test_R2")), "Test RMSE": n(r.get("test_RMSE")),
        "Test sMAPE": n(r.get("test_sMAPE"), 1),
    })

# ---- additional non-linear baseline: results/regression/histgbr/*.csv -- #
# Additive relative to the frozen, six-algorithm results/tabular_results.csv
# (unchanged by this); one of the 56 principal regression / 63 principal
# experiments overall. HistGradientBoosting is an established, additional
# non-linear ensemble baseline evaluated under the identical pipeline as the
# six principal regression algorithms -- not a novel algorithm or a research
# contribution of this work.
for hp in sorted(glob.glob(os.path.join(R, "regression", "histgbr", "*.csv"))):
    h = pd.read_csv(hp)
    for _, r in h.iterrows():
        ds, tgt, model = r["dataset"], r["target"], r["model"]
        d = DS_DIR[ds]
        rows.append({
            "Dataset": ds, "Module": "Regression", "Target": tgt, "Algorithm": model,
            "Source File": f"src/{d}/{ALGO_SRC[model]}.py",
            "Notebook": f"notebooks/{d}/{DS_NB[ds]}_{ALGO_NB[model]}_{TGT_NB[tgt]}.ipynb",
            "Result File": (
                f"results/regression/histgbr/{d}__hist_gradient_boosting.csv  "
                "(+ results/tabular_results_histgbr.csv)"
            ),
            "Status": (
                "verified - part of the 56 principal regression experiments (additional "
                "HistGradientBoosting baseline, additive relative to the frozen "
                "results/tabular_results.csv, which remains the six-algorithm 48-row file "
                "by design; see results/regression/histgbr/ and "
                "results/tabular_results_histgbr.csv)"
            ),
            "Train MAE": n(r.get("train_MAE")), "Train MAPE": n(r.get("train_MAPE"), 1),
            "Train R2": n(r.get("train_R2")),
            "Test MAE": n(r.get("test_MAE")), "Test MAPE": n(r.get("test_MAPE"), 1),
            "Test R2": n(r.get("test_R2")), "Test RMSE": n(r.get("test_RMSE")),
            "Test sMAPE": n(r.get("test_sMAPE"), 1),
        })

# ---- forecasting: results/sequence_results.csv ------------------------- #
sp = os.path.join(R, "sequence_results.csv")
if os.path.exists(sp):
    s = pd.read_csv(sp)
    for _, r in s.iterrows():
        model, util = r["model"], r["target"]           # cpu_util / mem_util
        tok = "CPU" if util.startswith("cpu") else "Memory"
        rows.append({
            "Dataset": "Alibaba 2017", "Module": "Forecasting", "Target": util,
            "Algorithm": model,
            "Source File": f"src/alibaba_2017/{model.lower()}.py",
            "Notebook": f"notebooks/alibaba_2017/Alibaba_2017_{model}_{tok}.ipynb",
            "Result File": f"results/sequence/alibaba_2017__{model.lower()}.csv  (+ results/sequence_results.csv)",
            "Status": "verified - reproduces within TensorFlow CPU variance (~+-0.05 R2)",
            "Train MAE": n(r.get("train_MAE")), "Train MAPE": n(r.get("train_MAPE"), 1),
            "Train R2": n(r.get("train_R2")),
            "Test MAE": n(r.get("test_MAE")), "Test MAPE": n(r.get("test_MAPE"), 1),
            "Test R2": n(r.get("test_R2")), "Test RMSE": n(r.get("test_RMSE")),
            "Test sMAPE": n(r.get("test_sMAPE"), 1),
        })

# ---- temporal diagnostic --------------------------------------------- #
rows.append({
    "Dataset": "Google 2019 Cluster", "Module": "Temporal Diagnostic", "Target": "-",
    "Algorithm": "ACF / noise-band",
    "Source File": "src/google_2019/temporal_diagnostic.py",
    "Notebook": "notebooks/google_2019/Google_2019_Temporal_Diagnostic.ipynb",
    "Result File": "results/temporal_diagnostic.json",
    "Status": "verified - exact reproduction of results/temporal_diagnostic.json (not a model; no train/test metrics)",
    "Train MAE": "", "Train MAPE": "", "Train R2": "", "Test MAE": "", "Test MAPE": "",
    "Test R2": "", "Test RMSE": "", "Test sMAPE": "",
})

df = pd.DataFrame(rows)
df.to_csv(os.path.join(HERE, "MASTER_EXPERIMENT_TABLE.csv"), index=False)

n_reg = int((df["Module"] == "Regression").sum())
n_fc = int((df["Module"] == "Forecasting").sum())
with open(os.path.join(HERE, "MASTER_EXPERIMENT_TABLE.md"), "w", encoding="utf-8") as fh:
    n_diag = int((df["Module"] == "Temporal Diagnostic").sum())
    fh.write("# Master experiment table\n\n")
    fh.write(f"{len(df)} rows: {n_reg} regression (seven algorithms -- linear, ridge, lasso, "
             "elastic net, stepwise, random forest, and the additional HistGradientBoosting "
             f"baseline -- over four targets on each of two traces) + {n_fc} forecasting + "
             f"{n_diag} temporal diagnostic. This is the **authoritative numerical inventory** "
             "of the 63 principal experiments; the `Status` column records verification. "
             "HistGradientBoosting is an established, additional non-linear ensemble baseline, "
             "evaluated under the identical pipeline as the six principal regression "
             "algorithms; it is not a novel algorithm or a research contribution of this work. "
             "`notebooks/MASTER_TABLE.md` describes the same experiments as a "
             "path-only map (no metrics). MAPE is undefined for zero actuals and unstable "
             "near zero (~12% of Google targets are exactly 0); it is retained from the "
             "result files for completeness but is **not** a principal metric — use "
             "MAE / RMSE / R2 / sMAPE.\n\n"
             "The prior six-algorithm-only inventory (55 rows) is superseded by this "
             "63-row table; the six-algorithm subset (48 rows) is still available "
             "directly from results/tabular_results.csv.\n\n")
    fh.write("| " + " | ".join(df.columns) + " |\n")
    fh.write("|" + "|".join(["---"] * len(df.columns)) + "|\n")
    for _, r in df.iterrows():
        fh.write("| " + " | ".join(str(x) for x in r) + " |\n")

print(f"MASTER_EXPERIMENT_TABLE: {len(df)} rows ({n_reg} reg, {n_fc} fc, 1 diag)")
if len(df) != 63:
    if n_fc < 6:
        reason = "forecasting results not yet generated"
    elif n_reg < 56:
        reason = "HistGradientBoosting results not yet generated"
    else:
        reason = "check results/ for missing files"
    print(f"  NOTE: {63 - len(df)} rows short of 63 (56 regression incl. "
          f"HistGradientBoosting + 6 forecasting + 1 diagnostic) — {reason}")
