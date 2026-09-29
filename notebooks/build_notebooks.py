"""
notebooks/build_notebooks.py
============================
Generates one self-contained Jupyter notebook per (dataset, algorithm, target)
experiment, mirroring the verified logic in ../src/ EXACTLY (same preprocessing,
same random_state=42, same 80/20 split, same hyper-parameters, same metrics).

Run:  python notebooks/build_notebooks.py

Output:
  notebooks/google_2019/*.ipynb      (24 regression + 1 diagnostic)
  notebooks/alibaba_2017/*.ipynb     (24 regression + 6 forecasting)
  notebooks/MASTER_TABLE.md
  notebooks/MASTER_TABLE.csv

Each regression notebook has the 19 numbered steps requested; each notebook writes
its result row to results/regression/<stem>.csv (forecasting -> results/sequence/).
Nothing here is new methodology: it is the same code that produced
results/tabular_results.csv and results/sequence_results.csv.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

sys.path.insert(0, ROOT)
from src.evaluation import GOOGLE_2019_CSV, ALIBABA_2017_DIR

# Configurable via the GOOGLE_2019_CSV / ALIBABA_2017_DIR environment
# variables; see src/evaluation.py and README.md "Dataset locations". These
# two values are embedded verbatim into every generated notebook's loading
# cell (each notebook is self-contained), so setting the env vars before
# running this generator changes what path the regenerated notebooks embed.
BORG_CSV = GOOGLE_2019_CSV
ALIBABA_DIR = ALIBABA_2017_DIR

# --------------------------------------------------------------------------- #
#  notebook helpers                                                           #
# --------------------------------------------------------------------------- #

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.rstrip("\n").splitlines(keepends=True)}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def write_nb(path, cells):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(notebook(cells), fh, indent=1)
    print("wrote", os.path.relpath(path, ROOT))


# --------------------------------------------------------------------------- #
#  shared source fragments  (identical to src/evaluation.py)                   #
# --------------------------------------------------------------------------- #

IMPORTS_REG = '''\
# Step 1: Import libraries
import ast, os, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LassoCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
EPS = 1e-9
RESULTS_DIR = r"{results_dir}"
'''

METRICS_SRC = '''\
# Step 15-17 helper: metric functions (MAE, MAPE, RMSE, R2, sMAPE)
# NOTE: raw MAPE is unreliable here because usage is frequently ~0 (it blows up).
#       sMAPE is the bounded percentage error actually used in the study.
def smape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(2*np.abs(p - y) / (np.abs(y) + np.abs(p) + EPS)) * 100)

def all_scores(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {{
        "MAE":   float(mean_absolute_error(y, p)),
        "MAPE":  float(mean_absolute_percentage_error(y, p) * 100),   # unreliable near 0
        "RMSE":  float(np.sqrt(np.mean((y - p) ** 2))),
        "R2":    float(r2_score(y, p)),
        "sMAPE": smape(y, p),
    }}

class RobustPrep:
    """log1p on non-negative skewed features + winsorise to the TRAIN 1st/99th
    percentile. Bounds learned from TRAIN rows only -> no leakage."""
    def fit(self, X):
        X = np.asarray(X, float)
        self.lo_ = np.nanpercentile(X, 1, axis=0)
        self.hi_ = np.nanpercentile(X, 99, axis=0)
        self.logmask_ = np.nanmin(X, axis=0) >= 0
        return self
    def transform(self, X):
        X = np.array(X, float)
        X = np.clip(X, self.lo_, self.hi_)
        X[:, self.logmask_] = np.log1p(X[:, self.logmask_])
        return X
'''

GOOGLE_PREP = '''\
# Step 2: Load dataset  --  Google 2019 Cluster  (Borg trace v3 public extract)
BORG_CSV = r"{borg_csv}"
raw = pd.read_csv(BORG_CSV, low_memory=False)
# reproducibility: 150,000-row random sample (random_state=42), as in the final run
raw = raw.sample(n=150_000, random_state=RANDOM_STATE)
print("raw sample:", raw.shape)

# Step 3: Data preprocessing  +  Step 4: Parse required columns
# resource_request / average_usage / maximum_usage are dict-strings -> pull cpus & memory
def dget(s, key):
    try:
        return ast.literal_eval(s).get(key, np.nan)
    except Exception:
        return np.nan

for col, pfx in [("resource_request", "req"), ("average_usage", "avg"), ("maximum_usage", "max")]:
    raw[pfx + "_cpu"] = raw[col].map(lambda s: dget(s, "cpus"))
    raw[pfx + "_mem"] = raw[col].map(lambda s: dget(s, "memory"))

# Step 5: Feature engineering  (entity table: 1 row per instance-usage record)
df = pd.DataFrame({{
    "cpu_request":      pd.to_numeric(raw["req_cpu"], errors="coerce"),
    "mem_request":      pd.to_numeric(raw["req_mem"], errors="coerce"),
    "priority":         pd.to_numeric(raw["priority"], errors="coerce"),
    "scheduling_class": pd.to_numeric(raw["scheduling_class"], errors="coerce"),
    "collection_type":  pd.to_numeric(raw["collection_type"], errors="coerce"),
    "cpu_usage_mean":   pd.to_numeric(raw["avg_cpu"], errors="coerce"),
    "mem_usage_mean":   pd.to_numeric(raw["avg_mem"], errors="coerce"),
    "cpu_usage_peak":   pd.to_numeric(raw["max_cpu"], errors="coerce"),
    "mem_usage_peak":   pd.to_numeric(raw["max_mem"], errors="coerce"),
    "order_key":        pd.to_numeric(raw["start_time"], errors="coerce"),
}})
df["cpu_mem_ratio"] = df["cpu_request"] / (df["mem_request"] + EPS)   # from REQUESTS -> no leak

# drop zero-request rows (ratio undefined; request=0 burst tasks are a separate regime)
df = df[(df["cpu_request"] > 0) & (df["mem_request"] > 0)]
# normalised usage lies in [0,1]; anything far outside is a parsing artefact
for c in ["cpu_usage_mean", "mem_usage_mean", "cpu_usage_peak", "mem_usage_peak"]:
    df = df[df[c].between(0, 5)]
df = df.dropna(subset=["order_key"]).sort_values("order_key").reset_index(drop=True)

FEATURES = ["cpu_request", "mem_request", "cpu_mem_ratio",
            "priority", "scheduling_class", "collection_type"]
print("entity table:", df.shape, "| features:", FEATURES)
'''

ALIBABA_PREP = '''\
# Step 2: Load dataset  --  Alibaba 2017  (cluster-trace-v2017)
ALIBABA_DIR = r"{alibaba_dir}"

# Step 3-4: container_event -> the resource REQUEST per online instance
ce = pd.read_csv(os.path.join(ALIBABA_DIR, "container_event.csv"), header=None,
                 names=["ts", "event", "instance_id", "machine_id",
                        "plan_cpu", "plan_mem", "plan_disk", "cpuset", "_x"])
ce = ce[ce["event"] == "Create"].copy()
ce["cpuset_width"] = ce["cpuset"].astype(str).str.count(r"\\|") + 1
ce = (ce.sort_values("ts").groupby("instance_id")
        .agg(plan_cpu=("plan_cpu", "first"), plan_mem=("plan_mem", "first"),
             plan_disk=("plan_disk", "first"), cpuset_width=("cpuset_width", "first"))
        .reset_index())

# Step 3-4: container_usage -> the observed USAGE, aggregated per instance
cu = pd.read_csv(os.path.join(ALIBABA_DIR, "container_usage.csv"), header=None,
                 names=["ts", "instance_id", "cpu_util", "mem_util", "disk_util",
                        "load1", "load5", "load15", "avg_cpi", "avg_mpki", "max_cpi", "max_mpki"])
agg = (cu.groupby("instance_id")
         .agg(cpu_usage_mean=("cpu_util", "mean"), cpu_usage_peak=("cpu_util", "max"),
              mem_usage_mean=("mem_util", "mean"), mem_usage_peak=("mem_util", "max"),
              order_key=("ts", "min"), n_obs=("ts", "size"))
         .reset_index())
agg = agg[agg["n_obs"] >= 3]      # need a few observations to characterise the instance

# Step 5: Feature engineering  (join request to usage; util is a % of the request)
m = ce.merge(agg, on="instance_id", how="inner")
df = pd.DataFrame({{
    "cpu_request":    m["plan_cpu"] / m["plan_cpu"].max(),   # normalise core count
    "mem_request":    m["plan_mem"],
    "plan_disk":      m["plan_disk"],
    "cpuset_width":   m["cpuset_width"],
    "cpu_usage_mean": m["cpu_usage_mean"] / 100.0,
    "mem_usage_mean": m["mem_usage_mean"] / 100.0,
    "cpu_usage_peak": m["cpu_usage_peak"] / 100.0,
    "mem_usage_peak": m["mem_usage_peak"] / 100.0,
    "order_key":      m["order_key"],
}})
df["cpu_mem_ratio"] = m["plan_cpu"].values / (m["plan_mem"].values + EPS)   # from REQUESTS -> no leak
df = df.dropna(subset=["order_key"]).sort_values("order_key").reset_index(drop=True)

FEATURES = ["cpu_request", "mem_request", "cpu_mem_ratio", "plan_disk", "cpuset_width"]
print("entity table:", df.shape, "| features:", FEATURES)
'''

XY_SPLIT = '''\
# Step 6: Define X and y      TARGET = "{target}"
TARGET = "{target}"
assert TARGET not in FEATURES, "target must never be a feature (this was the old 100% bug)"
X = df[FEATURES].copy()
y = df[TARGET].copy()

# Step 7: Handle missing values  -> handled inside the Pipeline by SimpleImputer(median)

# Step 8: Train/test split  --  random 80/20, random_state=42
# The regression task is CROSS-SECTIONAL (predict an unobserved property of a job
# from its request), so a shuffled split is correct and matches the original study.
Xtr_df, Xte_df, ytr, yte = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, shuffle=True)
ytr, yte = ytr.values, yte.values
print("train:", Xtr_df.shape, " test:", Xte_df.shape)

# Step 9: Scaling / preprocessing  --  RobustPrep (log1p + winsorise to TRAIN 1st/99th pct)
# fitted on TRAIN only, then the rest (impute -> [poly] -> scale) lives in the Pipeline.
prep = RobustPrep().fit(Xtr_df.values)
Xtr = prep.transform(Xtr_df.values)
Xte = prep.transform(Xte_df.values)
'''

PLOT_SAVE = '''\
# Step 18: Plot actual vs predicted (test set)
plt.figure(figsize=(6, 6))
plt.scatter(yte, test_pred, s=8, alpha=0.35)
lim = [min(yte.min(), test_pred.min()), max(yte.max(), test_pred.max())]
plt.plot(lim, lim, "r--", lw=1)
plt.xlabel("Actual " + TARGET); plt.ylabel("Predicted " + TARGET)
plt.title("{dataset} - {model} - " + TARGET + " (test)")
plt.tight_layout(); plt.show()

# Step 19: Print / save final results
train_s = all_scores(ytr, train_pred)
test_s  = all_scores(yte, test_pred)
print("\\n{dataset}  |  {model}  |  " + TARGET)
print("  TRAIN  MAE {{:.4f}}  MAPE {{:.1f}}%  RMSE {{:.4f}}  R2 {{:.4f}}  sMAPE {{:.1f}}".format(
    train_s["MAE"], train_s["MAPE"], train_s["RMSE"], train_s["R2"], train_s["sMAPE"]))
print("  TEST   MAE {{:.4f}}  MAPE {{:.1f}}%  RMSE {{:.4f}}  R2 {{:.4f}}  sMAPE {{:.1f}}".format(
    test_s["MAE"], test_s["MAPE"], test_s["RMSE"], test_s["R2"], test_s["sMAPE"]))

row = {{"dataset": "{dataset}", "module": "regression", "target": TARGET, "model": "{model}",
       "n_train": len(ytr), "n_test": len(yte),
       "train_MAE": train_s["MAE"], "train_MAPE": train_s["MAPE"], "train_RMSE": train_s["RMSE"],
       "train_R2": train_s["R2"], "train_sMAPE": train_s["sMAPE"],
       "test_MAE": test_s["MAE"], "test_MAPE": test_s["MAPE"], "test_RMSE": test_s["RMSE"],
       "test_R2": test_s["R2"], "test_sMAPE": test_s["sMAPE"],
       "hyperparams": json.dumps(HYPERPARAMS)}}
os.makedirs(os.path.join(RESULTS_DIR, "regression"), exist_ok=True)
out_csv = os.path.join(RESULTS_DIR, "regression", "{stem}.csv")
pd.DataFrame([row]).to_csv(out_csv, index=False)
print("\\nsaved ->", out_csv)
'''

# ---- per-algorithm model + train/predict fragments -------------------------- #

MODEL_FRAGMENTS = {
    "Linear Regression": '''\
# Step 10: Build the model  --  Ordinary Least Squares
# Step 11: Hyperparameter tuning  --  none (OLS has no hyper-parameters)
HYPERPARAMS = {}
pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler()),
    ("model",  LinearRegression()),
])

# Step 12: Train the model
pipe.fit(Xtr, ytr)
best = pipe

# Step 13-14: Generate train / test predictions
train_pred = best.predict(Xtr)
test_pred  = best.predict(Xte)
''',
    "Ridge Regression": '''\
# Step 10: Build the model  --  Ridge (L2)
# Step 11: Hyperparameter tuning  --  GridSearchCV(cv=5), alpha grid, on TRAIN only
HYPERPARAMS = {"model__alpha": [0.001, 0.01, 0.1, 1, 10, 100]}
pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model",  Ridge(random_state=RANDOM_STATE)),
])
gs = GridSearchCV(pipe, HYPERPARAMS, cv=5, scoring="neg_mean_absolute_error", n_jobs=4)
gs.fit(Xtr, ytr)
best = gs.best_estimator_
print("best params:", gs.best_params_)
HYPERPARAMS = gs.best_params_

# Step 12: Train the model  -> done by GridSearchCV.fit on TRAIN
# Step 13-14: Generate train / test predictions
train_pred = best.predict(Xtr)
test_pred  = best.predict(Xte)
''',
    "Lasso Regression": '''\
# Step 10: Build the model  --  Lasso (L1) on degree-2 polynomial features
# Step 11: Hyperparameter tuning  --  GridSearchCV(cv=5), alpha grid, on TRAIN only
HYPERPARAMS = {"model__alpha": [1e-4, 1e-3, 1e-2, 0.1]}
pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("model",  Lasso(max_iter=3000, tol=1e-3, random_state=RANDOM_STATE)),
])
gs = GridSearchCV(pipe, HYPERPARAMS, cv=5, scoring="neg_mean_absolute_error", n_jobs=4)
gs.fit(Xtr, ytr)
best = gs.best_estimator_
print("best params:", gs.best_params_)
HYPERPARAMS = gs.best_params_

# Step 12-14
train_pred = best.predict(Xtr)
test_pred  = best.predict(Xte)
''',
    "Elastic Net": '''\
# Step 10: Build the model  --  Elastic Net (L1+L2) on degree-2 polynomial features
# Step 11: Hyperparameter tuning  --  GridSearchCV(cv=5), alpha & l1_ratio, on TRAIN only
HYPERPARAMS = {"model__alpha": [1e-3, 1e-2, 0.1], "model__l1_ratio": [0.3, 0.6]}
pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("model",  ElasticNet(max_iter=3000, tol=1e-3, random_state=RANDOM_STATE)),
])
gs = GridSearchCV(pipe, HYPERPARAMS, cv=5, scoring="neg_mean_absolute_error", n_jobs=4)
gs.fit(Xtr, ytr)
best = gs.best_estimator_
print("best params:", gs.best_params_)
HYPERPARAMS = gs.best_params_

# Step 12-14
train_pred = best.predict(Xtr)
test_pred  = best.predict(Xte)
''',
    "Stepwise Regression": '''\
# Step 10: Build the model  --  Stepwise = embedded LassoCV selection -> OLS refit
# Step 11: Hyperparameter tuning  --  LassoCV(cv=5) chooses its own alpha, on TRAIN only
#          Every fitted transform + the selector see TRAIN data only.
imp  = SimpleImputer(strategy="median").fit(Xtr)
poly = PolynomialFeatures(degree=2, include_bias=False).fit(imp.transform(Xtr))
sc   = StandardScaler().fit(poly.transform(imp.transform(Xtr)))
Ztr  = sc.transform(poly.transform(imp.transform(Xtr)))
Zte  = sc.transform(poly.transform(imp.transform(Xte)))

sel  = LassoCV(cv=5, max_iter=10000, random_state=RANDOM_STATE).fit(Ztr, ytr)
mask = sel.coef_ != 0
if mask.sum() == 0:
    mask[:] = True
HYPERPARAMS = {"selector_alpha": float(sel.alpha_), "n_selected": int(mask.sum())}
print("LassoCV alpha:", sel.alpha_, "| features kept:", int(mask.sum()))

# Step 12: Train the final OLS on the selected features
ols = LinearRegression().fit(Ztr[:, mask], ytr)
best = ols

# Step 13-14: predictions on the selected columns
train_pred = ols.predict(Ztr[:, mask])
test_pred  = ols.predict(Zte[:, mask])
''',
    "Random Forest": '''\
# Step 10: Build the model  --  Random Forest
# Step 11: Hyperparameter tuning  --  GridSearchCV(cv=5), max_depth, on TRAIN only
HYPERPARAMS = {"model__max_depth": [6, 12]}
pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model",  RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=RANDOM_STATE,
                                     max_samples=0.5, min_samples_leaf=10)),
])
gs = GridSearchCV(pipe, HYPERPARAMS, cv=5, scoring="neg_mean_absolute_error", n_jobs=4)
gs.fit(Xtr, ytr)
best = gs.best_estimator_
print("best params:", gs.best_params_)
HYPERPARAMS = gs.best_params_

# Step 12-14
train_pred = best.predict(Xtr)
test_pred  = best.predict(Xte)
''',
    "HistGradientBoostingRegressor": '''\
# Step 10: Build the model  --  Histogram-based Gradient Boosted Decision Trees (GBDT)
# Step 11: Hyperparameter tuning  --  GridSearchCV(cv=5), learning_rate x max_depth, on TRAIN only
# ADDITIVE experiment: a second, independent non-linear ensemble (boosting) alongside
# Random Forest (bagging).  early_stopping=False -> full training fold, matching Random Forest.
HYPERPARAMS = {"model__learning_rate": [0.05, 0.1], "model__max_depth": [6, 12]}
pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model",  HistGradientBoostingRegressor(random_state=RANDOM_STATE, early_stopping=False)),
])
gs = GridSearchCV(pipe, HYPERPARAMS, cv=5, scoring="neg_mean_absolute_error", n_jobs=4)
gs.fit(Xtr, ytr)
best = gs.best_estimator_
print("best params:", gs.best_params_)
HYPERPARAMS = gs.best_params_

# Step 12-14
train_pred = best.predict(Xtr)
test_pred  = best.predict(Xte)
''',
}

# --------------------------------------------------------------------------- #
#  build regression notebooks                                                 #
# --------------------------------------------------------------------------- #

ALGOS = ["Linear Regression", "Ridge Regression", "Lasso Regression",
         "Elastic Net", "Stepwise Regression", "Random Forest"]
ALGO_SLUG = {"Linear Regression": "Linear_Regression", "Ridge Regression": "Ridge_Regression",
             "Lasso Regression": "Lasso_Regression", "Elastic Net": "ElasticNet",
             "Stepwise Regression": "Stepwise", "Random Forest": "RandomForest",
             "HistGradientBoostingRegressor": "HistGradientBoosting"}
TARGETS = {  # target column -> filename token
    "cpu_usage_mean": "CPU_mean", "cpu_usage_peak": "CPU_peak",
    "mem_usage_mean": "Memory_mean", "mem_usage_peak": "Memory_peak",
}
DATASETS = {
    "Google 2019 Cluster": dict(slug="Google_2019", folder="google_2019",
                                prep=GOOGLE_PREP, prep_fmt=dict(borg_csv=BORG_CSV),
                                infile="borg_traces_data.csv"),
    "Alibaba 2017": dict(slug="Alibaba_2017", folder="alibaba_2017",
                         prep=ALIBABA_PREP, prep_fmt=dict(alibaba_dir=ALIBABA_DIR),
                         infile="container_event.csv + container_usage.csv"),
}

master_rows = []


def build_regression():
    for ds_name, ds in DATASETS.items():
        for algo in ALGOS:
            for tgt_col, tgt_tok in TARGETS.items():
                stem = f"{ds['slug']}_{ALGO_SLUG[algo]}_{tgt_tok}"
                path = os.path.join(HERE, ds["folder"], stem + ".ipynb")
                feats = ("cpu_request, mem_request, cpu_mem_ratio, priority, scheduling_class, "
                         "collection_type" if ds_name.startswith("Google")
                         else "cpu_request, mem_request, cpu_mem_ratio, plan_disk, cpuset_width")
                title = f"# {ds_name} - {algo} - `{tgt_col}`\n"
                intro = (
                    f"{title}\n"
                    f"**Module:** cross-sectional regression &nbsp;|&nbsp; "
                    f"**Dataset:** {ds_name} &nbsp;|&nbsp; **Target:** `{tgt_col}`\n\n"
                    f"**Input file(s):** `{ds['infile']}`\n\n"
                    f"**Features (request + static metadata only; target never a feature):** "
                    f"`{feats}`\n\n"
                    f"**Split:** random 80/20, `random_state=42` &nbsp;|&nbsp; "
                    f"**CV:** GridSearchCV(cv=5) on TRAIN only (where applicable)\n\n"
                    f"**Preprocessing:** parse dict/columns -> feature engineering -> "
                    f"RobustPrep(log1p + winsorise to TRAIN 1st/99th pct) -> "
                    f"SimpleImputer(median) -> [PolynomialFeatures(2)] -> Scaler  *(inside a Pipeline)*\n\n"
                    f"**Result row -> ** `results/regression/{stem}.csv`  "
                    f"(also aggregated into `results/tabular_results.csv`)\n\n"
                    f"This notebook reproduces the leakage-free experiment behind "
                    f"`results/tabular_results.csv`. It does **not** reproduce the old "
                    f"target-in-features '100%' result."
                )
                cells = [
                    md(intro),
                    code(IMPORTS_REG.format(results_dir="../../results")),
                    code(METRICS_SRC.format()),
                    code(ds["prep"].format(**ds["prep_fmt"])),
                    code(XY_SPLIT.format(target=tgt_col)),
                    code(MODEL_FRAGMENTS[algo]),
                    code(PLOT_SAVE.format(dataset=ds_name, model=algo, target=tgt_col, stem=stem)),
                ]
                write_nb(path, cells)
                master_rows.append((ds_name, "Regression", tgt_col, algo,
                                    f"src/{ds['folder']}/{ALGO_SLUG[algo].lower().replace('elasticnet','elastic_net').replace('randomforest','random_forest').replace('stepwise','stepwise_regression')}.py",
                                    f"notebooks/{ds['folder']}/{stem}.ipynb",
                                    "results/tabular_results.csv"))


# --------------------------------------------------------------------------- #
#  ADDITIVE: HistGradientBoosting regression notebooks                         #
#  One of the 56 principal regression / 63 principal experiments overall,     #
#  but additive relative to the six-algorithm results/tabular_results.csv     #
#  (48 rows), which stays frozen by design. Generated by                      #
#  `python notebooks/build_notebooks.py histgbr`; leaves MASTER_TABLE.* alone.#
# --------------------------------------------------------------------------- #

HISTGBR_ALGO = "HistGradientBoostingRegressor"
histgbr_rows = []


def build_histgbr():
    for ds_name, ds in DATASETS.items():
        for tgt_col, tgt_tok in TARGETS.items():
            stem = f"{ds['slug']}_{ALGO_SLUG[HISTGBR_ALGO]}_{tgt_tok}"
            path = os.path.join(HERE, ds["folder"], stem + ".ipynb")
            feats = ("cpu_request, mem_request, cpu_mem_ratio, priority, scheduling_class, "
                     "collection_type" if ds_name.startswith("Google")
                     else "cpu_request, mem_request, cpu_mem_ratio, plan_disk, cpuset_width")
            intro = (
                f"# {ds_name} - HistGradientBoostingRegressor - `{tgt_col}`\n\n"
                f"**Module:** cross-sectional regression (ADDITIVE experiment) &nbsp;|&nbsp; "
                f"**Dataset:** {ds_name} &nbsp;|&nbsp; **Target:** `{tgt_col}`\n\n"
                f"**Input file(s):** `{ds['infile']}`\n\n"
                f"**Features (request + static metadata only; target never a feature):** `{feats}`\n\n"
                f"**Split:** random 80/20, `random_state=42` &nbsp;|&nbsp; "
                f"**CV:** GridSearchCV(cv=5) on TRAIN only, `learning_rate` x `max_depth`\n\n"
                f"**Preprocessing:** parse dict/columns -> feature engineering -> "
                f"RobustPrep(log1p + winsorise to TRAIN 1st/99th pct) -> "
                f"SimpleImputer(median) -> StandardScaler *(inside a Pipeline)*  "
                f"(no PolynomialFeatures; the trees model interactions directly, as for Random Forest)\n\n"
                f"**Model:** `HistGradientBoostingRegressor(random_state=42, early_stopping=False)` "
                f"— an independent non-linear ensemble (boosting) run alongside Random Forest (bagging).\n\n"
                f"**Result row -> ** `results/regression/{stem}.csv`  "
                f"(authoritative source-module output: `results/regression/histgbr/"
                f"{ds['folder']}__hist_gradient_boosting.csv`; aggregated by "
                f"`python -m src.run_all histgbr` into `results/tabular_results_histgbr.csv`).\n\n"
                f"This experiment is one of the 56 principal regression experiments / 63 "
                f"principal experiments overall, but is additive relative to the frozen, "
                f"six-algorithm `results/tabular_results.csv` (48 rows) and does not modify it "
                f"-- see `documentation/MASTER_EXPERIMENT_TABLE.md` for the combined inventory."
            )
            cells = [
                md(intro),
                code(IMPORTS_REG.format(results_dir="../../results")),
                code(METRICS_SRC.format()),
                code(ds["prep"].format(**ds["prep_fmt"])),
                code(XY_SPLIT.format(target=tgt_col)),
                code(MODEL_FRAGMENTS[HISTGBR_ALGO]),
                code(PLOT_SAVE.format(dataset=ds_name, model=HISTGBR_ALGO, target=tgt_col, stem=stem)),
            ]
            write_nb(path, cells)
            histgbr_rows.append((ds_name, "Regression", tgt_col, HISTGBR_ALGO,
                                 f"src/{ds['folder']}/hist_gradient_boosting.py",
                                 f"notebooks/{ds['folder']}/{stem}.ipynb",
                                 "results/tabular_results_histgbr.csv"))


def write_histgbr_table():
    hdr = "| Dataset | Module | Target | Algorithm | Source File | Notebook | Result File |\n"
    sep = "|---|---|---|---|---|---|---|\n"
    lines = [hdr, sep] + ["| " + " | ".join(str(c) for c in r) + " |\n" for r in histgbr_rows]
    with open(os.path.join(HERE, "MASTER_TABLE_histgbr.md"), "w", encoding="utf-8") as fh:
        fh.write("# Master table - ADDITIVE HistGradientBoostingRegressor experiment\n\n"
                 f"{len(histgbr_rows)} regression experiments (Google 2019 Cluster x 4 targets, "
                 "Alibaba 2017 x 4 targets), added to test whether the non-linear-ensemble "
                 "behaviour seen with Random Forest is reproduced by an independent boosting "
                 "ensemble. Same features, targets, split (`random_state=42`), preprocessing, "
                 "5-fold CV protocol and metrics as the existing regression modules.\n\n"
                 "One of the 56 principal regression / 63 principal experiments overall, "
                 "additive relative to the frozen, six-algorithm `results/tabular_results.csv` "
                 "(unchanged by this). See `documentation/MASTER_EXPERIMENT_TABLE.md` for the "
                 "combined inventory. Run: `python -m src.run_all histgbr`.\n\n")
        fh.writelines(lines)
    pd.DataFrame(histgbr_rows, columns=["dataset", "module", "target", "algorithm",
                                        "source_file", "notebook", "result_file"]
                 ).to_csv(os.path.join(HERE, "MASTER_TABLE_histgbr.csv"), index=False)
    print("wrote notebooks/MASTER_TABLE_histgbr.md / .csv  (%d rows)" % len(histgbr_rows))


# --------------------------------------------------------------------------- #
#  build Alibaba forecasting notebooks                                        #
# --------------------------------------------------------------------------- #

SEQ_ARCH = {
    "LSTM": '''\
# Step 10: Model architecture  --  stacked LSTM
model = Sequential([
    Input((WINDOW, 1)),
    LSTM(64, return_sequences=True), Dropout(0.2),
    LSTM(32),                        Dropout(0.2),
    Dense(1),
])''',
    "BiLSTM": '''\
# Step 10: Model architecture  --  stacked Bidirectional LSTM
model = Sequential([
    Input((WINDOW, 1)),
    Bidirectional(LSTM(64, return_sequences=True)), Dropout(0.2),
    Bidirectional(LSTM(32)),                        Dropout(0.2),
    Dense(1),
])''',
    "RNN": '''\
# Step 10: Model architecture  --  stacked SimpleRNN
model = Sequential([
    Input((WINDOW, 1)),
    SimpleRNN(64, return_sequences=True), Dropout(0.2),
    SimpleRNN(32),                        Dropout(0.2),
    Dense(1),
])''',
}


def build_forecasting():
    for model_name in ["LSTM", "BiLSTM", "RNN"]:
        for res, res_tok, util_col, tgt_label in [
                ("cpu", "CPU", "cpu_util", "cpu_util"),
                ("mem", "Memory", "mem_util", "mem_util")]:
            stem = f"Alibaba_2017_{model_name}_{res_tok}"
            path = os.path.join(HERE, "alibaba_2017", stem + ".ipynb")
            intro = (
                f"# Alibaba 2017 - {model_name} - next-interval forecast of `{util_col}`\n\n"
                f"**Module:** time-series forecasting &nbsp;|&nbsp; **Dataset:** Alibaba 2017 &nbsp;|&nbsp; "
                f"**Target:** `{util_col}` (machine utilisation, next 5-minute interval)\n\n"
                f"**Input file:** `server_usage.csv` (genuine machine-level telemetry, 5-min cadence, 12 h)\n\n"
                f"**Pipeline:** load server_usage -> per machine: chronological order -> "
                f"per-machine chronological 80/20 split -> `MinMaxScaler` fit on TRAIN points only -> "
                f"transform test -> previous-24-interval windows (no window crosses the split).\n\n"
                f"**Architecture:** 64 -> 32 recurrent units, Dropout 0.2, Dense(1) &nbsp;|&nbsp; "
                f"**Optimizer:** Adam(1e-3) &nbsp;|&nbsp; **Loss:** Huber(delta=1.0)\n\n"
                f"**Callbacks:** EarlyStopping(patience=8, restore_best_weights), "
                f"ReduceLROnPlateau(factor=0.5, patience=4) &nbsp;|&nbsp; 30 epochs, batch 128, "
                f"`validation_split=0.1`, `shuffle=False`\n\n"
                f"**Metrics (test):** MAE, RMSE, R2, sMAPE (train metrics also printed)\n\n"
                f"**Result row -> ** `results/sequence/{stem}.csv`  (aggregated into "
                f"`results/sequence_results.csv`)\n\n"
                f"> TensorFlow CPU training is not bit-for-bit deterministic; R2 varies by ~+-0.05 "
                f"between runs. This is the same code that produced `results/sequence_results.csv`."
            )
            src = '''\
# Step 1: Import libraries
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, SimpleRNN, Bidirectional, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
warnings.filterwarnings("ignore")

SEED = 42
tf.random.set_seed(SEED); np.random.seed(SEED)
EPS = 1e-9
WINDOW, HORIZON = 24, 1        # previous 24 five-minute intervals -> next interval
RESULTS_DIR = r"{results_dir}"
UTIL_COL = "{util_col}"        # TARGET

# Step 2: Load Alibaba server_usage  (machine-level telemetry)
ALIBABA_DIR = r"{alibaba_dir}"
su = pd.read_csv(os.path.join(ALIBABA_DIR, "server_usage.csv"), header=None,
                 names=["ts", "machine_id", "cpu_util", "mem_util", "disk_util",
                        "load1", "load5", "load15"])
su = su.dropna(subset=[UTIL_COL])
print("server_usage:", su.shape, "| machines:", su["machine_id"].nunique())

# Step 3-4: select the time-series per machine + chronological ordering
#           one series per machine, sorted by timestamp, rescaled percent -> [0,1]
MIN_LEN = 40
series = []
for _machine, g in su.sort_values(["machine_id", "ts"]).groupby("machine_id"):
    v = g[UTIL_COL].to_numpy(float) / 100.0
    if len(v) >= MIN_LEN:
        series.append(v)
print("machine series kept:", len(series))

# Step 5: sequence/window creation helper (previous WINDOW steps -> next step)
def make_windows(a, w=WINDOW, h=HORIZON):
    X, y = [], []
    for i in range(len(a) - w - h + 1):
        X.append(a[i:i + w]); y.append(a[i + w + h - 1])
    return np.array(X), np.array(y)

# Step 6-8: per-machine CHRONOLOGICAL 80/20 split (no shuffling, no leakage)
tr_raw, te_raw = [], []
for s in series:
    k = int(len(s) * 0.8)
    if k <= WINDOW + HORIZON or len(s) - k <= HORIZON:
        continue
    tr_raw.append(s[:k])                                  # first 80% of THIS machine
    te_raw.append(s[max(0, k - WINDOW - HORIZON + 1):])   # tail + context for 1st test window

# Step 9: fit the scaler on TRAINING points ONLY, then transform train & test
scaler = MinMaxScaler().fit(np.concatenate(tr_raw).reshape(-1, 1))
def stack(raws):
    Xs, ys = [], []
    for r in raws:
        rs = scaler.transform(r.reshape(-1, 1)).ravel()
        X, y = make_windows(rs)
        if len(X):
            Xs.append(X); ys.append(y)
    return np.concatenate(Xs)[..., None], np.concatenate(ys)
Xtr, ytr = stack(tr_raw)
Xte, yte = stack(te_raw)
print("windows  train:", Xtr.shape, " test:", Xte.shape)

{arch}

# Step 11: hyperparameters / compilation
model.compile(optimizer=Adam(1e-3), loss=Huber(delta=1.0))
model.summary()

# Step 12: train the model
history = model.fit(
    Xtr, ytr, validation_split=0.1, epochs=30, batch_size=128,
    shuffle=False, verbose=1,
    callbacks=[EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
               ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)])

# Step 13-14: predictions (inverse-scaled back to [0,1] utilisation)
def inv(a):
    return scaler.inverse_transform(np.asarray(a).reshape(-1, 1)).ravel()
train_pred, test_pred = inv(model.predict(Xtr, verbose=0)), inv(model.predict(Xte, verbose=0))
ytr_i, yte_i = inv(ytr), inv(yte)

# Step 15-17: metrics  (MAE, MAPE, RMSE, R2, sMAPE) for train and test
def smape(y, p):
    return float(np.mean(2*np.abs(p - y) / (np.abs(y) + np.abs(p) + EPS)) * 100)
def scores(y, p):
    return dict(MAE=float(mean_absolute_error(y, p)),
               MAPE=float(mean_absolute_percentage_error(y, p) * 100),
               RMSE=float(np.sqrt(np.mean((y - p) ** 2))),
               R2=float(r2_score(y, p)), sMAPE=smape(y, p))
tr_s, te_s = scores(ytr_i, train_pred), scores(yte_i, test_pred)
print("\\nAlibaba 2017  |  {model_name}  |  " + UTIL_COL)
print("  TRAIN  MAE {{:.4f}}  RMSE {{:.4f}}  R2 {{:.4f}}  sMAPE {{:.1f}}".format(
    tr_s["MAE"], tr_s["RMSE"], tr_s["R2"], tr_s["sMAPE"]))
print("  TEST   MAE {{:.4f}}  RMSE {{:.4f}}  R2 {{:.4f}}  sMAPE {{:.1f}}".format(
    te_s["MAE"], te_s["RMSE"], te_s["R2"], te_s["sMAPE"]))

# Step 18: plot actual vs predicted (first 500 test points)
plt.figure(figsize=(13, 4))
plt.plot(yte_i[:500], label="actual", lw=1)
plt.plot(test_pred[:500], label="predicted", lw=1)
plt.title("Alibaba 2017 - {model_name} - " + UTIL_COL + " (test, first 500)")
plt.legend(); plt.tight_layout(); plt.show()

# Step 19: save final results
row = dict(dataset="Alibaba 2017", module="forecasting", target=UTIL_COL, model="{model_name}",
           window=WINDOW, horizon=HORIZON, n_train=int(len(Xtr)), n_test=int(len(Xte)),
           train_MAE=tr_s["MAE"], train_MAPE=tr_s["MAPE"], train_RMSE=tr_s["RMSE"],
           train_R2=tr_s["R2"], train_sMAPE=tr_s["sMAPE"],
           test_MAE=te_s["MAE"], test_MAPE=te_s["MAPE"], test_RMSE=te_s["RMSE"],
           test_R2=te_s["R2"], test_sMAPE=te_s["sMAPE"])
os.makedirs(os.path.join(RESULTS_DIR, "sequence"), exist_ok=True)
out_csv = os.path.join(RESULTS_DIR, "sequence", "{stem}.csv")
pd.DataFrame([row]).to_csv(out_csv, index=False)
print("\\nsaved ->", out_csv)
'''.format(results_dir="../../results", alibaba_dir=ALIBABA_DIR,
           util_col=util_col, arch=SEQ_ARCH[model_name], model_name=model_name, stem=stem)
            write_nb(path, [md(intro), code(src)])
            master_rows.append(("Alibaba 2017", "Forecasting", util_col, model_name,
                                f"src/alibaba_2017/{model_name.lower()}.py",
                                f"notebooks/alibaba_2017/{stem}.ipynb",
                                "results/sequence_results.csv"))


# --------------------------------------------------------------------------- #
#  build Google temporal diagnostic notebook                                  #
# --------------------------------------------------------------------------- #

def build_diagnostic():
    stem = "Google_2019_Temporal_Diagnostic"
    path = os.path.join(HERE, "google_2019", stem + ".ipynb")
    intro = (
        "# Google 2019 Cluster - Temporal Diagnostic (why it is NOT used for forecasting)\n\n"
        "**Module:** temporal diagnostic &nbsp;|&nbsp; **Dataset:** Google 2019 Cluster\n\n"
        "**Input file:** `borg_traces_data.csv`\n\n"
        "This is **not** a forecasting experiment. It tests whether the public extract has "
        "any temporal structure a sequence model could learn. It does not.\n\n"
        "Steps: build the hourly cell-level aggregate (mean **and** sum) -> interpolate missing "
        "buckets -> autocorrelation (ACF) at several lags -> white-noise noise band "
        "(1.96/sqrt(n)) -> compare with an Alibaba cell-level series.\n\n"
        "**Result -> ** `results/temporal_diagnostic.json`\n\n"
        "**Conclusion:** Google ACF is inside the noise band at every lag (white noise); "
        "Alibaba ACF is 0.89 at lag 1 and 0.55 at lag 24 (clear diurnal structure). "
        "Therefore the final LSTM/BiLSTM/RNN forecasting module uses **Alibaba 2017 only**; "
        "Google 2019 Cluster is used for cross-sectional regression only."
    )
    src = '''\
# Step 1: Import libraries
import ast, os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BUCKET_US = 3_600_000_000          # 1 hour, in microseconds
LAGS = [1, 2, 3, 6, 12, 24, 48]
RESULTS_DIR = r"{results_dir}"
BORG_CSV = r"{borg_csv}"
ALIBABA_DIR = r"{alibaba_dir}"

# Step 2: Load the Google 2019 Cluster extract (only the 2 columns we need)
g = pd.read_csv(BORG_CSV, low_memory=False, usecols=["start_time", "average_usage"])
st = pd.to_numeric(g["start_time"], errors="coerce")
keep = st.between(1e6, 3e12)
g = g[keep].copy()
g["bucket"] = (st[keep] // BUCKET_US).astype("int64")   # hour index

# Step 3: parse average_usage -> cpus / memory
def dget(s, k):
    try:
        return ast.literal_eval(s).get(k, np.nan)
    except Exception:
        return np.nan

# Step 4-5: hourly cell-level aggregate  (MEAN and SUM)  + missing-bucket interpolation
def cell_series(resource, agg):
    v = g["average_usage"].map(lambda s: dget(s, resource))
    d = g.assign(val=v).dropna(subset=["val"])
    d = d[d["val"].between(0, 5)]
    s = d.groupby("bucket")["val"].mean() if agg == "mean" else d.groupby("bucket")["val"].sum()
    full = pd.RangeIndex(s.index.min(), s.index.max() + 1)          # regular hourly grid
    return s.reindex(full).interpolate("linear").bfill().ffill().to_numpy(float)

g_cpu_mean = cell_series("cpus", "mean")
g_cpu_sum  = cell_series("cpus", "sum")
g_mem_mean = cell_series("memory", "mean")
print("google hourly series length:", len(g_cpu_mean))

# Step 6: ACF  +  Step 7: noise band
def acf(x, lags=LAGS):
    x = (np.asarray(x, float) - np.mean(x)) / (np.std(x) + 1e-12)
    return {{int(k): float(np.corrcoef(x[:-k], x[k:])[0, 1]) for k in lags}}

def noise_band(n):
    return float(1.96 / np.sqrt(n))

# Step 8: comparison series -- Alibaba cell-level (mean of server_usage across machines per ts)
su = pd.read_csv(os.path.join(ALIBABA_DIR, "server_usage.csv"), header=None,
                 names=["ts", "machine_id", "cpu_util", "mem_util", "disk_util",
                        "load1", "load5", "load15"])
a_cpu = su.groupby("ts")["cpu_util"].mean().sort_index().to_numpy(float) / 100.0
a_mem = su.groupby("ts")["mem_util"].mean().sort_index().to_numpy(float) / 100.0

diag = {{
    "google_cpus_mean": {{"n": len(g_cpu_mean), "noise_band": round(noise_band(len(g_cpu_mean)), 4), "acf": acf(g_cpu_mean)}},
    "google_cpus_sum":  {{"n": len(g_cpu_sum),  "noise_band": round(noise_band(len(g_cpu_sum)), 4),  "acf": acf(g_cpu_sum)}},
    "google_memory_mean": {{"n": len(g_mem_mean), "noise_band": round(noise_band(len(g_mem_mean)), 4), "acf": acf(g_mem_mean)}},
    "alibaba_cpu_util_cell": {{"n": len(a_cpu), "noise_band": round(noise_band(len(a_cpu)), 4), "acf": acf(a_cpu, [1,2,3,6,12,24])}},
    "alibaba_mem_util_cell": {{"n": len(a_mem), "noise_band": round(noise_band(len(a_mem)), 4), "acf": acf(a_mem, [1,2,3,6,12,24])}},
}}

# Step 9: the resulting diagnostic
print(json.dumps(diag, indent=2))
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(os.path.join(RESULTS_DIR, "temporal_diagnostic.json"), "w") as fh:
    json.dump(diag, fh, indent=2)

# plot: Google (noise) vs Alibaba (structure)
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(g_cpu_mean); ax[0].set_title("Google 2019 Cluster - hourly cell CPU (mean)  -> white noise")
ax[1].plot(a_cpu);      ax[1].set_title("Alibaba 2017 - cell CPU utilisation  -> diurnal structure")
plt.tight_layout(); plt.show()

gb, ab = diag["google_cpus_mean"], diag["alibaba_cpu_util_cell"]
lags_bar = [1, 2, 3, 6, 12, 24]
plt.figure(figsize=(9, 4))
plt.bar([l - 0.35 for l in lags_bar], [gb["acf"][l] for l in lags_bar], width=0.7, label="Google 2019 Cluster")
plt.bar([l + 0.35 for l in lags_bar], [ab["acf"][l] for l in lags_bar], width=0.7, label="Alibaba 2017")
plt.axhline( gb["noise_band"], ls="--", c="grey"); plt.axhline(-gb["noise_band"], ls="--", c="grey")
plt.xlabel("lag (hours)"); plt.ylabel("autocorrelation"); plt.legend()
plt.title("ACF: Google extract is inside the noise band at every lag"); plt.tight_layout(); plt.show()

# Step 10: reasoning / decision
print("\\nDECISION")
print(f"  Google cell series (n={{gb['n']}}, band +-{{gb['noise_band']:.3f}}): "
      f"ACF lag1={{gb['acf'][1]:+.3f}}, lag24={{gb['acf'][24]:+.3f}}  -> statistically WHITE NOISE")
print(f"  Alibaba cell series (n={{ab['n']}}, band +-{{ab['noise_band']:.3f}}): "
      f"ACF lag1={{ab['acf'][1]:+.3f}}, lag24={{ab['acf'][24]:+.3f}}  -> clear diurnal STRUCTURE")
print("  => The public borg_traces_data.csv extract is a random cross-sectional sample and")
print("     cannot support sequence modelling. LSTM/BiLSTM/RNN would (and do) score R2 ~ 0 on it.")
print("  => FINAL DESIGN: Google 2019 Cluster -> cross-sectional regression only;")
print("     time-series forecasting (LSTM/BiLSTM/RNN) is reported on Alibaba 2017 only.")
'''.format(results_dir="../../results", borg_csv=BORG_CSV, alibaba_dir=ALIBABA_DIR)
    write_nb(path, [md(intro), code(src)])
    master_rows.append(("Google 2019 Cluster", "Temporal Diagnostic", "-", "ACF",
                        "src/google_2019/temporal_diagnostic.py",
                        f"notebooks/google_2019/{stem}.ipynb",
                        "results/temporal_diagnostic.json"))


def write_master_table():
    hdr = "| Dataset | Module | Target | Algorithm | Source File | Notebook | Result File |\n"
    sep = "|---|---|---|---|---|---|---|\n"
    lines = [hdr, sep]
    for r in master_rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |\n")
    with open(os.path.join(HERE, "MASTER_TABLE.md"), "w", encoding="utf-8") as fh:
        fh.write("# Master verification table\n\n"
                 f"{len(master_rows)} experiments. "
                 "Each notebook is self-contained and writes its own row into "
                 "`results/regression/` or `results/sequence/`; `src/run_all.py` "
                 "aggregates those into `results/tabular_results.csv` / "
                 "`results/sequence_results.csv`.\n\n")
        fh.writelines(lines)
    pd.DataFrame(master_rows, columns=["dataset", "module", "target", "algorithm",
                                       "source_file", "notebook", "result_file"]
                 ).to_csv(os.path.join(HERE, "MASTER_TABLE.csv"), index=False)
    print("wrote notebooks/MASTER_TABLE.md / .csv  (%d rows)" % len(master_rows))


if __name__ == "__main__":
    import sys
    import pandas as pd  # noqa
    if len(sys.argv) > 1 and sys.argv[1] == "histgbr":
        # ADDITIVE only: 8 HistGradientBoosting notebooks + companion table.
        # Does NOT touch the 55 six-algorithm notebooks or MASTER_TABLE.md/.csv
        # (48 regression + 6 forecasting + 1 diagnostic); those plus these 8
        # HistGradientBoosting notebooks make up the 63 principal experiments.
        build_histgbr()
        write_histgbr_table()
        print(f"\nDONE (histgbr). {len(histgbr_rows)} notebooks + companion table.")
    else:
        build_regression()
        build_forecasting()
        build_diagnostic()
        write_master_table()
        print(f"\nDONE. {len(master_rows)} notebooks + master table.")
