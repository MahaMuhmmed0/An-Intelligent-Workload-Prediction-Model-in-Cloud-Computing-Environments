"""
src/evaluation.py
================
Shared evaluation / preprocessing primitives used by every algorithm file.

This is the SAME logic that produced the numbers in results/  (it was lifted
verbatim from common/pipeline.py and common/sequence.py, which generated the
original results). Nothing here changes the methodology; it only organises it so
each algorithm can live in its own file.

Contents
--------
Regression (cross-sectional: predict a job's usage from its request)
    RANDOM_STATE, EPS
    RobustPrep                 log1p + winsorise to TRAIN 1st/99th pct (no leak)
    split_random(df)           shuffled 80/20 split, random_state=42
    regression_scores(y, yhat) -> MAE, MAPE, RMSE, R2, sMAPE
    fit_pipeline_model(...)    SimpleImputer -> [PolynomialFeatures] -> Scaler -> est
                               with optional GridSearchCV(cv=5)
    fit_stepwise(...)          PolynomialFeatures(2) -> StandardScaler ->
                               LassoCV(cv=5) selection -> OLS refit
    run_regression(...)        driver: load -> prep -> fit -> score -> save
    TARGETS                    the four regression targets

Forecasting (time series: predict the next interval from the previous 24)
    WINDOW = 24, HORIZON = 1
    make_windows(series, w, h)
    sequence_scores(y, yhat, scaler) -> MAE, MAPE, RMSE, R2, sMAPE (inverse-scaled)
    save_sequence_result(...)
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (PolynomialFeatures, StandardScaler,
                                   MinMaxScaler, RobustScaler)
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (mean_absolute_error, r2_score,
                             mean_absolute_percentage_error)

RANDOM_STATE = 42
EPS = 1e-9

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Dataset locations. Configurable via environment variables so the project
# runs on any machine without editing source files; falls back to a
# repository-relative "data/" layout if the variables are not set. See
# README.md "Dataset locations" and .env.example.
GOOGLE_2019_CSV = os.environ.get(
    "GOOGLE_2019_CSV",
    os.path.join(PROJECT_ROOT, "data", "google_2019", "borg_traces_data.csv"),
)
ALIBABA_2017_DIR = os.environ.get(
    "ALIBABA_2017_DIR",
    os.path.join(PROJECT_ROOT, "data", "alibaba_2017"),
)

# The four cross-sectional regression targets (same feature set for all of them).
TARGETS = ["cpu_usage_mean", "mem_usage_mean", "cpu_usage_peak", "mem_usage_peak"]


# --------------------------------------------------------------------------- #
#  REGRESSION preprocessing / metrics                                         #
# --------------------------------------------------------------------------- #

class RobustPrep:
    """log1p on non-negative skewed features + winsorise to the TRAIN 1st/99th
    percentile.  Bounds are learned from training rows only -> no leakage.
    Keeps degree-2 polynomial regressors numerically stable on heavy-tailed
    request data (without it, Linear/Stepwise on Google memory explode to R2 ~ -5)."""

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


def split_random(df, frac=0.8):
    """Shuffled 80/20 split.  The regression task is cross-sectional (predict an
    unobserved property of a job from its request), so a random split is correct
    and matches the original study.  random_state=42 -> reproducible."""
    tr, te = train_test_split(df, test_size=1 - frac,
                              random_state=RANDOM_STATE, shuffle=True)
    return tr.copy(), te.copy()


def _smape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(2 * np.abs(p - y) / (np.abs(y) + np.abs(p) + EPS)) * 100)


def regression_scores(y, p):
    """MAE, MAPE, RMSE, R2, sMAPE.

    NOTE ON MAPE: usage is frequently ~0, so raw MAPE is astronomically large and
    must not be reported as a headline metric.  It is included only because the
    brief asks for it; sMAPE is the bounded percentage error to actually use."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "MAE":   float(mean_absolute_error(y, p)),
        "MAPE":  float(mean_absolute_percentage_error(y, p) * 100),   # unreliable near 0
        "RMSE":  float(np.sqrt(np.mean((y - p) ** 2))),
        "R2":    float(r2_score(y, p)),
        "sMAPE": _smape(y, p),
    }


def _make_pipeline(estimator, use_poly, scaler):
    steps = [("impute", SimpleImputer(strategy="median"))]
    if use_poly:
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
    steps.append(("scaler", {"minmax": MinMaxScaler(),
                             "standard": StandardScaler(),
                             "robust": RobustScaler()}[scaler]))
    steps.append(("model", estimator))
    return Pipeline(steps)


def fit_pipeline_model(estimator, param_grid, use_poly, scaler,
                       Xtr, ytr, Xte, yte):
    """Fit  SimpleImputer -> [PolynomialFeatures(2)] -> Scaler -> estimator.
    If param_grid is non-empty, wrap in GridSearchCV(cv=5, scoring=neg MAE)
    on the TRAINING partition only."""
    pipe = _make_pipeline(estimator, use_poly, scaler)
    if param_grid:
        gs = GridSearchCV(pipe, param_grid, cv=5,
                          scoring="neg_mean_absolute_error", n_jobs=4)
        gs.fit(Xtr, ytr)
        best, params = gs.best_estimator_, gs.best_params_
    else:
        pipe.fit(Xtr, ytr)
        best, params = pipe, {}
    return {"train": regression_scores(ytr, best.predict(Xtr)),
            "test":  regression_scores(yte, best.predict(Xte)),
            "params": params}


def fit_stepwise(Xtr, ytr, Xte, yte):
    """Stepwise = embedded LassoCV feature selection on degree-2 polynomial
    features, then an OLS refit on the retained features.  Every fitted transform
    and the selector see TRAINING data only."""
    imp = SimpleImputer(strategy="median").fit(Xtr)
    poly = PolynomialFeatures(degree=2, include_bias=False).fit(imp.transform(Xtr))
    sc = StandardScaler().fit(poly.transform(imp.transform(Xtr)))
    Ztr = sc.transform(poly.transform(imp.transform(Xtr)))
    Zte = sc.transform(poly.transform(imp.transform(Xte)))
    sel = LassoCV(cv=5, max_iter=10000, random_state=RANDOM_STATE).fit(Ztr, ytr)
    mask = sel.coef_ != 0
    if mask.sum() == 0:
        mask[:] = True
    ols = LinearRegression().fit(Ztr[:, mask], ytr)
    return {"train": regression_scores(ytr, ols.predict(Ztr[:, mask])),
            "test":  regression_scores(yte, ols.predict(Zte[:, mask])),
            "params": {"n_selected": int(mask.sum()), "alpha": float(sel.alpha_)}}


def _save_csv(df, path):
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        alt = path.replace(".csv", f"_{int(time.time())}.csv")
        df.to_csv(alt, index=False)
        print(f"  (‘{os.path.basename(path)}’ is locked/open — wrote {alt} instead)")
        return alt


def run_regression(dataset_name, model_name, out_csv,
                   load_entity_table, features,
                   estimator=None, param_grid=None,
                   use_poly=False, scaler="standard", stepwise=False):
    """One algorithm, all four targets.  Prints train/test metrics and writes
    <out_csv> with one row per target.  Returns the rows."""
    if not os.path.isabs(out_csv):
        out_csv = os.path.join(PROJECT_ROOT, out_csv)
    df, feats_default, _meta = load_entity_table()
    features = features or feats_default
    tr, te = split_random(df)

    prep = RobustPrep().fit(tr[features].values)
    Xtr_all = prep.transform(tr[features].values)
    Xte_all = prep.transform(te[features].values)

    rows = []
    print(f"\n=== {dataset_name} · {model_name} "
          f"(features={features}; train={len(tr)}, test={len(te)}) ===")
    hdr = f"{'target':<16}{'tr_MAE':>10}{'tr_R2':>9}{'te_MAE':>10}{'te_R2':>9}{'te_RMSE':>10}{'te_sMAPE':>10}"
    print(hdr)
    for tgt in TARGETS:
        ytr, yte = tr[tgt].values, te[tgt].values
        if stepwise:
            r = fit_stepwise(Xtr_all, ytr, Xte_all, yte)
        else:
            r = fit_pipeline_model(estimator, param_grid or {}, use_poly, scaler,
                                   Xtr_all, ytr, Xte_all, yte)
        print(f"{tgt:<16}{r['train']['MAE']:>10.4f}{r['train']['R2']:>9.4f}"
              f"{r['test']['MAE']:>10.4f}{r['test']['R2']:>9.4f}"
              f"{r['test']['RMSE']:>10.4f}{r['test']['sMAPE']:>10.1f}")
        rows.append({
            "dataset": dataset_name, "module": "regression", "target": tgt,
            "model": model_name, "n_train": len(tr), "n_test": len(te),
            "train_MAE": r["train"]["MAE"], "train_MAPE": r["train"]["MAPE"],
            "train_RMSE": r["train"]["RMSE"], "train_R2": r["train"]["R2"],
            "train_sMAPE": r["train"]["sMAPE"],
            "test_MAE": r["test"]["MAE"], "test_MAPE": r["test"]["MAPE"],
            "test_RMSE": r["test"]["RMSE"], "test_R2": r["test"]["R2"],
            "test_sMAPE": r["test"]["sMAPE"],
            "hyperparams": json.dumps(r["params"]),
        })
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    _save_csv(pd.DataFrame(rows), out_csv)
    print(f"  -> {out_csv}")
    return rows


# --------------------------------------------------------------------------- #
#  FORECASTING preprocessing / metrics                                        #
# --------------------------------------------------------------------------- #

WINDOW = 24          # previous 24 five-minute intervals  (= 2 hours)
HORIZON = 1          # predict the next interval
SEQ_SEED = 42


def make_windows(arr, w=WINDOW, h=HORIZON):
    """Sliding windows over a single 1-D series: X[i] = arr[i:i+w], y[i] = arr[i+w+h-1]."""
    X, y = [], []
    for i in range(len(arr) - w - h + 1):
        X.append(arr[i:i + w])
        y.append(arr[i + w + h - 1])
    return np.array(X), np.array(y)


def sequence_scores(y, p, scaler):
    """Inverse-scale predictions and targets, then MAE / MAPE / RMSE / R2 / sMAPE."""
    y = scaler.inverse_transform(np.asarray(y).reshape(-1, 1)).ravel()
    p = scaler.inverse_transform(np.asarray(p).reshape(-1, 1)).ravel()
    return {
        "MAE":   float(mean_absolute_error(y, p)),
        "MAPE":  float(mean_absolute_percentage_error(y, p) * 100),
        "RMSE":  float(np.sqrt(np.mean((y - p) ** 2))),
        "R2":    float(r2_score(y, p)),
        "sMAPE": _smape(y, p),
    }


def save_sequence_result(rows, out_csv):
    if not os.path.isabs(out_csv):
        out_csv = os.path.join(PROJECT_ROOT, out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    _save_csv(pd.DataFrame(rows), out_csv)
    print(f"  -> {out_csv}")
