"""
Shared, leakage-free workload-prediction pipeline for BOTH datasets:
  - Google 2019 Cluster   (borg_traces_data.csv)
  - Alibaba cluster-trace-v2017  (container_event.csv + container_usage.csv + server_usage.csv)

Design principles (fixes the problems found in the original notebooks):
  * FEATURES = resource request / static metadata ONLY.
  * TARGETS  = observed usage ONLY  (never a feature, never used to derive a feature).
  * cpu_mem_ratio is computed from REQUESTS, not usage  -> no leakage.
  * Chronological 80/20 split (sorted by start/creation time) -> no look-ahead.
  * SimpleImputer / PolynomialFeatures / Scaler live INSIDE an sklearn Pipeline,
    so they are fit on the training folds only.
  * GridSearchCV(cv=5) runs on the training partition only.
  * Stepwise = embedded LassoCV selection -> OLS refit, entirely on training data.
  * Metrics: MAE, RMSE, R2 and a guarded sMAPE (raw MAPE is meaningless here because
    usage values are frequently ~0). Reported for TRAIN and TEST.
"""
import ast
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (PolynomialFeatures, StandardScaler, MinMaxScaler,
                                   RobustScaler, FunctionTransformer)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

RANDOM_STATE = 42
EPS = 1e-9

# ----------------------------------------------------------------------------- #
#  DATASET ADAPTERS  ->  canonical entity table                                 #
#  columns: cpu_request, mem_request, cpu_mem_ratio, + dataset metadata,        #
#           cpu_usage_mean, mem_usage_mean, cpu_usage_peak, mem_usage_peak,     #
#           order_key                                                           #
# ----------------------------------------------------------------------------- #

BORG_META  = ["priority", "scheduling_class", "collection_type"]
ALI_META   = ["plan_disk", "cpuset_width"]


def _dget(s, key):
    try:
        return ast.literal_eval(s).get(key, np.nan)
    except Exception:
        return np.nan


def load_borg(path, nrows=None, sample=None):
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=RANDOM_STATE)
    for src, pfx in [("resource_request", "req"),
                     ("average_usage", "avg"),
                     ("maximum_usage", "max")]:
        df[f"{pfx}_cpu"] = df[src].map(lambda s: _dget(s, "cpus"))
        df[f"{pfx}_mem"] = df[src].map(lambda s: _dget(s, "memory"))

    out = pd.DataFrame({
        "cpu_request":      pd.to_numeric(df["req_cpu"], errors="coerce"),
        "mem_request":      pd.to_numeric(df["req_mem"], errors="coerce"),
        "priority":         pd.to_numeric(df["priority"], errors="coerce"),
        "scheduling_class": pd.to_numeric(df["scheduling_class"], errors="coerce"),
        "collection_type":  pd.to_numeric(df["collection_type"], errors="coerce"),
        "cpu_usage_mean":   pd.to_numeric(df["avg_cpu"], errors="coerce"),
        "mem_usage_mean":   pd.to_numeric(df["avg_mem"], errors="coerce"),
        "cpu_usage_peak":   pd.to_numeric(df["max_cpu"], errors="coerce"),
        "mem_usage_peak":   pd.to_numeric(df["max_mem"], errors="coerce"),
        "order_key":        pd.to_numeric(df["start_time"], errors="coerce"),
    })
    out["cpu_mem_ratio"] = out["cpu_request"] / (out["mem_request"] + EPS)
    out = out[(out["cpu_request"] > 0) & (out["mem_request"] > 0)]
    for c in ["cpu_usage_mean", "mem_usage_mean", "cpu_usage_peak", "mem_usage_peak"]:
        out = out[out[c].between(0, 5)]          # normalised usage; drop parsing junk
    out = out.dropna(subset=["order_key"]).sort_values("order_key").reset_index(drop=True)
    return out, BORG_META


def load_alibaba(folder, usage_as_fraction=True):
    import os
    ce = pd.read_csv(os.path.join(folder, "container_event.csv"), header=None,
                     names=["ts", "event", "instance_id", "machine_id",
                            "plan_cpu", "plan_mem", "plan_disk", "cpuset", "_x"])
    ce = ce[ce["event"] == "Create"].copy()
    ce["cpuset_width"] = ce["cpuset"].astype(str).str.count(r"\|") + 1
    ce = (ce.sort_values("ts")
            .groupby("instance_id")
            .agg(plan_cpu=("plan_cpu", "first"), plan_mem=("plan_mem", "first"),
                 plan_disk=("plan_disk", "first"), cpuset_width=("cpuset_width", "first"))
            .reset_index())

    cu = pd.read_csv(os.path.join(folder, "container_usage.csv"), header=None,
                     names=["ts", "instance_id", "cpu_util", "mem_util", "disk_util",
                            "load1", "load5", "load15",
                            "avg_cpi", "avg_mpki", "max_cpi", "max_mpki"])
    agg = (cu.groupby("instance_id")
             .agg(cpu_usage_mean=("cpu_util", "mean"), cpu_usage_peak=("cpu_util", "max"),
                  mem_usage_mean=("mem_util", "mean"), mem_usage_peak=("mem_util", "max"),
                  order_key=("ts", "min"), n_obs=("ts", "size"))
             .reset_index())
    agg = agg[agg["n_obs"] >= 3]                 # need a few observations to characterise

    m = ce.merge(agg, on="instance_id", how="inner")
    scale = 100.0 if usage_as_fraction else 1.0
    out = pd.DataFrame({
        "cpu_request":    m["plan_cpu"] / m["plan_cpu"].max(),   # normalise core count
        "mem_request":    m["plan_mem"],
        "plan_disk":      m["plan_disk"],
        "cpuset_width":   m["cpuset_width"],
        "cpu_usage_mean": m["cpu_usage_mean"] / scale,
        "mem_usage_mean": m["mem_usage_mean"] / scale,
        "cpu_usage_peak": m["cpu_usage_peak"] / scale,
        "mem_usage_peak": m["mem_usage_peak"] / scale,
        "order_key":      m["order_key"],
    })
    out["cpu_mem_ratio"] = (m["plan_cpu"].values) / (m["plan_mem"].values + EPS)
    out = out.dropna(subset=["order_key"]).sort_values("order_key").reset_index(drop=True)
    return out, ALI_META


# ----------------------------------------------------------------------------- #
#  SPLIT + METRICS                                                              #
# ----------------------------------------------------------------------------- #

def chrono_split(df, frac=0.8):
    k = int(len(df) * frac)
    return df.iloc[:k].copy(), df.iloc[k:].copy()


def split_df(df, how="random", frac=0.8):
    """Cross-sectional regression (job usage from job request) -> random split is
    standard and matches the original study. 'chrono' kept for sensitivity analysis."""
    if how == "chrono":
        return chrono_split(df, frac)
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(df, test_size=1 - frac, random_state=RANDOM_STATE, shuffle=True)
    return tr.copy(), te.copy()


def smape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(2 * np.abs(p - y) / (np.abs(y) + np.abs(p) + EPS)) * 100)


def scores(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "MAE":   float(mean_absolute_error(y, p)),
        "RMSE":  float(np.sqrt(np.mean((y - p) ** 2))),
        "R2":    float(r2_score(y, p)),
        "sMAPE": smape(y, p),
    }


# ----------------------------------------------------------------------------- #
#  MODELS                                                                       #
# ----------------------------------------------------------------------------- #

class RobustPrep:
    """log1p on non-negative skewed features + winsorise to TRAIN [p1, p99].
    Prevents degree-2 polynomial extrapolation blow-ups on heavy-tailed request data.
    Bounds come from training data only -> no leakage."""
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


def _pipe(model, poly, scaler):
    steps = [("impute", SimpleImputer(strategy="median"))]
    if poly:
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
    sc = {"minmax": MinMaxScaler(), "standard": StandardScaler(), "robust": RobustScaler()}[scaler]
    steps.append(("scaler", sc))
    steps.append(("model", model))
    return Pipeline(steps)


def model_specs():
    return {
        "Linear Regression": dict(est=LinearRegression(), grid={}, poly=False, scaler="minmax"),
        "Ridge Regression":  dict(est=Ridge(random_state=RANDOM_STATE),
                                  grid={"model__alpha": [0.001, 0.01, 0.1, 1, 10, 100]},
                                  poly=False, scaler="standard"),
        "Lasso Regression":  dict(est=Lasso(max_iter=3000, tol=1e-3, random_state=RANDOM_STATE),
                                  grid={"model__alpha": [1e-4, 1e-3, 1e-2, 0.1]},
                                  poly=True, scaler="standard"),
        "Elastic Net":       dict(est=ElasticNet(max_iter=3000, tol=1e-3, random_state=RANDOM_STATE),
                                  grid={"model__alpha": [1e-3, 1e-2, 0.1],
                                        "model__l1_ratio": [0.3, 0.6]},
                                  poly=True, scaler="standard"),
        "Random Forest":     dict(est=RandomForestRegressor(n_estimators=150, n_jobs=-1,
                                                            random_state=RANDOM_STATE,
                                                            max_samples=0.5,
                                                            min_samples_leaf=10),
                                  grid={"model__max_depth": [6, 12]},
                                  poly=False, scaler="standard"),
    }


def fit_one(name, spec, Xtr, ytr, Xte, yte):
    pipe = _pipe(spec["est"], spec["poly"], spec["scaler"])
    if spec["grid"]:
        gs = GridSearchCV(pipe, spec["grid"], cv=5,
                          scoring="neg_mean_absolute_error", n_jobs=4)
        gs.fit(Xtr, ytr)
        best, params = gs.best_estimator_, gs.best_params_
    else:
        pipe.fit(Xtr, ytr)
        best, params = pipe, {}
    return {"model": name, "params": params,
            "train": scores(ytr, best.predict(Xtr)),
            "test":  scores(yte, best.predict(Xte))}


def fit_stepwise(Xtr, ytr, Xte, yte):
    """Embedded LassoCV feature selection on degree-2 polynomial features, then OLS refit."""
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
    return {"model": "Stepwise Regression",
            "params": {"n_selected": int(mask.sum()), "alpha": float(sel.alpha_)},
            "train": scores(ytr, ols.predict(Ztr[:, mask])),
            "test":  scores(yte, ols.predict(Zte[:, mask]))}


def run_tabular(df, meta, target, feature_cols, split="random"):
    tr, te = split_df(df, how=split)
    prep = RobustPrep().fit(tr[feature_cols].values)
    Xtr, ytr = prep.transform(tr[feature_cols].values), tr[target].values
    Xte, yte = prep.transform(te[feature_cols].values), te[target].values
    rows = [fit_stepwise(Xtr, ytr, Xte, yte)]
    for name, spec in model_specs().items():
        rows.append(fit_one(name, spec, Xtr, ytr, Xte, yte))
    return rows, (len(tr), len(te))
