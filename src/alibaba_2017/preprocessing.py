"""
src/alibaba_2017/preprocessing.py
=================================
Two preprocessing entry points for the Alibaba cluster-trace-v2017 data.

1. load_entity_table()  -- CROSS-SECTIONAL REGRESSION
   One row per online container instance. Joins the request (container_event)
   to the instance's observed usage (container_usage, aggregated per instance).

   FEATURES (inputs -- request + static metadata only)
       cpu_request     plan_cpu / max(plan_cpu)     (core count, normalised)
       mem_request     plan_mem                     (normalised)
       cpu_mem_ratio   plan_cpu / plan_mem          (from REQUESTS -> no leak)
       plan_disk       normalised disk request
       cpuset_width    number of pinned CPUs (from the '|'-delimited cpuset)

   TARGETS (usage only; util is a % of the instance's own request, rescaled to [0,1])
       cpu_usage_mean / mem_usage_mean   mean of cpu_util% / mem_util% per instance
       cpu_usage_peak / mem_usage_peak   max  of cpu_util% / mem_util% per instance

   ~10 185 instances (>=3 usage observations each). This is the exact logic
   (`load_alibaba`) behind results/tabular_results.csv.

2. load_server_usage_series(resource) -- TIME-SERIES FORECASTING
   Per-machine utilisation series from server_usage.csv (dense, 5-min cadence,
   12 h -> 144 points per machine). Used by lstm.py / bilstm.py / rnn.py.

3. build_forecasting_windows(resource) -- the full sequence pipeline in one place:
   load server_usage -> per machine: chronological order -> chronological 80/20
   split of THAT machine's timeline -> fit MinMaxScaler on TRAIN points only ->
   transform train & test -> previous-24 windows (no window crosses the split).
"""
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.evaluation import WINDOW, HORIZON, make_windows, ALIBABA_2017_DIR

# Configurable via the ALIBABA_2017_DIR environment variable; see
# src/evaluation.py and README.md "Dataset locations".
ALIBABA_DIR = ALIBABA_2017_DIR

FEATURES = ["cpu_request", "mem_request", "cpu_mem_ratio", "plan_disk", "cpuset_width"]
META = ["plan_disk", "cpuset_width"]

EPS = 1e-9
MIN_SERIES_LEN = 40           # keep machines with >=40 of the 144 intervals


# --------------------------------------------------------------------------- #
#  1. cross-sectional regression entity table                                 #
# --------------------------------------------------------------------------- #

_ENTITY_CACHE = {}


def _load_entity_table_impl(folder=ALIBABA_DIR):
    ce = pd.read_csv(os.path.join(folder, "container_event.csv"), header=None,
                     names=["ts", "event", "instance_id", "machine_id",
                            "plan_cpu", "plan_mem", "plan_disk", "cpuset", "_x"])
    ce = ce[ce["event"] == "Create"].copy()
    ce["cpuset_width"] = ce["cpuset"].astype(str).str.count(r"\|") + 1
    ce = (ce.sort_values("ts").groupby("instance_id")
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
    agg = agg[agg["n_obs"] >= 3]

    m = ce.merge(agg, on="instance_id", how="inner")
    out = pd.DataFrame({
        "cpu_request":    m["plan_cpu"] / m["plan_cpu"].max(),
        "mem_request":    m["plan_mem"],
        "plan_disk":      m["plan_disk"],
        "cpuset_width":   m["cpuset_width"],
        "cpu_usage_mean": m["cpu_usage_mean"] / 100.0,
        "mem_usage_mean": m["mem_usage_mean"] / 100.0,
        "cpu_usage_peak": m["cpu_usage_peak"] / 100.0,
        "mem_usage_peak": m["mem_usage_peak"] / 100.0,
        "order_key":      m["order_key"],
    })
    out["cpu_mem_ratio"] = m["plan_cpu"].values / (m["plan_mem"].values + EPS)
    out = out.dropna(subset=["order_key"]).sort_values("order_key").reset_index(drop=True)
    return out, FEATURES, META


# --------------------------------------------------------------------------- #
#  2 + 3. time-series forecasting input                                        #
# --------------------------------------------------------------------------- #

def load_server_usage_series(resource, folder=ALIBABA_DIR, min_len=MIN_SERIES_LEN):
    """Return a list of 1-D arrays, one per machine, of that machine's
    utilisation over time (values rescaled from percent to [0,1]), ordered
    chronologically by timestamp."""
    col = {"cpu": "cpu_util", "mem": "mem_util"}[resource]
    su = pd.read_csv(os.path.join(folder, "server_usage.csv"), header=None,
                     names=["ts", "machine_id", "cpu_util", "mem_util", "disk_util",
                            "load1", "load5", "load15"])
    su = su.dropna(subset=[col])
    series = []
    for _machine, g in su.sort_values(["machine_id", "ts"]).groupby("machine_id"):
        v = g[col].to_numpy(float) / 100.0            # percent -> [0,1]
        if len(v) >= min_len:
            series.append(v)
    return series


def build_forecasting_windows(resource, w=WINDOW, h=HORIZON, split=0.8):
    """Full leakage-free sequence pipeline.

    For every machine series:
      * split THAT machine's timeline chronologically at 80 %
      * training points = s[:k]; test context = s[k-w-h+1:]  (so the first test
        window is fully populated but still targets a post-split point)
    Then:
      * fit ONE MinMaxScaler on the concatenation of all training points only
      * transform train and test points with it
      * cut previous-`w` windows -> (X, y); no window straddles a split boundary
    Returns Xtr, ytr, Xte, yte, scaler  (X shaped [n, w, 1]).
    """
    series = load_server_usage_series(resource)
    tr_raw, te_raw = [], []
    for s in series:
        k = int(len(s) * split)
        if k <= w + h or len(s) - k <= h:
            continue
        tr_raw.append(s[:k])
        te_raw.append(s[max(0, k - w - h + 1):])

    scaler = MinMaxScaler().fit(np.concatenate(tr_raw).reshape(-1, 1))

    def stack(raws):
        Xs, ys = [], []
        for r in raws:
            rs = scaler.transform(r.reshape(-1, 1)).ravel()
            X, y = make_windows(rs, w, h)
            if len(X):
                Xs.append(X)
                ys.append(y)
        return np.concatenate(Xs)[..., None], np.concatenate(ys)

    Xtr, ytr = stack(tr_raw)
    Xte, yte = stack(te_raw)
    return Xtr, ytr, Xte, yte, scaler




def load_entity_table(*args, **kwargs):
    key = (args, tuple(sorted(kwargs.items())))
    if key not in _ENTITY_CACHE:
        _ENTITY_CACHE[key] = _load_entity_table_impl(*args, **kwargs)
    df, feats, meta = _ENTITY_CACHE[key]
    return df.copy(), list(feats), list(meta)
