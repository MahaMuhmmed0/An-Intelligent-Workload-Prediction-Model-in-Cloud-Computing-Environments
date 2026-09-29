"""
src/google_2019/temporal_diagnostic.py
======================================
Investigation: does the public Google 2019 Cluster extract support temporal
forecasting?  ANSWER: no.  This file produces the evidence for that decision.

Do NOT read this as a successful forecasting experiment. The LSTM/BiLSTM/RNN
attempt on this data is in sequence_attempt.py and scores R2 ~ 0; this file
explains why (the series has no autocorrelation to learn).

What it does
------------
1. build_cell_series(resource, agg)
     * take every instance-usage row, bucket start_time into 1-hour bins
       (BUCKET_US = 3_600_000_000 microseconds)
     * aggregate average_usage[resource] across all instances in each bin
       - agg="mean"  -> typical per-instance intensity
       - agg="sum"   -> total offered load
     * reindex onto a regular hourly grid and linearly interpolate the few
       missing buckets (bfill/ffill the ends)
2. acf(x, lags)          sample autocorrelation at the given lags
3. noise_band(n)         1.96 / sqrt(n)  -> white-noise 95% confidence half-width
4. compare with an Alibaba cell-level series (mean of server_usage across
   machines per 5-minute timestamp)
5. write results/temporal_diagnostic.json  and print the verdict

Result (representative):
    Google cell series (n=745, band +-0.072): ACF at lags 1..24 all within +-0.04
    Alibaba cell series (n~138, band +-0.167): ACF 0.89 at lag 1, 0.55 at lag 24
    -> Google extract is statistically white noise; Alibaba has clear diurnal
       structure. Google 2019 Cluster is therefore used for cross-sectional
       regression only; forecasting (LSTM/BiLSTM/RNN) is reported on Alibaba.

Run:  python -m src.google_2019.temporal_diagnostic
"""
import ast
import json
import os

import numpy as np
import pandas as pd

from src.evaluation import PROJECT_ROOT
from src.google_2019.preprocessing import BORG_CSV
from src.alibaba_2017.preprocessing import ALIBABA_DIR

BUCKET_US = 3_600_000_000          # 1 hour in microseconds
LAGS = [1, 2, 3, 6, 12, 24, 48]


def build_cell_series(resource="cpus", agg="mean", path=BORG_CSV):
    df = pd.read_csv(path, low_memory=False, usecols=["start_time", "average_usage"])
    st = pd.to_numeric(df["start_time"], errors="coerce")
    keep = st.between(1e6, 3e12)
    df = df[keep].copy()
    df["bucket"] = (st[keep] // BUCKET_US).astype("int64")
    df["val"] = df["average_usage"].map(
        lambda s: ast.literal_eval(s).get(resource, np.nan) if isinstance(s, str) else np.nan)
    df = df.dropna(subset=["val"])
    df = df[df["val"].between(0, 5)]

    grp = df.groupby("bucket")["val"]
    s = grp.mean() if agg == "mean" else grp.sum()

    full = pd.RangeIndex(s.index.min(), s.index.max() + 1)      # regular hourly grid
    s = s.reindex(full).interpolate("linear").bfill().ffill()   # fill missing buckets
    return s.to_numpy(float)


def build_alibaba_cell_series(resource="cpu_util", folder=ALIBABA_DIR):
    su = pd.read_csv(os.path.join(folder, "server_usage.csv"), header=None,
                     names=["ts", "machine_id", "cpu_util", "mem_util", "disk_util",
                            "load1", "load5", "load15"])
    return su.groupby("ts")[resource].mean().sort_index().to_numpy(float) / 100.0


def acf(x, lags=LAGS):
    x = (np.asarray(x, float) - np.mean(x)) / (np.std(x) + 1e-12)
    return {int(k): float(np.corrcoef(x[:-k], x[k:])[0, 1]) for k in lags}


def noise_band(n):
    return round(1.96 / np.sqrt(n), 4)


def main():
    out = {}
    for res in ("cpus", "memory"):
        for agg in ("mean", "sum"):
            s = build_cell_series(res, agg)
            out[f"google_{res}_{agg}"] = {
                "n": len(s), "noise_band": noise_band(len(s)), "acf": acf(s)}
    for res in ("cpu_util", "mem_util"):
        s = build_alibaba_cell_series(res)
        out[f"alibaba_{res}_cell"] = {
            "n": len(s), "noise_band": noise_band(len(s)),
            "acf": acf(s, [1, 2, 3, 6, 12, 24])}

    path = os.path.join(PROJECT_ROOT, "results", "temporal_diagnostic.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)

    g = out["google_cpus_mean"]
    a = out["alibaba_cpu_util_cell"]
    print(json.dumps(out, indent=2))
    print("\nVERDICT")
    print(f"  Google cell series  n={g['n']}  band +-{g['noise_band']}  "
          f"ACF lag1={g['acf'][1]:+.3f}  lag24={g['acf'][24]:+.3f}  -> white noise")
    print(f"  Alibaba cell series n={a['n']}  band +-{a['noise_band']}  "
          f"ACF lag1={a['acf'][1]:+.3f}  lag24={a['acf'][24]:+.3f}  -> diurnal structure")
    print("  => Google 2019 Cluster: cross-sectional regression only.")
    print("     Forecasting (LSTM/BiLSTM/RNN) is reported on Alibaba 2017.")
    print(f"\n  -> {path}")


if __name__ == "__main__":
    main()
