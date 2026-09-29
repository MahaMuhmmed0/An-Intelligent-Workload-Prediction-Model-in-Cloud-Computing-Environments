"""
src/google_2019/preprocessing.py
================================
Builds the canonical *entity table* for the Google 2019 Cluster trace
(Borg v3 public extract, ``borg_traces_data.csv``).

One row per instance-usage record.

FEATURES (model inputs)  -- resource request + static scheduling metadata only
    cpu_request        resource_request['cpus']            (NCU, normalised [0,1])
    mem_request        resource_request['memory']          (normalised bytes [0,1])
    cpu_mem_ratio      cpu_request / mem_request            (from REQUESTS -> no leak)
    priority           scheduling priority
    scheduling_class   0..3  latency sensitivity
    collection_type    0 = job, 1 = alloc set

TARGETS (never used as features, never used to derive a feature)
    cpu_usage_mean     average_usage['cpus']
    mem_usage_mean     average_usage['memory']
    cpu_usage_peak     maximum_usage['cpus']
    mem_usage_peak     maximum_usage['memory']

order_key = start_time  (kept only for the chronological sensitivity split; the
headline results use the random split).

This is the exact logic (``load_borg``) that produced results/tabular_results.csv.
The 150 000-row sample + random_state=42 make it reproducible; the filtered table
has ~145 700 rows (train 116 586 / test 29 147).
"""
import ast
import os

import numpy as np
import pandas as pd

from src.evaluation import RANDOM_STATE, EPS, GOOGLE_2019_CSV

# Configurable via the GOOGLE_2019_CSV environment variable; see
# src/evaluation.py and README.md "Dataset locations".
BORG_CSV = GOOGLE_2019_CSV

FEATURES = ["cpu_request", "mem_request", "cpu_mem_ratio",
            "priority", "scheduling_class", "collection_type"]
META = ["priority", "scheduling_class", "collection_type"]
SAMPLE_ROWS = 150_000          # rows read from the CSV before filtering (reproducibility)


def _dget(s, key):
    try:
        return ast.literal_eval(s).get(key, np.nan)
    except Exception:
        return np.nan


_ENTITY_CACHE = {}


def _load_entity_table_impl(path=BORG_CSV, sample=SAMPLE_ROWS):
    df = pd.read_csv(path, low_memory=False)
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=RANDOM_STATE)

    for src_col, pfx in [("resource_request", "req"),
                         ("average_usage", "avg"),
                         ("maximum_usage", "max")]:
        df[f"{pfx}_cpu"] = df[src_col].map(lambda s: _dget(s, "cpus"))
        df[f"{pfx}_mem"] = df[src_col].map(lambda s: _dget(s, "memory"))

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

    # zero-request rows dropped: cpu_mem_ratio is undefined and the request->usage
    # mapping has no meaning (the spec notes request=0 is legitimate for burst tasks)
    out = out[(out["cpu_request"] > 0) & (out["mem_request"] > 0)]
    # normalised usage lies in [0,1]; anything far outside is a parsing artdefact
    for c in ["cpu_usage_mean", "mem_usage_mean", "cpu_usage_peak", "mem_usage_peak"]:
        out = out[out[c].between(0, 5)]

    out = out.dropna(subset=["order_key"]).sort_values("order_key").reset_index(drop=True)
    return out, FEATURES, META




def load_entity_table(*args, **kwargs):
    key = (args, tuple(sorted(kwargs.items())))
    if key not in _ENTITY_CACHE:
        _ENTITY_CACHE[key] = _load_entity_table_impl(*args, **kwargs)
    df, feats, meta = _ENTITY_CACHE[key]
    return df.copy(), list(feats), list(meta)
