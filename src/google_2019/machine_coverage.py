"""
src/google_2019/machine_coverage.py
===================================
Per-machine temporal-coverage statistic for the Google 2019 Cluster extract.

WHY THIS EXISTS
---------------
Section 5 of the paper states that the public extract cannot be turned into a
per-machine utilisation series because each machine is observed in only a handful
of disjoint hours. That claim rested on a one-off calculation that was reported in
`documentation/TEMPORAL_DIAGNOSTIC.md` but had no committed, re-runnable script.
This file is that script. It changes no data, no threshold, and no reported
number; it only makes the already-reported statistic reproducible.

WHAT IT CALCULATES
------------------
From `borg_traces_data.csv` (one row per instance-usage measurement window):

  1. parse `start_time` (microseconds) and keep rows with 1e6 <= start_time <= 3e12
     -- the same validity filter used by src/google_2019/temporal_diagnostic.py;
  2. assign each row to a 1-hour bucket:  hour = start_time // 3_600_000_000
     -- the same BUCKET_US = 3_600_000_000 us used by the temporal diagnostic;
  3. group by `machine_id` and count the number of DISTINCT hour buckets each
     machine appears in  ("distinct hours observed per machine");
  4. summarise that distribution:
        - number of distinct machines
        - observation span in hours and days (max hour bucket - min hour bucket)
        - median / mean / p90 / max distinct-hours-per-machine
        - fraction of machines observed in <= 6 distinct hours
        - number of machines observed in >= 40 distinct hours
          (40 = the minimum series length the forecasting pipeline requires,
           MIN_SERIES_LEN in src/alibaba_2017/preprocessing.py)
        - number of machines observed in >= 144 distinct hours
          (144 = one full Alibaba-style series length)

DETERMINISM
-----------
No sampling, no randomness. The result depends only on the input CSV and is
byte-identical on every run.

OUTPUT
------
results/machine_coverage.json   (a NEW file; nothing existing is touched)

Run:  python -m src.google_2019.machine_coverage
"""
import json
import os

import numpy as np
import pandas as pd

from src.evaluation import PROJECT_ROOT
from src.google_2019.preprocessing import BORG_CSV

BUCKET_US = 3_600_000_000          # 1 hour in microseconds  (matches temporal_diagnostic.py)
FORECAST_MIN_HOURS = 40           # MIN_SERIES_LEN in src/alibaba_2017/preprocessing.py
FULL_SERIES_HOURS = 144          # one full Alibaba-style per-machine series


def compute(path=BORG_CSV):
    df = pd.read_csv(path, usecols=["machine_id", "start_time"], low_memory=False)
    st = pd.to_numeric(df["start_time"], errors="coerce")
    keep = st.between(1e6, 3e12)
    df = df.loc[keep].copy()
    df["hour"] = (st.loc[keep] // BUCKET_US).astype("int64")

    hours_per_machine = df.groupby("machine_id")["hour"].nunique()
    span_hours = int(df["hour"].max() - df["hour"].min())

    return {
        "source": "borg_traces_data.csv (one row per instance-usage window)",
        "bucket_microseconds": BUCKET_US,
        "start_time_filter": "1e6 <= start_time <= 3e12",
        "definition": "distinct 1-hour buckets in which each machine_id appears",
        "n_machines": int(hours_per_machine.size),
        "span_hours": span_hours,
        "span_days": round(span_hours / 24, 1),
        "distinct_hours_per_machine": {
            "median": float(hours_per_machine.median()),
            "mean": round(float(hours_per_machine.mean()), 2),
            "p90": float(hours_per_machine.quantile(0.90)),
            "max": int(hours_per_machine.max()),
        },
        "fraction_machines_le_6_hours": round(float((hours_per_machine <= 6).mean()), 4),
        "n_machines_ge_40_hours": int((hours_per_machine >= FORECAST_MIN_HOURS).sum()),
        "n_machines_ge_144_hours": int((hours_per_machine >= FULL_SERIES_HOURS).sum()),
    }


def main():
    out = compute()
    path = os.path.join(PROJECT_ROOT, "results", "machine_coverage.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)

    print(json.dumps(out, indent=2))
    d = out["distinct_hours_per_machine"]
    print("\nSUMMARY (matches the statistic reported in Section 5 of the paper)")
    print(f"  {out['n_machines']:,} distinct machines over a {out['span_hours']}-hour "
          f"({out['span_days']:.0f}-day) span")
    print(f"  median distinct hours per machine : {d['median']:.0f}")
    print(f"  machines observed in <= 6 hours   : {100 * out['fraction_machines_le_6_hours']:.1f}%")
    print(f"  machines observed in >= 40 hours  : {out['n_machines_ge_40_hours']:,}")
    print(f"  => no per-machine sequence of usable length can be assembled;")
    print(f"     the Google 2019 Cluster extract is used for cross-sectional regression only.")
    print(f"\n  -> {path}")


if __name__ == "__main__":
    main()
