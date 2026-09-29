"""
Run the shared leakage-free sequence pipeline (LSTM / BiLSTM / RNN) on both datasets.

Usage:
    python run_sequence.py                     # cpu + mem, both datasets
    python run_sequence.py --only alibaba
    python run_sequence.py --borg-nrows 300000
"""
import argparse
import json
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common.sequence import borg_series, alibaba_series, run_sequence
from src.evaluation import GOOGLE_2019_CSV, ALIBABA_2017_DIR

# Configurable via the GOOGLE_2019_CSV / ALIBABA_2017_DIR environment
# variables; see src/evaluation.py and README.md "Dataset locations".
BORG_CSV = GOOGLE_2019_CSV
ALI_DIR = ALIBABA_2017_DIR
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")


def report(name, target, rows):
    print(f"\n=== {name}  target={target} ===")
    print(f"{'model':<8} {'tr_MAE':>9} {'tr_R2':>8} {'te_MAE':>9} {'te_R2':>8} {'te_RMSE':>9} {'te_sMAPE':>9}")
    for r in rows:
        print(f"{r['model']:<8} {r['train']['MAE']:>9.4f} {r['train']['R2']:>8.4f} "
              f"{r['test']['MAE']:>9.4f} {r['test']['R2']:>8.4f} "
              f"{r['test']['RMSE']:>9.4f} {r['test']['sMAPE']:>9.1f}")



def _save_csv(df, path):
    import time
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        alt = path.replace(".csv", f"_{int(time.time())}.csv")
        df.to_csv(alt, index=False)
        print(f"  (primary CSV locked; wrote {alt})")
        return alt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["borg", "alibaba"], default=None)
    ap.add_argument("--borg-nrows", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    results = []

    jobs = []
    if a.only in (None, "alibaba"):
        jobs += [("Alibaba 2017", "cpu_util", lambda: alibaba_series(ALI_DIR, "cpu")),
                 ("Alibaba 2017", "mem_util", lambda: alibaba_series(ALI_DIR, "mem"))]
    if a.only in (None, "borg"):
        jobs += [("Google 2019 Cluster", "cpu_usage", lambda: borg_series(BORG_CSV, "cpus", a.borg_nrows)),
                 ("Google 2019 Cluster", "mem_usage", lambda: borg_series(BORG_CSV, "memory", a.borg_nrows))]

    for name, target, mk in jobs:
        series = mk()
        print(f"[{name}/{target}] {len(series)} machine series")
        rows = run_sequence(series, epochs=a.epochs)
        for r in rows:
            r["dataset"], r["target"] = name, target
        report(name, target, rows)
        results += rows

    with open(os.path.join(OUT_DIR, "sequence_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    flat = [{
        "dataset": r["dataset"], "target": r["target"], "model": r["model"],
        "n_train": r["n_train"], "n_test": r["n_test"],
        "train_MAE": r["train"]["MAE"], "train_RMSE": r["train"]["RMSE"],
        "train_R2": r["train"]["R2"], "train_sMAPE": r["train"]["sMAPE"],
        "test_MAE": r["test"]["MAE"], "test_RMSE": r["test"]["RMSE"],
        "test_R2": r["test"]["R2"], "test_sMAPE": r["test"]["sMAPE"],
    } for r in results]
    pd.DataFrame(flat).to_csv(os.path.join(OUT_DIR, "sequence_results.csv"), index=False)
    print(f"\nSaved -> {OUT_DIR}\\sequence_results.csv / .json")


if __name__ == "__main__":
    main()
