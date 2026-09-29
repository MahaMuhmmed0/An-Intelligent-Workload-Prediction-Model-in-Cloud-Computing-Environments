"""
Run the shared leakage-free tabular pipeline on BOTH datasets and both targets.

Usage:
    python run_tabular.py                # both datasets, mean targets
    python run_tabular.py --peak         # also run peak-usage targets
    python run_tabular.py --borg-nrows 300000
"""
import argparse
import json
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common.pipeline import (load_borg, load_alibaba, run_tabular,
                             BORG_META, ALI_META)
from src.evaluation import GOOGLE_2019_CSV, ALIBABA_2017_DIR

# Configurable via the GOOGLE_2019_CSV / ALIBABA_2017_DIR environment
# variables; see src/evaluation.py and README.md "Dataset locations".
BORG_CSV = GOOGLE_2019_CSV
ALI_DIR  = ALIBABA_2017_DIR
SPLIT = "random"
OUT_DIR  = os.path.join(os.path.dirname(__file__), "results")


def feats(meta):
    return ["cpu_request", "mem_request", "cpu_mem_ratio"] + meta


def do(name, df, meta, targets):
    fcols = feats(meta)
    all_rows = []
    for tgt in targets:
        rows, (ntr, nte) = run_tabular(df, meta, tgt, fcols, split=SPLIT)
        for r in rows:
            r["dataset"], r["target"] = name, tgt
            r["n_train"], r["n_test"] = ntr, nte
        all_rows += rows
        print(f"\n=== {name}  target={tgt}   (train={ntr}, test={nte}) ===")
        print(f"{'model':<20} {'tr_MAE':>9} {'tr_R2':>8} {'te_MAE':>9} {'te_R2':>8} {'te_RMSE':>9} {'te_sMAPE':>9}")
        for r in rows:
            print(f"{r['model']:<20} {r['train']['MAE']:>9.4f} {r['train']['R2']:>8.4f} "
                  f"{r['test']['MAE']:>9.4f} {r['test']['R2']:>8.4f} "
                  f"{r['test']['RMSE']:>9.4f} {r['test']['sMAPE']:>9.1f}")
    return all_rows



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
    ap.add_argument("--peak", action="store_true")
    ap.add_argument("--borg-nrows", type=int, default=None)
    ap.add_argument("--borg-sample", type=int, default=None)
    ap.add_argument("--only", choices=["borg", "alibaba"], default=None)
    ap.add_argument("--split", choices=["random", "chrono"], default="random")
    a = ap.parse_args()
    global SPLIT
    SPLIT = a.split

    targets = ["cpu_usage_mean", "mem_usage_mean"]
    if a.peak:
        targets += ["cpu_usage_peak", "mem_usage_peak"]

    os.makedirs(OUT_DIR, exist_ok=True)
    results = []

    if a.only in (None, "alibaba"):
        adf, ameta = load_alibaba(ALI_DIR)
        print(f"[alibaba] entity table: {adf.shape}")
        results += do("Alibaba 2017", adf, ameta, targets)

    if a.only in (None, "borg"):
        bdf, bmeta = load_borg(BORG_CSV, nrows=a.borg_nrows, sample=a.borg_sample)
        print(f"[google] entity table: {bdf.shape}")
        results += do("Google 2019 Cluster", bdf, bmeta, targets)

    with open(os.path.join(OUT_DIR, "tabular_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    flat = [{
        "dataset": r["dataset"], "target": r["target"], "model": r["model"],
        "n_train": r["n_train"], "n_test": r["n_test"],
        "train_MAE": r["train"]["MAE"], "train_RMSE": r["train"]["RMSE"],
        "train_R2": r["train"]["R2"], "train_sMAPE": r["train"]["sMAPE"],
        "test_MAE": r["test"]["MAE"], "test_RMSE": r["test"]["RMSE"],
        "test_R2": r["test"]["R2"], "test_sMAPE": r["test"]["sMAPE"],
        "params": json.dumps(r["params"]),
    } for r in results]
    pd.DataFrame(flat).to_csv(os.path.join(OUT_DIR, "tabular_results.csv"), index=False)
    print(f"\nSaved -> {OUT_DIR}\\tabular_results.csv / .json")


if __name__ == "__main__":
    main()
