"""Evidence that the public borg_traces_data.csv extract has no temporal structure,
so sequential models (LSTM/BiLSTM/RNN) are not applicable to it.
Compares against Alibaba server_usage, which does have structure."""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common.sequence import borg_series, alibaba_series
from src.evaluation import GOOGLE_2019_CSV, ALIBABA_2017_DIR

# Configurable via the GOOGLE_2019_CSV / ALIBABA_2017_DIR environment
# variables; see src/evaluation.py and README.md "Dataset locations".
BORG_CSV = GOOGLE_2019_CSV
ALI_DIR = ALIBABA_2017_DIR
LAGS = [1, 2, 3, 6, 12, 24, 48]


def acf(x, lags):
    x = (np.asarray(x, float) - np.mean(x)) / (np.std(x) + 1e-12)
    return {int(k): float(np.corrcoef(x[:-k], x[k:])[0, 1]) for k in lags}


def main():
    out = {}
    for res in ("cpus", "memory"):
        s = borg_series(BORG_CSV, res, agg="mean")[0]
        out[f"borg_{res}"] = {"n": len(s), "noise_band": round(1.96 / np.sqrt(len(s)), 4),
                              "acf": acf(s, LAGS)}
    # Alibaba: mean across machines per 5-min timestamp -> cell-level series
    for res in ("cpu", "mem"):
        series = alibaba_series(ALI_DIR, res, min_len=1)
        L = min(len(a) for a in series)
        cell = np.mean(np.vstack([a[:L] for a in series]), axis=0)
        out[f"alibaba_{res}_cell"] = {"n": int(L), "noise_band": round(1.96 / np.sqrt(L), 4),
                                      "acf": acf(cell, [1, 2, 3, 6, 12, 24])}
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "temporal_diagnostic.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
