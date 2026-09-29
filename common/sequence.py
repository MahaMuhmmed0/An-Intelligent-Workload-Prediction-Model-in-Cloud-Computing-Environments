"""
Shared leakage-free SEQUENCE pipeline (LSTM / BiLSTM / RNN) for both datasets.

Key corrections vs the original notebooks:
  * A GENUINE time series is built per machine (ordered by time), instead of feeding
    a shuffled bag of independent instance records into create_sequences().
  * Each machine's series is split chronologically (first 80% -> train windows,
    last 20% -> test windows). No window ever straddles the boundary.
  * The scaler is fit on TRAIN windows only, then applied to test.
  * Validation data = last 10% of the training windows (never the test set).
  * Metrics (MAE, RMSE, R2, guarded sMAPE) are computed on inverse-scaled values,
    for TRAIN and TEST.

Datasets:
  Google 2019 Cluster  -> per-machine sequence of average_usage['cpus'] / ['memory'],
                ordered by start_time.
  Alibaba 2017 -> per-machine server_usage cpu_util% / mem_util%, ordered by ts.
"""
import ast
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score

EPS = 1e-9
WINDOW = 24
HORIZON = 1
SEED = 42


# --------------------------- series construction --------------------------- #

BORG_BUCKET_US = 3_600_000_000          # 1 hour, in microseconds


def borg_series(path, resource="cpus", nrows=None, min_len=40, agg="mean"):
    """Cell-level aggregate utilization time series.

    The public Google 2019 Cluster (Borg v3) extract samples *instances* sparsely (median ~4 distinct
    hours per machine), so a per-machine series is not viable. Instead we bucket
    every instance-usage window by hour and aggregate `average_usage[resource]`
    across all instances active in that hour -> one continuous ~31-day series.
    `agg='mean'` = typical per-instance intensity; `agg='sum'` = total offered load.
    """
    df = pd.read_csv(path, nrows=nrows, low_memory=False,
                     usecols=["start_time", "average_usage"])
    st = pd.to_numeric(df["start_time"], errors="coerce")
    df = df[st.between(1e6, 3e12)].copy()
    df["bucket"] = (st[st.between(1e6, 3e12)] // BORG_BUCKET_US).astype("int64")
    df["val"] = df["average_usage"].map(
        lambda s: (ast.literal_eval(s).get(resource, np.nan)
                   if isinstance(s, str) else np.nan))
    df = df.dropna(subset=["val"])
    df = df[df["val"].between(0, 5)]
    s = df.groupby("bucket")["val"].mean() if agg == "mean" else df.groupby("bucket")["val"].sum()
    # reindex onto a regular hourly grid; interpolate the few missing buckets
    full = pd.RangeIndex(s.index.min(), s.index.max() + 1)
    s = s.reindex(full).interpolate("linear").bfill().ffill()
    arr = s.to_numpy(float)
    return [arr] if len(arr) >= min_len else []


def alibaba_series(folder, resource="cpu", min_len=40):
    col = {"cpu": "cpu_util", "mem": "mem_util"}[resource]
    su = pd.read_csv(os.path.join(folder, "server_usage.csv"), header=None,
                     names=["ts", "machine_id", "cpu_util", "mem_util", "disk_util",
                            "load1", "load5", "load15"])
    su = su.dropna(subset=[col])
    series = []
    for _, g in su.sort_values(["machine_id", "ts"]).groupby("machine_id"):
        v = (g[col].to_numpy(float)) / 100.0
        if len(v) >= min_len:
            series.append(v)
    return series


# --------------------------- windowing + scaling --------------------------- #

def _windows(arr, w, h):
    X, y = [], []
    for i in range(len(arr) - w - h + 1):
        X.append(arr[i:i + w])
        y.append(arr[i + w + h - 1])
    return np.array(X), np.array(y)


def build_xy(series, w=WINDOW, h=HORIZON, split=0.8):
    """Chronological per-series split, then scale using TRAIN statistics only."""
    tr_raw, te_raw = [], []
    for s in series:
        k = int(len(s) * split)
        if k <= w + h or len(s) - k <= h:
            continue
        tr_raw.append(s[:k])
        te_raw.append(s[max(0, k - w - h + 1):])   # keep context for first test window
    scaler = MinMaxScaler().fit(np.concatenate(tr_raw).reshape(-1, 1))

    def stack(raws):
        Xs, ys = [], []
        for r in raws:
            r2 = scaler.transform(r.reshape(-1, 1)).ravel()
            X, y = _windows(r2, w, h)
            if len(X):
                Xs.append(X); ys.append(y)
        X = np.concatenate(Xs)[..., None]
        y = np.concatenate(ys)
        return X, y

    Xtr, ytr = stack(tr_raw)
    Xte, yte = stack(te_raw)
    return Xtr, ytr, Xte, yte, scaler


# --------------------------- models --------------------------- #

def _make(kind, w):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, SimpleRNN, Bidirectional, Dropout, Dense, Input
    tf.random.set_seed(SEED)
    m = Sequential([Input((w, 1))])
    if kind == "LSTM":
        m.add(LSTM(64, return_sequences=True)); m.add(Dropout(0.2))
        m.add(LSTM(32)); m.add(Dropout(0.2))
    elif kind == "BiLSTM":
        m.add(Bidirectional(LSTM(64, return_sequences=True))); m.add(Dropout(0.2))
        m.add(Bidirectional(LSTM(32))); m.add(Dropout(0.2))
    elif kind == "RNN":
        m.add(SimpleRNN(64, return_sequences=True)); m.add(Dropout(0.2))
        m.add(SimpleRNN(32)); m.add(Dropout(0.2))
    m.add(Dense(1))
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import Huber
    m.compile(optimizer=Adam(1e-3), loss=Huber(1.0))
    return m


def _scores(y, p, scaler):
    y = scaler.inverse_transform(np.asarray(y).reshape(-1, 1)).ravel()
    p = scaler.inverse_transform(np.asarray(p).reshape(-1, 1)).ravel()
    sm = float(np.mean(2 * np.abs(p - y) / (np.abs(y) + np.abs(p) + EPS)) * 100)
    return {"MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
            "R2": float(r2_score(y, p)), "sMAPE": sm}


def run_sequence(series, kinds=("LSTM", "BiLSTM", "RNN"), w=WINDOW, epochs=60):
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    Xtr, ytr, Xte, yte, scaler = build_xy(series, w=w)
    out = []
    for kind in kinds:
        model = _make(kind, w)
        cb = [EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
              ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)]
        model.fit(Xtr, ytr, validation_split=0.1, epochs=epochs, batch_size=128,
                  shuffle=False, verbose=0, callbacks=cb)
        out.append({"model": kind,
                    "train": _scores(ytr, model.predict(Xtr, verbose=0), scaler),
                    "test": _scores(yte, model.predict(Xte, verbose=0), scaler),
                    "n_train": int(len(Xtr)), "n_test": int(len(Xte))})
    return out
