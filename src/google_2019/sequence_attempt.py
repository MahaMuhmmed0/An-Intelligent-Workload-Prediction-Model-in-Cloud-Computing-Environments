"""
src/google_2019/sequence_attempt.py
===================================
!!! NOT A FINAL EXPERIMENT — NEGATIVE RESULT ONLY !!!

This runs LSTM / BiLSTM / RNN on the Google 2019 Cluster hourly cell-level series
(the correct construction: chronological split, scaler on train windows only,
validation = tail of train). It is included ONLY to document that the models
cannot learn on this data — every configuration scores R2 ~ 0 (to slightly
negative).

The reason is in temporal_diagnostic.py: the public borg_traces_data.csv extract
is a random cross-sectional sample with no temporal autocorrelation, so there is
no signal for a recurrent model to fit. The final forecasting module of the study
uses Alibaba 2017 only (src/alibaba_2017/{lstm,bilstm,rnn}.py).

INPUT       : temporal_diagnostic.build_cell_series("cpus"/"memory", agg="mean")
              -> single ~745-point hourly series
SPLIT       : chronological 80/20 on that one series
SCALER      : MinMaxScaler fit on the training segment only
WINDOWS     : previous 24 hours -> next hour
ARCHITECTURE: same as the Alibaba forecasters (64 -> 32 recurrent, Dropout 0.2)
TRAIN       : 80 epochs, batch 64, validation_split=0.1, EarlyStopping,
              ReduceLROnPlateau, shuffle=False
OUTPUT      : results/sequence/google_2019__attempt.csv   (flagged not_used=True)

Run:  python -m src.google_2019.sequence_attempt
"""
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, SimpleRNN, Bidirectional, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.evaluation import WINDOW, HORIZON, SEQ_SEED, make_windows, sequence_scores, save_sequence_result
from src.google_2019.temporal_diagnostic import build_cell_series


def _build(kind, w):
    net = {"LSTM": LSTM, "RNN": SimpleRNN}
    m = Sequential([Input((w, 1))])
    if kind == "BiLSTM":
        m.add(Bidirectional(LSTM(64, return_sequences=True))); m.add(Dropout(0.2))
        m.add(Bidirectional(LSTM(32))); m.add(Dropout(0.2))
    else:
        cell = net[kind]
        m.add(cell(64, return_sequences=True)); m.add(Dropout(0.2))
        m.add(cell(32)); m.add(Dropout(0.2))
    m.add(Dense(1))
    return m


def main():
    rows = []
    for resource in ("cpus", "memory"):
        series = build_cell_series(resource, agg="mean")
        k = int(len(series) * 0.8)
        tr_raw, te_raw = series[:k], series[max(0, k - WINDOW - HORIZON + 1):]
        scaler = MinMaxScaler().fit(tr_raw.reshape(-1, 1))
        Xtr, ytr = make_windows(scaler.transform(tr_raw.reshape(-1, 1)).ravel())
        Xte, yte = make_windows(scaler.transform(te_raw.reshape(-1, 1)).ravel())
        Xtr, Xte = Xtr[..., None], Xte[..., None]

        for kind in ("LSTM", "BiLSTM", "RNN"):
            tf.random.set_seed(SEQ_SEED); np.random.seed(SEQ_SEED)
            model = _build(kind, WINDOW)
            model.compile(optimizer=Adam(1e-3), loss=Huber(1.0))
            model.fit(Xtr, ytr, validation_split=0.1, epochs=80, batch_size=64,
                      shuffle=False, verbose=0,
                      callbacks=[EarlyStopping(monitor="val_loss", patience=10,
                                               restore_best_weights=True),
                                 ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                   patience=5, min_lr=1e-6)])
            te = sequence_scores(yte, model.predict(Xte, verbose=0), scaler)
            tr = sequence_scores(ytr, model.predict(Xtr, verbose=0), scaler)
            print(f"{kind:7s} Google {resource:6s}  test R2={te['R2']:+.4f}  MAE={te['MAE']:.4f}  (NOT USED)")
            rows.append({
                "dataset": "Google 2019 Cluster", "module": "forecasting_attempt",
                "target": f"{resource}_hourly_cell", "model": kind, "not_used": True,
                "train_R2": tr["R2"], "test_R2": te["R2"],
                "test_MAE": te["MAE"], "test_RMSE": te["RMSE"], "test_sMAPE": te["sMAPE"],
            })
    save_sequence_result(rows, "results/sequence/google_2019__attempt.csv")
    print("\nAll R2 ~ 0  ->  Google 2019 Cluster is NOT used for the forecasting module.")


if __name__ == "__main__":
    main()
