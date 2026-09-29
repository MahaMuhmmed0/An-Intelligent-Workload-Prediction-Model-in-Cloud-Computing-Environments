"""
src/alibaba_2017/bilstm.py
==========================
Bidirectional-LSTM forecaster — Alibaba 2017, per-machine server utilisation.
Runs BOTH targets: BiLSTM - CPU  and  BiLSTM - Memory.

Identical data pipeline, compilation, training schedule, prediction and metrics
as src/alibaba_2017/lstm.py (see _forecasting.py). Only the architecture differs:
the two recurrent layers are wrapped in Bidirectional().

ARCHITECTURE: Bidirectional(LSTM(64, return_sequences)) -> Dropout(0.2)
              -> Bidirectional(LSTM(32)) -> Dropout(0.2) -> Dense(1)
OUTPUT      : results/sequence/alibaba_2017__bilstm.csv   (2 rows)

Representative result: CPU test R2 ~ 0.76, Memory test R2 ~ 0.93.

Run:  python -m src.alibaba_2017.bilstm
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dropout, Dense

from src.alibaba_2017._forecasting import run_forecaster


def build_model(window):
    return Sequential([
        Input((window, 1)),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(1),
    ])


if __name__ == "__main__":
    run_forecaster(build_model, "BiLSTM")
