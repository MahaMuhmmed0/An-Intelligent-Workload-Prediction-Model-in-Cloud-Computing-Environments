"""
src/alibaba_2017/rnn.py
=======================
Simple (vanilla) RNN forecaster — Alibaba 2017, per-machine server utilisation.
Runs BOTH targets: RNN - CPU  and  RNN - Memory.

Identical data pipeline, compilation, training schedule, prediction and metrics
as src/alibaba_2017/lstm.py (see _forecasting.py). Only the recurrent cell
differs: SimpleRNN instead of LSTM.

ARCHITECTURE: SimpleRNN(64, return_sequences) -> Dropout(0.2)
              -> SimpleRNN(32) -> Dropout(0.2) -> Dense(1)
OUTPUT      : results/sequence/alibaba_2017__rnn.csv   (2 rows)

Representative result: CPU test R2 ~ 0.78, Memory test R2 ~ 0.91.

Run:  python -m src.alibaba_2017.rnn
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, Dropout, Dense

from src.alibaba_2017._forecasting import run_forecaster


def build_model(window):
    return Sequential([
        Input((window, 1)),
        SimpleRNN(64, return_sequences=True),
        Dropout(0.2),
        SimpleRNN(32),
        Dropout(0.2),
        Dense(1),
    ])


if __name__ == "__main__":
    run_forecaster(build_model, "RNN")
