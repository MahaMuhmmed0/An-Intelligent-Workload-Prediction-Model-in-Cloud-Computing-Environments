"""
src/alibaba_2017/lstm.py
========================
LSTM forecaster — Alibaba 2017, per-machine server utilisation.
Runs BOTH targets: LSTM - CPU  and  LSTM - Memory.

TASK        : predict the next 5-minute interval from the previous 24 intervals
INPUT       : one scalar per step (machine utilisation, scaled to [0,1])
DATA PATH   : server_usage.csv -> per machine -> chronological order ->
              per-machine chronological 80/20 split -> MinMaxScaler fit on TRAIN
              points only -> previous-24 windows            (preprocessing.py)
ARCHITECTURE: LSTM(64, return_sequences) -> Dropout(0.2)
              -> LSTM(32) -> Dropout(0.2) -> Dense(1)
COMPILE     : Adam(1e-3), Huber(delta=1.0)                  (_forecasting.py)
TRAIN       : 30 epochs, batch 128, validation_split=0.1,
              EarlyStopping(patience=8, restore_best_weights),
              ReduceLROnPlateau(factor=0.5, patience=4), shuffle=False
PREDICT     : model.predict(train) and model.predict(test), inverse-scaled
METRICS     : train & test  MAE, MAPE, RMSE, R2, sMAPE
OUTPUT      : results/sequence/alibaba_2017__lstm.csv   (2 rows: cpu_util, mem_util)

Representative result: CPU test R2 ~ 0.73-0.80, Memory test R2 ~ 0.91-0.93.

Run:  python -m src.alibaba_2017.lstm
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense

from src.alibaba_2017._forecasting import run_forecaster


def build_model(window):
    return Sequential([
        Input((window, 1)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1),
    ])


if __name__ == "__main__":
    run_forecaster(build_model, "LSTM")
