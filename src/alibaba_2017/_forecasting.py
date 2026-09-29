"""
src/alibaba_2017/_forecasting.py
================================
Shared training / evaluation loop for the three Alibaba forecasters
(lstm.py, bilstm.py, rnn.py). Each of those files only defines `build_model()`
(the architecture) and calls `run_forecaster(build_model, "LSTM")`.

Pipeline steps and where they live
----------------------------------
load server_usage ........... preprocessing.load_server_usage_series
select machine .............. preprocessing (groupby machine_id)
chronological ordering ...... preprocessing (sort_values ["machine_id","ts"])
previous-24 input windows ... evaluation.make_windows (w=24, h=1)
train/test split ............ preprocessing.build_forecasting_windows (per-machine 80/20)
scaler fit on TRAIN only .... preprocessing.build_forecasting_windows (MinMaxScaler on train pts)
transform test .............. preprocessing.build_forecasting_windows
model architecture ......... the calling file's build_model()
compilation ................ here: Adam(1e-3), Huber(delta=1.0)
training ................... here: 30 epochs, batch 128, validation_split=0.1,
                             EarlyStopping(patience=8), ReduceLROnPlateau, shuffle=False
prediction ................. here: model.predict(train), model.predict(test)
MAE / RMSE / R2 / sMAPE ..... evaluation.sequence_scores (on inverse-scaled values)

NOTE: TensorFlow CPU training is not bit-for-bit deterministic even with a fixed
seed; R2 varies by ~+-0.05 between runs. The saved CSV is one representative run.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.evaluation import (WINDOW, SEQ_SEED, sequence_scores, save_sequence_result)
from src.alibaba_2017.preprocessing import build_forecasting_windows

EPOCHS = 30
BATCH = 128


def run_forecaster(build_model, model_name):
    rows = []
    for resource, target_label in [("cpu", "cpu_util"), ("mem", "mem_util")]:
        tf.random.set_seed(SEQ_SEED)
        np.random.seed(SEQ_SEED)

        Xtr, ytr, Xte, yte, scaler = build_forecasting_windows(resource, w=WINDOW)

        model = build_model(WINDOW)
        model.compile(optimizer=Adam(1e-3), loss=Huber(delta=1.0))
        model.fit(
            Xtr, ytr,
            validation_split=0.1, epochs=EPOCHS, batch_size=BATCH,
            shuffle=False, verbose=0,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
            ],
        )
        tr = sequence_scores(ytr, model.predict(Xtr, verbose=0), scaler)
        te = sequence_scores(yte, model.predict(Xte, verbose=0), scaler)

        print(f"\n{model_name}  Alibaba 2017  {target_label}")
        print(f"  train  MAE {tr['MAE']:.4f}  RMSE {tr['RMSE']:.4f}  R2 {tr['R2']:.4f}  sMAPE {tr['sMAPE']:.1f}")
        print(f"  test   MAE {te['MAE']:.4f}  RMSE {te['RMSE']:.4f}  R2 {te['R2']:.4f}  sMAPE {te['sMAPE']:.1f}")

        rows.append({
            "dataset": "Alibaba 2017", "module": "forecasting", "target": target_label,
            "model": model_name, "window": WINDOW, "horizon": 1,
            "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
            "train_MAE": tr["MAE"], "train_MAPE": tr["MAPE"], "train_RMSE": tr["RMSE"],
            "train_R2": tr["R2"], "train_sMAPE": tr["sMAPE"],
            "test_MAE": te["MAE"], "test_MAPE": te["MAPE"], "test_RMSE": te["RMSE"],
            "test_R2": te["R2"], "test_sMAPE": te["sMAPE"],
        })

    out = f"results/sequence/alibaba_2017__{model_name.lower()}.csv"
    save_sequence_result(rows, out)
    return rows
