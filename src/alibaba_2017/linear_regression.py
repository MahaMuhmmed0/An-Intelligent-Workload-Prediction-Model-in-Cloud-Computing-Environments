"""
src/alibaba_2017/linear_regression.py
=====================================
Ordinary Least Squares — Alibaba 2017, cross-sectional regression.

INPUT FEATURES : cpu_request, mem_request, cpu_mem_ratio, plan_disk, cpuset_width
TARGET(S)      : cpu_usage_mean, mem_usage_mean, cpu_usage_peak, mem_usage_peak
                 (util as a fraction of the instance's request)
PREPROCESSING  : RobustPrep -> SimpleImputer(median) -> MinMaxScaler   [Pipeline]
TRAIN/TEST     : random 80/20, random_state=42
HYPERPARAMS    : none        GRIDSEARCHCV : not used
TRAINING       : Pipeline.fit(X_train, y_train)
PREDICTION     : Pipeline.predict on train and test
METRICS        : train & test  MAE, MAPE, RMSE, R2, sMAPE
OUTPUT         : results/regression/alibaba_2017__linear.csv

Run:  python -m src.alibaba_2017.linear_regression
"""
from sklearn.linear_model import LinearRegression

from src.evaluation import run_regression
from src.alibaba_2017.preprocessing import load_entity_table, FEATURES

DATASET = "Alibaba 2017"
MODEL_NAME = "Linear Regression"
OUT_CSV = "results/regression/alibaba_2017__linear.csv"

ESTIMATOR = LinearRegression()
PARAM_GRID = {}
USE_POLY = False
SCALER = "minmax"

if __name__ == "__main__":
    run_regression(DATASET, MODEL_NAME, OUT_CSV, load_entity_table, FEATURES,
                   estimator=ESTIMATOR, param_grid=PARAM_GRID,
                   use_poly=USE_POLY, scaler=SCALER)
