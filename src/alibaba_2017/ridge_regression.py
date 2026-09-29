"""
src/alibaba_2017/ridge_regression.py
====================================
Ridge (L2) regression — Alibaba 2017, cross-sectional.

INPUT FEATURES : cpu_request, mem_request, cpu_mem_ratio, plan_disk, cpuset_width
TARGET(S)      : cpu_usage_mean, mem_usage_mean, cpu_usage_peak, mem_usage_peak
PREPROCESSING  : RobustPrep -> SimpleImputer(median) -> StandardScaler   [Pipeline]
TRAIN/TEST     : random 80/20, random_state=42
HYPERPARAMS    : alpha
GRIDSEARCHCV   : GridSearchCV(cv=5, scoring="neg_mean_absolute_error"),
                 alpha in [0.001, 0.01, 0.1, 1, 10, 100], on TRAIN only
TRAINING/PRED  : GridSearchCV.fit(train) -> best_estimator_.predict(train/test)
METRICS        : train & test  MAE, MAPE, RMSE, R2, sMAPE
OUTPUT         : results/regression/alibaba_2017__ridge.csv

Run:  python -m src.alibaba_2017.ridge_regression
"""
from sklearn.linear_model import Ridge

from src.evaluation import run_regression, RANDOM_STATE
from src.alibaba_2017.preprocessing import load_entity_table, FEATURES

DATASET = "Alibaba 2017"
MODEL_NAME = "Ridge Regression"
OUT_CSV = "results/regression/alibaba_2017__ridge.csv"

ESTIMATOR = Ridge(random_state=RANDOM_STATE)
PARAM_GRID = {"model__alpha": [0.001, 0.01, 0.1, 1, 10, 100]}
USE_POLY = False
SCALER = "standard"

if __name__ == "__main__":
    run_regression(DATASET, MODEL_NAME, OUT_CSV, load_entity_table, FEATURES,
                   estimator=ESTIMATOR, param_grid=PARAM_GRID,
                   use_poly=USE_POLY, scaler=SCALER)
