"""
src/alibaba_2017/random_forest.py
=================================
Random Forest regression — Alibaba 2017, cross-sectional.

INPUT FEATURES : cpu_request, mem_request, cpu_mem_ratio, plan_disk, cpuset_width
TARGET(S)      : cpu_usage_mean, mem_usage_mean, cpu_usage_peak, mem_usage_peak
PREPROCESSING  : RobustPrep -> SimpleImputer(median) -> StandardScaler   [Pipeline]
TRAIN/TEST     : random 80/20, random_state=42
HYPERPARAMS    : n_estimators=150, min_samples_leaf=10, max_samples=0.5,
                 n_jobs=-1, random_state=42;  max_depth tuned
GRIDSEARCHCV   : GridSearchCV(cv=5, scoring="neg_mean_absolute_error"),
                 max_depth in [6, 12], on TRAIN only
TRAINING/PRED  : GridSearchCV.fit(train) -> best_estimator_.predict(train/test)
METRICS        : train & test  MAE, MAPE, RMSE, R2, sMAPE
OUTPUT         : results/regression/alibaba_2017__random_forest.csv

Best model on Alibaba too, but test R2 only ~ 0.33 (CPU) / ~ 0.32 (memory):
Alibaba usage is a percentage OF the request, so the request cancels out - this
is the genuine over-provisioning gap, reported as a finding.

Run:  python -m src.alibaba_2017.random_forest
"""
from sklearn.ensemble import RandomForestRegressor

from src.evaluation import run_regression, RANDOM_STATE
from src.alibaba_2017.preprocessing import load_entity_table, FEATURES

DATASET = "Alibaba 2017"
MODEL_NAME = "Random Forest"
OUT_CSV = "results/regression/alibaba_2017__random_forest.csv"

ESTIMATOR = RandomForestRegressor(n_estimators=150, n_jobs=-1,
                                  random_state=RANDOM_STATE,
                                  max_samples=0.5, min_samples_leaf=10)
PARAM_GRID = {"model__max_depth": [6, 12]}
USE_POLY = False
SCALER = "standard"

if __name__ == "__main__":
    run_regression(DATASET, MODEL_NAME, OUT_CSV, load_entity_table, FEATURES,
                   estimator=ESTIMATOR, param_grid=PARAM_GRID,
                   use_poly=USE_POLY, scaler=SCALER)
