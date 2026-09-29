"""
src/alibaba_2017/elastic_net.py
===============================
Elastic Net (L1 + L2) regression — Alibaba 2017, cross-sectional.

INPUT FEATURES : cpu_request, mem_request, cpu_mem_ratio, plan_disk, cpuset_width
TARGET(S)      : cpu_usage_mean, mem_usage_mean, cpu_usage_peak, mem_usage_peak
PREPROCESSING  : RobustPrep -> SimpleImputer(median)
                 -> PolynomialFeatures(degree=2) -> StandardScaler   [Pipeline]
TRAIN/TEST     : random 80/20, random_state=42
HYPERPARAMS    : alpha, l1_ratio; ElasticNet(max_iter=3000, tol=1e-3)
GRIDSEARCHCV   : GridSearchCV(cv=5, scoring="neg_mean_absolute_error"),
                 alpha in [1e-3, 1e-2, 0.1], l1_ratio in [0.3, 0.6], on TRAIN only
TRAINING/PRED  : GridSearchCV.fit(train) -> best_estimator_.predict(train/test)
METRICS        : train & test  MAE, MAPE, RMSE, R2, sMAPE
OUTPUT         : results/regression/alibaba_2017__elastic_net.csv

Run:  python -m src.alibaba_2017.elastic_net
"""
from sklearn.linear_model import ElasticNet

from src.evaluation import run_regression, RANDOM_STATE
from src.alibaba_2017.preprocessing import load_entity_table, FEATURES

DATASET = "Alibaba 2017"
MODEL_NAME = "Elastic Net"
OUT_CSV = "results/regression/alibaba_2017__elastic_net.csv"

ESTIMATOR = ElasticNet(max_iter=3000, tol=1e-3, random_state=RANDOM_STATE)
PARAM_GRID = {"model__alpha": [1e-3, 1e-2, 0.1], "model__l1_ratio": [0.3, 0.6]}
USE_POLY = True
SCALER = "standard"

if __name__ == "__main__":
    run_regression(DATASET, MODEL_NAME, OUT_CSV, load_entity_table, FEATURES,
                   estimator=ESTIMATOR, param_grid=PARAM_GRID,
                   use_poly=USE_POLY, scaler=SCALER)
