"""
src/google_2019/ridge_regression.py
===================================
Ridge (L2-penalised) regression — Google 2019 Cluster, cross-sectional.

INPUT FEATURES : cpu_request, mem_request, cpu_mem_ratio, priority,
                 scheduling_class, collection_type
TARGET(S)      : cpu_usage_mean, mem_usage_mean, cpu_usage_peak, mem_usage_peak
PREPROCESSING  : RobustPrep -> SimpleImputer(median) -> StandardScaler   [Pipeline]
                 no PolynomialFeatures
TRAIN/TEST     : random 80/20, random_state=42
HYPERPARAMS    : alpha  (L2 strength)
GRIDSEARCHCV   : GridSearchCV(cv=5, scoring="neg_mean_absolute_error"),
                 alpha in [0.001, 0.01, 0.1, 1, 10, 100], on TRAIN only
TRAINING       : best_estimator_.fit  (via GridSearchCV.fit on train)
PREDICTION     : best_estimator_.predict on train and test
METRICS        : train & test  MAE, MAPE, RMSE, R2, sMAPE
OUTPUT         : results/regression/google_2019__ridge.csv

Run:  python -m src.google_2019.ridge_regression
"""
from sklearn.linear_model import Ridge

from src.evaluation import run_regression, RANDOM_STATE
from src.google_2019.preprocessing import load_entity_table, FEATURES

DATASET = "Google 2019 Cluster"
MODEL_NAME = "Ridge Regression"
OUT_CSV = "results/regression/google_2019__ridge.csv"

ESTIMATOR = Ridge(random_state=RANDOM_STATE)
PARAM_GRID = {"model__alpha": [0.001, 0.01, 0.1, 1, 10, 100]}
USE_POLY = False
SCALER = "standard"

if __name__ == "__main__":
    run_regression(DATASET, MODEL_NAME, OUT_CSV, load_entity_table, FEATURES,
                   estimator=ESTIMATOR, param_grid=PARAM_GRID,
                   use_poly=USE_POLY, scaler=SCALER)
