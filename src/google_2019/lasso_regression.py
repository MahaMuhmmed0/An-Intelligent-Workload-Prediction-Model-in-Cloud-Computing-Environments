"""
src/google_2019/lasso_regression.py
===================================
Lasso (L1-penalised) regression — Google 2019 Cluster, cross-sectional.

INPUT FEATURES : cpu_request, mem_request, cpu_mem_ratio, priority,
                 scheduling_class, collection_type
TARGET(S)      : cpu_usage_mean, mem_usage_mean, cpu_usage_peak, mem_usage_peak
PREPROCESSING  : RobustPrep -> SimpleImputer(median)
                 -> PolynomialFeatures(degree=2) -> StandardScaler     [Pipeline]
TRAIN/TEST     : random 80/20, random_state=42
HYPERPARAMS    : alpha (L1 strength); Lasso(max_iter=3000, tol=1e-3)
GRIDSEARCHCV   : GridSearchCV(cv=5, scoring="neg_mean_absolute_error"),
                 alpha in [1e-4, 1e-3, 1e-2, 0.1], on TRAIN only
TRAINING       : GridSearchCV.fit on train -> best_estimator_
PREDICTION     : best_estimator_.predict on train and test
METRICS        : train & test  MAE, MAPE, RMSE, R2, sMAPE
OUTPUT         : results/regression/google_2019__lasso.csv

Run:  python -m src.google_2019.lasso_regression
"""
from sklearn.linear_model import Lasso

from src.evaluation import run_regression, RANDOM_STATE
from src.google_2019.preprocessing import load_entity_table, FEATURES

DATASET = "Google 2019 Cluster"
MODEL_NAME = "Lasso Regression"
OUT_CSV = "results/regression/google_2019__lasso.csv"

ESTIMATOR = Lasso(max_iter=3000, tol=1e-3, random_state=RANDOM_STATE)
PARAM_GRID = {"model__alpha": [1e-4, 1e-3, 1e-2, 0.1]}
USE_POLY = True
SCALER = "standard"

if __name__ == "__main__":
    run_regression(DATASET, MODEL_NAME, OUT_CSV, load_entity_table, FEATURES,
                   estimator=ESTIMATOR, param_grid=PARAM_GRID,
                   use_poly=USE_POLY, scaler=SCALER)
