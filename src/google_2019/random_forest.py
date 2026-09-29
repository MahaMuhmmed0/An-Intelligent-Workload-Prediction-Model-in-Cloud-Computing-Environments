"""
src/google_2019/random_forest.py
================================
Random Forest regression — Google 2019 Cluster, cross-sectional.

INPUT FEATURES : cpu_request, mem_request, cpu_mem_ratio, priority,
                 scheduling_class, collection_type
TARGET(S)      : cpu_usage_mean, mem_usage_mean, cpu_usage_peak, mem_usage_peak
PREPROCESSING  : RobustPrep -> SimpleImputer(median) -> StandardScaler   [Pipeline]
                 no PolynomialFeatures (the trees model interactions directly)
TRAIN/TEST     : random 80/20, random_state=42
HYPERPARAMS    : n_estimators=150, min_samples_leaf=10, max_samples=0.5,
                 n_jobs=-1, random_state=42;  max_depth tuned
GRIDSEARCHCV   : GridSearchCV(cv=5, scoring="neg_mean_absolute_error"),
                 max_depth in [6, 12], on TRAIN only
TRAINING       : GridSearchCV.fit on train -> best_estimator_
PREDICTION     : best_estimator_.predict on train and test
METRICS        : train & test  MAE, MAPE, RMSE, R2, sMAPE
OUTPUT         : results/regression/google_2019__random_forest.csv

This is the strongest model on Google (test R2 ~ 0.83 CPU, ~ 0.94 memory) because
Google request and usage share the same normalised scale.

Run:  python -m src.google_2019.random_forest
"""
from sklearn.ensemble import RandomForestRegressor

from src.evaluation import run_regression, RANDOM_STATE
from src.google_2019.preprocessing import load_entity_table, FEATURES

DATASET = "Google 2019 Cluster"
MODEL_NAME = "Random Forest"
OUT_CSV = "results/regression/google_2019__random_forest.csv"

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
