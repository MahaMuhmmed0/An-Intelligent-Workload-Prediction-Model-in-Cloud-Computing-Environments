"""
src/google_2019/linear_regression.py
====================================
Ordinary Least Squares — Google 2019 Cluster, cross-sectional regression.

INPUT FEATURES : cpu_request, mem_request, cpu_mem_ratio, priority,
                 scheduling_class, collection_type          (request + metadata only)
TARGET(S)      : cpu_usage_mean, mem_usage_mean, cpu_usage_peak, mem_usage_peak
                 (run for all four; the peak runs reuse this exact code with the
                 peak target)
PREPROCESSING  : RobustPrep (log1p + winsorise to TRAIN 1st/99th pct)
                 -> SimpleImputer(median) -> MinMaxScaler        [inside a Pipeline]
                 no PolynomialFeatures for plain OLS
TRAIN/TEST     : random 80/20, random_state=42  (cross-sectional task)
HYPERPARAMS    : none (OLS has none)
GRIDSEARCHCV   : not used
TRAINING       : Pipeline.fit(X_train, y_train)
PREDICTION     : Pipeline.predict on train and test
METRICS        : train & test  MAE, MAPE, RMSE, R2, sMAPE   (MAPE unreliable ~0)
OUTPUT         : results/regression/google_2019__linear.csv   (4 rows)

Run:  python -m src.google_2019.linear_regression
"""
from sklearn.linear_model import LinearRegression

from src.evaluation import run_regression
from src.google_2019.preprocessing import load_entity_table, FEATURES

DATASET = "Google 2019 Cluster"
MODEL_NAME = "Linear Regression"
OUT_CSV = "results/regression/google_2019__linear.csv"

ESTIMATOR = LinearRegression()
PARAM_GRID = {}
USE_POLY = False
SCALER = "minmax"

if __name__ == "__main__":
    run_regression(DATASET, MODEL_NAME, OUT_CSV, load_entity_table, FEATURES,
                   estimator=ESTIMATOR, param_grid=PARAM_GRID,
                   use_poly=USE_POLY, scaler=SCALER)
