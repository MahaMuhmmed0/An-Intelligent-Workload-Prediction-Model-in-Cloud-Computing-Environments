"""
src/google_2019/hist_gradient_boosting.py
=========================================
Histogram-based Gradient Boosted Decision Trees (GBDT) regression
— Google 2019 Cluster, cross-sectional.

ADDITIVE EXPERIMENT.  This module is NOT part of the frozen 48-experiment set
behind results/tabular_results.csv.  It was added to test whether the
non-linear-ensemble behaviour observed with Random Forest is reproduced by an
independent boosting ensemble.  Its rows are written to
results/regression/google_2019__hist_gradient_boosting.csv and are aggregated by
`python -m src.run_all histgbr` into results/tabular_results_histgbr.csv only —
never into results/tabular_results.csv.

INPUT FEATURES : cpu_request, mem_request, cpu_mem_ratio, priority,
                 scheduling_class, collection_type
TARGET(S)      : cpu_usage_mean, mem_usage_mean, cpu_usage_peak, mem_usage_peak
PREPROCESSING  : RobustPrep -> SimpleImputer(median) -> StandardScaler   [Pipeline]
                 no PolynomialFeatures (the trees model interactions directly)
TRAIN/TEST     : random 80/20, random_state=42        (identical to Random Forest)
HYPERPARAMS    : HistGradientBoostingRegressor(random_state=42, early_stopping=False);
                 all other params = scikit-learn defaults (loss='squared_error',
                 max_iter=100, max_leaf_nodes=31, l2_regularization=0.0).
                 early_stopping=False so the full training fold is used, matching
                 Random Forest and keeping runs deterministic.
GRIDSEARCHCV   : GridSearchCV(cv=5, scoring="neg_mean_absolute_error"),
                 learning_rate in {0.05, 0.1} x max_depth in {6, 12}, on TRAIN only
TRAINING/PRED  : GridSearchCV.fit(train) -> best_estimator_.predict(train/test)
METRICS        : train & test  MAE, MAPE, RMSE, R2, sMAPE
OUTPUT         : results/regression/histgbr/google_2019__hist_gradient_boosting.csv

Run:  python -m src.google_2019.hist_gradient_boosting
"""
from sklearn.ensemble import HistGradientBoostingRegressor

from src.evaluation import run_regression, RANDOM_STATE
from src.google_2019.preprocessing import load_entity_table, FEATURES

DATASET = "Google 2019 Cluster"
MODEL_NAME = "HistGradientBoostingRegressor"
OUT_CSV = "results/regression/histgbr/google_2019__hist_gradient_boosting.csv"

ESTIMATOR = HistGradientBoostingRegressor(random_state=RANDOM_STATE, early_stopping=False)
PARAM_GRID = {"model__learning_rate": [0.05, 0.1], "model__max_depth": [6, 12]}
USE_POLY = False
SCALER = "standard"

if __name__ == "__main__":
    run_regression(DATASET, MODEL_NAME, OUT_CSV, load_entity_table, FEATURES,
                   estimator=ESTIMATOR, param_grid=PARAM_GRID,
                   use_poly=USE_POLY, scaler=SCALER)
