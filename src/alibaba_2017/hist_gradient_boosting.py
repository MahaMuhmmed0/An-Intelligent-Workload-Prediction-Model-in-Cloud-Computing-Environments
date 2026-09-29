"""
src/alibaba_2017/hist_gradient_boosting.py
=========================================
Histogram-based Gradient Boosted Decision Trees (GBDT) regression
— Alibaba 2017, cross-sectional.

ADDITIVE EXPERIMENT.  This module is NOT part of the frozen 48-experiment set
behind results/tabular_results.csv.  It was added to test whether the
non-linear-ensemble behaviour observed with Random Forest is reproduced by an
independent boosting ensemble.  Its rows are written to
results/regression/histgbr/alibaba_2017__hist_gradient_boosting.csv and are
aggregated by `python -m src.run_all histgbr` into
results/tabular_results_histgbr.csv only — never into results/tabular_results.csv.

INPUT FEATURES : cpu_request, mem_request, cpu_mem_ratio, plan_disk, cpuset_width
TARGET(S)      : cpu_usage_mean, mem_usage_mean, cpu_usage_peak, mem_usage_peak
PREPROCESSING  : RobustPrep -> SimpleImputer(median) -> StandardScaler   [Pipeline]
TRAIN/TEST     : random 80/20, random_state=42        (identical to Random Forest)
HYPERPARAMS    : HistGradientBoostingRegressor(random_state=42, early_stopping=False);
                 all other params = scikit-learn defaults.
GRIDSEARCHCV   : GridSearchCV(cv=5, scoring="neg_mean_absolute_error"),
                 learning_rate in {0.05, 0.1} x max_depth in {6, 12}, on TRAIN only
TRAINING/PRED  : GridSearchCV.fit(train) -> best_estimator_.predict(train/test)
METRICS        : train & test  MAE, MAPE, RMSE, R2, sMAPE
OUTPUT         : results/regression/histgbr/alibaba_2017__hist_gradient_boosting.csv

As with Random Forest, test R2 stays ~0.33 (CPU) / ~0.32 (memory): the Alibaba
target is a percentage OF the request, so the request magnitude cancels out.

Run:  python -m src.alibaba_2017.hist_gradient_boosting
"""
from sklearn.ensemble import HistGradientBoostingRegressor

from src.evaluation import run_regression, RANDOM_STATE
from src.alibaba_2017.preprocessing import load_entity_table, FEATURES

DATASET = "Alibaba 2017"
MODEL_NAME = "HistGradientBoostingRegressor"
OUT_CSV = "results/regression/histgbr/alibaba_2017__hist_gradient_boosting.csv"

ESTIMATOR = HistGradientBoostingRegressor(random_state=RANDOM_STATE, early_stopping=False)
PARAM_GRID = {"model__learning_rate": [0.05, 0.1], "model__max_depth": [6, 12]}
USE_POLY = False
SCALER = "standard"

if __name__ == "__main__":
    run_regression(DATASET, MODEL_NAME, OUT_CSV, load_entity_table, FEATURES,
                   estimator=ESTIMATOR, param_grid=PARAM_GRID,
                   use_poly=USE_POLY, scaler=SCALER)
