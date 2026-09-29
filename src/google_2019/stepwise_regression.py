"""
src/google_2019/stepwise_regression.py
======================================
Stepwise regression — Google 2019 Cluster, cross-sectional.

Implemented as EMBEDDED selection: degree-2 polynomial expansion, then LassoCV
picks the non-zero-coefficient terms, then an OLS model is refit on exactly those
terms. Selection and every fitted transform see TRAINING data only (this is the
fix for the original notebooks, which selected features against the test set).

INPUT FEATURES : cpu_request, mem_request, cpu_mem_ratio, priority,
                 scheduling_class, collection_type
TARGET(S)      : cpu_usage_mean, mem_usage_mean, cpu_usage_peak, mem_usage_peak
PREPROCESSING  : RobustPrep -> SimpleImputer(median)
                 -> PolynomialFeatures(degree=2) -> StandardScaler
                 (fit on train, applied to test)
TRAIN/TEST     : random 80/20, random_state=42
SELECTOR       : LassoCV(cv=5, max_iter=10000, random_state=42)  -> non-zero mask
FINAL MODEL    : LinearRegression().fit(Z_train[:, mask], y_train)
HYPERPARAMS    : LassoCV chooses its own alpha by 5-fold CV; n_selected reported
GRIDSEARCHCV   : not used (LassoCV is the internal CV)
PREDICTION     : OLS.predict on the selected columns of train and test
METRICS        : train & test  MAE, MAPE, RMSE, R2, sMAPE
OUTPUT         : results/regression/google_2019__stepwise.csv

Run:  python -m src.google_2019.stepwise_regression
"""
from src.evaluation import run_regression
from src.google_2019.preprocessing import load_entity_table, FEATURES

DATASET = "Google 2019 Cluster"
MODEL_NAME = "Stepwise Regression"
OUT_CSV = "results/regression/google_2019__stepwise.csv"

if __name__ == "__main__":
    run_regression(DATASET, MODEL_NAME, OUT_CSV, load_entity_table, FEATURES,
                   stepwise=True)
