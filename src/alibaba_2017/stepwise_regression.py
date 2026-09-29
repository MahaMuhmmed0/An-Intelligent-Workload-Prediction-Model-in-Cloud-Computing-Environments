"""
src/alibaba_2017/stepwise_regression.py
=======================================
Stepwise regression — Alibaba 2017, cross-sectional.

Embedded selection: degree-2 polynomial expansion -> LassoCV(cv=5) keeps the
non-zero terms -> OLS refit on those terms. Selection sees TRAIN data only.

INPUT FEATURES : cpu_request, mem_request, cpu_mem_ratio, plan_disk, cpuset_width
TARGET(S)      : cpu_usage_mean, mem_usage_mean, cpu_usage_peak, mem_usage_peak
PREPROCESSING  : RobustPrep -> SimpleImputer(median)
                 -> PolynomialFeatures(degree=2) -> StandardScaler (fit on train)
TRAIN/TEST     : random 80/20, random_state=42
SELECTOR       : LassoCV(cv=5, max_iter=10000, random_state=42) -> non-zero mask
FINAL MODEL    : LinearRegression().fit(Z_train[:, mask], y_train)
GRIDSEARCHCV   : not used (LassoCV is the internal CV)
METRICS        : train & test  MAE, MAPE, RMSE, R2, sMAPE
OUTPUT         : results/regression/alibaba_2017__stepwise.csv

Run:  python -m src.alibaba_2017.stepwise_regression
"""
from src.evaluation import run_regression
from src.alibaba_2017.preprocessing import load_entity_table, FEATURES

DATASET = "Alibaba 2017"
MODEL_NAME = "Stepwise Regression"
OUT_CSV = "results/regression/alibaba_2017__stepwise.csv"

if __name__ == "__main__":
    run_regression(DATASET, MODEL_NAME, OUT_CSV, load_entity_table, FEATURES,
                   stepwise=True)
