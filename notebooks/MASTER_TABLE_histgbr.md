# Master table - ADDITIVE HistGradientBoostingRegressor experiment

> **These 8 rows have been merged into the combined `notebooks/MASTER_TABLE.md`
> (now 63 rows).** This file is retained as the standalone, historical record
> of the additive HistGradientBoosting notebooks and is superseded as a
> stand-alone map — use the combined table for the current path map.

8 regression experiments (Google 2019 Cluster x 4 targets, Alibaba 2017 x 4 targets), added to test whether the non-linear-ensemble behaviour seen with Random Forest is reproduced by an independent boosting ensemble. Same features, targets, split (`random_state=42`), preprocessing, 5-fold CV protocol and metrics as the existing regression modules.

One of the 56 principal regression / 63 principal experiments overall, additive relative to the frozen, six-algorithm `results/tabular_results.csv` (unchanged by this). See `documentation/MASTER_EXPERIMENT_TABLE.md` for the combined numeric inventory. Run: `python -m src.run_all histgbr`.

| Dataset | Module | Target | Algorithm | Source File | Notebook | Result File |
|---|---|---|---|---|---|---|
| Google 2019 Cluster | Regression | cpu_usage_mean | HistGradientBoostingRegressor | src/google_2019/hist_gradient_boosting.py | notebooks/google_2019/Google_2019_HistGradientBoosting_CPU_mean.ipynb | results/tabular_results_histgbr.csv |
| Google 2019 Cluster | Regression | cpu_usage_peak | HistGradientBoostingRegressor | src/google_2019/hist_gradient_boosting.py | notebooks/google_2019/Google_2019_HistGradientBoosting_CPU_peak.ipynb | results/tabular_results_histgbr.csv |
| Google 2019 Cluster | Regression | mem_usage_mean | HistGradientBoostingRegressor | src/google_2019/hist_gradient_boosting.py | notebooks/google_2019/Google_2019_HistGradientBoosting_Memory_mean.ipynb | results/tabular_results_histgbr.csv |
| Google 2019 Cluster | Regression | mem_usage_peak | HistGradientBoostingRegressor | src/google_2019/hist_gradient_boosting.py | notebooks/google_2019/Google_2019_HistGradientBoosting_Memory_peak.ipynb | results/tabular_results_histgbr.csv |
| Alibaba 2017 | Regression | cpu_usage_mean | HistGradientBoostingRegressor | src/alibaba_2017/hist_gradient_boosting.py | notebooks/alibaba_2017/Alibaba_2017_HistGradientBoosting_CPU_mean.ipynb | results/tabular_results_histgbr.csv |
| Alibaba 2017 | Regression | cpu_usage_peak | HistGradientBoostingRegressor | src/alibaba_2017/hist_gradient_boosting.py | notebooks/alibaba_2017/Alibaba_2017_HistGradientBoosting_CPU_peak.ipynb | results/tabular_results_histgbr.csv |
| Alibaba 2017 | Regression | mem_usage_mean | HistGradientBoostingRegressor | src/alibaba_2017/hist_gradient_boosting.py | notebooks/alibaba_2017/Alibaba_2017_HistGradientBoosting_Memory_mean.ipynb | results/tabular_results_histgbr.csv |
| Alibaba 2017 | Regression | mem_usage_peak | HistGradientBoostingRegressor | src/alibaba_2017/hist_gradient_boosting.py | notebooks/alibaba_2017/Alibaba_2017_HistGradientBoosting_Memory_peak.ipynb | results/tabular_results_histgbr.csv |
