# Master verification table

63 experiments (56 regression across seven algorithms, incl. the additional HistGradientBoosting baseline, + 6 Alibaba forecasting + 1 Google temporal diagnostic). Each notebook is self-contained and writes its own row into `results/regression/` or `results/sequence/`; `src/run_all.py` aggregates those into `results/tabular_results.csv` / `results/sequence_results.csv` (HistGradientBoosting is additive, via `src/run_all.py histgbr` -> `results/tabular_results_histgbr.csv`; it does not modify `results/tabular_results.csv`). The prior 55-row, six-algorithm-only map is superseded by this 63-row map; the six-algorithm subset is still available directly from `results/tabular_results.csv`.

| dataset | module | target | algorithm | source_file | notebook | result_file |
|---|---|---|---|---|---|---|
| Google 2019 Cluster | Regression | cpu_usage_mean | Linear Regression | src/google_2019/linear_regression.py | notebooks/google_2019/Google_2019_Linear_Regression_CPU_mean.ipynb | results/tabular_results.csv |
| Google 2019 Cluster | Regression | cpu_usage_peak | Linear Regression | src/google_2019/linear_regression.py | notebooks/google_2019/Google_2019_Linear_Regression_CPU_peak.ipynb | results/tabular_results.csv |
| Google 2019 Cluster | Regression | mem_usage_mean | Linear Regression | src/google_2019/linear_regression.py | notebooks/google_2019/Google_2019_Linear_Regression_Memory_mean.ipynb | results/tabular_results.csv |
| Google 2019 Cluster | Regression | mem_usage_peak | Linear Regression | src/google_2019/linear_regression.py | notebooks/google_2019/Google_2019_Linear_Regression_Memory_peak.ipynb | results/tabular_results.csv |
| Google 2019 Cluster | Regression | cpu_usage_mean | Ridge Regression | src/google_2019/ridge_regression.py | notebooks/google_2019/Google_2019_Ridge_Regression_CPU_mean.ipynb | results/tabular_results.csv |
| Google 2019 Cluster | Regression | cpu_usage_peak | Ridge Regression | src/google_2019/ridge_regression.py | notebooks/google_2019/Google_2019_Ridge_Regression_CPU_peak.ipynb | results/tabular_results.csv |
| Google 2019 Cluster | Regression | mem_usage_mean | Ridge Regression | src/google_2019/ridge_regression.py | notebooks/google_2019/Google_2019_Ridge_Regression_Memory_mean.ipynb | results/tabular_results.csv |
| Google 2019 Cluster | Regression | mem_usage_peak | Ridge Regression | src/google_2019/ridge_regression.py | notebooks/google_2019/Google_2019_Ridge_Regression_Memory_peak.ipynb | results/tabular_results.csv |
| Google 2019 Cluster | Regression | cpu_usage_mean | Lasso Regression | src/google_2019/lasso_regression.py | notebooks/google_2019/Google_2019_Lasso_Regression_CPU_mean.ipynb | results/tabular_results.csv |
| Google 2019 Cluster | Regression | cpu_usage_peak | Lasso Regression | src/google_2019/lasso_regression.py | notebooks/google_2019/Google_2019_Lasso_Regression_CPU_peak.ipynb | results/tabular_results.csv |
| Google 2019 Cluster | Regression | mem_usage_mean | Lasso Regression | src/google_2019/lasso_regression.py | notebooks/google_2019/Google_2019_Lasso_Regression_Memory_mean.ipynb | results/tabular_results.csv |
| Google 2019 Cluster | Regression | mem_usage_peak | Lasso Regression | src/google_2019/lasso_regression.py | notebooks/google_2019/Google_2019_Lasso_Regression_Memory_peak.ipynb | results/tabular_results.csv |
| Google 2019 Cluster | Regression | cpu_usage_mean | Elastic Net | src/google_2019/elastic_net.py | notebooks/google_2019/Google_2019_ElasticNet_CPU_mean.ipynb | results/tabular_results.csv |
| Google 2019 Cluster | Regression | cpu_usage_peak | Elastic Net | src/google_2019/elastic_net.py | notebooks/google_2019/Google_2019_ElasticNet_CPU_peak.ipynb | results/tabular_results.csv |
| Google 2019 Cluster | Regression | mem_usage_mean | Elastic Net | src/google_2019/elastic_net.py | notebooks/google_2019/Google_2019_ElasticNet_Memory_mean.ipynb | results/tabular_results.csv |
| Google 2019 Cluster | Regression | mem_usage_peak | Elastic Net | src/google_2019/elastic_net.py | notebooks/google_2019/Google_2019_ElasticNet_Memory_peak.ipynb | results/tabular_results.csv |
| Google 2019 Cluster | Regression | cpu_usage_mean | Stepwise Regression | src/google_2019/stepwise_regression.py | notebooks/google_2019/Google_2019_Stepwise_CPU_mean.ipynb | results/tabular_results.csv |
| Google 2019 Cluster | Regression | cpu_usage_peak | Stepwise Regression | src/google_2019/stepwise_regression.py | notebooks/google_2019/Google_2019_Stepwise_CPU_peak.ipynb | results/tabular_results.csv |
| Google 2019 Cluster | Regression | mem_usage_mean | Stepwise Regression | src/google_2019/stepwise_regression.py | notebooks/google_2019/Google_2019_Stepwise_Memory_mean.ipynb | results/tabular_results.csv |
| Google 2019 Cluster | Regression | mem_usage_peak | Stepwise Regression | src/google_2019/stepwise_regression.py | notebooks/google_2019/Google_2019_Stepwise_Memory_peak.ipynb | results/tabular_results.csv |
| Google 2019 Cluster | Regression | cpu_usage_mean | Random Forest | src/google_2019/random_forest.py | notebooks/google_2019/Google_2019_RandomForest_CPU_mean.ipynb | results/tabular_results.csv |
| Google 2019 Cluster | Regression | cpu_usage_peak | Random Forest | src/google_2019/random_forest.py | notebooks/google_2019/Google_2019_RandomForest_CPU_peak.ipynb | results/tabular_results.csv |
| Google 2019 Cluster | Regression | mem_usage_mean | Random Forest | src/google_2019/random_forest.py | notebooks/google_2019/Google_2019_RandomForest_Memory_mean.ipynb | results/tabular_results.csv |
| Google 2019 Cluster | Regression | mem_usage_peak | Random Forest | src/google_2019/random_forest.py | notebooks/google_2019/Google_2019_RandomForest_Memory_peak.ipynb | results/tabular_results.csv |
| Alibaba 2017 | Regression | cpu_usage_mean | Linear Regression | src/alibaba_2017/linear_regression.py | notebooks/alibaba_2017/Alibaba_2017_Linear_Regression_CPU_mean.ipynb | results/tabular_results.csv |
| Alibaba 2017 | Regression | cpu_usage_peak | Linear Regression | src/alibaba_2017/linear_regression.py | notebooks/alibaba_2017/Alibaba_2017_Linear_Regression_CPU_peak.ipynb | results/tabular_results.csv |
| Alibaba 2017 | Regression | mem_usage_mean | Linear Regression | src/alibaba_2017/linear_regression.py | notebooks/alibaba_2017/Alibaba_2017_Linear_Regression_Memory_mean.ipynb | results/tabular_results.csv |
| Alibaba 2017 | Regression | mem_usage_peak | Linear Regression | src/alibaba_2017/linear_regression.py | notebooks/alibaba_2017/Alibaba_2017_Linear_Regression_Memory_peak.ipynb | results/tabular_results.csv |
| Alibaba 2017 | Regression | cpu_usage_mean | Ridge Regression | src/alibaba_2017/ridge_regression.py | notebooks/alibaba_2017/Alibaba_2017_Ridge_Regression_CPU_mean.ipynb | results/tabular_results.csv |
| Alibaba 2017 | Regression | cpu_usage_peak | Ridge Regression | src/alibaba_2017/ridge_regression.py | notebooks/alibaba_2017/Alibaba_2017_Ridge_Regression_CPU_peak.ipynb | results/tabular_results.csv |
| Alibaba 2017 | Regression | mem_usage_mean | Ridge Regression | src/alibaba_2017/ridge_regression.py | notebooks/alibaba_2017/Alibaba_2017_Ridge_Regression_Memory_mean.ipynb | results/tabular_results.csv |
| Alibaba 2017 | Regression | mem_usage_peak | Ridge Regression | src/alibaba_2017/ridge_regression.py | notebooks/alibaba_2017/Alibaba_2017_Ridge_Regression_Memory_peak.ipynb | results/tabular_results.csv |
| Alibaba 2017 | Regression | cpu_usage_mean | Lasso Regression | src/alibaba_2017/lasso_regression.py | notebooks/alibaba_2017/Alibaba_2017_Lasso_Regression_CPU_mean.ipynb | results/tabular_results.csv |
| Alibaba 2017 | Regression | cpu_usage_peak | Lasso Regression | src/alibaba_2017/lasso_regression.py | notebooks/alibaba_2017/Alibaba_2017_Lasso_Regression_CPU_peak.ipynb | results/tabular_results.csv |
| Alibaba 2017 | Regression | mem_usage_mean | Lasso Regression | src/alibaba_2017/lasso_regression.py | notebooks/alibaba_2017/Alibaba_2017_Lasso_Regression_Memory_mean.ipynb | results/tabular_results.csv |
| Alibaba 2017 | Regression | mem_usage_peak | Lasso Regression | src/alibaba_2017/lasso_regression.py | notebooks/alibaba_2017/Alibaba_2017_Lasso_Regression_Memory_peak.ipynb | results/tabular_results.csv |
| Alibaba 2017 | Regression | cpu_usage_mean | Elastic Net | src/alibaba_2017/elastic_net.py | notebooks/alibaba_2017/Alibaba_2017_ElasticNet_CPU_mean.ipynb | results/tabular_results.csv |
| Alibaba 2017 | Regression | cpu_usage_peak | Elastic Net | src/alibaba_2017/elastic_net.py | notebooks/alibaba_2017/Alibaba_2017_ElasticNet_CPU_peak.ipynb | results/tabular_results.csv |
| Alibaba 2017 | Regression | mem_usage_mean | Elastic Net | src/alibaba_2017/elastic_net.py | notebooks/alibaba_2017/Alibaba_2017_ElasticNet_Memory_mean.ipynb | results/tabular_results.csv |
| Alibaba 2017 | Regression | mem_usage_peak | Elastic Net | src/alibaba_2017/elastic_net.py | notebooks/alibaba_2017/Alibaba_2017_ElasticNet_Memory_peak.ipynb | results/tabular_results.csv |
| Alibaba 2017 | Regression | cpu_usage_mean | Stepwise Regression | src/alibaba_2017/stepwise_regression.py | notebooks/alibaba_2017/Alibaba_2017_Stepwise_CPU_mean.ipynb | results/tabular_results.csv |
| Alibaba 2017 | Regression | cpu_usage_peak | Stepwise Regression | src/alibaba_2017/stepwise_regression.py | notebooks/alibaba_2017/Alibaba_2017_Stepwise_CPU_peak.ipynb | results/tabular_results.csv |
| Alibaba 2017 | Regression | mem_usage_mean | Stepwise Regression | src/alibaba_2017/stepwise_regression.py | notebooks/alibaba_2017/Alibaba_2017_Stepwise_Memory_mean.ipynb | results/tabular_results.csv |
| Alibaba 2017 | Regression | mem_usage_peak | Stepwise Regression | src/alibaba_2017/stepwise_regression.py | notebooks/alibaba_2017/Alibaba_2017_Stepwise_Memory_peak.ipynb | results/tabular_results.csv |
| Alibaba 2017 | Regression | cpu_usage_mean | Random Forest | src/alibaba_2017/random_forest.py | notebooks/alibaba_2017/Alibaba_2017_RandomForest_CPU_mean.ipynb | results/tabular_results.csv |
| Alibaba 2017 | Regression | cpu_usage_peak | Random Forest | src/alibaba_2017/random_forest.py | notebooks/alibaba_2017/Alibaba_2017_RandomForest_CPU_peak.ipynb | results/tabular_results.csv |
| Alibaba 2017 | Regression | mem_usage_mean | Random Forest | src/alibaba_2017/random_forest.py | notebooks/alibaba_2017/Alibaba_2017_RandomForest_Memory_mean.ipynb | results/tabular_results.csv |
| Alibaba 2017 | Regression | mem_usage_peak | Random Forest | src/alibaba_2017/random_forest.py | notebooks/alibaba_2017/Alibaba_2017_RandomForest_Memory_peak.ipynb | results/tabular_results.csv |
| Google 2019 Cluster | Regression | cpu_usage_mean | HistGradientBoostingRegressor | src/google_2019/hist_gradient_boosting.py | notebooks/google_2019/Google_2019_HistGradientBoosting_CPU_mean.ipynb | results/tabular_results_histgbr.csv |
| Google 2019 Cluster | Regression | cpu_usage_peak | HistGradientBoostingRegressor | src/google_2019/hist_gradient_boosting.py | notebooks/google_2019/Google_2019_HistGradientBoosting_CPU_peak.ipynb | results/tabular_results_histgbr.csv |
| Google 2019 Cluster | Regression | mem_usage_mean | HistGradientBoostingRegressor | src/google_2019/hist_gradient_boosting.py | notebooks/google_2019/Google_2019_HistGradientBoosting_Memory_mean.ipynb | results/tabular_results_histgbr.csv |
| Google 2019 Cluster | Regression | mem_usage_peak | HistGradientBoostingRegressor | src/google_2019/hist_gradient_boosting.py | notebooks/google_2019/Google_2019_HistGradientBoosting_Memory_peak.ipynb | results/tabular_results_histgbr.csv |
| Alibaba 2017 | Regression | cpu_usage_mean | HistGradientBoostingRegressor | src/alibaba_2017/hist_gradient_boosting.py | notebooks/alibaba_2017/Alibaba_2017_HistGradientBoosting_CPU_mean.ipynb | results/tabular_results_histgbr.csv |
| Alibaba 2017 | Regression | cpu_usage_peak | HistGradientBoostingRegressor | src/alibaba_2017/hist_gradient_boosting.py | notebooks/alibaba_2017/Alibaba_2017_HistGradientBoosting_CPU_peak.ipynb | results/tabular_results_histgbr.csv |
| Alibaba 2017 | Regression | mem_usage_mean | HistGradientBoostingRegressor | src/alibaba_2017/hist_gradient_boosting.py | notebooks/alibaba_2017/Alibaba_2017_HistGradientBoosting_Memory_mean.ipynb | results/tabular_results_histgbr.csv |
| Alibaba 2017 | Regression | mem_usage_peak | HistGradientBoostingRegressor | src/alibaba_2017/hist_gradient_boosting.py | notebooks/alibaba_2017/Alibaba_2017_HistGradientBoosting_Memory_peak.ipynb | results/tabular_results_histgbr.csv |
| Alibaba 2017 | Forecasting | cpu_util | LSTM | src/alibaba_2017/lstm.py | notebooks/alibaba_2017/Alibaba_2017_LSTM_CPU.ipynb | results/sequence_results.csv |
| Alibaba 2017 | Forecasting | mem_util | LSTM | src/alibaba_2017/lstm.py | notebooks/alibaba_2017/Alibaba_2017_LSTM_Memory.ipynb | results/sequence_results.csv |
| Alibaba 2017 | Forecasting | cpu_util | BiLSTM | src/alibaba_2017/bilstm.py | notebooks/alibaba_2017/Alibaba_2017_BiLSTM_CPU.ipynb | results/sequence_results.csv |
| Alibaba 2017 | Forecasting | mem_util | BiLSTM | src/alibaba_2017/bilstm.py | notebooks/alibaba_2017/Alibaba_2017_BiLSTM_Memory.ipynb | results/sequence_results.csv |
| Alibaba 2017 | Forecasting | cpu_util | RNN | src/alibaba_2017/rnn.py | notebooks/alibaba_2017/Alibaba_2017_RNN_CPU.ipynb | results/sequence_results.csv |
| Alibaba 2017 | Forecasting | mem_util | RNN | src/alibaba_2017/rnn.py | notebooks/alibaba_2017/Alibaba_2017_RNN_Memory.ipynb | results/sequence_results.csv |
| Google 2019 Cluster | Temporal Diagnostic | - | ACF | src/google_2019/temporal_diagnostic.py | notebooks/google_2019/Google_2019_Temporal_Diagnostic.ipynb | results/temporal_diagnostic.json |
