# Limitations

Only limitations actually supported by the experiments and data are listed.

## L1 — The two datasets are not structurally equivalent
Google 2019 Cluster is used through a single joined CSV extract of instance-usage
records; Alibaba 2017 is used through its native relational tables. Their "usage"
quantities differ: an absolute normalised value for Google (fraction of the
largest machine's capacity) versus a request-relative percentage for Alibaba.
**Consequence:** only R² is comparable across the two datasets; MAE, RMSE and
sMAPE are compared only within a dataset. The comparison is between *modelling
behaviour on two traces*, not a controlled A/B on one workload.

## L2 — Different usage definitions and scales
Because Alibaba usage is normalised by each instance's own request, the request
magnitude is largely divided out of the target. The Google target retains the
request scale. This is the principal reason the R² ceilings differ so much
(Google Random Forest ≈ 0.83–0.94; Alibaba ≈ 0.30–0.35) and it is a property of
the published data, not of the models.

## L3 — Google 2019 Cluster temporal sparsity
The public extract records one row per instance-usage window, not per machine over
time. The median machine is observed in roughly four distinct hours across the
31-day span, and the hourly cell-level aggregate has autocorrelation within the
white-noise band at every lag (`TEMPORAL_DIAGNOSTIC.md`). A defensible
sequence-forecasting experiment cannot be built on it.

## L4 — No equivalent sequence forecasting on Google
Following L3, LSTM/BiLSTM/RNN are evaluated on Alibaba only. The regression-vs-
forecasting comparison is therefore complete for Alibaba but one-sided for Google
(regression only). We do not claim what a Google forecasting model *would* achieve
on the full trace.

## L5 — Different temporal resolution and horizon
Alibaba `server_usage` provides 12 hours at 5-minute resolution (144 points per
machine). Forecasting is one-step-ahead (next 5-minute interval) from a 24-step
window. Longer horizons, multi-step forecasting and longer histories are outside
the available coverage and are not evaluated.

## L6 — Sampling of the Google extract
For tractability the Google experiments use a fixed 150,000-row random sample
(`random_state=42`), reduced to ≈145,700 entities after filtering. Results on the
full extract could differ, though the sample is large relative to the 5–6
dimensional feature space.

## L7 — Small, fixed hyper-parameter search
Regression grids are deliberately small (Ridge/Lasso/Elastic Net over a few
`alpha` values; Random Forest over two `max_depth` values). Recurrent-network
hyper-parameters (units, dropout, learning rate, loss, window length) are fixed a
priori and **not** searched, to keep the three architectures comparable and the
compute bounded. A wider search might shift individual numbers; the qualitative
findings (tree ensembles best for regression; recurrent models strong for Alibaba
forecasting) are unlikely to reverse but are not guaranteed optimal.

## L8 — Feature set is intentionally minimal
Features are restricted to the resource request plus static scheduling metadata.
Richer signals (co-location, machine attributes, historical usage of the same
user/collection) are excluded — some because they are unavailable in one or both
traces, some to keep the two datasets on a common footing, and some because using
past usage of the same entity risks reintroducing leakage. The regression R²
therefore reflects *how well the request alone predicts usage*, which is the
intended research question, not the best achievable prediction.

## L9 — Online containers only (Alibaba)
The Alibaba regression uses online-service containers. Batch-task usage is not
modelled because `batch_instance.csv` (which holds actual batch usage) is not in
the provided data; `batch_task.csv` contains requests only.

## L10 — Reproducibility bounds
Regression results are exactly reproducible within one scikit-learn version;
CV-tuned models can vary by ~±0.02 in R² across versions due to internal
tie-breaking. Forecasting results vary by ~±0.05 in R² between runs because
TensorFlow CPU training is not bit-for-bit deterministic even with a fixed seed.
The reported numbers are single representative runs; the direction and rough
magnitude of every finding is stable across repetitions.

## L11 — Predictive, not causal
The models establish that resource requests and recent utilisation history are
*predictive* of future usage to a measurable degree. They do not establish
causal mechanisms, and the study performs no statistical significance testing or
causal inference.

## L12 — Single cell / single period per trace
Each trace is one cluster over one time window (a Google cell over 31 days; an
Alibaba cluster over 12 hours). Generalisation to other providers, cluster sizes,
scheduling policies or time periods is not tested.

## L13 — Alibaba CPU-request normalisation constant
The Alibaba CPU request feature is normalised as
`cpu_request = plan_cpu / plan_cpu.max()`, where the maximum is computed over the
complete extracted dataset (train + test partitions). This constant is
independent of the target variable and of any usage measurement, and the audit
determined its numerical impact on the results to be negligible, so it does
**not** introduce target leakage. A strictly train-only preprocessing
implementation could instead estimate this normalisation constant from the
training partition. The Google 2019 Cluster requests are already published in
normalised units and are not re-scaled by our code, so this point applies to
Alibaba only.
