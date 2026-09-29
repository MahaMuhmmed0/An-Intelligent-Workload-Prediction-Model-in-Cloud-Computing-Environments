# Results & Discussion (paper-ready draft)

All numbers come from `results/tabular_results.csv`, `results/sequence_results.csv`
and `results/temporal_diagnostic.json`. Exact tables are in `RESULTS_TABLES.md`;
this document is the narrative. Claims are limited to what the results support.

---

## 4. Results

### 4.1 Overview

Across the 48 cross-sectional regression experiments there is no evidence of
substantial over-fitting: the mean absolute train–test R² gap is approximately
0.02, with a maximum of approximately 0.07 on the noisiest target (Alibaba CPU
peak), and no test R² approaches 1. Target leakage — which produced the near-100 %
test scores of the earlier internal experiments — is ruled out separately, on the
basis of the feature construction and the training pipeline rather than the
train–test performance gap (Section on leakage in the methodology;
`LEAKAGE_CORRECTION.md`).

### 4.2 Google 2019 Cluster — regression

Predicting an instance's usage from its resource request and static metadata:

* **The two non-linear ensembles — Random Forest and the additional
  HistGradientBoosting baseline — are clearly the strongest models** on every
  target, reaching a test R² of about **0.83–0.84 for mean CPU usage** and
  about **0.94 for mean memory usage**, with the peak targets close behind
  (≈ 0.81–0.83 CPU, ≈ 0.94 memory). The two ensembles agree to within 0.03 R²
  on every target (within the ±0.02 cross-version reproducibility tolerance
  on seven of eight) and are treated as equivalent; HistGradientBoosting is
  an established, additional baseline, not a novel algorithm, and neither
  ensemble is a universal winner.
* The linear family is markedly weaker (test R² ≈ 0.34 for CPU, ≈ 0.50 for
  memory). L1/elastic-net regularisation on degree-2 polynomial features helps
  (Lasso/Elastic-Net test R² ≈ 0.39–0.41 CPU, ≈ 0.60–0.62 memory), and the
  stepwise selection is the strongest of the six principal (non-ensemble)
  algorithms (≈ 0.47 CPU, ≈ 0.64 memory).
* Memory usage is consistently more predictable than CPU usage for every model.

### 4.3 Alibaba 2017 — regression

The same protocol on the Alibaba online-container instances:

* All seven algorithms land in a narrow band of **test R² ≈ 0.19–0.34**. The
  two non-linear ensembles are again the strongest (Random Forest ≈ 0.33 CPU
  mean, ≈ 0.32 memory mean; HistGradientBoosting ≈ 0.34 CPU mean, ≈ 0.33 CPU
  peak), but their margin over the linear models is small (≈ 0.05–0.10), and
  for the memory-peak target stepwise, Random Forest, and HistGradientBoosting
  are all within ≈ 0.004 of each other.
* The gap between the non-linear ensembles and the linear models is far
  smaller than on Google, indicating that little additional non-linear
  structure is available in the Alibaba feature–target relationship.

### 4.4 Alibaba 2017 — forecasting

One-step-ahead forecasting of machine utilisation from the previous 24 intervals
(≈ 118,500 training / 38,000 test windows):

* All three recurrent models forecast **memory utilisation very well** (test R²
  0.900–0.933) and **CPU utilisation well** (test R² 0.768–0.796). Best per
  target: LSTM for CPU (0.796), BiLSTM for memory (0.933).
* Differences between LSTM, BiLSTM and RNN are within the run-to-run variance
  (≈ ±0.05 R²); none is consistently superior, and the vanilla RNN is competitive
  (CPU 0.783, memory 0.900).
* Forecasting test R² substantially exceeds the Alibaba regression test R² on
  the corresponding resource: memory ≈ 0.93 (forecast) vs ≈ 0.32 (regression);
  CPU ≈ 0.79 vs ≈ 0.33.

### 4.5 Google 2019 Cluster — temporal diagnostic (no forecasting model)

The hourly cell-level series (n = 745) has autocorrelation within the ±0.072
white-noise band at every lag, including lag 24; the equivalent Alibaba series
has autocorrelation 0.87 at lag 1 and 0.55 at lag 24. Google forecasting is
therefore excluded from the study (`TEMPORAL_DIAGNOSTIC.md`).

### 4.6 Highest test R² per dataset/target

Not a "best model" in a statistical sense: the two non-linear ensembles agree
to within 0.03 R² on every regression target (within the ±0.02 cross-version
reproducibility tolerance on seven of eight) and are treated as equivalent;
the three forecasting architectures are within their ±0.05 run-to-run
variation of each other. Neither is a universal winner.

| Dataset | Target | Highest test R² model | Test R² | Runner-up (within tolerance) |
|---|---|---|---|---|
| Google 2019 Cluster | CPU mean | HistGradientBoosting | ≈ 0.84 | Random Forest ≈ 0.83 |
| Google 2019 Cluster | CPU peak | HistGradientBoosting | ≈ 0.83 | Random Forest ≈ 0.81 |
| Google 2019 Cluster | Memory mean | HistGradientBoosting | ≈ 0.94 | Random Forest ≈ 0.94 |
| Google 2019 Cluster | Memory peak | HistGradientBoosting | ≈ 0.94 | Random Forest ≈ 0.94 |
| Alibaba 2017 | CPU mean | HistGradientBoosting | ≈ 0.34 | Random Forest ≈ 0.33 |
| Alibaba 2017 | CPU peak | HistGradientBoosting | ≈ 0.34 | Random Forest ≈ 0.31 |
| Alibaba 2017 | Memory mean | HistGradientBoosting | ≈ 0.33 | Random Forest ≈ 0.32 |
| Alibaba 2017 | Memory peak | HistGradientBoosting | ≈ 0.33 | Stepwise ≈ 0.33, Random Forest ≈ 0.32 |
| Alibaba 2017 (forecast) | CPU utilisation | LSTM ≈ RNN ≈ BiLSTM | ≈ 0.73–0.80 | — |
| Alibaba 2017 (forecast) | Memory utilisation | BiLSTM ≈ LSTM ≈ RNN | ≈ 0.91–0.93 | — |

*(Exact values in `RESULTS_TABLES.md`.)*

---

## 5. Discussion

### 5.1 Why the non-linear ensembles differ from the linear models

On Google, the non-linear ensembles improve test R² by roughly 0.3–0.5 over the
best linear model. The engineered features are few (five to six) and the
polynomial-augmented linear models already capture pairwise interactions, so
the gain is attributable to the trees modelling **non-monotone and threshold
effects** in the request → usage mapping (for example, low-priority
collections systematically under-using their request) — an interpretation of
the pattern, not a demonstrated mechanism, since the frozen study performs no
feature-importance or ablation analysis. The additional HistGradientBoosting
baseline reproduces this margin, so the ensemble advantage is not specific to
Random Forest. On Alibaba the ensembles gain only ≈ 0.05–0.10 over the linear
family, which indicates the Alibaba request → usage relationship is close to
what a regularised linear model can already express — there is little
additional structure to exploit, not merely that a different model is needed.
This is not a demonstrated cause: the datasets also differ in size, feature
content, workload population, and usage aggregation, which the experiments do
not separate.

### 5.2 Why Google and Alibaba behave so differently

The decisive factor is the **definition of the target** (`DATASET_METHODOLOGY.md`
§6). Google usage is an absolute normalised quantity on the same scale as the
request, so a job that requests more tends to use more and the request is
genuinely informative — the non-linear ensembles reach R² ≈ 0.83–0.94. Alibaba usage is a
**percentage of the instance's own request**, so the request magnitude is
divided out of the target; what remains to predict is how *efficiently* an
instance uses whatever it asked for, which the request itself barely constrains —
hence R² ≈ 0.30. This is a measured consequence of the two traces' encoding, and
it should not be read as one dataset being "better".

### 5.3 Cross-sectional prediction versus temporal forecasting

For Alibaba, forecasting the next interval from recent history is far more
accurate than predicting usage from the request (memory ≈ 0.92 vs ≈ 0.31; CPU
≈ 0.77 vs ≈ 0.33). Recent utilisation is a strong autoregressive signal
(`temporal_diagnostic.json`: lag-1 autocorrelation ≈ 0.87), whereas the request
is a weak one. **Interpretation:** where a short history is available, reactive
forecasting is the more reliable basis for scaling decisions; the request is
useful mainly at admission time, before any history exists. The two paradigms are
complementary rather than competing.

### 5.4 The predictive value of resource requests

On Google the request explains a large share of usage variance (up to ≈ 94 % for
memory with the non-linear ensembles); on Alibaba it explains little (≈ 30 %). Taken
together, **the request is informative in proportion to how directly it is tied
to the usage scale**. Where usage is reported relative to the request, the
request alone is a poor predictor and other signals (history, co-location,
workload class) would be needed.

### 5.5 Over-provisioning

On both traces, mean CPU usage is a small fraction of the request (the Google
sMAPE for CPU exceeds 100 %, reflecting predictions and actuals that are both
close to zero relative to the request; the Alibaba mean `cpu_util` in the raw
data is ≈ 9–10 % of the request). This is consistent with the well-documented
over-provisioning of cloud reservations. **A predictive caveat:** our models
estimate *expected* usage; using them to reclaim head-room would additionally
require modelling the upper tail, which the peak-usage targets only partially
address and which is left for future work.

### 5.6 The effect of temporal structure

The Alibaba forecasting results and the Google diagnostic together show that the
presence of a genuine machine-level time series is what enables accurate
short-horizon prediction. The Google extract, lacking such continuity, cannot
support forecasting regardless of model choice — the near-zero R² of the Google
sequence *attempt* is caused by the absent autocorrelation, not by the LSTM.

### 5.7 Implications for workload prediction and resource provisioning

* **Admission-time right-sizing** benefits from request-based regression only
  where the platform reports absolute usage (Google-like); where usage is
  request-relative (Alibaba-like), request-based estimates are weak and should be
  treated as lower-confidence.
* **Reactive autoscaling** is well served by simple recurrent models on
  5-minute telemetry: even a vanilla RNN reaches R² ≈ 0.78 (CPU) / ≈ 0.91
  (memory) here, and the added complexity of LSTM/BiLSTM buys little on these
  short univariate series.
* **Dataset choice matters for method evaluation:** a workload-prediction method
  that looks strong on one trace can look weak on another purely because of how
  usage is normalised, so cross-trace evaluation should compare R² and state the
  usage definition explicitly.

All statements above describe predictive relationships observed in these two
traces. No causal claims are made, and no significance tests were performed.
