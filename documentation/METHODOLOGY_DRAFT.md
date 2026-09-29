# Methodology (paper-ready draft)

Draft for the Master's thesis / paper. Academic register. Numbers are filled from
`results/` in `RESULTS_TABLES.md`; this section describes *how*, not *what was
obtained*. Novelty is not overstated: the contribution is a careful, leakage-free
comparative study across two public cluster traces and two prediction paradigms.

---

## 3.1 Research methodology

This study evaluates machine-learning models for cloud workload prediction under
two complementary paradigms and on two independent public cluster traces. The
first paradigm, **cross-sectional regression**, predicts the resource *usage* of
an individual workload from the resource *request* declared at submission time;
it is relevant to admission control and right-sizing. The second paradigm,
**one-step-ahead time-series forecasting**, predicts a machine's next-interval
utilisation from its recent history; it is relevant to reactive autoscaling.

The two datasets — the Google 2019 Cluster trace (Borg, trace v3) and the Alibaba
`cluster-trace-v2017` — are treated with an identical *methodology* (feature
contract, split protocol, preprocessing discipline, evaluation metrics) but with
*feature content* that reflects what each trace records. The datasets are not
assumed to share a schema; the mapping between them, and its limits, is stated
explicitly (Section 3.2).

A central methodological concern is **leakage avoidance**. Preliminary experiments
that included the prediction target, or statistics derived from it, among the
model inputs produced near-perfect scores that were artefacts rather than
results. The design described below removes every such path and is validated by
the close agreement between training and test performance (Section 4).

All experiments are packaged for reproduction: each is a self-contained Jupyter
notebook and an importable Python module, and every reported number is written to
a result file that the notebook regenerates (Section 3.17).

## 3.2 Dataset description

### 3.2.1 Google 2019 Cluster (Borg trace v3)

The trace was collected from a Google Borg cell over a 31-day period in 2019 and
is published as a set of relational tables. The extract used here,
`borg_traces_data.csv`, is a row-level join of the instance-events,
instance-usage and collection-metadata tables, with one row per instance-usage
measurement window (approximately 405,900 rows, 34 columns). CPU is expressed in
Normalised Compute Units (NCU): Google Compute Units divided by the largest
machine's GCU capacity, so values lie in [0, 1]; memory is normalised bytes on the
same principle. The resource request represents an upper *limit* on consumption.

### 3.2.2 Alibaba cluster-trace-v2017

The trace covers a co-located production cluster over 12 continuous hours of usage
data (`trace_201708`). It is published as separate tables; this study uses
`container_event.csv` (online-instance requests), `container_usage.csv`
(online-instance usage, sampled every 60 s and averaged over 300 s), and
`server_usage.csv` (per-machine utilisation at 5-minute resolution). CPU requests
are integer core counts; memory and disk requests are normalised. Container usage
(`cpu_util`, `mem_util`) is expressed as a **percentage of the instance's own
request** (0–100).

### 3.2.3 Consequences of the differences

The two traces differ in structure, in temporal coverage, and — critically — in
what "usage" denotes: an absolute normalised quantity for Google, a
request-relative ratio for Alibaba. Only the coefficient of determination (R²) is
therefore comparable across datasets; absolute-error metrics are not. A full
field-by-field comparison, distinguishing documented facts from interpretation,
is given in `DATASET_METHODOLOGY.md`.

## 3.3 Data preprocessing

**Google 2019 Cluster.** A fixed 150,000-row random sample
(`random_state = 42`) is drawn for tractability. The dict-encoded columns
`resource_request`, `average_usage` and `maximum_usage` are parsed with
`ast.literal_eval` to extract their `cpus` and `memory` fields. Rows with a zero
CPU or memory request are removed (the request-to-usage relationship is undefined
and the ratio feature cannot be formed). Usage values outside a wide plausible
band are removed as parsing artefacts. The filtered table contains approximately
145,700 entities.

**Alibaba 2017.** `container_event` is filtered to `Create` events and reduced to
one row per instance (first observed request). `container_usage` is aggregated to
one row per instance (mean and maximum of `cpu_util` and `mem_util`, and the count
of observations). Instances with fewer than three usage observations are dropped.
The request and usage tables are inner-joined on `instance_id`, yielding
approximately 10,185 instances.

**Both.** Missing feature values are imputed with the column median. To keep the
polynomial-augmented linear models numerically stable on heavy-tailed request
data, each feature is winsorised to the training 1st/99th percentile and then
`log1p`-transformed; both operations learn their parameters from the training
partition only. Imputation, winsorisation, polynomial expansion and scaling are
composed as a scikit-learn `Pipeline` so that no transform is ever fitted on test
data.

## 3.4 Feature engineering

The feature set is restricted to the resource request and static scheduling
metadata — never a usage quantity, and never a quantity derived from a target.

* **Google 2019 Cluster:** `cpu_request`, `mem_request`,
  `cpu_mem_ratio = cpu_request / (mem_request + ε)`, `priority`,
  `scheduling_class`, `collection_type`.
* **Alibaba 2017:** `cpu_request` (core count rescaled to [0, 1]), `mem_request`,
  `cpu_mem_ratio = plan_cpu / (plan_mem + ε)`, `plan_disk`, `cpuset_width` (the
  number of pinned CPUs).

The ratio feature is formed from requests only. No `network_io` feature is used
(the Alibaba trace contains no network column; earlier notebooks that used one
were in error).

## 3.5 Target definition

Four regression targets are predicted independently:

| Target | Google 2019 Cluster | Alibaba 2017 |
|---|---|---|
| `cpu_usage_mean` | `average_usage.cpus` | mean of `cpu_util` per instance, ÷ 100 |
| `cpu_usage_peak` | `maximum_usage.cpus` | max of `cpu_util` per instance, ÷ 100 |
| `mem_usage_mean` | `average_usage.memory` | mean of `mem_util` per instance, ÷ 100 |
| `mem_usage_peak` | `maximum_usage.memory` | max of `mem_util` per instance, ÷ 100 |

For forecasting, the target is the next-interval machine utilisation
(`cpu_util` or `mem_util`, rescaled to [0, 1]).

## 3.6 Cross-sectional regression methodology

Each (dataset, target) pair is modelled by seven regressors (the six principal
algorithms plus the additional, established HistGradientBoosting baseline). The data are split
80/20 with a **shuffled** `train_test_split` (`random_state = 42`): the task is
cross-sectional — each row is an independent workload entity, not a time step — so
a random split is appropriate and matches the protocol of the earlier internal
study. A chronological split is retained only for sensitivity analysis. All
preprocessing and hyper-parameter selection use the training partition; the test
partition is untouched until final evaluation.

## 3.7 Regression algorithms

| Algorithm | Regularisation | Polynomial degree | Scaler | Tuned hyper-parameter(s) |
|---|---|---|---|---|
| Linear Regression | none | 1 | MinMax | — |
| Ridge Regression | L2 | 1 | Standard | `alpha ∈ {10⁻³ … 10²}` |
| Lasso Regression | L1 | 2 | Standard | `alpha ∈ {10⁻⁴ … 10⁻¹}` |
| Elastic Net | L1 + L2 | 2 | Standard | `alpha ∈ {10⁻³ … 10⁻¹}`, `l1_ratio ∈ {0.3, 0.6}` |
| Stepwise Regression | embedded (LassoCV) | 2 | Standard | `alpha` chosen internally by `LassoCV` |
| Random Forest | ensemble | 1 | Standard | `max_depth ∈ {6, 12}` |
| HistGradientBoosting *(additional baseline)* | ensemble | 1 | Standard | `learning_rate ∈ {0.05, 0.1}`, `max_depth ∈ {6, 12}` |

HistGradientBoostingRegressor is an established, additional non-linear
ensemble baseline evaluated under the identical pipeline as the six principal
regression algorithms, to check whether behaviour attributed to Random Forest
is specific to that model. It is not a novel algorithm and is not a research
contribution of this work.

Stepwise regression is implemented as embedded selection: on the degree-2
polynomial feature space, `LassoCV` (5-fold) selects the non-zero-coefficient
terms and an ordinary least-squares model is refitted on exactly those terms.
Linear models were selected as interpretable baselines; regularised variants test
whether shrinkage or sparsity helps on a small, correlated feature set; Random
Forest tests whether non-linear feature interactions carry additional signal.

## 3.8 Time-series methodology (Alibaba 2017)

Forecasting is performed on the Alibaba `server_usage` telemetry, which records
per-machine utilisation at 5-minute resolution (144 intervals per machine). One
series is built per machine, ordered chronologically and rescaled from percent to
[0, 1]; machines observed in fewer than 40 of the 144 intervals are discarded,
leaving 1,310 series. For each machine the first 80 % of its timeline forms the
training segment and the last 20 % the test segment; a single `MinMaxScaler` is
fitted on the concatenation of all training points and applied to the test points.
Sliding windows of the previous 24 intervals predict the next interval; no window
crosses a machine's train/test boundary. This yields approximately 118,500
training and 38,000 test windows. Validation during training uses the final 10 %
of the training windows — never the test set.

## 3.9 LSTM architecture

A stacked long short-term memory network: `LSTM(64, return_sequences=True)` →
`Dropout(0.2)` → `LSTM(32)` → `Dropout(0.2)` → `Dense(1)`. Input shape
`(24, 1)`.

## 3.10 BiLSTM architecture

As the LSTM, with both recurrent layers wrapped in a bidirectional layer:
`Bidirectional(LSTM(64, return_sequences=True))` → `Dropout(0.2)` →
`Bidirectional(LSTM(32))` → `Dropout(0.2)` → `Dense(1)`.

## 3.11 RNN architecture

As the LSTM, with `SimpleRNN` cells replacing the `LSTM` cells:
`SimpleRNN(64, return_sequences=True)` → `Dropout(0.2)` → `SimpleRNN(32)` →
`Dropout(0.2)` → `Dense(1)`.

All three are compiled with the Adam optimiser (learning rate 10⁻³) and the Huber
loss (δ = 1.0), and trained for up to 30 epochs with batch size 128,
`EarlyStopping` on the validation loss (patience 8, best weights restored) and
`ReduceLROnPlateau` (factor 0.5, patience 4). Sequence ordering is preserved
(`shuffle=False`).

## 3.12 Hyper-parameter tuning

Regression hyper-parameters are tuned by `GridSearchCV` with 5-fold
cross-validation on the training partition, scoring by negative mean absolute
error. The grids are deliberately small (Table in Section 3.7) — the feature set
has five to six dimensions and larger grids gave no measurable benefit in
preliminary runs. Recurrent-network hyper-parameters (units, dropout, learning
rate, loss) are fixed a priori from common practice for short univariate
utilisation series and are not searched, to keep the three architectures directly
comparable and the computational budget bounded.

## 3.13 Train/test methodology

* Regression: 80/20 shuffled split, `random_state = 42`; 5-fold CV inside the
  training partition; test partition evaluated once.
* Forecasting: per-machine 80/20 chronological split; 10 % of the training windows
  held out for validation; test windows evaluated once.
* No data point, and no statistic computed from a test data point, participates in
  fitting or model selection.

## 3.14 Evaluation metrics

Performance is reported with mean absolute error (MAE), root mean squared error
(RMSE) and the coefficient of determination (R²), for both the training and the
test partition. The symmetric mean absolute percentage error (sMAPE) is also
reported. The ordinary mean absolute percentage error (MAPE) is **not** a
meaningful metric here: both traces contain many usage values at or near zero
(a task may legitimately consume no CPU while binaries are staged), which makes
MAPE diverge. Where MAPE appears in the result files it is retained only for
completeness and is explicitly flagged as unreliable; interpretation uses MAE,
RMSE, R² and sMAPE.

Because the Google and Alibaba targets denote different physical quantities
(Section 3.2.3), only R² is compared across datasets; MAE, RMSE and sMAPE are
compared only within a dataset.

## 3.15 Leakage prevention

The following safeguards are applied and verified (details and evidence in
`LEAKAGE_CORRECTION.md`):

1. The feature list is an explicit request/metadata set; an assertion
   (`TARGET not in FEATURES`) is executed before every experiment.
2. No feature is derived from any target; the CPU-to-memory ratio uses requests
   only.
3. Imputation, winsorisation, polynomial expansion and scaling are `Pipeline`
   steps, fitted on training folds only.
4. `GridSearchCV` and the `LassoCV`-based stepwise selector receive training data
   only.
5. For forecasting, the scaler is fitted on training points only, the split is
   chronological per machine, no window crosses the split, and validation uses a
   held-out tail of the training windows rather than the test set.

The absence of leakage is established from the feature construction and the
training pipeline above, and corroborated by independent re-execution of
representative experiments and by the absence of implausibly high test scores.
Over-fitting is assessed separately: there is no evidence of substantial
over-fitting, the mean absolute train–test R² gap across the 56 regression
experiments (including the additional, established HistGradientBoosting
baseline) being approximately 0.02 with a maximum of approximately 0.07 on the
noisiest target (Alibaba CPU peak). Similar training and test performance is not
by itself treated as evidence against leakage.

## 3.16 Google 2019 Cluster temporal diagnostic

Before excluding the Google trace from the forecasting module we quantified its
temporal structure (full procedure in `TEMPORAL_DIAGNOSTIC.md`). The extract
records one row per instance-usage window rather than per machine over time; the
median machine is observed in roughly four distinct hours over the 31-day span,
so a per-machine series is not available. Aggregating all records into an hourly
cell-level series (n = 745) and computing the sample autocorrelation function,
every lag — including the 24-hour lag — lies within the ±1.96/√n white-noise band,
whereas the equivalent Alibaba `server_usage` cell series has an autocorrelation
of 0.87 at lag 1 and 0.55 at lag 24. The Google extract is therefore suitable for
cross-sectional regression but not for a defensible sequence-forecasting
experiment; no Google forecasting model is reported.

## 3.17 Reproducibility

The project (`Workload-Prediction-Rebuild/`) contains: `src/` (importable
per-algorithm modules and a shared `evaluation.py`), `notebooks/` (55
self-contained notebooks, one per experiment, each with 19 numbered steps),
`common/` (batch runners), `results/` (one CSV/JSON per experiment plus the
aggregated tables), and `documentation/`. Every notebook fixes `random_state = 42`,
reads the datasets from documented absolute paths, and writes its metrics to a
result file that `MASTER_EXPERIMENT_TABLE.csv` maps back to source. Regression
results are exactly reproducible; recurrent-network results vary by approximately
±0.05 in R² between runs because TensorFlow CPU training is not bit-for-bit
deterministic, and this is stated wherever those numbers are used.
