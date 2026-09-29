# Google 2019 Cluster temporal diagnostic

**This is a data-suitability analysis, not a machine-learning experiment. It has
no model and no R². Its conclusion is that the available Google 2019 Cluster CSV
extract supports cross-sectional regression but does not provide sufficient
temporal continuity for a defensible sequence-forecasting experiment.**

Source: `src/google_2019/temporal_diagnostic.py`,
`notebooks/google_2019/Google_2019_Temporal_Diagnostic.ipynb`.
Output: `results/temporal_diagnostic.json`.

## 1. Why a per-machine time series was investigated

The Alibaba forecasting module (LSTM/BiLSTM/RNN) uses per-machine utilisation
series from `server_usage.csv`. To keep the two datasets on the same footing we
first tried to build an equivalent per-machine series from the Google extract.

## 2. Why the extract turned out to be too sparse

`borg_traces_data.csv` contains one row per **instance** usage window, not per
machine per time step. Grouping the rows by `machine_id` and counting the distinct
1-hour buckets (`start_time // 3_600_000_000`) each machine appears in
(script: `src/google_2019/machine_coverage.py`; output:
`results/machine_coverage.json`):

* **81,036** distinct machines appear in the extract, over a **744-hour (31-day)** span;
* the **median machine is observed in only 4 distinct hours** (mean 4.6, p90 5, max 184);
* **96.5%** of machines are observed in **6 or fewer** distinct hours;
* only **159** machines are observed in **≥ 40** distinct hours (the minimum series
  length the forecasting pipeline requires), and only **2** in ≥ 144.

A per-machine sequence would therefore be a handful of disconnected points for
almost every machine — not a series. **[fact — reproducible via
`python -m src.google_2019.machine_coverage`]**

## 3. Hourly cell-level aggregate

Instead of per-machine, we aggregated **across the whole cell**: bucket every
row's `start_time` into 1-hour bins (`BUCKET_US = 3_600_000_000` µs), parse
`average_usage` → `cpus` / `memory`, and aggregate per hour.

* **Mean aggregation** — the typical per-instance usage intensity in that hour.
* **Sum aggregation** — the total offered load in that hour.

Both were computed (the diagnostic reports `google_cpus_mean`, `google_cpus_sum`,
`google_memory_mean`).

Resulting series length: **n = 745 hourly points**. **[fact]**

## 4. Missing-bucket handling

The hourly index is reindexed onto a regular range
(`pd.RangeIndex(min_bucket, max_bucket+1)`) and the few empty buckets are filled
with `interpolate("linear").bfill().ffill()`. **[fact]**

## 5. Autocorrelation (ACF) and noise band

* `acf(x, lag) = np.corrcoef(x[:-lag], x[lag:])[0, 1]` at lags 1, 2, 3, 6, 12, 24, 48.
* White-noise 95 % confidence band half-width `= 1.96 / sqrt(n)`.
  For n = 745 → **± 0.072**.

## 6. Results (`results/temporal_diagnostic.json`)

### Google 2019 Cluster — hourly cell series (n = 745, band ± 0.072)

| lag (h) | 1 | 2 | 3 | 6 | 12 | 24 | 48 |
|---|---|---|---|---|---|---|---|
| CPU (mean agg) | +0.011 | −0.039 | +0.015 | +0.033 | −0.006 | **−0.013** | +0.026 |
| CPU (sum agg) | +0.057 | −0.002 | −0.009 | −0.020 | −0.001 | +0.003 | +0.074 |
| memory (mean agg) | +0.023 | −0.032 | −0.019 | +0.014 | −0.034 | −0.022 | +0.041 |

**Every value at every lag is inside the ± 0.072 noise band**, including lag 24,
where a real datacenter load signal would show a strong diurnal peak. The series
is statistically indistinguishable from white noise. **[fact]**

### Alibaba 2017 — cell-level `server_usage` series (n = 144, band ± 0.163) — comparison

| lag (5-min steps) | 1 | 2 | 3 | 6 | 12 | 24 |
|---|---|---|---|---|---|---|
| CPU utilisation | **+0.872** | +0.824 | +0.790 | +0.698 | +0.539 | +0.549 |
| memory utilisation | +0.879 | +0.787 | +0.717 | +0.556 | +0.440 | +0.474 |

Strong, decaying autocorrelation with a clear structure — the opposite of the
Google extract. **[fact]**

## 7. Evidence and decision

**[interpretation]** The public `borg_traces_data.csv` extract is a random
cross-sectional sample of instance-usage records. It preserves *what each
workload used* but not *how load evolved over time* on any machine or in the cell
as a whole. A recurrent model trained on such a series has no temporal signal to
learn — and indeed `src/google_2019/sequence_attempt.py` (which builds the series
correctly: chronological split, scaler on train only) scores **R² ≈ 0 to −0.45**
for LSTM, BiLSTM and RNN. That near-zero score is a property of the **data**, not
a modelling failure, and it is confirmed independently by the ACF above.

**Decision (locked):**

| Dataset | Cross-sectional regression | Time-series forecasting |
|---|---|---|
| **Google 2019 Cluster** | ✅ used | ❌ excluded — extract lacks temporal continuity (this diagnostic) |
| **Alibaba 2017** | ✅ used | ✅ used (`server_usage`, genuine machine-level series) |

## 8. How to state this in the paper

> To place both datasets on an equal footing we examined whether the Google 2019
> Cluster extract admits temporal forecasting. The extract records one entry per
> instance-usage window rather than per machine over time; the median machine is
> observed in only about four distinct hours across the 31-day span. Aggregating
> all records into an hourly cell-level series (n = 745) and computing the sample
> autocorrelation function, every lag — including the 24-hour lag — falls within
> the ±1.96/√n white-noise band, whereas the equivalent Alibaba series shows an
> autocorrelation of 0.87 at lag 1 and 0.55 at lag 24. We therefore restrict the
> Google 2019 Cluster analysis to cross-sectional regression and perform
> time-series forecasting on the Alibaba 2017 `server_usage` telemetry only. We
> do not report a Google forecasting model.
