# Dataset methodology — Google 2019 Cluster vs Alibaba 2017

**The two traces do not have the same structure.** They are aligned only at the
level of the modelling contract (features = resource request + metadata;
targets = observed usage). Everything below marked **[fact]** is directly
supported by the dataset files or the official documentation; **[interpretation]**
is our reading of the consequences.

Sources:
* Google 2019 Cluster — Wilkes, J.: *Google cluster-usage traces v3*, Technical report, Google Inc. (2020), https://github.com/google/cluster-data; local extract `borg_traces_data.csv` (see README.md "Dataset locations" for the configurable local path).
* Alibaba 2017 — the official schema and trace documentation published at https://github.com/alibaba/clusterdata (`cluster-trace-v2017`); local extract in the `container_event.csv` / `container_usage.csv` / `server_usage.csv` files (see README.md "Dataset locations" for the configurable local path).

---

## 1. Source and data structure

| | Google 2019 Cluster | Alibaba 2017 |
|---|---|---|
| **Origin** [fact] | Public extract of the Google Borg cluster-manager trace, 2019 (trace v3). Distributed here as a single pre-processed CSV, `borg_traces_data.csv`. | Alibaba `cluster-trace-v2017` (`trace_201708`), distributed as several headerless CSVs. |
| **Native form** [fact] | v3 is published as sharded JSON tables (`instance_events`, `instance_usage`, `collection_events`, `machine_events`, …). The CSV used here is a **row-level join** of instance events + instance usage + collection metadata — one row per instance-usage measurement window. | Separate relational tables: `server_event`, `server_usage`, `container_event`, `container_usage`, `batch_task`, `batch_instance`. |
| **Files used** [fact] | `borg_traces_data.csv` only. | `container_event.csv`, `container_usage.csv`, `server_usage.csv`. (`batch_*` not used — `batch_instance.csv`, which holds actual batch usage, is not in the provided folder.) |
| **Record count used** [fact] | ≈ 405 900 instance-usage rows (34 columns). After parsing and filtering: ≈ 145 700 entities (150 000-row sample, `random_state=42`; train 116 586 / test 29 147). | Regression: ≈ 10 185 online-container instances (those with ≥ 3 usage observations and a matching `Create` event). Forecasting: 1 310 machine series from `server_usage`. |

> **[fact] Row-count note.** `wc -l borg_traces_data.csv` reports ~1.32 M lines, but
> the file has literal newlines inside quoted array fields (`cpu_usage_distribution`).
> The true record count is ≈ 405 900 (pandas parses this correctly).

---

## 2. Resource-request information

| | Google 2019 Cluster | Alibaba 2017 |
|---|---|---|
| Field(s) [fact] | `resource_request` — a dict-string `{'cpus': …, 'memory': …}` | `container_event.plan_cpu`, `plan_mem`, `plan_disk` |
| CPU request unit [fact] | **NCU** (Normalised Compute Unit): GCUs ÷ largest GCU machine capacity → range [0, 1]. 1 GCU ≈ one core on a nominal machine. | **Number of CPUs** requested per instance (integer core count, e.g. 4, 8). **Not** normalised. |
| Memory request unit [fact] | Normalised bytes: RAM ÷ largest machine memory → range [0, 1]. | Normalised memory (the trace states memory and disk are normalised; CPU core count is not). |
| Disk request [fact] | not present in this CSV | `plan_disk` (normalised) |
| Semantics [fact] | the request is the **limit** — the maximum the instance may use; tasks may exceed the CPU limit using free machine capacity, and a request of 0 is legitimate for latency-insensitive burst tasks. | `plan_cpu` / `plan_mem` are the per-instance request used by the scheduler. |

**Rebuild handling** [fact]:
* Google — `cpu_request`, `mem_request` taken directly (already in [0, 1]).
* Alibaba — `cpu_request = plan_cpu / max(plan_cpu)` (rescaled to [0, 1] for
  comparability of the feature ranges); `mem_request = plan_mem` (already
  normalised); `plan_disk` kept as a feature; `cpuset_width` = number of pinned
  CPUs, from the `|`-delimited `cpuset` string.
* Zero-request Google rows are dropped (ratio undefined).

---

## 3. CPU usage information

| | Google 2019 Cluster | Alibaba 2017 |
|---|---|---|
| Fields [fact] | `average_usage` (`{'cpus', 'memory'}`), `maximum_usage` (`{'cpus', 'memory'}`), `cpu_usage_distribution` (11 percentiles: 0, 10, …, 100), `tail_cpu_usage_distribution` (91–99 %), `random_sampled_usage` (CPU only). | `container_usage.cpu_util` — one row per 5-minute measurement window per instance. |
| Meaning [fact] | `average_usage.cpus` = mean CPU rate over the 5-minute window in **NCU-seconds/second**; `maximum_usage.cpus` = the largest 1-second sample rate; the distribution vector is the CDF of per-second CPU samples within the window. | `cpu_util` = **used percent of the instance's requested CPUs** (0–100). |
| Scale [fact] | absolute normalised usage (fraction of the largest machine) | ratio of usage to the instance's own request |

**Rebuild targets** [fact]:
* Google — `cpu_usage_mean = average_usage.cpus`, `cpu_usage_peak = maximum_usage.cpus`.
* Alibaba — `cpu_usage_mean` / `cpu_usage_peak` = mean / max of `cpu_util` over the
  instance's windows, then ÷ 100 so the target lies in [0, 1].

---

## 4. Memory usage information

| | Google 2019 Cluster | Alibaba 2017 |
|---|---|---|
| Fields [fact] | `average_usage.memory`, `maximum_usage.memory`, `assigned_memory`, `page_cache_memory` | `container_usage.mem_util` |
| Meaning [fact] | `average_usage.memory` = area under the memory-vs-time curve ÷ window duration, normalised bytes; `maximum_usage.memory` = max observed. | `mem_util` = used percent of the instance's requested memory (0–100). |

**Rebuild targets** [fact]: Google `mem_usage_mean/peak = average_usage.memory /
maximum_usage.memory`; Alibaba = mean / max of `mem_util` ÷ 100.

> **[fact] Corrected from the original.** The pre-rebuild Google memory notebooks
> predicted `memory_request`, not memory usage. The rebuild predicts usage
> (`average_usage.memory`), matching the CPU treatment and the trace spec.

---

## 5. Temporal information

| | Google 2019 Cluster | Alibaba 2017 |
|---|---|---|
| Time field [fact] | `time` (µs since 600 s before trace start), `start_time` / `end_time` of each usage window | `ts` (seconds since trace start) |
| Coverage [fact] | 31-day span; usage windows are ≈ 300 s | 12 continuous hours of usage; sampled every 60 s and averaged over 300 s |
| Machine-level series? [fact] | **No.** The CSV rows are per-*instance* usage windows. Many unrelated short-lived instances per machine; median ≈ 4 distinct hours observed per machine over 81 000 machines. | **Yes.** `server_usage.csv` = per-machine utilisation, one row per machine per 5-minute bucket → 144 points per machine, 1 310 machines. |

**[interpretation]** The Google extract is a cross-sectional sample: it records
*what each workload used*, not *how a machine's load evolved*. The Alibaba
`server_usage` table is exactly a machine-level time series. This is the structural
reason forecasting is done on Alibaba only (quantified in `TEMPORAL_DIAGNOSTIC.md`).

---

## 6. Normalisation / units — comparability

| Quantity | Google 2019 Cluster | Alibaba 2017 | Comparable? |
|---|---|---|---|
| CPU request | NCU ∈ [0, 1] | core count → rescaled to [0, 1] | ranges comparable; **absolute meaning differs** |
| Memory request | normalised bytes ∈ [0, 1] | normalised ∈ [0, 1] | comparable range |
| CPU / memory **usage** (target) | absolute normalised (fraction of largest machine) ∈ [0, 1] | **% of the instance's request** ÷ 100 ∈ [0, 1] | **not the same quantity** |

**[interpretation]** Because the two targets measure different things, **only R² is
comparable across datasets** — MAE, RMSE and sMAPE are on different scales and
must not be compared Google-vs-Alibaba. This is also the leading explanation for
the very different R² ceilings (see `RESULTS_DISCUSSION_DRAFT.md`).

---

## 7. What maps between the datasets, and what does not

### Maps [fact]

| Concept | Google | Alibaba |
|---|---|---|
| CPU request | `resource_request.cpus` | `plan_cpu` (normalised) |
| Memory request | `resource_request.memory` | `plan_mem` |
| CPU : memory request ratio | derived from the two above | derived from the two above |
| Mean CPU usage (target) | `average_usage.cpus` | mean of `cpu_util` |
| Mean memory usage (target) | `average_usage.memory` | mean of `mem_util` |
| Peak CPU / memory usage (target) | `maximum_usage.*` | max of `cpu_util` / `mem_util` |
| A coarse time index | `start_time` | `ts` |

### Does not map [fact]

| Concept | Reason |
|---|---|
| **Network I/O** | not present in Alibaba 2017 at all (the original Alibaba notebooks invented a `network_io` column) |
| **Priority, scheduling class** | Google has `priority`, `scheduling_class`; Alibaba exposes priority for **batch tasks only**, not for the online containers used here |
| **`cpu_usage_distribution` / `tail_cpu_usage_distribution`** | Google-specific pre-computed percentile vectors; no Alibaba equivalent (comparable statistics would have to be computed from each instance's own `container_usage` rows) |
| **Disk request** | Alibaba has `plan_disk`; the Google CSV does not |
| **`cpuset` width** | Alibaba-specific (pinned-CPU set); Google has placement *constraints*, not a cpuset |
| **CPI / MPKI as inputs** | Alibaba records CPI/MPKI only as usage *outputs*, not as request-time inputs |
| **Machine-level utilisation time series** | Google extract does not contain one (§5) |

**[interpretation]** The shared feature set is therefore deliberately small —
`cpu_request`, `mem_request`, `cpu_mem_ratio` plus each dataset's available static
metadata. The methodology is held constant across datasets; the *feature content*
is what each trace can support.
