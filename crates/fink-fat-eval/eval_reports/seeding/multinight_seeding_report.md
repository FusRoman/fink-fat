# Multi-night seeding report (pairs & triplets)

## Experiment summary

This report documents a **multi-night evaluation** of the **intra-night seeding stage**
(*pairs + triplets*) in **Fink-FAT**.

For each night (`nid`), the binary:

1. scans + ingests only that night's alerts into an `AlertStoreWithTruth`,
2. builds the spatio-temporal bucket index,
3. generates **pairs** (and **triplets**),
4. computes quality metrics against truth association,
5. records counts and timings into a CSV summary.

The goal is to assess **variability and scaling across nights**, using a **fixed**
seeding configuration.

### Dataset

- Input: `../../test_exp/ztf_dataset_2025.parquet`
- Mode: `--mode fink-truth`
- Nights processed: **87**
- Total alerts processed: **3,323,222**
- Total pairs generated: **1,676,956**
- Total triplets generated: **5,359**
  (non-zero on **36** night(s))

### Command line

```bash
cargo run --release -p fink-fat-eval --bin pairs-multinight -- \
  ../../test_exp/ztf_dataset_2025.parquet \
  --out-dir multi_nights \
  --mode fink-truth \
  --allow-same-timebin \
  --max-dt 0.1 \
  --gen-max-omega "35 arcmin/day"
```

## Configuration highlights

The key kinematic gate used by pair seeding is:

\[
\mathrm{sep}(a,b) \le \omega_{max}\, \Delta t
\]

with:

- \(\Delta t \le\) **0.1 day** (≈ 2.4 hours)
- \(\omega_{max} =\) **35 arcmin/day** (≈ 0.01018 rad/day)
- `--allow-same-timebin`: enabled (pairs may come from the same discretized time bin)

Triplets are built from the generated pairs using the default `TripletConfig`.

## Outputs

This run writes:

- `multinight_seeding_summary.csv` — per-night counts, metrics, and timings
- The following plots (embedded below):
  - `multinight_alerts_per_night.png` — alert volume per night
  - `multinight_pairs_per_alert.png` — pair combinatorics normalized by alert count
  - `multinight_triplets_per_alert.png` — triplet combinatorics normalized by alert count
  - `multinight_runtime_vs_alerts.png` — scaling of bucket-index / pairs / triplets runtime vs alerts
  - `multinight_quality_tradeoff_pairs.png` — per-night pair quality tradeoff (bubble size ~ pairs/alert)

---

## Global summary statistics

### Alert volume and combinatorics

- **Alerts per night**: min **130.000**, p10 **9.86e+03**, median **3.01e+04**, p90 **6.89e+04**, max **2.02e+05**
- **Pairs per alert**: min **0.00342**, p10 **0.107**, median **0.447**, p90 **0.817**, max **0.938**
- **Triplets per alert**: min **0.000**, p10 **0.000**, median **0.000**, p90 **0.00286**, max **0.037**

### Pair quality (truth-based)

- `pair_purity_overall`: min **0.783**, p10 **0.953**, median **0.974**, p90 **0.990**, max **1.000**
- `pair_precision_on_truth`: min **0.956**, p10 **0.971**, median **0.984**, p90 **0.997**, max **1.000**
- `pair_consecutive_recall` (completeness proxy): min **0.881**, p10 **0.928**, median **0.984**, p90 **0.997**, max **1.000**

### Timings (per night)

- `dt_bucket_ms`: min **0.021**, p10 **1.055**, median **3.361**, p90 **7.220**, max **16.356** ms
- `dt_pairs_ms`: min **0.266**, p10 **13.575**, median **38.098**, p90 **86.910**, max **160.966** ms
- `dt_triplets_ms`: min **0.019**, p10 **4.570**, median **16.900**, p90 **51.069**, max **61.190** ms
- **Total runtime** (`bucket+pairs+triplets`): min **0.305**, p10 **19.135**, median **60.405**, p90 **146.222**, max **203.309** ms

**Scaling hint (log-log correlation):**
- corr(log10(alerts), log10(total_runtime_ms)) ≈ **0.988**
- corr(log10(pairs),  log10(total_runtime_ms)) ≈ **0.941**

> These correlations are not a model fit, but they match what is visible in the runtime scatter plot:
> runtime increases strongly with alert volume / pair count, and shows a saturating trend at high counts.

---

## Results and interpretation (plots)

### 1) Alert volume per night (`multinight_alerts_per_night.png`)

![Multi-night: alerts per night](../../multi_nights/multinight_alerts_per_night.png)

Alert volume varies substantially across nights (from a few hundred alerts to ~2×10⁵).
This variability directly drives both **runtime** and **pair/triplet combinatorics**.

**Interpretation**
- Nights with very low alert counts tend to be dominated by fixed overheads (I/O + indexing).
- High-volume nights are the stress test for combinatorics and runtime.

### 2) Pairs per alert (`multinight_pairs_per_alert.png`)

![Multi-night: pairs per alert](../../multi_nights/multinight_pairs_per_alert.png)

Pairs per alert is a practical proxy for **combinatorial cost**.
The distribution spans a wide range, with a central band around ~0.5–0.9 pairs/alert,
and noticeably smaller values on low-volume nights.

**Interpretation**
- Higher pairs/alert often indicates denser sky regions, higher source density, or cadence/time-bin effects.
- `--allow-same-timebin` tends to increase short-Δt pairings (and thus combinatorics).

### 3) Triplets per alert (`multinight_triplets_per_alert.png`)

![Multi-night: triplets per alert](../../multi_nights/multinight_triplets_per_alert.png)

Triplets are much rarer than pairs (typically ≤ a few percent per alert), and many nights have **zero** triplets
under this configuration (non-zero on **36** nights).

**Interpretation**
- Requiring 3 temporally-consistent detections is a strong constraint.
- In survey cadence regimes with limited revisits inside \(\Delta t\le 0.1\) day, triplets are expected to be sparse.

**Triplet-only quality (nights with triplets)**
- triplets/alert: min **5.81e-05**, p10 **0.000146**, median **0.00174**, p90 **0.014**, max **0.037**
- triplet_purity_overall: min **0.750**, p10 **0.875**, median **0.956**, p90 **1.000**, max **1.000**
- triplet_precision_on_truth: min **0.750**, p10 **0.875**, median **0.973**, p90 **1.000**, max **1.000**
- triplet_consecutive_recall: min **0.000152**, p10 **0.000889**, median **0.00535**, p90 **0.066**, max **0.111**

> As expected, triplets tend to be **high-purity** when they exist, but the **recall proxy is low**
> because many true objects simply do not have 3 usable detections within the configured intra-night window.

### 4) Runtime scaling (`multinight_runtime_vs_alerts.png`)

![Multi-night: runtime vs alerts](../../multi_nights/multinight_runtime_vs_alerts.png)

The plot shows per-night runtime for:
- bucket index build,
- pair generation,
- triplet generation,

as a function of the number of alerts.

**Interpretation**
- Bucket indexing shows an increasing trend with alerts, with relatively low scatter.
- Pair and triplet generation show stronger growth and more variance, consistent with
  data-dependent neighbor counts and gating outcomes.

### 5) Pair quality tradeoff (`multinight_quality_tradeoff_pairs.png`)

![Multi-night: pair quality tradeoff](../../multi_nights/multinight_quality_tradeoff_pairs.png)

Each bubble is a night:
- x-axis: `pair_consecutive_recall` (completeness proxy)
- y-axis: `pair_purity_overall`
- bubble size: proportional to **pairs per alert**

**Interpretation**
- Most nights cluster at **high purity (≈0.95–1.0)** and **high recall proxy (≈0.9–1.0)**.
- A small number of nights show a clearer tradeoff (lower purity and/or recall), often corresponding
  to higher combinatorics (larger bubbles) or unusual cadence.

---

## Outliers (useful for debugging)

### Highest pairs per alert

|   nid |   n_alerts |   n_pairs |   pairs_per_alert |   pair_purity_overall |   pair_precision_on_truth |   pair_consecutive_recall |   total_runtime_ms |
|------:|-----------:|----------:|------------------:|----------------------:|--------------------------:|--------------------------:|-------------------:|
|  3132 |      73655 |     69106 |            0.9382 |                0.9735 |                    0.9744 |                    0.987  |             156.23 |
|  3122 |      68538 |     63514 |            0.9267 |                0.9717 |                    0.974  |                    0.9974 |             149.53 |
|  3133 |      82179 |     75298 |            0.9163 |                0.9709 |                    0.9716 |                    0.9637 |             161.55 |
|  3131 |      71377 |     64857 |            0.9087 |                0.9803 |                    0.9813 |                    0.9825 |             149.18 |
|  3121 |      59935 |     53012 |            0.8845 |                0.9743 |                    0.978  |                    0.9952 |             127.21 |
|  3124 |      66010 |     57218 |            0.8668 |                0.9694 |                    0.972  |                    0.9984 |             139.39 |

### Lowest pair purity overall

|   nid |   n_alerts |   n_pairs |   pairs_per_alert |   pair_purity_overall |   pair_precision_on_truth |   pair_consecutive_recall |   total_runtime_ms |
|------:|-----------:|----------:|------------------:|----------------------:|--------------------------:|--------------------------:|-------------------:|
|  3082 |      17188 |      2178 |            0.1267 |                0.7833 |                    0.9988 |                    0.9301 |              33.92 |
|  3074 |      38061 |       934 |            0.0245 |                0.8469 |                    0.9635 |                    0.9817 |              35.16 |
|  3093 |      44728 |      6782 |            0.1516 |                0.8502 |                    0.9826 |                    0.9911 |              87.31 |
|  3087 |      14210 |      2429 |            0.1709 |                0.86   |                    0.9971 |                    0.9575 |              24.43 |
|  3102 |      11556 |      2088 |            0.1807 |                0.863  |                    0.9907 |                    0.9876 |              21.1  |
|  3156 |     202165 |     10722 |            0.053  |                0.8632 |                    0.9753 |                    0.9884 |             203.31 |

### Highest total runtime

|   nid |   n_alerts |   n_pairs |   pairs_per_alert |   pair_purity_overall |   pair_precision_on_truth |   pair_consecutive_recall |   total_runtime_ms |
|------:|-----------:|----------:|------------------:|----------------------:|--------------------------:|--------------------------:|-------------------:|
|  3156 |     202165 |     10722 |            0.053  |                0.8632 |                    0.9753 |                    0.9884 |             203.31 |
|  3157 |      92512 |     51014 |            0.5514 |                0.9731 |                    0.9773 |                    0.9976 |             172    |
|  3133 |      82179 |     75298 |            0.9163 |                0.9709 |                    0.9716 |                    0.9637 |             161.55 |
|  3160 |      72542 |     35448 |            0.4887 |                0.9676 |                    0.9717 |                    0.9942 |             157.78 |
|  3132 |      73655 |     69106 |            0.9382 |                0.9735 |                    0.9744 |                    0.987  |             156.23 |
|  3163 |      66622 |     35366 |            0.5308 |                0.9816 |                    0.9825 |                    0.9962 |             153.38 |
