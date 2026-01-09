# Pair post-hoc sweep report (pairs_posthoc_sweep)

## Experiment summary

This report documents a **post-hoc threshold sweep** for the *pair seeding* stage in **Fink-FAT**.  
The goal is to tune the **angular-speed cut** used to retain candidate pairs, without re-running pair generation for each threshold.

### Dataset

- Input: `../../test_exp/ztf_dataset_2025.parquet`
- Night filter: `--nid 3122`
- Loading mode: `--mode oracle` (keep only asteroid alerts: trajectory_id > 0 or fink_class == "Solar System MPC")

### Command line

```bash
cargo run --release -p fink-fat-eval --bin pairs_posthoc_sweep -- ../../test_exp/ztf_dataset_2025.parquet --out-dir out_pairs --nid 3122 --angular-unit arcmin --gen-max-omega "35 arcmin/day" --sweep-min-omega "1 arcmin/day" --max-dt 0.1 --sweep-steps 100 --allow-same-timebin --mode oracle
```

### Configuration highlights

- Pair superset generation cut (applied once):

  \[
  \mathrm{sep}(a,b) \le \omega_{max} \; \Delta t
  \]

  with:
  - \(\omega_{max} =\) **35 arcmin/day**  
    (≈ **0.01018 rad/day**)
  - \(\Delta t \le\) **0.1 day** (≈ 2.4 hours)

- Post-hoc sweep cut (applied many times on the fixed superset):

  \[
  \mathrm{keep}(a,b;\omega) \iff \mathrm{sep}(a,b) \le \omega \; \Delta t
  \]

  with:
  - \(\omega\) swept from **1 arcmin/day** (≈ 0.000291 rad/day)
  - up to **35 arcmin/day** (the generation maximum)

- Binning / gating knobs (as used by the binary defaults unless overridden):
  - HEALPix depth: 8
  - Uniform time bin width: 0.01 day
  - `--allow-same-timebin`: enabled (pairs may come from the same time bin)

## Outputs

The binary wrote the following plots to `out_pairs/`:

- `pairs_dt_hist.png` — Δt distribution by truth label  
- `pairs_sep_hist.png` — separation distribution by truth label  
- `pairs_scatter_dt_sep.png` — Δt vs separation scatter  
- `pairs_omega_hist.png` — ω = sep / Δt histogram  
- `pairs_tradeoff_omega_threshold.png` — precision/recall vs ω threshold  
- `pairs_global_tradeoff_omega_threshold.png` — global purity/completeness vs ω threshold  
- `pairs_cost_vs_omega_threshold.png` — log10(#pairs kept) vs ω threshold

> **Note on labels:** because `--only-truth` was enabled, the `unknown` category should be empty (or near-empty).
> Any “unknown” bars you might see are typically due to plotting code always reserving a series.

## Results and interpretation

### 1) Δt distribution (`pairs_dt_hist.png`)

![Pairs: pairs dt histogram](../out_pairs/pairs_dt_hist.png)

The Δt histogram shows a **very strong spike at very small Δt**, followed by a structured distribution up to the configured maximum (**0.1 day**).

**Interpretation**
- The spike at Δt≈0 is consistent with `--allow-same-timebin`: you allow pairing detections that fall in the same discretized time bin.
- The “comb-like” structure (repeated peaks) is consistent with the survey cadence + discretization effects (time bins).

**Actionable note**
- If the goal is to reduce the combinatorics (and likely contamination), re-run with `--no-allow-same-timebin` and compare the cost/quality curves.

### 2) Separation distribution (`pairs_sep_hist.png`)

![Pairs: pairs sep histogram](../out_pairs/pairs_sep_hist.png)

Separation is strongly concentrated at **small values** for true pairs, with a visibly heavier tail for contaminated pairs.

Key qualitative patterns:
- The bulk of **true** pairs lies below ~**1.0–1.2 arcmin**.
- **Contaminated** pairs become much more prevalent in the **high-separation tail** (roughly above ~**1.4 arcmin**), extending up to ~**3 arcmin** in this dataset/night.

This is exactly what we want from a kinematic filter: large separations (given realistic Δt) are more likely to be spurious.

### 3) Δt vs separation scatter (`pairs_scatter_dt_sep.png`)

![Pairs: pairs scatter dt histogram](../out_pairs/pairs_scatter_dt_sep.png)

The scatter plot shows:
- A clear **positive envelope**: larger Δt allows larger separations.
- True pairs tend to occupy a **lower-separation band** for a given Δt.
- Contaminated pairs populate the **upper region** (large separations), especially at larger Δt.

You can visually interpret the kinematic cut \(\mathrm{sep} \le \omega\,\Delta t\) as a family of straight lines passing through the origin in this plot: lower \(\omega\) keeps only points below a steeper constraint.

### 4) Angular-speed distribution (`pairs_omega_hist.png`)

![Pairs: pairs omega histogram](../out_pairs/pairs_omega_hist.png)

With \(\omega = \mathrm{sep}/\Delta t\) (displayed in **arcmin/day**):
- The distribution peaks roughly around **12–15 arcmin/day** for true pairs.
- Contaminated pairs are more visible in the **high-ω tail** (extending towards the generation maximum at **35 arcmin/day**).

This provides an empirical justification for sweeping ω rather than a fixed separation.

### 5) Tradeoff curves (quality vs ω)

#### Precision/recall proxy (`pairs_tradeoff_omega_threshold.png`)

![Pairs: pairs tradeoff omega threshold](../out_pairs/pairs_tradeoff_omega_threshold.png)

- **Precision on truth** stays high across the sweep (**≈0.99**, slowly decreasing as ω increases).
- **Consecutive recall (proxy)** rises sharply and then saturates:
  - very low at ω ≲ 0.002 rad/day
  - steep rise around ω ≈ 0.003–0.005 rad/day
  - near-saturation beyond ω ≈ 0.006 rad/day

A practical “knee point” appears around:

- **ω ≈ 0.005 rad/day** (≈ **17.2 arcmin/day**)

At this value (from visual inspection of the curve):
- recall proxy is already around **~0.90**
- precision remains **>0.98**

#### Global purity/completeness (`pairs_global_tradeoff_omega_threshold.png`)

![Pairs: pairs global tradeoff omega threshold](../out_pairs/pairs_global_tradeoff_omega_threshold.png)

Because you ran with `--only-truth`, the global purity curve is very close to the truth-only precision curve. The same “knee” is visible.

### 6) Cost curve (`pairs_cost_vs_omega_threshold.png`)

![Pairs: pairs cost vs omega threshold](../out_pairs/pairs_cost_vs_omega_threshold.png)

The cost curve reports **log10(n_pairs_kept)** as ω increases:

- At small ω, very few pairs are kept.
- The curve rises rapidly and plateaus around **log10(n_pairs_kept) ≈ 4.78–4.80**, i.e.:

  \[
  n_{\text{pairs}} \approx 10^{4.8} \approx 6.3\times 10^4
  \]

So, going all the way to 35 arcmin/day keeps on the order of **~60k pairs** for this night and configuration.

## Recommended operating points (for this dataset/night)

Below are **reasonable** choices depending on whether you prioritize recall or cost.

### Balanced knee (recommended first candidate)

- **ω ≈ 0.005 rad/day** ≈ **17.2 arcmin/day**
- Motivation:
  - hits the recall “knee” (proxy recall ~0.9)
  - keeps precision high (>0.98)
  - cost is near the plateau (you accept most of what the current superset can provide)

### More conservative (reduce combinatorics)

- **ω ≈ 0.004 rad/day** ≈ **13.8 arcmin/day**
- Motivation:
  - substantially fewer pairs than the plateau region
  - but recall proxy is noticeably lower (curve suggests ~0.7)

### More permissive (maximize recall)

- **ω ≈ 0.006 rad/day** ≈ **20.6 arcmin/day**
- Motivation:
  - recall proxy is close to saturation (~0.95+)
  - precision decreases slightly
  - cost is essentially at the plateau already
