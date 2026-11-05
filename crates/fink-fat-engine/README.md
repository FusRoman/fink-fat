# fink-fat-engine

Core library of Fink-FAT. It turns a stream of photometric detections into
multi-night candidate trajectories of Solar System objects. The approach is
**angles-only multi-hypothesis tracking**:

- every intra-night tracklet seeds a bank of topocentric Kalman filters that
  covers the unknown range / range-rate;
- every lineage branches on the candidate detections of the following
  exposures;
- branches compete through a log-likelihood ratio against the local alert
  density.

The crate does no I/O of its own beyond reading and writing its snapshot and
archive files. The CLI (`fink-fat`), the evaluation tools (`fink-fat-eval`)
and the web explorer (`fink-fat-explorer`) all sit on top of it. See the
[workspace README](../../README.md) for how to run the pipeline.

> Mathematical notation below follows the code. The rustdoc of each module
> (`cargo doc -p fink-fat-engine`, with the KaTeX header) gives the full
> derivations and the reasons behind each design choice. Every symbol is
> explained in the [Notation](#notation) section.

## Contents

1. [Notation](#notation)
2. [Pipeline at a glance](#pipeline-at-a-glance)
3. [Data model](#data-model)
4. [Intra-night seeding: the tracklet linker](#intra-night-seeding-the-tracklet-linker)
5. [Admissible-region seed grid](#admissible-region-seed-grid)
6. [The topocentric Kalman filter](#the-topocentric-kalman-filter)
7. [The Kalman filter bank](#the-kalman-filter-bank)
8. [Predicted search region](#predicted-search-region)
9. [Night-to-night multi-hypothesis tracking](#night-to-night-multi-hypothesis-tracking)
10. [Fragment linkage](#fragment-linkage)
11. [Persistence](#persistence)
12. [Configuration](#configuration)
13. [Logging](#logging)
14. [Parallelism and determinism](#parallelism-and-determinism)
15. [Module map](#module-map)
16. [Status and known limitations](#status-and-known-limitations)

---

## Notation

The tables below explain every symbol used in the equations of this README,
grouped by the section where it first appears. Each section with equations
links back to its table.

**Conventions**

- Bold lower-case letters ($\mathbf r$, $\mathbf v$, $\mathbf x$) are vectors;
  a hat ($\hat{\mathbf u}$) marks a unit vector.
- A dot ($\dot\rho$, $\dot\alpha$) is a time derivative.
- $^\top$ is the transpose, $I$ the identity matrix ($I_2$: 2×2) and
  $\mathrm{diag}(\dots)$ a diagonal matrix.
- Superscripts $^-$ / $^+$ mean "before" / "after" the Kalman update of the
  current observation.
- $\Delta t$ is always a time difference in days (TT).
- Index $i$ runs over the hypotheses of a bank, index $p$ over the orbital
  populations of the prior.

> **Overloaded symbols.** The notation follows the code and astronomy usage,
> so a few letters have two meanings. The context always makes it clear:
>
> | Symbol | Meaning 1 | Meaning 2 |
> | --- | --- | --- |
> | $\alpha$ | right ascension | phase angle (Sun–object–observer), in $\Phi_{HG}(\alpha)$; also the EMA factor $\alpha = 0.4$ |
> | $H$ | measurement matrix $[I_2 \mid 0]$ (Kalman filter) | absolute magnitude (photometry) |
> | $\Phi$ | state transition matrix (Kalman prediction) | H-G phase function $\Phi_{HG}$ |
> | $r$ | heliocentric distance $\lvert\mathbf r\rvert$ | search radius (tracklet linker, search region) |
> | $\lambda$ | covariance inflation factor | $\lambda_\mathrm{max}$: largest eigenvalue; $\lambda_\mathrm{clutter}$: alert density |
> | $w$ | weight ($w_i$, $w_p$, $w(a)$) | width of the completeness curve (mag) |
> | $\Delta$ | topocentric distance (= $\rho$) | $\Delta t$, and the LLR increments $\Delta_\mathrm{obs}$, $\Delta_\mathrm{phot}$, $\Delta_\mathrm{null}$ |

### Notation: sky positions and tracklet linker

| Symbol | Meaning | Unit |
| --- | --- | --- |
| $\alpha, \delta$ | right ascension, declination (equatorial J2000) | rad |
| $\sigma_\alpha, \sigma_\delta$ | 1-σ astrometric errors of an alert on $\alpha$ and $\delta$ | rad |
| $m_\mathrm{obs}$, $m_\mathrm{last}$ | apparent magnitude of the new observation / of the track's last detection | mag |
| $\dot\theta_\mathrm{max}$ | maximum angular speed on the sky (`max_angular_speed`) | rad/day |
| $\Delta t_\mathrm{max}$ | maximum time gap between two detections of a track (`max_dt`) | day |
| $\texttt{sep\_cap}$ | largest angular distance an object can travel in $\Delta t_\mathrm{max}$ | rad |
| $r_\mathrm{cell}$ | radius of one HEALPix cell of the spatial index | rad |
| $r$ | search radius around an observation | rad |
| $\alpha_0, \delta_0$ | position of the track's first observation (tangent-plane origin) | rad |
| $\mathrm{wrap}(\cdot)$ | brings an angle back into $(-\pi, \pi]$ (handles the RA 0/2π jump) | — |
| $x, y$ | flat tangent-plane coordinates relative to $(\alpha_0, \delta_0)$ | rad |
| $\hat p(t)$ | position predicted at time $t$ by the track's linear fit | — |
| $p_\mathrm{obs}$ | position of the candidate observation | — |
| $\mathrm{sep}(\cdot,\cdot)$ | angular separation between two sky positions | rad |
| $\sigma^2_\mathrm{pred}(t)$ | variance of the fit's prediction at time $t$ | rad² |
| $\sigma^2_\mathrm{obs}$ | variance of one observation, $(\sigma_\alpha\cos\delta)^2 + \sigma_\delta^2$ | rad² |
| $\bar\sigma^2_\mathrm{obs}$ | mean $\sigma^2_\mathrm{obs}$ over the track's points | rad² |
| $n$ | number of points in the track | — |
| $t$, $\bar t$ | observation time, mean time of the track's points | day |
| $S_{xx}$ | spread of the times, $\sum_k (t_k - \bar t)^2$ | day² |

### Notation: orbital geometry and admissible region

| Symbol | Meaning | Unit |
| --- | --- | --- |
| $\rho$, $\dot\rho$ | topocentric range (observer–object distance) and its rate | au, au/day |
| $\hat{\mathbf u}$, $\dot{\hat{\mathbf u}}$ | unit line-of-sight vector from the observer to the object, and its time derivative | —, 1/day |
| $\mathbf r_\mathrm{obs}$, $\mathbf v_\mathrm{obs}$ | heliocentric position and velocity of the observer | au, au/day |
| $\mathbf r$, $\mathbf v$ | heliocentric position and velocity of the object | au, au/day |
| $r$ | heliocentric distance of the object, $\lvert\mathbf r\rvert$ | au |
| $k$ | Gaussian gravitational constant ($k^2 = GM_\odot$) | au$^{3/2}$/day |
| $\varepsilon$ | specific orbital energy; the orbit is bound (elliptic) when $\varepsilon < 0$ | au²/day² |
| $\mathbf v_T$ | part of $\mathbf v$ that does not depend on $\dot\rho$ | au/day |
| $D$ | discriminant of the quadratic in $\dot\rho$; no bound orbit when $D \le 0$ | au²/day² |
| $a$ | semi-major axis of the orbit, $-k^2/(2\varepsilon)$ | au |
| $a_p$, $\sigma_p$, $w_p$ | centre, width and weight of population $p$ in the prior | au, au, — |
| $w(a)$ | prior weight given to a grid node of semi-major axis $a$ | — |
| $\sigma_\rho$, $\sigma_{\dot\rho}$ | initial 1-σ uncertainty on $\rho$ and $\dot\rho$ | au, au/day |

### Notation: Kalman filter and bank

| Symbol | Meaning | Unit |
| --- | --- | --- |
| $\mathbf x$ | state (attributable) vector $(\alpha, \delta, \dot\alpha, \dot\delta, \rho, \dot\rho)$ | mixed |
| $P$ | 6×6 state covariance ($P^-$ predicted, $P^+$ updated) | mixed |
| $\tau$ | light time, travel time of light from the object to the observer | day |
| $c$ | speed of light | au/day |
| $J$ | 6×6 Jacobian of the attributable → Cartesian conversion; $J_\mathrm{old}$ at the start epoch, $J_\mathrm{new}$ at the target epoch | — |
| $\Phi$ | state transition matrix (STM) of the two-body motion | — |
| $f, g$, $\dot f, \dot g$ | Lagrange coefficients and their time derivatives | —, day, 1/day, — |
| $Q$ | process-noise covariance (models the unmodelled accelerations) | — |
| $q_0$ | base process-noise intensity | au²/day³ |
| $q_\mathrm{eff}$ | intensity actually used, grows with $\Delta t$ | au²/day³ |
| $\Delta t_\mathrm{ref}$ | time scale of that growth | day |
| $\lambda$ | fading-memory inflation factor applied to the predicted covariance | — |
| $\mathrm{NIS}$ | normalised innovation squared, $\nu^\top S^{-1}\nu$ (≈ 2 on average for a well-tuned filter) | — |
| $\mathrm{NIS}_\mathrm{EMA}$ | exponential moving average of the NIS; 5.991 is the 95 % χ² quantile with 2 degrees of freedom | — |
| $z$ | measured position $(\alpha, \delta)$ of the alert | rad |
| $H$ | measurement matrix $[I_2 \mid 0]$: picks $(\alpha, \delta)$ out of $\mathbf x$ | — |
| $R$ | measurement-noise covariance $\mathrm{diag}(\sigma_\alpha^2, \sigma_\delta^2)$ | rad² |
| $\nu$ | innovation: measured minus predicted position | rad |
| $S$ | innovation covariance (uncertainty on $\nu$) | rad² |
| $K$ | Kalman gain | — |
| $d^2$ | squared Mahalanobis distance of the observation, $\nu^\top S^{-1}\nu$ | — |
| $\ell$ | log-likelihood of the observation for one hypothesis | — |
| $w_i$ | weight of hypothesis $i$ in the bank (they sum to 1) | — |
| $\bar\ell_h$ | mean of $\ell$ over the sliding window for hypothesis $h$ | — |
| $n_\mathrm{steps}$ | number of observations the bank has absorbed | — |

### Notation: photometry, search region and MHT scoring

| Symbol | Meaning | Unit |
| --- | --- | --- |
| $m$ | apparent magnitude of a detection | mag |
| $H$ | absolute magnitude of the object | mag |
| $\Delta$ | topocentric distance (equal to $\rho$) | au |
| $\alpha$ | phase angle Sun–object–observer | rad |
| $\Phi_{HG}(\alpha)$, $G$ | H-G phase function and its slope parameter | mag, — |
| $P_i$ | covariance of hypothesis $i$ predicted to the visit epoch | — |
| $S_i$ | sky-plane covariance of hypothesis $i$ | rad² |
| $R_\mathrm{search}$ | measurement noise added when building the search region | rad² |
| $\delta_i$ | offset of hypothesis $i$'s predicted position from the mixture mean | rad |
| $S_\mathrm{mix}$ | covariance of the whole mixture (spread of each component + spread between them) | rad² |
| $\lambda_\mathrm{max}(\cdot)$ | largest eigenvalue of a matrix (semi-major axis² of the ellipse) | rad² |
| $\chi^2$ | χ² threshold setting the size of the region (`search_region_chi2`) | — |
| $L(z)$ | predictive mixture likelihood of the bank at the candidate position $z$ | 1/sr |
| $\lambda_\mathrm{clutter}$ | local density of alerts (false-alarm density) | 1/sr |
| $m_\mathrm{pred}$ | apparent magnitude predicted from $H$ and the geometry | mag |
| $\sigma_\mathrm{mag}$ | assumed 1-σ spread of $m_\mathrm{obs} - m_\mathrm{pred}$ (`photometric_sigma_mag`) | mag |
| $P_D$ | probability that the object is detected in the visit | — |
| $m_\mathrm{lim}$, $w$ | magnitude of 50 % completeness and width of the completeness curve | mag |
| $\Delta_\mathrm{obs}$, $\Delta_\mathrm{phot}$, $\Delta_\mathrm{null}$ | LLR increments for an astrometric match, the photometric match, and a missed detection | — |

---


## Pipeline at a glance

One call, `BranchCollection::advance_one_night`, runs a whole night. Night 0
and night N use the same code path: when the collection is empty, the update
phase is simply skipped.

```text
               night N observations (ObsDataset)
                              │
                ┌─────────────┴──────────────┐
                ▼                            │
     group into visits (exposures)           │
                │                            │
   ┌────────────▼─────────────┐              │
   │ for each visit, for each │  (rayon)     │
   │ live lineage:            │              │
   │  quick HEALPix prefilter │              │
   │  predict bank → region   │              │
   │  candidate search + gate │              │
   │  branch: 1 per candidate │              │
   │          + 1 null branch │              │
   │  LLR scoring             │              │
   └────────────┬─────────────┘              │
                ▼                            │
       top-B cap per lineage                 │
                │  (after every visit)       │
                ▼                            │
       N-scan pruning (once per night)       │
                ▼                            │
       stale-lineage purge → archive         │
                ▼                            ▼
       unclaimed observations ──► tracklet linker ──► new KF banks
                │                                         │
                └──────────────► BranchCollection ◄───────┘
```

## Data model

| Concept | Type | Description |
| --- | --- | --- |
| Observation | `photom::observation_dataset::ObsDataset` | Detections with RA/Dec and their errors (rad), MJD TT, magnitude, band and observer. Provided by the [`photom`](https://crates.io/crates/photom) crate. |
| Pair | `seeding::pairs::Pair` | Two references to observations of the same night: the first (`a`) and the last (`b`) member of a tracklet. Carries no state of its own. |
| Filter state | `topocentric_kf::single_kalman::KFState` | One topocentric Kalman filter: 6-D attributable state, covariance, epoch, observer state, NIS EMA. `KFStateSnapshot` is its rkyv form. |
| Hypothesis | `topocentric_kf::kalman_bank::hypothesis::Hypothesis` | One `KFState` plus a log-weight and a sliding window of recent log-likelihoods. |
| Bank | `topocentric_kf::kalman_bank::KFBank` | A population of hypotheses that share **one** association history (`track_ids`), plus a running absolute-magnitude estimate. |
| Branch | `topocentric_kf::branching::Branch` | A full `KFBank` clone with one specific association history, its cumulative LLR, and its ancestry (`lineage_id`, `branch_id`, `parent_branch_id`, N-scan anchor, last real update). |
| Lineage | `lineage_id: u64` | Identity of a candidate object, inherited from the tracklet that seeded it. Several branches can share a lineage. |
| Collection | `topocentric_kf::branching::BranchCollection` | All live branches, the night counter, the IDs consumed then pruned last night, and the trajectories archived during the current night. |
| Archived trajectory | `topocentric_kf::branching::archive::ArchivedTrajectory` | Result of a lineage that stopped being propagated: designation, `track_ids`, cumulative LLR, MAP state, H estimate. |

**Why branch at the bank level?** All hypotheses inside a bank must agree on
the observations they have consumed. The bank resolves the `(ρ, ρ̇)`
ambiguity against a fixed history. A branch decides *which* observation that
history contains. Branching below the bank level would break the bank's
invariant.

**Designations.** Designations are deterministic and built from the content of
the track:
- Lineage: `FF{YYYY}{12 letters}`. The letters are a seeded `ahash` of the
  `track_ids`, and `YYYY` is the TT year.
- Branch: `{lineage}-{8 letters}`.

The hash seeds are frozen constants in `branching/branch_id.rs`. Changing them
would rename every existing object. Numeric `branch_id`s come from an atomic
counter, so they depend on thread scheduling. They are never used for
ordering.

## Intra-night seeding: the tracklet linker

*Symbols: see [Notation](#notation-sky-positions-and-tracklet-linker).*

`seeding::tracklet_linker::link_tracklets` links each object's intra-night
detections into a track. It works as a **time-ordered sweep**, which avoids
enumerating all pairs (`O(n²)` on a night with many revisits):

1. Sort the night's observations by `mjd_tt`. Index the active tracks by the
   HEALPix cell of their last detection.
2. For each observation, look up the active tracks in the neighbouring cells
   within $r = \texttt{sep\_cap} + r_\mathrm{cell}$, where
   $\texttt{sep\_cap} = \dot\theta_\mathrm{max} \cdot \Delta t_\mathrm{max}$.
   A track is a candidate only if its last detection is less than `max_dt`
   old.
3. Common gate: $\Delta t > 0$ and $|m_\mathrm{obs} - m_\mathrm{last}| \le$
   `max_mag_difference`.
4. Score each candidate track:
   - **Fitted** (track with ≥ 2 points): the track carries an incremental
     least-squares linear fit on the tangent plane, with
     $x = \mathrm{wrap}(\alpha - \alpha_0)\cos\delta_0$ and
     $y = \delta - \delta_0$. The candidate is accepted if
     ```math
     \frac{\mathrm{sep}\big(\hat p(t),\, p_\mathrm{obs}\big)}{\sqrt{\sigma^2_\mathrm{pred}(t) + \sigma^2_\mathrm{obs}}} \le \texttt{tracklet\_residual\_sigma}
     ```
     The prediction variance follows the usual linear-regression form
     $\bar\sigma^2_\mathrm{obs}\left(\tfrac1n + \tfrac{(t-\bar t)^2}{S_{xx}}\right)$.
     The observation variance is $(\sigma_\alpha\cos\delta)^2 + \sigma_\delta^2$.
   - **Fallback** (single-point track): the angular rate
     $\mathrm{sep}/\Delta t$ must not exceed `max_angular_speed`.

   A fitted match always beats a fallback match. Among fitted matches the
   smallest normalised residual wins. Among fallback matches the smallest
   $\Delta t$ wins. Each observation joins at most one track. If none
   matches, it opens a new track.
5. Every track with at least 2 members yields one **maximal-baseline pair**
   (first, last). Singleton tracks are dropped.

Each pair then seeds one bank through `KFBank::from_grid`
(`kalman_bank::from_seeds::build_kf_bank_collection_from_observations`). A
bank that fails to build is logged and skipped.

`seeding::pairs::generate_pairs` (pairwise time / rate / magnitude gate) and
`seeding::triplets` remain available as standalone tools. The production
pipeline does not use them.

## Admissible-region seed grid

*Symbols: see [Notation](#notation-orbital-geometry-and-admissible-region).*

A tracklet constrains $(\alpha, \delta, \dot\alpha, \dot\delta)$ but not the
topocentric range $\rho$ nor the range-rate $\dot\rho$. Instead of guessing
one value, `kalman_bank::seed_grid::admissible_region_grid` samples the
**admissible region**, the set of $(\rho, \dot\rho)$ values that give a bound
heliocentric orbit:

- **ρ nodes**: `n_rho` values spaced logarithmically in
  `[rho_min, rho_max]`. Defaults: 28 nodes in 0.1 – 60 au.
- **ρ̇ interval**: for each ρ, the orbit is bound
  ($\varepsilon = \tfrac12|\mathbf v|^2 - k^2/r < 0$) only for $\dot\rho$
  between the two roots of a quadratic:
  ```math
  \dot\rho \in \Big[-\mathbf v_T\!\cdot\hat{\mathbf u} - \sqrt{D},\; -\mathbf v_T\!\cdot\hat{\mathbf u} + \sqrt{D}\Big],\qquad D = (\mathbf v_T\!\cdot\hat{\mathbf u})^2 - \Big(|\mathbf v_T|^2 - \frac{2k^2}{r}\Big)
  ```
  Here $\mathbf v_T = \mathbf v_\mathrm{obs} + \rho\,\dot{\hat{\mathbf u}}$.
  A ρ with $D \le 0$ is skipped. `n_rho_dot` (default 11) nodes are placed at
  the centres of equal sub-intervals.
- **Population prior**: every node with $\varepsilon < 0$ gets its semi-major
  axis $a = -k^2/(2\varepsilon)$ and a weight from a Gaussian mixture in $a$:
  ```math
  w(a) = \sum_p w_p \exp\!\Big[-\tfrac12\Big(\frac{a - a_p}{\sigma_p}\Big)^2\Big]
  ```
  Nodes below `weight_floor` (1e-6) are dropped.
- **Initial covariance**: diagonal in attributable space. The angular blocks
  come from the pair's astrometric errors. $\sigma_\rho$ is the half-width of
  the log cell and $\sigma_{\dot\rho}$ is half the ρ̇ step, with floors
  `sigma_pos_au_floor` and `sigma_rho_dot_floor`.

Default populations (`engine_config::grid_population::default_populations`):

| Population | $a_p$ (au) | $\sigma_p$ (au) | $w_p$ |
| --- | --- | --- | --- |
| NEO | 1.50 | 0.60 | 0.02 |
| MBA inner / middle / outer | 2.20 / 2.70 / 3.10 | 0.15 | 0.60 / 1.00 / 0.70 |
| Cybele | 3.42 | 0.10 | 0.04 |
| Hilda | 3.95 | 0.15 | 0.05 |
| Trojan | 5.20 | 0.30 | 0.10 |
| Centaur | 15.0 | 5.0 | 0.03 |
| KBO cold / hot / resonant | 44.0 / 45.0 / 39.4 | 2.0 / 8.0 / 1.0 | 0.04 / 0.03 / 0.02 |
| SDO | 70.0 | 20.0 | 0.02 |
| Detached | 90.0 | 10.0 | 0.01 |

The populations are configurable. The prior is broad on purpose: rare
populations (NEOs, Centaurs, TNOs) keep live hypotheses until the data rules
them out.

## The topocentric Kalman filter

*Symbols: see [Notation](#notation-kalman-filter-and-bank).*

`topocentric_kf::single_kalman`.

**State.** The state is the attributable vector
$\mathbf x = (\alpha, \delta, \dot\alpha, \dot\delta, \rho, \dot\rho)$ in
equatorial J2000, with angles in rad, rates in rad/day (not multiplied by
$\cos\delta$), $\rho$ in au and $\dot\rho$ in au/day. The heliocentric
Cartesian state is
$\mathbf r = \mathbf r_\mathrm{obs} + \rho\hat{\mathbf u}$ and
$\mathbf v = \mathbf v_\mathrm{obs} + \dot\rho\hat{\mathbf u} + \rho\dot{\hat{\mathbf u}}$
(`conversion.rs`, with the analytic 6×6 Jacobian).

**Observer.** The `KalmanContext` loads the JPL ephemeris (default
`naif:DE440`) and the UT1 / EOP data at start-up. It downloads them if they
are missing. For each observation it gives the observer's heliocentric state,
built through `outfit`'s observer cache (`observer_state.rs`).

**Prediction** (`propagate.rs`):
- Two-body Keplerian propagation with universal variables
  (`outfit::kepler::propagate_universal`), warm-started from the previous
  universal anomaly.
- **Light-time correction**: the stored angles are apparent, so the arc runs
  from emission time to emission time. The light time
  $\tau = |\mathbf r - \mathbf r_\mathrm{obs}|/c$ is refined by a bounded
  fixed-point iteration (at most 5 iterations, tolerance 1e-9 day).
  Annual aberration is not modelled.
- The covariance is moved to Cartesian space through the Jacobian $J$,
  propagated with the Lagrange-coefficient STM
  $\Phi = \left(\begin{smallmatrix} f I & g I \\ \dot f I & \dot g I\end{smallmatrix}\right)$,
  then mapped back:
  ```math
  P^- = J_\mathrm{new}^{-1}\big(\lambda\,\Phi\,J_\mathrm{old}PJ_\mathrm{old}^\top\Phi^\top + Q\big)J_\mathrm{new}^{-\top}
  ```
- **Process noise**: state noise compensation (SNC) with
  $q_\mathrm{eff} = q_0\,(1 + (\Delta t/\Delta t_\mathrm{ref})^2)$ and
  $Q = q_\mathrm{eff}\left(\begin{smallmatrix}\Delta t^3/3\,I & \Delta t^2/2\,I\\ \Delta t^2/2\,I & \Delta t\,I\end{smallmatrix}\right)$.
  Defaults: $q_0 = 10^{-16}$ au²/day³, $\Delta t_\mathrm{ref} = 1$ day.
- **Fading-memory inflation**: $\lambda = \mathrm{clamp}(\mathrm{NIS}_\mathrm{EMA}/5.991,\,1,\,5)$.
  The filter widens its covariance when its recent innovations are larger
  than expected.
- If $\rho$ leaves $[10^{-4}, 10^{6}]$ au, the propagation is rejected as
  ill-conditioned.

**Update** (`update.rs`). The measurement is linear in the state:
$H = [I_2 \mid 0]$ and $R = \mathrm{diag}(\sigma_\alpha^2, \sigma_\delta^2)$,
taken from the alert's astrometric errors.

```math
\nu = z - H\mathbf x^-\ (\text{RA wrapped}),\quad S = HP^-H^\top + R,\quad K = P^-H^\top S^{-1}
```
```math
\mathbf x^+ = \mathbf x^- + K\nu,\qquad P^+ = (I-KH)P^-(I-KH)^\top + KRK^\top \ \ \text{(Joseph form)}
```

Numerical safeguards:
- $S$ is symmetrised, then regularised with an escalating jitter when it is
  near-singular.
- A floor keeps the angular variances of $P^+$ from collapsing below a
  fraction of $R$ (`min_angular_variance_ratio`).
- $\mathrm{NIS} = \nu^\top S^{-1}\nu$ feeds an EMA ($\alpha = 0.4$) that
  drives the inflation above.

## The Kalman filter bank

*Symbols: see [Notation](#notation-kalman-filter-and-bank).*

`topocentric_kf::kalman_bank::KFBank`. One bank holds all the $(\rho,\dot\rho)$
hypotheses of one association history. At each step (`step`):

1. **Predict** every hypothesis to the observation epoch.
2. **Gate**: compute $d^2 = \nu^\top S^{-1}\nu$ and reject the hypothesis if
   $d^2 >$ `gate_chi2` (default 23.0). The `min_hypotheses` best hypotheses
   (default 5) are **protected** from this gate. The gate's job is to thin a
   large population, not to kill the best surviving estimate at the moment
   it most needs to adapt. Non-finite $d^2$ is always rejected.
3. **Weight**: Bayesian update in log space with the predictive likelihood,
   $\ell = -\tfrac12(\ln\det S + d^2)$, so $\log w_i \leftarrow \log w_i + \ell_i$.
   Weights are then normalised with log-sum-exp. Each $\ell$ also enters a
   sliding window of length `likelihood_window` (default 3).
4. **Kalman update** of every surviving hypothesis.
5. **Cleanup**:
   - *Smoothed pruning*: a hypothesis is dropped when the mean log-likelihood
     over its window falls below $\max_h \bar\ell_h + \ln(\texttt{weight\_floor})$.
     Averaging over the window means one bad epoch does not kill an
     otherwise consistent hypothesis.
   - *Scheduled cap*: keep the top
     $\max(\mathrm{cap}(n_\mathrm{steps}),\ \texttt{min\_hypotheses})$
     hypotheses by weight. `HypothesisCapSchedule` is one of `Fixed` (default
     1000), `Linear`, `Logarithmic` or `Exponential` decay with the number of
     observations.
   - *Mode merging*: hypotheses whose Cartesian positions are closer than
     `merge_position_au` (default 0.02 au) are merged by moment matching.

The bank also keeps a **running absolute-magnitude estimate**. After each real
association it computes $H = m - 5\log_{10}(r\Delta) - \Phi_{HG}(\alpha)$
from the MAP hypothesis ($r$ heliocentric distance, $\Delta = \rho$, H-G
phase function with $G$ = `slope_parameter_g`, default 0.15). This estimate
drives the detection probability and the photometric score described below.

## Predicted search region

*Symbols: see [Notation](#notation-photometry-search-region-and-mht-scoring).*

`kalman_bank::ellipse_region_finder`. To find where a bank expects its next
detection, `predict_to` propagates it read-only to the visit epoch. A subset
of the predicted mixture is kept (`top_k`, default
`WeightThreshold(0.99)`: the smallest set of hypotheses holding 99 % of the
weight). Each kept hypothesis becomes a 2-D Gaussian component
$S_i = HP_iH^\top + R_\mathrm{search}$.

- **Centre**: the sky position of the MAP hypothesis.
- **Radius** (`RadiusStrategy`):
  - `MixtureCovariance`:
    $r = \sqrt{\chi^2\,\lambda_\mathrm{max}(S_\mathrm{mix})}$ with
    $S_\mathrm{mix} = \sum w_i(S_i + \delta_i\delta_i^\top)$.
  - `MaxEllipse`: the largest component ellipse plus its offset from the mean.
  - `Clamped`: either of the above, capped (default 30′).
  - `AdaptiveConvergence`: switches between the two according to the number
    of hypotheses.

  $\chi^2$ is `search_region_chi2` (default 400).
- **Multi-cone coverage**: when the radius is pinned at the clamp, a greedy
  cover can split the region into several cones (`max_search_cones`,
  disabled by default).

## Night-to-night multi-hypothesis tracking

*Symbols: see [Notation](#notation-photometry-search-region-and-mht-scoring).*

`topocentric_kf::branching`. This is track-oriented MHT branched at the bank
level, with top-B and N-scan pruning.

### 1. Visits

`visit::group_observations_into_visits` splits a night into **exposure
epochs**: observations sorted by time join the current visit while they lie
within `visit_epoch_tolerance_days` (default 1 s) of its first observation.
At LSST cadence a night holds hundreds of visits.

### 2. Per-visit branching

`orchestrate::advance_bank_collection_one_night`. For each visit, every live
lineage is processed in parallel:

1. **Quick prefilter**: once per night, each lineage's MAP state is propagated
   to an anchor epoch. A linear extrapolation from that anchor then decides
   whether the lineage can fall in the visit's HEALPix footprint. If not, the
   lineage is carried over unchanged.
2. **Prediction and candidate search**: the bank is predicted to the visit
   epoch and its search region is built. The visit's HEALPix index is queried
   with a cone (`branching::candidate_search`). Candidates pass a two-step
   gate: Mahalanobis distance within `gate_chi2` of at least one component,
   then mixture likelihood ≥ `likelihood_threshold`.
3. **Optional candidate filters** (`candidate_filters`): photometric
   consistency, cross-track distance, relative likelihood and top-K by
   combined score. Each filter is off when its parameter is 0.
4. **Branching**: each candidate spawns a child branch
   (`Branch::from_observation`: bank step, `n_real_updates += 1`). One
   **null branch** (`Branch::from_null`) is also spawned for the hypothesis
   "the object was not detected in this visit".
5. **Scoring**: the cumulative LLR is the sum of per-visit increments:
   ```math
   \Delta_\mathrm{obs} = \ln L(z) - \ln\lambda_\mathrm{clutter},\qquad
   \Delta_\mathrm{phot} = -\tfrac12\Big(\frac{m_\mathrm{obs}-m_\mathrm{pred}}{\sigma_\mathrm{mag}}\Big)^2,\qquad
   \Delta_\mathrm{null} = \ln(1 - P_D)
   ```
   - $L(z)$ is the predictive mixture likelihood at the candidate.
   - $\lambda_\mathrm{clutter}$ is the local alert density, in alerts per
     steradian: the number of alerts in the candidate's HEALPix cell of the
     visit, divided by the pixel area (`spacetime_bucket::clutter_density`).
   - $m_\mathrm{pred} = H + 5\log_{10}(r\Delta) + \Phi_{HG}(\alpha)$.
   - The detection probability is a logistic completeness curve,
     $P_D = 1/(1 + e^{(m_\mathrm{pred} - m_\mathrm{lim})/w})$, with
     defaults $m_\mathrm{lim} = 21$ and $w = 0.4$ mag.
   - Without an H estimate, $P_D = 0.5$ and $\Delta_\mathrm{phot} = 0$.
6. **Top-B**: after each visit, each lineage keeps at most `branch_cap`
   (default 4) branches by cumulative LLR.

### 3. Night-level pruning

- **N-scan** (`pruning::apply_n_scan_pruning`): branches are grouped by their
  ancestor at the scan horizon. Once that ancestor is `n_scan` nights old
  (default 1), only the best descendant survives and becomes the new anchor.
  This forces a decision on old ambiguities while keeping recent ones open.
- **Stale-lineage purge** (`pruning::purge_stale_lineages`): a lineage is
  stale when its best branch has not had a real update for
  `max_lineage_lifetime_nights` and its LLR is below `stale_llr_floor`, or,
  whatever its LLR, after `hard_stale_nights`. A purged lineage is not lost:
  its best branch becomes an `ArchivedTrajectory` if it has at least
  `archive_min_real_updates` real updates. Stopping the propagation of a
  coasting lineage (whose error box keeps growing and mis-associating) and
  keeping its reconstruction are two separate concerns.

### 4. Discovery

`discovery::seed_new_lineages_from_leftovers`. A night's observations that
belong to no surviving branch are passed to the tracklet linker. Each new
bank becomes a new lineage (`Branch::seed`). Observations consumed by a
branch that was pruned the same night count as unclaimed, so they can seed
again.

## Fragment linkage

`branching::merge`. An object's arc often splits into several lineages: the
association breaks (gate rejection, cadence gap) and the orphan detections
are re-seeded. On a 200-night ZTF-cadence run, 42 % of trackable trajectories
were fragmented, most into individually pure pieces. The merge module decides
whether two arcs are the same object, through a cascade of tests:

1. The predictor arc (the longer one) has converged (≥ `min_arc_points`,
   default 6).
2. The two arcs are disjoint in time, with a gap ≤ `max_gap_days`
   (default 60).
3. Their H estimates are compatible (`max_delta_h`, optional).
4. **Sequential fit**: the predictor is propagated through each observation
   of the other arc and absorbs it. Each residual must stay below
   `max_residual_arcsec` (default 300″), with an optional reduced-χ² cut.

Candidate pairs come from a space-time index (1-day slices, HEALPix depth 5)
with a rate-difference cut. Accepted pairs are grouped by union-find.
Components larger than `max_component_size` (default 8) are dissolved.

> **Status:** the module is implemented and tested, but `advance_one_night`
> does not call it yet.

## Persistence

| File (under `storage_path`) | Content |
| --- | --- |
| `branch_collection.rkyv` | `BranchCollectionSnapshot`: the live branches, `current_step` and last night's consumed-then-pruned IDs. Serialised with rkyv and loaded via `mmap`. The CLI writes a `.tmp` file, then renames it. |
| `archived_trajectories.rkyvlog` | Append-only log. Each record is a little-endian `u64` length followed by an rkyv `Vec<ArchivedTrajectory>` (one batch per night). A truncated tail is ignored on read. A corrupt complete record is an error. |

The `KalmanContext` (ephemeris, UT1) and the configuration are not
persisted. They are rebuilt at start-up, so a snapshot can be resumed with a
new configuration.

## Configuration

`engine_config::EngineConfig` is loaded from YAML by
`EngineConfig::load_engine_config_validated`:

1. **Load / merge**: the YAML file, then environment variables prefixed with
   `FINK_FAT__` (`__` separates nesting levels, e.g.
   `FINK_FAT__KFBANK_CONFIG__GATE_CHI2`). Environment values win.
2. **Deserialise**: every struct is `#[serde(default, deny_unknown_fields)]`.
   Missing keys take Rust defaults and unknown keys are typos, reported as
   errors.
3. **Validate**: the `Validate` trait checks ranges and cross-field
   invariants and **collects every failure** before reporting
   (`ConfigError::Validation`).

| Section | Type | Controls |
| --- | --- | --- |
| `pairs` | `PairConfig` | Tracklet linker: `max_dt` (86.4 min), `max_angular_speed`, `max_mag_difference`, `tracklet_residual_sigma` (5). |
| `triplets` | `TripletConfig` | Standalone triplet generator (not used by the pipeline). |
| `kalman_shared_context` | `KalmanContextConfig` | Ephemeris file, UT1 version, single-filter tuning (`q0`, `dt_ref`, Kepler solver, inflation threshold / max, angular variance floor). |
| `kfbank_config` | `KFBankConfig` | `gate_chi2`, `weight_floor`, `min_hypotheses`, `cap_schedule`, `likelihood_window`, `search_region_chi2`, `merge_position_au`, `slope_parameter_g`. |
| `seeding_grid_config` | `GridConfig` | ρ / ρ̇ grid bounds and resolution, populations, covariance floors. |
| `advance_params` | `NightAdvanceParams` | Visits, prefilter, search region (`top_k`, `radius_strategy`, cones), candidate filters, `branch_cap`, `n_scan`, stale / archive policy, photometric model (`limiting_magnitude`, `completeness_width_mag`, `photometric_sigma_mag`). |
| `merge_params` | `MergeParams` | Fragment linkage thresholds. |
| `healpix_depth` | `u8` | Spatial index depth (default 8, ≈ 13.7′ cells). |
| `storage_path` | path | Snapshot, archive and log directory (default `./storage`). |
| `log_level`, `log_targets`, `log_retention_days` | | Logging (used by the CLI). |

**Units** (`engine_config::units`). A field takes either a bare number in the
canonical unit (days, radians, au, au/day, rad², or arcsec for the `*_arcsec`
fields) or a string with a unit: `"86.4 min"`, `"35 arcmin/day"`,
`"0.02 au"`, `"2 arcsec"`. Time, angle, length, angular-rate,
angular-acceleration and angular-variance tokens are supported, including
French aliases (`jour`, `heure`, `ua`, …). For a variance field, a string is
read as a sigma and squared.

Reference configurations for ZTF and LSST cadence are at the workspace root
(`engine_config.best.yaml`, `engine_config_lsst.best.yaml`).

## Logging

Each module defines a typed event enum with an `emit()` method on its own
`tracing` target. `logging::registry::all_targets()` lists them for the CLI's
`--list-log-targets`:

`bank`, `bank_build`, `candidate_search`, `collection`,
`detection_probability`, `discovery`, `ellipse_region`, `init`, `llr_score`,
`orchestrate`, `propagation`, `pruning`, `seed_grid`, `seeding`,
`spacetime_bucket`, `update`, `visit`.

The engine never installs a subscriber. The binary does.

## Parallelism and determinism

The library uses `rayon` for:
- the per-night prefilter anchors;
- the per-lineage loop inside each visit;
- fragment-pair search and evaluation.

Results are merged sequentially in a fixed order. Designations and
`track_ids` do not depend on thread scheduling. Only the numeric `branch_id`
values do, and they are never used for ordering. Floating-point ordering
always uses `f64::total_cmp`.

## Module map

```text
src/
├── engine_config/        YAML schema, units, validation
│   ├── main_config.rs    EngineConfig (root), load + validate
│   ├── pair_config.rs, triplet_config.rs
│   ├── kalman_context.rs, single_kalman_config.rs
│   ├── kf_bank_config.rs, hypothesis_cap.rs
│   ├── grid_population.rs, night_advance_params.rs
│   └── units.rs, validate_helpers.rs, error.rs
├── seeding/              intra-night pairing
│   ├── tracklet_linker/  time-ordered sweep + incremental linear fit
│   ├── pairs.rs          Pair, standalone pairwise gate
│   └── triplets.rs       standalone triplet generator
├── spacetime_bucket/     (HEALPix cell, time bin) index, clutter density
├── topocentric_kf/
│   ├── conversion.rs, observer_state.rs, pairs_state.rs, constants.rs
│   ├── single_kalman/    KFState: init, propagate (Kepler + light time), update
│   ├── kalman_bank/      KFBank, hypotheses, seed grid, search region, from_seeds
│   └── branching/        MHT: visit, orchestrate, candidate search / filters,
│                         LLR, detection probability, pruning, discovery,
│                         archive, merge, BranchCollection
├── logging/              LogTarget trait, target registry
└── error.rs              EngineError and shared error types
```

## Status and known limitations

- With the default configuration, the stale-lineage purge and archiving, the
  candidate filters and multi-cone coverage are **disabled**. The reference
  YAML files set the values actually used in production.
- Fragment linkage (`merge`) is not yet called by the night loop.
- Not implemented: inter-bank deduplication ("promotion") and arbitration
  between overlapping lineages by orbital fit.
- The `pairs.acc_prior_var` and `time_binner_width` keys are validated but
  not read by the pipeline. `triplets` is only used by its own tests.
- The motion model has no annual aberration and no planetary perturbations
  (two-body heliocentric propagation).

## Testing

```bash
cargo test -p fink-fat-engine
RUST_TEST_THREADS=1 RAYON_NUM_THREADS=2 cargo test -p fink-fat-engine   # as in CI
```

Unit tests and `proptest` property tests live next to the code. Anything that
holds a `KFBank` needs a live `KalmanContext` (network-loaded ephemeris), so
end-to-end tracking is tested at the workspace level (`tests/`) and through
`fink-fat-eval`.
