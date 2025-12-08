//! High-level **inter-night linking** orchestration (snapshots → pairwise link → stitching).
//!
//! # Overview
//! This module provides a thin, testable façade around the lower-level building
//! blocks (seeding, feature extraction, scoring, assignment) to run **night-to-night
//! linking** in a rolling fashion. It defines:
//!
//! - [`NightSnapshot`] — a minimal per-night payload to persist and resume,
//! - `continue_linking` — link `prev` (night *N*) to `curr` (night *N+1*),
//! - [`RollingLinkState`] — a convenience state machine to ingest nights incrementally,
//!   keep pairwise results in memory, update a [`TrackRegistry`], and emit summaries
//!   or DataFrame-ready exports.
//!
//! Pairwise linking delegates to the **engine** layer (candidate retrieval → scoring
//! with hard gates → Top-K → 1-to-1 assignment), and stitching afterwards builds
//! **disjoint chains** across multiple nights.
//!
//! ## Determinism
//! Deterministic as long as inputs (alerts, seeds, binner iteration order) and
//! configuration are deterministic. All logs use `println!` and are meant for debug.
//!
//! ## Units & Conventions
//! - Angles in **radians**, times in **days** (MJD TT), costs are **dimensionless**,
//! - IDs are opaque (`SeedId`, `NightId`) and treated as stable integers.
//!
//! ## Failure Modes
//! - If you forget to register a previous-night `AlertStore`, the track update will
//!   `expect` it (see `run_nightly_step`) — make sure each night is ingested once,
//! - Very large Top-K or lax scoring gates can explode the edge count and solver time,
//! - `stitch_tracks` is **greedy** and produces one successor/predecessor per node;
//!   it is not combinatorial by design (see engine docs for alternatives).
//!
//! ## See also
//! - `crate::propagation::engine` — candidate retrieval, scoring, assignment, stitching,
//! - `crate::propagation::scoring` — interpretable gated scores (position/velocity/flux/gap),
//! - `crate::propagation::solver` — pluggable bipartite solvers (Greedy, Hungarian stub),
//! - `crate::track_registry` — detection-to-track assignment and exports.

use ahash::AHashMap;
use numpy::PyReadonlyArray1;
use pyo3::{
    pyclass, pymethods,
    types::{PyDict, PyDictMethods, PyList, PyListMethods},
    Py, PyResult, Python,
};

use crate::{
    alerts::AlertStore,
    params::{engine_params::InterNightLinkConfig, params_binding::PyFinkFatParams},
    propagation::{
        engine::{build_id_to_index, link_pair_with_binner, LinkResult},
        features::SeedNode,
        solver::{AssignmentSolver, GreedySolver},
    },
    seeding::{healpix_binners::HealpixBinner, space_time_bucket::SpatialBinner, Pairs, Triplets},
    track_registry::{DetectConflictPolicy, TrackRegistry},
    NightId,
};

/* ------------------------------- Snapshot ------------------------------- */

/// Minimal per-night payload to persist and resume **inter-night linking**.
///
/// ### Why this shape?
/// At link time you only need the **seed list** of night *N* as “left” and the
/// seed list of night *N+1* as “right”. Persisting `pairs`/`triplets` along with
/// the seeds is convenient if you ever want to **re-extract** features later,
/// but for standard runs the stored `seeds` are sufficient for `N → N+1` linking.
///
/// ### Fields
/// - `night_id` — integer night identifier (monotonic),
/// - `pairs`, `triplets` — optional intra-night seeds (diagnostics/repro),
/// - `seeds` — feature records used by the inter-night linker.
///
/// ### Determinism
/// Deterministic if the seed extraction and ordering are deterministic.
#[derive(Clone, Debug)]
pub struct NightSnapshot {
    /// Night identifier (e.g., NID).
    pub night_id: NightId,
    /// Optional intra-night seeds for reproducibility/diagnostics.
    pub pairs: Pairs,
    pub triplets: Triplets,
    /// Feature records used by the inter-night linker.
    pub seeds: Vec<SeedNode>,
}

/* --------------------------- Pairwise continuation --------------------------- */

/// Continue inter-night linking from the **previous snapshot (N)** to a **current
/// snapshot (N+1)**. Returns the **pairwise** linking result for this step.
///
/// ### Typical usage per night
/// 1) Load `prev: NightSnapshot` from disk,  
/// 2) Build `curr = build_snapshot_from_store(...)`,  
/// 3) Call `continue_linking(&prev, &curr, ...)`,  
/// 4) Persist `curr` as the next `prev`.
///
/// ### Complexity (typical)
/// - Spatial index build: O(R),
/// - Candidate retrieval: ~O(L·log R),
/// - Scoring: O(E) after Top-K,
/// - Greedy assignment: O(E log E). Hungarian (dense LAP): ~O(n³).
///
/// ### Notes
/// - Internally builds a right-side `seed_id → index` lookup and delegates to
///   [`link_pair_with_binner`].
pub fn continue_linking<S, B>(
    prev: &NightSnapshot,
    curr: &NightSnapshot,
    cfg: &InterNightLinkConfig,
    solver: &S,
    binner: &B,
) -> LinkResult
where
    S: AssignmentSolver,
    B: SpatialBinner,
{
    // Right-side lookup for dense `seed_id`
    let right_lookup = build_id_to_index(&curr.seeds);
    link_pair_with_binner(&prev.seeds, &curr.seeds, cfg, solver, binner, &right_lookup)
}

/* -------------------------- Rolling linking state -------------------------- */

/// Rolling state helper to ingest nights incrementally, keep pairwise results,
/// maintain a [`TrackRegistry`], and export results.
///
/// ### Lifecycle
/// - Call [`RollingLinkState::run_nightly_step`] for each night (in order),
/// - Use [`RollingLinkState::stats`] to summarize costs/edges,
/// - Use [`RollingLinkState::export_linked_detections_dict`] for a DataFrame-ready dict.
///
/// ### Memory model
/// - `stores_by_night` holds per-night `AlertStore`s; beware of memory usage if you
///   retain **many** nights. Persist to disk as needed.
///
/// ### Determinism
/// Deterministic given deterministic ingestion order and configs.
#[pyclass(module = "fink_fat")]
#[derive(Default, Debug, Clone)]
pub struct RollingLinkState {
    /// Last available per-night snapshot (serves as "prev" on the next night).
    pub last: Option<NightSnapshot>,
    /// Accumulated pairwise results (N0→N1, N1→N2, …).
    pub pair_results: Vec<LinkResult>,
    /// Registry of all created tracks (if you want to access them by ID).
    tracks: TrackRegistry,
    /// Optional: keep all per-night stores in memory if you want to access them later.
    /// Beware of memory usage if you run over many nights!
    stores_by_night: AHashMap<NightId, AlertStore>,
}

impl RollingLinkState {
    /// Advance the state with a **current** night (store + params), build its snapshot,
    /// and link it against the stored `last` snapshot if present.
    ///
    /// ### Returns
    /// - `(Some(LinkResult), snapshot)` if a previous night exists and was linked,
    /// - `(None, snapshot)` on the very first night (no previous night to link from).
    ///
    /// ### Side effects
    /// - Updates `self.last`,
    /// - Appends to `self.pair_results` when a link occurs,
    /// - Does **not** modify the track registry here; see `run_nightly_step` for that.
    #[allow(clippy::too_many_arguments)]
    pub fn step_with_current<S, B>(
        &mut self,
        curr_store: &AlertStore,
        curr_night_id: NightId,
        params: &PyFinkFatParams,
        solver: &S,
        binner: &B,
    ) -> (Option<LinkResult>, NightSnapshot)
    where
        S: AssignmentSolver,
        B: SpatialBinner,
    {
        let curr_snap = curr_store.build_snapshot_from_store(curr_night_id, &params.inner);

        println!(
            "Built NightSnapshot for night_id={} with {} seeds",
            curr_snap.night_id,
            curr_snap.seeds.len()
        );

        let pair_res = if let Some(prev_snap) = &self.last {
            let res = continue_linking(prev_snap, &curr_snap, &params.inner.link, solver, binner);
            self.pair_results.push(res.clone());

            println!(
                "Linked night_id={} ({} seeds) → night_id={} ({} seeds) with {} matches",
                prev_snap.night_id,
                prev_snap.seeds.len(),
                curr_snap.night_id,
                curr_snap.seeds.len(),
                res.matches.len()
            );

            Some(res)
        } else {
            println!(
                "No previous night to link from; starting fresh at night_id={} with {} seeds",
                curr_snap.night_id,
                curr_snap.seeds.len()
            );
            None // no previous night to link from (first night in the sequence)
        };

        // Update rolling state
        self.last = Some(curr_snap.clone());

        (pair_res, curr_snap)
    }
}

#[pymethods]
impl RollingLinkState {
    /// Python constructor: `RollingLinkState(conflict_policy)`
    ///
    /// Parameters
    /// ----------
    /// conflict_policy : DetectConflictPolicy
    ///     Strategy for handling detection-to-track conflicts in the registry.
    #[new]
    pub fn new(conflict_policy: DetectConflictPolicy) -> Self {
        Self {
            last: None,
            pair_results: Vec::new(),
            tracks: TrackRegistry::new(conflict_policy),
            stores_by_night: AHashMap::new(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RollingLinkState(last={}, pair_results={})",
            self.last
                .as_ref()
                .map(|s| s.night_id.to_string())
                .unwrap_or_else(|| "None".into()),
            self.pair_results.len()
        )
    }

    /// Ingest one night: build store & snapshot, link to previous if present,
    /// update the track registry, and register the store for later exports.
    ///
    /// Parameters
    /// ----------
    /// dia_source_id, ra, ra_err, dec, dec_err, mjd_tt, flux, flux_err, band : ndarray
    ///     LSST DIA columns (1D, matching length).
    /// night_id : int
    ///     Night identifier for this batch.
    /// params : PyFinkFatParams
    ///    Global configuration (seeding, features, linking).
    /// Notes
    /// -----
    /// - On the **first night**, we “seed” all seeds to fresh tracks so their
    ///   detections carry a stable `trajectory_id` immediately.
    /// - On subsequent nights, we **link** previous→current, merge trajectories
    ///   if chains join, then assign all member detections to the final track id.
    #[pyo3(
        text_signature = "($self, dia_source_id, ra, ra_err, dec, dec_err, mjd_tt, flux, flux_err, band, night_id, seeding_params, feature_params, pair_config)"
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn run_nightly_step(
        &mut self,
        dia_source_id: PyReadonlyArray1<u64>,
        ra: PyReadonlyArray1<f64>,
        ra_err: PyReadonlyArray1<f64>,
        dec: PyReadonlyArray1<f64>,
        dec_err: PyReadonlyArray1<f64>,
        mjd_tt: PyReadonlyArray1<f64>,
        flux: PyReadonlyArray1<f32>,
        flux_err: PyReadonlyArray1<f32>,
        band: PyReadonlyArray1<u8>,
        night_id: NightId,
        params: &PyFinkFatParams,
    ) -> PyResult<()> {
        // 1) Build the per-night store from numpy
        let curr_store = AlertStore::from_numpy(
            dia_source_id,
            ra,
            ra_err,
            dec,
            dec_err,
            mjd_tt,
            flux,
            flux_err,
            band,
        )?;

        println!(
            "Running nightly step for night_id={} with {} alerts",
            night_id,
            curr_store.alerts.len()
        );

        // 2) Build the snapshot (seeding + features)
        let sb = HealpixBinner::new(params.healpix_depth());
        let curr_snap = curr_store.build_snapshot_from_store(night_id, &params.inner);

        println!(
            "Built NightSnapshot for night_id={} with {} seeds",
            curr_snap.night_id,
            curr_snap.seeds.len()
        );

        // 3) Link to previous if present, and update TrackRegistry
        if let Some(prev_snap) = &self.last {
            let res = continue_linking(
                prev_snap,
                &curr_snap,
                &params.inner.link,
                &GreedySolver,
                &sb,
            );

            println!(
                "Linked night_id={} ({} seeds) → night_id={} ({} seeds) with {} matches",
                prev_snap.night_id,
                prev_snap.seeds.len(),
                curr_snap.night_id,
                curr_snap.seeds.len(),
                res.matches.len()
            );

            // Track update needs both left & right stores
            let left_store = self.stores_by_night.get(&prev_snap.night_id).expect(
                "missing AlertStore for previous night; did you call run_nightly_step on it?",
            );

            self.tracks
                .update_from_link(prev_snap, &curr_snap, &res, left_store, &curr_store);

            self.pair_results.push(res);
        } else {
            // First night: give stable trajectory ids to all detections in that night.
            self.tracks.seed_all_of_snapshot(&curr_snap, &curr_store);
            println!(
                "No previous night to link from; seeded {} seeds at night_id={}",
                curr_snap.seeds.len(),
                curr_snap.night_id
            );
        }

        // 4) Update rolling “last” snapshot and register the store for later
        self.last = Some(curr_snap);
        self.stores_by_night.insert(night_id, curr_store);

        Ok(())
    }

    /// Summarize inter-night linking results into a **Python dict**.
    ///
    /// Returns
    /// -------
    /// dict
    ///     A nested dictionary with global and per-pair statistics:
    /// ```text
    /// - "total_pairs": int
    /// - "total_matches": int
    /// - "edges_kept_total": int
    /// - "pairs": list of dicts, each with:
    ///     - "night_left": int
    ///     - "night_right": int
    ///     - "n_matches": int
    ///     - "edges_kept": int
    ///     - "cost": { "count","min","max","mean","median","p90","p95","p99" }
    /// - "cost": global cost summary with the same fields as above
    /// - "edges_kept": { "total","mean_per_pair","median_per_pair","max_per_pair" }
    /// ```
    ///
    /// Notes
    /// -----
    /// - Only uses information available in `pair_results` (no per-night seed counts).
    ///   If you need match rates, consider extending [`LinkResult`] to carry
    ///   `left_count` / `right_count`.
    #[pyo3(text_signature = "($self)")]
    pub fn stats(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let out = PyDict::new(py);

        let total_pairs = self.pair_results.len();
        out.set_item("total_pairs", total_pairs)?;

        let mut total_matches = 0usize;
        let mut edges_kept_total = 0usize;

        let mut all_costs: Vec<f64> =
            Vec::with_capacity(self.pair_results.iter().map(|lr| lr.matches.len()).sum());

        let mut edges_kept_per_pair: Vec<f64> = Vec::with_capacity(total_pairs);

        let pairs_list = PyList::empty(py);

        for lr in &self.pair_results {
            let n_matches = lr.matches.len();
            total_matches += n_matches;
            edges_kept_total += lr.edges_kept;
            edges_kept_per_pair.push(lr.edges_kept as f64);

            let mut costs = Vec::with_capacity(n_matches);
            for m in &lr.matches {
                costs.push(m.cost);
                all_costs.push(m.cost);
            }

            let per_pair = PyDict::new(py);
            per_pair.set_item("night_left", lr.night_left)?;
            per_pair.set_item("night_right", lr.night_right)?;
            per_pair.set_item("n_matches", n_matches)?;
            per_pair.set_item("edges_kept", lr.edges_kept)?;
            per_pair.set_item("cost", pair_cost_summary(py, &costs)?)?;

            pairs_list.append(per_pair)?;
        }

        out.set_item("total_matches", total_matches)?;
        out.set_item("pairs", pairs_list)?;

        // Global cost summary
        out.set_item("cost", pair_cost_summary(py, &all_costs)?)?;

        // Edges-kept summary
        let edges_dict = PyDict::new(py);
        edges_dict.set_item("total", edges_kept_total)?;

        if edges_kept_per_pair.is_empty() {
            edges_dict.set_item("mean_per_pair", f64::NAN)?;
            edges_dict.set_item("median_per_pair", f64::NAN)?;
            edges_dict.set_item("max_per_pair", f64::NAN)?;
        } else {
            let mean_edges = mean(&edges_kept_per_pair);
            let mut sorted_edges = edges_kept_per_pair.clone();
            sorted_edges.sort_by(|a, b| a.total_cmp(b));
            edges_dict.set_item("mean_per_pair", mean_edges)?;
            edges_dict.set_item("median_per_pair", quantile_sorted(&sorted_edges, 0.5))?;
            edges_dict.set_item("max_per_pair", sorted_edges[sorted_edges.len() - 1])?;
        }
        out.set_item("edges_kept", edges_dict)?;

        Ok(out.into())
    }

    /// Return a `dict[str, list]` suitable for `pandas.DataFrame(d)`.
    ///
    /// Columns
    /// -------
    /// - `candid` : int
    /// - `ra` : float (deg)
    /// - `dec` : float (deg)
    /// - `jd` : float (TT)
    /// - `mjd_tt` : float (TT)
    /// - `trajectory_id` : int
    ///
    /// Notes
    /// -----
    /// Uses the internal [`TrackRegistry`] and the registered `AlertStore`s to
    /// reconstruct detection-level rows associated with each linked track.
    pub fn export_linked_detections_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        self.tracks
            .export_detection_dict_py(py, &self.stores_by_night)
    }
}

/* ------------------------------ Utilities ------------------------------ */

/// Simple mean; returns NaN on empty input.
fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        f64::NAN
    } else {
        xs.iter().sum::<f64>() / (xs.len() as f64)
    }
}

/// Linear-interpolated quantile (R-7 style) on a **sorted** slice.
/// Returns NaN on empty input. `q` must be in `[0, 1]`.
fn quantile_sorted(xs: &[f64], q: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let n = xs.len();
    if n == 1 {
        return xs[0];
    }
    let q = q.clamp(0.0, 1.0);
    let h = (n as f64 - 1.0) * q;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    if lo == hi {
        xs[lo]
    } else {
        xs[lo] + (h - lo as f64) * (xs[hi] - xs[lo])
    }
}

/// Build a small per-pair cost summary as a Python dict.
///
/// Fields
/// ------
/// - `count`, `min`, `max`, `mean`, `median`, `p90`, `p95`, `p99`.
///
/// Notes
/// -----
/// Returns a bound `PyDict` ready to be inserted into larger summaries.
fn pair_cost_summary<'py>(py: Python<'py>, costs: &[f64]) -> PyResult<pyo3::Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    if costs.is_empty() {
        d.set_item("count", 0)?;
        d.set_item("min", f64::NAN)?;
        d.set_item("max", f64::NAN)?;
        d.set_item("mean", f64::NAN)?;
        d.set_item("median", f64::NAN)?;
        d.set_item("p90", f64::NAN)?;
        d.set_item("p95", f64::NAN)?;
        d.set_item("p99", f64::NAN)?;
        return Ok(d);
    }
    let mut s = costs.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    d.set_item("count", s.len())?;
    d.set_item("min", s[0])?;
    d.set_item("max", s[s.len() - 1])?;
    d.set_item("mean", mean(&s))?;
    d.set_item("median", quantile_sorted(&s, 0.5))?;
    d.set_item("p90", quantile_sorted(&s, 0.90))?;
    d.set_item("p95", quantile_sorted(&s, 0.95))?;
    d.set_item("p99", quantile_sorted(&s, 0.99))?;
    Ok(d)
}
