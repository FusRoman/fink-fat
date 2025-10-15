// -----------------------------------------------------------------------------
// Rolling Min-Cost Flow (MCF) state: incremental, horizon-limited orchestration
// -----------------------------------------------------------------------------

use std::collections::BTreeMap;

use crate::alerts::AlertStore;
use crate::params::params_binding::PyFinkFatParams;
use crate::propagation::engine::{build_id_to_index, generate_topk_edges, median_epoch};
use crate::propagation::features::{SeedId, SeedNode, SeedSpatialIndex};
use crate::propagation::flow::{
    solve_and_extract, FlowBuilder, FlowUpdate, MinCostFlowSolver, NullFlowSolver,
};
use crate::seeding::healpix_binners::HealpixBinner;
use crate::seeding::space_time_bucket::SpatialBinner;
use crate::track_registry::{TrackRegistry, TrajectoryId};
use crate::NightId;
use ahash::AHashMap;
use pyo3::{pyclass, pymethods, PyResult};

/// Rolling MCF orchestrator: ingest nights, add link arcs within a horizon,
/// solve the min-cost flow, and update the TrackRegistry.
///
/// Design
/// ------
/// * On the **first** night: we ingest the layer and add Start/End arcs (no links yet),
///   then we can solve (will typically return zero trajectories if you require at
///   least one hop).
/// * On each **subsequent** night `k`: we ingest the layer, then for each previous
///   night `i` in the rolling window `k - horizon ≤ i < k`, we **generate Top-K
///   edges** `i → k` with the same binner/scoring used by the bipartite engine,
///   add them to the MCF graph, and finally **solve** the global problem.
/// * We return a compact `FlowUpdate` (counts + path skeletons). The `TrackRegistry`
///   is updated at **seed-level** (one representative id per path). Assigning
///   **detections** to trajectories can be done separately using your snapshots.
///
/// Determinism
/// -----------
/// Deterministic given stable seed slices, binner iteration, and solver.
///
/// See also
/// --------
/// * `crate::propagation::flow` (FlowProblem/FlowBuilder/MinCostFlowSolver)
/// * `crate::propagation::engine::generate_topk_edges_between` (Top-K edges)
#[pyclass(module = "fink_fat")]
#[derive(Default, Debug, Clone)]
pub struct RollingFlowState {
    /// Time-expanded flow builder (holds Source/Sink, layers, arcs).
    pub builder: FlowBuilder,
    /// Keep per-night seeds (needed to regenerate Top-K edges on demand).
    seeds_by_night: BTreeMap<NightId, Vec<SeedNode>>,
    /// Global track registry (seed-level unions).
    tracks: TrackRegistry,
}

#[pymethods]
impl RollingFlowState {
    /// Create a new rolling MCF orchestrator.
    ///
    /// Parameters
    /// ----------
    /// horizon_nights : int
    ///     Number of previous nights to link to the current night (Δ=1..H).
    ///     Example: `3` means add edges from `k-1`, `k-2`, `k-3` to `k`.
    #[new]
    #[pyo3(text_signature = "(horizon_nights=3)")]
    pub fn new() -> Self {
        Self {
            builder: FlowBuilder::new(Default::default()),
            seeds_by_night: BTreeMap::new(),
            tracks: TrackRegistry::default(),
        }
    }

    /// Python-friendly façade that mirrors `RollingLinkState::run_nightly_step`,
    /// but performs a **global MCF** solve instead of a bipartite pairwise link.
    ///
    /// Returns
    /// -------
    /// dict
    ///     A compact summary of the MCF update (counts + simple trajectory skeletons).
    #[pyo3(
        text_signature = "($self, dia_source_id, ra, ra_err, dec, dec_err, mjd_tt, flux, flux_err, band, night_id, params, mcf_solver, binner)"
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn run_nightly_step_mcf(
        &mut self,
        night_id: NightId,
        alert_store: &AlertStore,
        params: &PyFinkFatParams,
    ) -> PyResult<FlowUpdate> {
        let spatial_binner = HealpixBinner::new(params.healpix_depth());
        // 2) Borrow typed backends from Python.
        let mcf_solver = NullFlowSolver {};

        // 3) Orchestration step (ingest → edges within horizon → solve → registry).
        let upd = self.step_with_current_mcf(
            alert_store,
            night_id,
            params,
            &mcf_solver,
            &spatial_binner,
        )?;

        Ok(upd)
    }
}

impl RollingFlowState {
    /// Internal: ingest the current night, add link arcs from previous nights within horizon,
    /// then solve the min-cost flow and update the registry.
    ///
    /// Notes
    /// -----
    /// - Generic over the **MCF solver** and the **spatial binner** (same traits que l’engine).
    /// - Uses the **same** scoring/gating/Top-K as pairwise via `generate_topk_edges_between`.
    pub fn step_with_current_mcf<S, B>(
        &mut self,
        curr_store: &AlertStore,
        curr_night_id: NightId,
        params: &PyFinkFatParams,
        mcf_solver: &S,
        binner: &B,
    ) -> PyResult<FlowUpdate>
    where
        S: MinCostFlowSolver,
        B: SpatialBinner,
    {
        // Build seeds for the current night from the store (same as pairwise path).
        let curr_snap = curr_store.build_snapshot_from_store(curr_night_id, params);

        // 1) Ingest current layer (adds Start/End arcs).
        let _layer_summary = self
            .builder
            .ingest_night(curr_snap.night_id, &curr_snap.seeds)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // 2) Register seeds & store for this night (needed to build edges).
        self.seeds_by_night
            .insert(curr_snap.night_id, curr_snap.seeds.clone());

        // 3) For each previous night within the horizon, add Top-K edges i→k.
        //    We always generate edges with **right = current** to avoid duplicates:
        //    pairs for (i→i+1) were already added when (i+1) arrived.
        let mut added_links_total = 0usize;
        let mut prev_keys: Vec<NightId> = self
            .seeds_by_night
            .keys()
            .copied()
            .filter(|&n| n < curr_snap.night_id)
            .collect();
        prev_keys.sort(); // chronological

        // Limit to the H most recent nights.
        let start_at = prev_keys
            .len()
            .saturating_sub(params.inner.link.mcf.horizon_nights);

        println!(
            "MCF step: night_id={} | seeds={} | linking to {} previous nights (horizon={})",
            curr_snap.night_id,
            curr_snap.seeds.len(),
            prev_keys.len() - start_at,
            params.inner.link.mcf.horizon_nights
        );

        // PRECOMPUTE once for the current right-night
        let right = &curr_snap.seeds;
        let index_right = SeedSpatialIndex::build(right, binner); // moved out of the loop
        let t_right_med = median_epoch(right);
        let right_id_to_index: AHashMap<SeedId, usize> = build_id_to_index(right);

        for &prev_night in &prev_keys[start_at..] {
            println!(" - linking from night {}", prev_night);

            let left = self
                .seeds_by_night
                .get(&prev_night)
                .expect("left night present");
            let right = &curr_snap.seeds;

            // Build O(1) deref for the right partition and generate sparse Top-K edges.
            let edges = generate_topk_edges(
                left,
                right,
                &params.inner.link,
                binner,
                &index_right,
                t_right_med,
                &right_id_to_index,
            );

            println!(
                "Generated {} candidate edges between nights {} → {}",
                edges.len(),
                prev_night,
                curr_snap.night_id
            );

            // Push edges into the time-expanded graph as Link arcs prev_night→curr_night.
            let created = self
                .builder
                .add_links_between(prev_night, curr_snap.night_id, &edges)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            println!(
                "Added {} link arcs between nights {} → {}",
                created, prev_night, curr_snap.night_id
            );

            added_links_total += created;
        }

        println!("Total added link arcs: {}", added_links_total);

        // 4) Solve the **global** min-cost flow and extract trajectories.
        let upd = solve_and_extract(&self.builder, mcf_solver)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // 5) Update the TrackRegistry at **seed-level** using the recovered paths.
        //    (One representative TrajectoryId per path; you can then assign detections.)
        self.update_registry_from_flow_paths(&upd)?;

        Ok(upd)
    }

    /// Merge seed-paths into the TrackRegistry (seed-level only).
    fn update_registry_from_flow_paths(&mut self, upd: &FlowUpdate) -> PyResult<Vec<TrajectoryId>> {
        // We perform the same logic as `FlowSolution::into_registry`, but we apply it
        // to the paths present in `upd` to avoid coupling this struct to the internal
        // solution object.
        let mut reps = Vec::with_capacity(upd.trajectories.len());
        for path in upd.trajectories.iter() {
            if path.is_empty() {
                continue;
            }
            let first = path[0];
            let mut keep = self
                .tracks
                .ensure_seed_traj(crate::track_registry::SeedKey {
                    night_id: first.night,
                    seed_id: first.seed,
                });
            for sk in path.iter().skip(1) {
                let tid = self
                    .tracks
                    .ensure_seed_traj(crate::track_registry::SeedKey {
                        night_id: sk.night,
                        seed_id: sk.seed,
                    });
                keep = self.tracks.merge_traj(keep, tid);
            }
            reps.push(keep);
        }
        Ok(reps)
    }
}
