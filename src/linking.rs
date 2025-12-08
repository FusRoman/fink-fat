use std::collections::BTreeSet;

use ahash::AHashMap;
use pyo3::types::PyDict;
// src/python/py_rolling_graph_builder.rs
use pyo3::{pyclass, pymethods, Py, PyResult, Python};

use crate::alerts::{Alert, AlertStore};
use crate::graph::components::connected_components;
use crate::graph::graph::InterNightGraph;
use crate::graph::ingest::{NightIngestionReport, RollingGraphBuilder, SeedExtractor};
use crate::graph::stats::GraphStats;
use crate::graph::Horizon;
use crate::params::params_binding::PyFinkFatParams;
use crate::params::FinkFatParams;
use crate::propagation::features::SeedNode;
use crate::seeding::healpix_binners::HealpixBinner;
use crate::track_registry::{DetectKey, TrackRegistry};
use crate::NightId;

#[derive(Clone, Default, Debug)]
struct GraphSeedExtractor;

impl SeedExtractor for GraphSeedExtractor {
    fn extract(&self, store: &AlertStore, night: NightId, params: &FinkFatParams) -> Vec<SeedNode> {
        let snapshot = store.build_snapshot_from_store(night, &params);
        snapshot.seeds
    }
}

/* -------------------------------------------------------------------------- */
/*  Variant wrapper: type-erased RollingGraphBuilder                           */
/* -------------------------------------------------------------------------- */

/// Non-generic, type-tagged wrapper over concrete `RollingGraphBuilder<Bs, Ex>`.
///
/// Naming convention
/// -----------------
/// Use `BinnerExtractor` as `<SpatialBinner><SeedExtractor>` to make the variant
/// self-descriptive. Here: `HealpixBinner + GraphSeedExtractor`.
enum BuilderVariant {
    /// Healpix spatial binning + graph-based seed extractor.
    HealpixGraphSeeds(RollingGraphBuilder<HealpixBinner, GraphSeedExtractor>),
}

impl BuilderVariant {
    /// Construct the HealpixBinner + GraphSeedExtractor pipeline.
    ///
    /// Notes
    /// -----
    /// - Keeps a 3-night rolling horizon by default (tweak as needed).
    fn new_healpix_graphseeds(params: &FinkFatParams) -> Self {
        let binner = HealpixBinner::new(params.binning.healpix_depth);
        let extractor = GraphSeedExtractor {};
        let horizon = Horizon { horizon_nights: 3 };
        let graph = InterNightGraph::new(horizon);
        let inner = RollingGraphBuilder::new(graph, binner, extractor);
        BuilderVariant::HealpixGraphSeeds(inner)
    }

    fn get_graph_builder(&self) -> &RollingGraphBuilder<HealpixBinner, GraphSeedExtractor> {
        match self {
            BuilderVariant::HealpixGraphSeeds(b) => b,
        }
    }

    /// Ingest one night of alerts into the rolling graph.
    fn ingest_night(
        &mut self,
        night: NightId,
        store: &AlertStore,
        params: &FinkFatParams,
    ) -> NightIngestionReport {
        match self {
            BuilderVariant::HealpixGraphSeeds(b) => b.ingest_night(night, store, params),
        }
    }

    /// Compute high-level statistics about the current graph.
    fn stats(&self) -> GraphStats {
        match self {
            BuilderVariant::HealpixGraphSeeds(b) => GraphStats::from_graph(&b.graph),
        }
    }

    fn graph(&self) -> &InterNightGraph {
        match self {
            BuilderVariant::HealpixGraphSeeds(b) => &b.graph,
        }
    }
}

/* -------------------------------------------------------------------------- */
/*  Per-run report returned by `solve_graph`                                   */
/* -------------------------------------------------------------------------- */

#[pyclass(module = "fink_fat", name = "SolveRunReport")]
#[derive(Clone, Debug)]
pub struct SolveRunReport {
    #[pyo3(get)]
    pub n_components: usize,
    #[pyo3(get)]
    pub n_paths: usize,
    #[pyo3(get)]
    pub n_detections: usize,

    #[pyo3(get)]
    pub path_len_min: usize,
    #[pyo3(get)]
    pub path_len_max: usize,
    #[pyo3(get)]
    pub path_len_mean: f64,

    #[pyo3(get)]
    pub det_min: usize,
    #[pyo3(get)]
    pub det_max: usize,
    #[pyo3(get)]
    pub det_mean: f64,

    #[pyo3(get)]
    pub registry_traj_count: usize,
    #[pyo3(get)]
    pub registry_det_count: usize,

    /// Canonical (DSU-root) trajectory ids that were **touched** (created/updated)
    /// by this solve-run. Use these to restrict the subsequent IOD to the
    /// freshly solved trajectories only.
    #[pyo3(get)]
    pub traj_ids: Vec<u64>,
}

impl SolveRunReport {
    fn empty() -> Self {
        Self {
            n_components: 0,
            n_paths: 0,
            n_detections: 0,
            path_len_min: 0,
            path_len_max: 0,
            path_len_mean: 0.0,
            det_min: 0,
            det_max: 0,
            det_mean: 0.0,
            registry_traj_count: 0,
            registry_det_count: 0,
            traj_ids: Vec::new(),
        }
    }
}

#[pymethods]
impl SolveRunReport {
    /// Human-friendly single-line summary (repr-like).
    ///
    /// Returns
    /// -------
    /// str
    ///     Reconstructible-ish representation with key fields.
    fn __repr__(&self) -> pyo3::PyResult<String> {
        // Keep it short and reconstructible-ish.
        Ok(format!(
            "SolveRunReport(n_components={}, n_paths={}, n_detections={}, \
             path_len_min={}, path_len_max={}, path_len_mean={:.3}, \
             det_min={}, det_max={}, det_mean={:.3}, \
             registry_traj_count={}, registry_det_count={}, traj_ids=[{}])",
            self.n_components,
            self.n_paths,
            self.n_detections,
            self.path_len_min,
            self.path_len_max,
            self.path_len_mean,
            self.det_min,
            self.det_max,
            self.det_mean,
            self.registry_traj_count,
            self.registry_det_count,
            // Print at most first 8 ids, then ellipsis if longer.
            {
                const MAX_IDS: usize = 8;
                if self.traj_ids.len() <= MAX_IDS {
                    self.traj_ids
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                } else {
                    let mut s = self
                        .traj_ids
                        .iter()
                        .take(MAX_IDS)
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    s.push_str(", …");
                    s
                }
            }
        ))
    }

    /// Pretty-print as a compact ASCII table.
    ///
    /// Returns
    /// -------
    /// str
    ///     Multi-line text table summarizing this run.
    fn __str__(&self) -> pyo3::PyResult<String> {
        // Helper to render a key/value row with aligned columns.
        fn row(k: &str, v: impl std::fmt::Display) -> String {
            //  - left column width ~28, right flexible
            format!("│ {:<28} │ {:>16} │\n", k, v)
        }

        // Build the ASCII table.
        let mut out = String::new();
        out.push_str("┌────────────────────────────────┬──────────────────┐\n");
        out.push_str("│ SolveRunReport                 │ value            │\n");
        out.push_str("├────────────────────────────────┼──────────────────┤\n");

        out.push_str(&row("components", self.n_components));
        out.push_str(&row("paths (hypotheses)", self.n_paths));
        out.push_str(&row("detections (upper bound)", self.n_detections));

        out.push_str("├────────────────────────────────┼──────────────────┤\n");
        out.push_str(&row("path_len.min", self.path_len_min));
        out.push_str(&row("path_len.max", self.path_len_max));
        out.push_str(&row("path_len.mean", format!("{:.3}", self.path_len_mean)));

        out.push_str("├────────────────────────────────┼──────────────────┤\n");
        out.push_str(&row("det.min", self.det_min));
        out.push_str(&row("det.max", self.det_max));
        out.push_str(&row("det.mean", format!("{:.3}", self.det_mean)));

        out.push_str("├────────────────────────────────┼──────────────────┤\n");
        out.push_str(&row("registry.traj_count", self.registry_traj_count));
        out.push_str(&row("registry.det_count", self.registry_det_count));

        // Trajectory ids: show a compact summary + first few ids.
        let total_ids = self.traj_ids.len();
        let preview = {
            const MAX_IDS: usize = 8;
            if total_ids == 0 {
                String::from("[]")
            } else if total_ids <= MAX_IDS {
                format!(
                    "[{}]",
                    self.traj_ids
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            } else {
                format!(
                    "[{} …] ({} ids)",
                    self.traj_ids
                        .iter()
                        .take(MAX_IDS)
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                    total_ids
                )
            }
        };

        out.push_str("├────────────────────────────────┼──────────────────┤\n");
        out.push_str(&row("traj_ids", preview));

        out.push_str("└────────────────────────────────┴──────────────────┘");

        Ok(out)
    }
}

/* -------------------------------------------------------------------------- */
/*  PyO3 façade : classe non générique pour Python                             */
/* -------------------------------------------------------------------------- */

/// Python wrapper over the rolling graph builder (type-erased).
///
/// This façade selects a concrete (binner, extractor) pair internally and
/// exposes a stable Python API to ingest nights from NumPy arrays.
#[pyclass(module = "fink_fat", name = "RollingGraphBuilder")]
pub struct PyRollingGraphBuilder {
    /// Inner type-erased graph builder.
    builder: BuilderVariant,
    /// Alerts per night (local AlertId space).
    alerts_by_night: AHashMap<NightId, AlertStore>,
    /// Track registry.
    pub registry: TrackRegistry,
}

impl PyRollingGraphBuilder {
    /// Retrieve the original `Alert`s corresponding to a list of `SeedNode`s.
    ///
    /// Parameters
    /// ----------
    /// seeds : &[SeedNode]
    ///   List of seed nodes whose alerts are to be retrieved.
    ///
    /// Returns
    /// -------
    /// Vec<Alert>
    ///    The corresponding alerts.
    pub fn get_alerts_from_seeds(&self, seeds: &[SeedNode]) -> Vec<&Alert> {
        let mut alerts = Vec::with_capacity(seeds.len());
        for seed in seeds {
            if let Some(store) = self.alerts_by_night.get(&seed.night_id) {
                // seed.members is a Vec<AlertId>; iterate over owned AlertId values
                // so they satisfy get_many's IntoIterator<Item = AlertId>.
                if let Some(iter) = store.get_many(seed.members.iter()) {
                    alerts.extend(iter);
                }
            }
        }
        alerts
    }
}

#[pymethods]
impl PyRollingGraphBuilder {
    /// Create a new rolling graph builder.
    ///
    /// Parameters
    /// ----------
    /// params : FinkFatParams
    ///     Global configuration (seeding, binning, scoring, limits).
    /// variant : str, default "healpix+h2"
    ///     Which concrete backend to use.
    ///     Supported: "healpix+h2" (default).
    #[new]
    pub fn new(params: &PyFinkFatParams) -> PyResult<Self> {
        // For now, only one variant is supported.
        let inner = BuilderVariant::new_healpix_graphseeds(&params.inner);
        Ok(Self {
            builder: inner,
            alerts_by_night: AHashMap::default(),
            registry: TrackRegistry::default(),
        })
    }

    /// Count canonical trajectories (DSU roots) and DISTINCT detections assigned.
    ///
    /// Notes
    /// -----
    /// - We count **distinct DetectKey** for the detection total
    ///   (not the number of (dk, tid) pairs), which is usually what you want.
    /// - We rely on `iter_detection_assignments()` which yields canonical tids.
    fn registry_totals(&self) -> (usize, usize) {
        let mut uniq_roots: BTreeSet<u64> = BTreeSet::new();
        let mut uniq_dk: BTreeSet<DetectKey> = BTreeSet::new();

        for (dk, tid) in self.registry.iter_detection_assignments() {
            uniq_dk.insert(dk);
            uniq_roots.insert(tid.0);
        }
        (uniq_roots.len(), uniq_dk.len())
    }

    /// Ingest one night of alerts from NumPy arrays and update the rolling graph.
    ///
    /// Parameters
    /// ----------
    /// night_id : int
    ///     Monotonic night identifier.
    /// dia_source_id : ndarray[int64]
    /// ra, dec, mjd_tt : ndarray[float64]
    /// ra_err, dec_err, flux, flux_err : ndarray[float64], optional
    /// band : ndarray[uint8]
    ///
    /// Returns
    /// -------
    /// dict
    ///     Summary counts after the update.
    pub fn add_night<'py>(
        &mut self,
        store: &AlertStore,
        night_id: NightId,
        params: &PyFinkFatParams,
        py: Python<'py>,
    ) -> PyResult<NightIngestionReport> {
        // Insert alerts store for this night
        self.alerts_by_night.insert(night_id, store.clone());

        // Heavy work without GIL
        Ok(py.detach(|| self.builder.ingest_night(night_id, &store, &params.inner)))
    }

    /// Solve all connected components, commit hypothesized trajectories into the registry,
    /// and return per-run statistics + the list of affected canonical TrajectoryIds.
    ///
    /// Returns
    /// -------
    /// SolveRunReport
    ///     Aggregated metrics for this run and the `traj_ids` to feed into IOD.
    pub fn solve_graph(&mut self, _params: &PyFinkFatParams) -> PyResult<SolveRunReport> {
        let graph = self.builder.graph();
        let comps = connected_components(graph);

        let n_components = comps.len();
        let mut n_paths = 0usize;
        let mut n_detections = 0usize;

        let mut path_len_min = usize::MAX;
        let mut path_len_max = 0usize;
        let mut path_len_sum = 0usize;

        let mut det_min = usize::MAX;
        let mut det_max = 0usize;
        let mut det_sum = 0usize;

        // Collect canonical (post-merge) trajectory ids touched this run
        use std::collections::BTreeSet;
        let mut affected_roots: BTreeSet<u64> = BTreeSet::new();

        for comp in comps {
            let sol = comp.solve_auto(graph, 2, 256, 10_000, 50_000, 12.0, 10, 3);

            let hyp_traj = self
                .builder
                .get_graph_builder()
                .reconstruct_from_component_solution(&sol);

            // Stats before commit
            for tr in &hyp_traj {
                n_paths += 1;

                let plen = tr.node_ids.len();
                path_len_min = path_len_min.min(plen);
                path_len_max = path_len_max.max(plen);
                path_len_sum += plen;

                let dlen = tr.detections.len();
                det_min = det_min.min(dlen);
                det_max = det_max.max(dlen);
                det_sum += dlen;
            }

            // Commit to registry (persistent across runs)
            let track_ids = self.registry.assign_reconstructed_many(&hyp_traj);

            // Record canonical roots affected this run
            for tid in track_ids {
                let root = self.registry.canonical_id_ro(tid);
                affected_roots.insert(root.0);
            }

            // Upper-bound of engaged detections this run
            n_detections += hyp_traj.iter().map(|tr| tr.detections.len()).sum::<usize>();
        }

        if n_paths == 0 {
            let mut rep = SolveRunReport::empty();
            let (t, d) = self.registry_totals();
            rep.n_components = n_components;
            rep.registry_traj_count = t;
            rep.registry_det_count = d;
            // traj_ids stays empty
            return Ok(rep);
        }

        let path_len_mean = (path_len_sum as f64) / (n_paths as f64);
        let det_mean = (det_sum as f64) / (n_paths as f64);

        if path_len_min == usize::MAX {
            path_len_min = 0;
        }
        if det_min == usize::MAX {
            det_min = 0;
        }

        let (registry_traj_count, registry_det_count) = self.registry_totals();

        let rep = SolveRunReport {
            n_components,
            n_paths,
            n_detections,
            path_len_min,
            path_len_max,
            path_len_mean,
            det_min,
            det_max,
            det_mean,
            registry_traj_count,
            registry_det_count,
            traj_ids: affected_roots.into_iter().collect(),
        };
        Ok(rep)
    }

    /// Get current graph statistics.
    pub fn stats(&self) -> PyResult<GraphStats> {
        Ok(self.builder.stats())
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
        self.registry
            .export_detection_dict_py(py, &self.alerts_by_night)
    }
}
