//! In-RAM snapshot of everything the homepage displays.
//!
//! The homepage used to answer every user interaction — a legend toggle, a
//! column sort, a page change, a keystroke in the search box — with fresh SQL.
//! Instead, the whole working set is loaded into an immutable [`Snapshot`] —
//! about 35 MB at 214k branches — and every homepage server function becomes
//! an in-memory scan.
//!
//! The build itself reads two narrow tables `fink-fat convert` prepares for
//! it: `branches` (with its precomputed observation statistics) and
//! `branch_best_state` (each branch's best hypothesis state). It used to
//! reach that state through a `LATERAL` best-hypothesis lookup plus a
//! `kf_state` primary-key probe *per branch* — ~214k random reads into tens
//! of GB, minutes per build. The remaining queries (fits, failures,
//! cross-matches, counts) are small and run concurrently.
//!
//! The data changes when `fink-fat convert` rewrites the bulk-loaded tables
//! (`TRUNCATE ... CASCADE` then `COPY`), and when the explorer records orbit
//! fits, fit failures or cross-matches. When to rebuild is decided by
//! [`crate::homepage::snapshot_refresh`], from the [`DataMarker`] each
//! snapshot is built from.
//!
//! Server-only: `main.rs` gates the module on the `server` feature, so none of
//! this reaches the wasm bundle.

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, PoisonError, RwLock};

use dioxus::server::axum::body::Bytes;
use sqlx::PgPool;

use crate::fit_pipeline::params;
use crate::get_pool;
use crate::homepage::dynamic_pop_plot::PlotSeries;
use crate::homepage::family::DynamicalFamily;
use crate::homepage::interaction::SortColumn;
use crate::homepage::quality_tier::{assign_quality_tier, FitMethod, LatestFit, QualityTier};
use crate::homepage::series_route::etag_for;
use crate::homepage::series_wire::{encode_series, SeriesWireError};
use crate::homepage::snapshot_refresh::{tracked, BuildStep};

/// Raw row shape of [`SNAPSHOT_QUERY`].
#[derive(sqlx::FromRow)]
struct SnapshotRow {
    branch_id: i64,
    lineage_id: i64,
    designation: String,
    lineage_designation: String,
    cumulative_llr: f64,
    n_real_updates: i64,
    arc_length_days: f64,
    n_nights: i64,
    median_inter_night_dt_days: Option<f64>,
    well_sampled_nights: i64,
    semi_major_axis: f64,
    eccentricity: f64,
    // Raw Kalman attributable state — not used by the (a, e) plot or the
    // lineage table, but carried through so the homepage's 3D view
    // (`orbit3d::server_fns::get_homepage_orbit3d`) can derive full
    // Keplerian elements (inclination, node, periapsis argument, mean
    // anomaly) without a schema migration: `kf_state` only stores
    // `semi_major_axis`/`eccentricity` as precomputed columns, but this
    // attributable state already carries everything needed to recompute the
    // *full* element set on demand, via the same
    // `attributable_to_cartesian` + `OrbitalElements::from_orbital_state`
    // path `homepage::family::classify_from_attributable_state` already
    // uses for family classification.
    ra: f64,
    dec: f64,
    ra_dot: f64,
    dec_dot: f64,
    rho: f64,
    rho_dot: f64,
    epoch: f64,
    r_obs_x: f64,
    r_obs_y: f64,
    r_obs_z: f64,
    v_obs_x: f64,
    v_obs_y: f64,
    v_obs_z: f64,
}

/// The one query the whole homepage is built from: every branch, joined to its
/// best hypothesis' state, precomputed by `fink-fat convert` in
/// `branch_best_state` (one row per branch, so this is a hash join of two
/// sequential scans). A branch without any hypothesis has no
/// `branch_best_state` row and is left out, as the former per-branch
/// best-hypothesis lookup did.
const SNAPSHOT_QUERY: &str = "
    SELECT b.branch_id, b.lineage_id, b.designation, b.lineage_designation,
           b.cumulative_llr, b.n_real_updates,
           b.arc_length_days, b.n_nights, b.median_inter_night_dt_days,
           b.well_sampled_nights,
           s.semi_major_axis, s.eccentricity,
           s.ra, s.dec, s.ra_dot, s.dec_dot, s.rho, s.rho_dot, s.epoch,
           s.r_obs_x, s.r_obs_y, s.r_obs_z, s.v_obs_x, s.v_obs_y, s.v_obs_z
    FROM branches b
    JOIN branch_best_state s ON s.branch_id = b.branch_id
";

/// The hypothesis count shown in the navbar, from the planner's row estimate
/// rather than a `COUNT(*)` over tens of millions of rows. `fink-fat convert`
/// runs `ANALYZE` at the end of every load, so the estimate is exact right
/// after a convert; `GREATEST` covers a never-analyzed table (`-1`).
const HYPOTHESIS_COUNT_QUERY: &str =
    "SELECT GREATEST(reltuples, 0)::bigint FROM pg_class WHERE oid = 'hypotheses'::regclass";

/// Cheap fingerprint of everything the snapshot is built from, see
/// [`DataMarker`].
const DATA_MARKER_QUERY: &str = "
    SELECT pg_relation_filenode('branches')::bigint AS branches_filenode,
           pg_relation_filenode('branch_best_state')::bigint AS best_state_filenode,
           COALESCE((
               SELECT sum(n_tup_ins + n_tup_upd + n_tup_del)
               FROM pg_stat_user_tables
               WHERE relid IN ('orbit_fits'::regclass, 'orbit_fit_failures'::regclass,
                               'cnd_queries'::regclass, 'skybot_obs_status'::regclass)
           ), 0)::bigint AS overlay_changes
";

/// Cheap fingerprint of the database content a [`Snapshot`] is built from:
/// two catalog lookups and one statistics view, no table scan. A snapshot
/// stores the marker read just before its build; a different marker means
/// the data changed since.
#[derive(Debug, Clone, PartialEq, Eq, sqlx::FromRow)]
pub struct DataMarker {
    /// Physical file of `branches`. `fink-fat convert` `TRUNCATE`s it, which
    /// always allocates a new file, so this changes on every convert.
    pub branches_filenode: i64,
    /// Same for `branch_best_state`, truncated in the same transaction.
    pub best_state_filenode: i64,
    /// Cumulative inserted + updated + deleted rows in the tables the
    /// explorer writes to while running (orbit fits, fit failures, CND and
    /// Skybot cross-matches), from Postgres' activity statistics. Only
    /// compared for equality: any write changes it. The statistics are
    /// flushed within about a second of a commit, and a statistics reset
    /// merely triggers one extra rebuild.
    pub overlay_changes: i64,
}

/// Reads the current [`DataMarker`].
///
/// # Arguments
/// * `pool` — the database pool.
///
/// # Return
/// The marker.
///
/// # Errors
/// The query failing, including when `branch_best_state` does not exist
/// yet, i.e. the database was loaded by a `fink-fat convert` that predates
/// it.
pub async fn read_data_marker(pool: &PgPool) -> Result<DataMarker, sqlx::Error> {
    sqlx::query_as(DATA_MARKER_QUERY).fetch_one(pool).await
}

/// One row of `params::ELIGIBLE_BRANCH_QUERY` — only `branch_id` is used
/// here, but that query also selects `lineage_designation` (needed by
/// `bulk_orbit_fit::run`, its other caller), so the shape has to match both
/// columns.
#[derive(sqlx::FromRow)]
struct EligibleBranchRow {
    branch_id: i64,
    #[allow(dead_code)]
    lineage_designation: String,
}

/// A branch's latest `orbit_fits` row, just the columns
/// [`assign_quality_tier`] needs. Mirrors what
/// `orbit_fit::latest::get_latest_orbit_fit_result` reads for the
/// single-lineage fit page, minus everything specific to rendering a fit
/// result.
const LATEST_ORBIT_FIT_QUERY: &str = "
    SELECT DISTINCT ON (branch_id) branch_id, fit_method, num_measurements, fitted_at
    FROM orbit_fits
    WHERE branch_id IS NOT NULL
    ORDER BY branch_id, fitted_at DESC
";

#[derive(sqlx::FromRow)]
struct LatestFitRow {
    branch_id: i64,
    fit_method: String,
    num_measurements: i32,
    fitted_at: chrono::DateTime<chrono::Utc>,
}

/// Each branch's best available `orbit_fits` row — n-body
/// differential-correction preferred over a Gauss-IOD-only fit whenever both
/// exist, via [`crate::best_orbit::PREFER_NBODY_ORDER_BY`], the same
/// priority `lineage_page::identity_card::best_orbit_fit` uses for the
/// identity card's own semi-major-axis/eccentricity display.
/// Distinct from [`LATEST_ORBIT_FIT_QUERY`] above, which deliberately stays
/// latest-by-time (it feeds quality-tier assessment: "what did the most
/// recent fit *attempt* do", not "what's the best orbit we have").
fn best_orbit_fit_query() -> String {
    format!(
        "SELECT DISTINCT ON (branch_id) branch_id, fit_method, keplerian
         FROM orbit_fits
         WHERE branch_id IS NOT NULL
         ORDER BY branch_id, {}",
        crate::best_orbit::PREFER_NBODY_ORDER_BY
    )
}

#[derive(sqlx::FromRow)]
struct BestOrbitFitRow {
    branch_id: i64,
    fit_method: String,
    keplerian: sqlx::types::Json<crate::fit_pipeline::fit::KeplerianView>,
}

/// Runs [`best_orbit_fit_query`] into a `branch_id -> OrbitCandidate` map. A
/// branch with no `orbit_fits` row at all is simply absent — [`assemble`]
/// falls back to the Kalman-bank state
/// (`SnapshotRow::semi_major_axis`/`::eccentricity`) for those, same as the
/// lineage page does when `best_orbit_fit` returns `None`.
///
/// # Errors
///
/// The query failing.
async fn build_best_orbit_index(
    pool: &PgPool,
) -> Result<HashMap<i64, crate::best_orbit::OrbitCandidate>, sqlx::Error> {
    use crate::best_orbit::OrbitCandidate;
    use crate::fit_pipeline::fit::FitMethod;

    let rows: Vec<BestOrbitFitRow> = sqlx::query_as(sqlx::AssertSqlSafe(best_orbit_fit_query()))
        .fetch_all(pool)
        .await?;
    Ok(rows
        .into_iter()
        .map(|r| {
            (
                r.branch_id,
                OrbitCandidate {
                    fit_method: FitMethod::from_column(&r.fit_method),
                    semi_major_axis_au: r.keplerian.0.semi_major_axis_au,
                    eccentricity: r.keplerian.0.eccentricity,
                },
            )
        })
        .collect())
}

/// A branch's latest recorded failed bulk-fit attempt, if any — see
/// `orbit_fit_failures` in `src/converter/sql.rs`.
const LATEST_ORBIT_FIT_FAILURE_QUERY: &str = "
    SELECT DISTINCT ON (branch_id) branch_id, attempted_at
    FROM orbit_fit_failures
    ORDER BY branch_id, attempted_at DESC
";

#[derive(sqlx::FromRow)]
struct LatestFailureRow {
    branch_id: i64,
    attempted_at: chrono::DateTime<chrono::Utc>,
}

/// Everything [`assign_quality_tier`] needs about every branch, keyed by
/// `branch_id`. Bundled into one struct — rather than threading four loose
/// maps through [`build`] and [`assemble`] — so the snapshot's quality-tier
/// inputs have one shape, fetched by [`build_quality_index`] and consumed
/// once per branch in [`assemble`].
struct QualityIndex {
    eligible: HashSet<i64>,
    latest_fit: HashMap<i64, LatestFit>,
    latest_failure_at: HashMap<i64, chrono::DateTime<chrono::Utc>>,
    /// Lineages with an active CND/Skybot cross-match hit — see
    /// [`crate::cross_match_status::build_cross_match_index`].
    cross_match: HashSet<Box<str>>,
}

impl QualityIndex {
    fn tier_for(&self, row: &SnapshotRow) -> QualityTier {
        assign_quality_tier(
            self.eligible.contains(&row.branch_id),
            self.latest_fit.get(&row.branch_id),
            self.latest_failure_at.get(&row.branch_id).copied(),
            row.n_nights,
            row.well_sampled_nights,
            self.cross_match.contains(row.lineage_designation.as_str()),
        )
    }
}

/// Branches the bulk orbit fit would attempt, see
/// [`params::ELIGIBLE_BRANCH_QUERY`].
///
/// # Errors
///
/// The query failing.
async fn fetch_eligible(pool: &PgPool) -> Result<HashSet<i64>, sqlx::Error> {
    let rows: Vec<EligibleBranchRow> = sqlx::query_as(params::ELIGIBLE_BRANCH_QUERY)
        .bind(params::MIN_OBSERVATIONS as i64)
        .bind(params::MIN_BASELINE_DAYS)
        .fetch_all(pool)
        .await?;
    Ok(rows.into_iter().map(|r| r.branch_id).collect())
}

/// Each branch's latest `orbit_fits` row, see [`LATEST_ORBIT_FIT_QUERY`].
///
/// # Errors
///
/// The query failing.
async fn fetch_latest_fits(pool: &PgPool) -> Result<HashMap<i64, LatestFit>, sqlx::Error> {
    let rows: Vec<LatestFitRow> = sqlx::query_as(LATEST_ORBIT_FIT_QUERY)
        .fetch_all(pool)
        .await?;
    Ok(rows
        .into_iter()
        .map(|r| {
            (
                r.branch_id,
                LatestFit {
                    fit_method: FitMethod::from_column(&r.fit_method),
                    num_measurements: r.num_measurements,
                    fitted_at: r.fitted_at,
                },
            )
        })
        .collect())
}

/// Each branch's latest failed bulk-fit attempt, see
/// [`LATEST_ORBIT_FIT_FAILURE_QUERY`].
///
/// # Errors
///
/// The query failing.
async fn fetch_latest_failures(
    pool: &PgPool,
) -> Result<HashMap<i64, chrono::DateTime<chrono::Utc>>, sqlx::Error> {
    let rows: Vec<LatestFailureRow> = sqlx::query_as(LATEST_ORBIT_FIT_FAILURE_QUERY)
        .fetch_all(pool)
        .await?;
    Ok(rows
        .into_iter()
        .map(|r| (r.branch_id, r.attempted_at))
        .collect())
}

/// Postgres sorts NaN as *larger* than any other float, Infinity included, so
/// an unsanitized `cumulative_llr` silently corrupted both "best branch per
/// lineage" selection and column sorting. The SQL `CASE` that used to do this
/// was duplicated across four queries; this is now the single copy, applied
/// once at snapshot build time.
fn sanitize_llr(llr: f64) -> f64 {
    if llr.is_finite() {
        llr
    } else {
        0.0
    }
}

/// One branch, carrying exactly the columns the table and the plot render.
/// `family` is the parsed enum (one byte) rather than the `TEXT` label, and
/// the orbital elements are `f32` because the plot only ever feeds them to
/// plotly as `f32`.
pub struct BranchRow {
    pub branch_id: i64,
    pub lineage_id: i64,
    pub designation: Box<str>,
    pub lineage_designation: Box<str>,
    /// Already passed through [`sanitize_llr`].
    pub cumulative_llr: f64,
    pub n_real_updates: i64,
    pub arc_length_days: f64,
    pub n_nights: i64,
    pub median_inter_night_dt_days: Option<f64>,
    pub family: DynamicalFamily,
    pub semi_major_axis: f32,
    pub eccentricity: f32,
    /// Resolved once at snapshot build time by [`assign_quality_tier`] from
    /// this branch's latest `orbit_fits`/`orbit_fit_failures` rows — see
    /// [`crate::homepage::quality_tier`].
    pub quality_tier: QualityTier,
    /// Raw Kalman attributable state, `f64` (unlike the `f32` orbital
    /// elements above, which only ever feed the (a, e) plot) — see the field
    /// comment on [`SnapshotRow`] for why this is carried at all. `epoch` is
    /// Modified Julian Date, Terrestrial Time.
    pub ra: f64,
    pub dec: f64,
    pub ra_dot: f64,
    pub dec_dot: f64,
    pub rho: f64,
    pub rho_dot: f64,
    pub epoch: f64,
    pub r_obs_x: f64,
    pub r_obs_y: f64,
    pub r_obs_z: f64,
    pub v_obs_x: f64,
    pub v_obs_y: f64,
    pub v_obs_z: f64,
}

/// One lineage: its best branch plus the others, held as indices into
/// [`Snapshot::branches`] so that grouping costs no allocation per lineage.
pub struct LineageEntry {
    pub lineage_id: i64,
    /// Index into [`Snapshot::branches`] of the highest-LLR branch — the one
    /// whose values the collapsed table row shows, and the one the family
    /// filter applies to.
    pub best: u32,
    /// Slice of [`Snapshot::others_idx`] holding this lineage's remaining
    /// branches, already ordered by descending LLR.
    pub others: Range<u32>,
    /// `lineage_designation` lowercased once, so the search box's
    /// case-insensitive substring match (the old `ILIKE '%…%'`) is a plain
    /// `str::contains` with no per-row allocation.
    search_key: Box<str>,
}

pub struct Snapshot {
    pub branches: Vec<BranchRow>,
    /// One entry per lineage, in ascending `lineage_id` order — which is also
    /// the listing's default order when no column sort is active.
    pub lineages: Vec<LineageEntry>,
    pub others_idx: Vec<u32>,
    /// Pre-sorted permutations of `lineages`, one per [`SortColumn`], indexed
    /// by [`SortColumn::index`] and always ascending; a descending sort walks
    /// the same permutation backwards. Sorting 214k entries per request would
    /// only cost ~20 ms, but paying it once at build time keeps every
    /// interaction bounded by the page size instead of the population.
    sorted: [Vec<u32>; SortColumn::ALL.len()],
    /// The (a, e) plot's series, already encoded with
    /// [`crate::homepage::series_wire::encode_series`]: one series per
    /// (family, quality tier) pair present in the data, ordered by family then
    /// tier — the order both legends list their chips in. Encoded once here so
    /// a request only clones this reference-counted buffer instead of cloning
    /// and serializing the whole population.
    pub series_payload: Bytes,
    /// HTTP entity tag of [`Snapshot::series_payload`], derived from its
    /// content (see [`crate::homepage::series_route::etag_for`]): a rebuild
    /// over unchanged data keeps it, so browsers keep their cached copy.
    pub series_etag: Box<str>,
    pub n_branches: i64,
    pub n_hypotheses: i64,
    pub n_lineages: i64,
    pub n_archived: i64,
    /// Increments on every successful build. The homepage's refresh button
    /// watches it to know when a rebuild it asked for has actually landed —
    /// until then the previous snapshot is still what gets served.
    pub version: u64,
    /// The database fingerprint read just before this snapshot's build.
    pub marker: DataMarker,
}

impl Snapshot {
    /// The lineage permutation to walk for a given sort, or `None` for the
    /// default `lineage_id ASC` order (which `lineages` is already in).
    pub fn order(&self, column: Option<SortColumn>) -> Option<&[u32]> {
        column.map(|c| self.sorted[c.index()].as_slice())
    }
}

static NEXT_VERSION: AtomicU64 = AtomicU64::new(1);

/// `None` until the first build finishes; callers surface that as a loading
/// state rather than blocking on it.
static SNAPSHOT: RwLock<Option<Arc<Snapshot>>> = RwLock::new(None);

/// The current snapshot, or `None` while the first build is still running.
///
/// Never blocks on a build, and never triggers one: builds are driven by
/// [`crate::homepage::snapshot_refresh`].
pub fn snapshot() -> Option<Arc<Snapshot>> {
    // Cloning an Arc is the whole critical section, so a plain std RwLock is
    // the right primitive; a poisoned lock still holds a valid `Option`.
    SNAPSHOT
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
}

/// Replaces the served snapshot.
///
/// # Arguments
/// * `snap` — the newly built snapshot.
pub(crate) fn publish(snap: Snapshot) {
    *SNAPSHOT.write().unwrap_or_else(PoisonError::into_inner) = Some(Arc::new(snap));
}

/// Why a snapshot build failed. The previous snapshot, if any, stays served.
#[derive(Debug, thiserror::Error)]
pub enum SnapshotBuildError {
    /// One of the build queries failed.
    #[error(transparent)]
    Sql(#[from] sqlx::Error),
    /// The (a, e) plot series could not be encoded.
    #[error(transparent)]
    Series(#[from] SeriesWireError),
    /// The CPU-bound assembly task panicked or was cancelled.
    #[error("snapshot assembly task failed: {0}")]
    Join(#[from] tokio::task::JoinError),
}

/// Reads everything the homepage needs and assembles a new [`Snapshot`].
///
/// The [`DataMarker`] is read first, so a change landing during the build is
/// picked up by the next check; the other queries then run concurrently
/// (the pool has room for all of them), and the CPU-bound assembly runs on
/// a blocking thread. Each step reports its completion through
/// [`tracked`].
///
/// # Return
/// The new snapshot; publishing it is left to the caller.
///
/// # Errors
/// [`SnapshotBuildError`] if any query, the series encoding or the assembly
/// task fails.
pub(crate) async fn build() -> Result<Snapshot, SnapshotBuildError> {
    let pool = get_pool().await;

    let marker = tracked(BuildStep::DataMarker, read_data_marker(pool)).await?;

    let (
        rows,
        n_hypotheses,
        n_archived,
        eligible,
        latest_fit,
        latest_failure_at,
        cross_match,
        best_orbits,
    ) = tokio::try_join!(
        tracked(
            BuildStep::Branches,
            sqlx::query_as::<_, SnapshotRow>(SNAPSHOT_QUERY).fetch_all(pool)
        ),
        tracked(
            BuildStep::HypothesisCount,
            sqlx::query_scalar::<_, i64>(HYPOTHESIS_COUNT_QUERY).fetch_one(pool)
        ),
        tracked(
            BuildStep::ArchivedCount,
            sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM archived_trajectories")
                .fetch_one(pool)
        ),
        tracked(BuildStep::Eligibility, fetch_eligible(pool)),
        tracked(BuildStep::LatestFits, fetch_latest_fits(pool)),
        tracked(BuildStep::FitFailures, fetch_latest_failures(pool)),
        tracked(
            BuildStep::CrossMatches,
            crate::cross_match_status::build_cross_match_index(pool)
        ),
        tracked(BuildStep::BestOrbits, build_best_orbit_index(pool)),
    )?;

    let quality = QualityIndex {
        eligible,
        latest_fit,
        latest_failure_at,
        cross_match,
    };
    let snap = tracked(
        BuildStep::Indexing,
        tokio::task::spawn_blocking(move || {
            assemble(
                rows,
                n_hypotheses,
                n_archived,
                &quality,
                &best_orbits,
                marker,
            )
        }),
    )
    .await??;
    Ok(snap)
}

/// Turns raw rows into the indexed, pre-sorted structure the server functions
/// read. Split out from [`build`] so it is exercisable without a database.
///
/// # Arguments
///
/// * `best_orbits` — see [`build_best_orbit_index`]; fed through
///   [`crate::best_orbit::resolve_best_orbit`] to prefer a branch's best
///   `orbit_fits` row over its Kalman-bank state for the (a, e) plot's
///   position and the family badge, matching the identity card's own
///   n-body-over-IOD-over-Kalman priority.
/// * `marker` — the [`DataMarker`] read before the queries, stored as-is.
///
/// # Errors
///
/// [`SeriesWireError`] if the (a, e) plot series cannot be encoded.
fn assemble(
    rows: Vec<SnapshotRow>,
    n_hypotheses: i64,
    n_archived: i64,
    quality: &QualityIndex,
    best_orbits: &HashMap<i64, crate::best_orbit::OrbitCandidate>,
    marker: DataMarker,
) -> Result<Snapshot, SeriesWireError> {
    let mut branches: Vec<BranchRow> = rows
        .into_iter()
        .map(|row| {
            // Prefer this branch's best `orbit_fits` row (n-body over
            // IOD-only) over the Kalman-bank state already joined into
            // `row`, via the same shared cascade the identity card uses.
            // Family is re-derived from whichever (a, e) wins rather than
            // trusting `row.dynamic_family` (which is only ever
            // Kalman-derived, baked in at `fink-fat convert` time) so the
            // badge never disagrees with the plotted point.
            let best_orbit = crate::best_orbit::resolve_best_orbit(
                best_orbits.get(&row.branch_id).copied(),
                (row.semi_major_axis, row.eccentricity),
            );
            let (semi_major_axis, eccentricity, family) = (
                best_orbit.semi_major_axis_au as f32,
                best_orbit.eccentricity as f32,
                best_orbit.family,
            );
            let quality_tier = quality.tier_for(&row);

            BranchRow {
                branch_id: row.branch_id,
                lineage_id: row.lineage_id,
                designation: row.designation.into_boxed_str(),
                lineage_designation: row.lineage_designation.into_boxed_str(),
                cumulative_llr: sanitize_llr(row.cumulative_llr),
                n_real_updates: row.n_real_updates,
                arc_length_days: row.arc_length_days,
                n_nights: row.n_nights,
                median_inter_night_dt_days: row.median_inter_night_dt_days,
                family,
                semi_major_axis,
                eccentricity,
                quality_tier,
                ra: row.ra,
                dec: row.dec,
                ra_dot: row.ra_dot,
                dec_dot: row.dec_dot,
                rho: row.rho,
                rho_dot: row.rho_dot,
                epoch: row.epoch,
                r_obs_x: row.r_obs_x,
                r_obs_y: row.r_obs_y,
                r_obs_z: row.r_obs_z,
                v_obs_x: row.v_obs_x,
                v_obs_y: row.v_obs_y,
                v_obs_z: row.v_obs_z,
            }
        })
        .collect();

    // Exactly what `DISTINCT ON (lineage_id) ... ORDER BY lineage_id,
    // cumulative_llr DESC` did: after this, each lineage's branches are
    // contiguous and its best one comes first.
    branches.sort_by(|a, b| {
        a.lineage_id.cmp(&b.lineage_id).then_with(|| {
            b.cumulative_llr
                .partial_cmp(&a.cumulative_llr)
                // Both sides are finite after `sanitize_llr`, so this arm is
                // unreachable; it keeps the comparator total regardless.
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });

    let mut lineages: Vec<LineageEntry> = Vec::new();
    let mut others_idx: Vec<u32> = Vec::new();

    for (i, branch) in branches.iter().enumerate() {
        let i = i as u32;
        match lineages.last_mut() {
            Some(last) if last.lineage_id == branch.lineage_id => {
                others_idx.push(i);
                last.others.end = others_idx.len() as u32;
            }
            _ => {
                let start = others_idx.len() as u32;
                lineages.push(LineageEntry {
                    lineage_id: branch.lineage_id,
                    best: i,
                    others: start..start,
                    search_key: branch.lineage_designation.to_lowercase().into_boxed_str(),
                });
            }
        }
    }

    let sorted = build_orders(&branches, &lineages);
    let series_payload = Bytes::from(encode_series(&build_series(&branches))?);
    let series_etag = etag_for(&series_payload).into_boxed_str();

    let n_branches = branches.len() as i64;
    let n_lineages = lineages.len() as i64;

    Ok(Snapshot {
        branches,
        lineages,
        others_idx,
        sorted,
        series_payload,
        series_etag,
        n_branches,
        n_hypotheses,
        n_lineages,
        n_archived,
        version: NEXT_VERSION.fetch_add(1, Ordering::SeqCst),
        marker,
    })
}

/// One ascending permutation of `lineages` per sortable column.
///
/// Every comparator falls back to `lineage_id` so the order is total and a
/// page boundary can never show or skip a row depending on sort stability.
fn build_orders(
    branches: &[BranchRow],
    lineages: &[LineageEntry],
) -> [Vec<u32>; SortColumn::ALL.len()] {
    let identity: Vec<u32> = (0..lineages.len() as u32).collect();

    SortColumn::ALL.map(|column| {
        let mut order = identity.clone();
        order.sort_by(|&a, &b| {
            let (la, lb) = (&lineages[a as usize], &lineages[b as usize]);
            let (ba, bb) = (&branches[la.best as usize], &branches[lb.best as usize]);

            let primary = match column {
                SortColumn::CumulativeLlr => total_cmp_f64(ba.cumulative_llr, bb.cumulative_llr),
                SortColumn::Updates => ba.n_real_updates.cmp(&bb.n_real_updates),
                // `DynamicalFamily`'s own `Ord` is declaration order, i.e.
                // increasing heliocentric distance — the same ranking the SQL
                // `CASE` over `family::ORDERED_LABELS` used to build.
                SortColumn::Family => ba.family.cmp(&bb.family),
                SortColumn::ArcLength => total_cmp_f64(ba.arc_length_days, bb.arc_length_days),
                SortColumn::Nights => ba.n_nights.cmp(&bb.n_nights),
                // Branches with <=1 night have no inter-night gap. `None`
                // sorts last here, and the descending sort reverses the
                // permutation *excluding* them (see `page`), reproducing the
                // old `NULLS LAST` in both directions.
                SortColumn::MedianInterNightDt => {
                    match (ba.median_inter_night_dt_days, bb.median_inter_night_dt_days) {
                        (Some(x), Some(y)) => total_cmp_f64(x, y),
                        (Some(_), None) => std::cmp::Ordering::Less,
                        (None, Some(_)) => std::cmp::Ordering::Greater,
                        (None, None) => std::cmp::Ordering::Equal,
                    }
                }
                // Reversed: `QualityTier`'s `Ord` ranks `WellSampledDiscovery`
                // (the best tier) as the *smallest* value, but every other
                // column here ranks "better" as larger — the default first
                // click is always `SortDirection::Desc`
                // (`branch_tab::toggle_sort`), which reverses this ascending
                // permutation, so a plain `cmp` would show `Ineligible`
                // first instead of `WellSampledDiscovery`.
                SortColumn::QualityTier => bb.quality_tier.cmp(&ba.quality_tier),
            };

            primary.then_with(|| la.lineage_id.cmp(&lb.lineage_id))
        });
        order
    })
}

/// `f64::total_cmp` is the right comparator here even though `sanitize_llr`
/// already removed the non-finite LLRs: `arc_length_days` and the median gap
/// come straight from the database and are only trusted to be numbers.
fn total_cmp_f64(a: f64, b: f64) -> std::cmp::Ordering {
    a.total_cmp(&b)
}

/// Groups every branch's (a, e) by (family, quality tier) for the population
/// plot: family drives the marker color, tier drives its shape/opacity/
/// border (`quality_tier::marker_for`), and plotly only offers one of each
/// per trace — so a family alone is no longer a fine enough grouping once
/// every branch also carries a tier. Each point's `lineage_designation` rides
/// along too, so the plot can label points on hover and link through to the
/// lineage page on click.
fn build_series(branches: &[BranchRow]) -> Vec<PlotSeries> {
    type GroupEntry = (Vec<f32>, Vec<f32>, Vec<Box<str>>);
    let mut grouped: HashMap<(DynamicalFamily, QualityTier), GroupEntry> = HashMap::new();
    for branch in branches {
        let entry = grouped
            .entry((branch.family, branch.quality_tier))
            .or_default();
        entry.0.push(branch.semi_major_axis);
        entry.1.push(branch.eccentricity);
        entry.2.push(branch.lineage_designation.clone());
    }

    let mut series: Vec<PlotSeries> = grouped
        .into_iter()
        .map(
            |((family, tier), (a, e, lineage_designations))| PlotSeries {
                family,
                tier,
                a,
                e,
                lineage_designations,
            },
        )
        .collect();
    // Family (increasing heliocentric distance) then tier (best to worst) —
    // the order both legends render their chips in.
    series.sort_by_key(|s| (s.family, s.tier));
    series
}

/// Which lineages a listing request selects, and in what order.
pub struct PageQuery<'a> {
    /// Already lowercased by the caller; empty means "match everything".
    pub search: &'a str,
    pub hidden_families: &'a [DynamicalFamily],
    pub hidden_tiers: &'a [QualityTier],
    pub sort_column: Option<SortColumn>,
    pub descending: bool,
    pub offset: usize,
    pub limit: usize,
}

impl Snapshot {
    /// Applies `query` and returns the matching lineage indices for the
    /// requested page, plus the total number of matches (which the pagination
    /// footer needs).
    ///
    /// Both the filter and the count run over every lineage — ~214k cheap
    /// predicate evaluations, well under a millisecond — while the ordering
    /// comes for free from the pre-sorted permutations.
    pub fn page(&self, query: &PageQuery<'_>) -> (Vec<u32>, i64) {
        let matches = |&idx: &u32| -> bool {
            let entry = &self.lineages[idx as usize];
            if !query.search.is_empty() && !entry.search_key.contains(query.search) {
                return false;
            }
            if !query.hidden_families.is_empty() {
                let family = self.branches[entry.best as usize].family;
                if query.hidden_families.contains(&family) {
                    return false;
                }
            }
            if !query.hidden_tiers.is_empty() {
                let tier = self.branches[entry.best as usize].quality_tier;
                if query.hidden_tiers.contains(&tier) {
                    return false;
                }
            }
            true
        };

        // The default (no column sort) order is ascending `lineage_id`, which
        // `lineages` is already stored in — no permutation needed, and no
        // direction toggle either, matching the old `ORDER BY lineage_id ASC`.
        let (order, descending): (Box<dyn Iterator<Item = u32> + '_>, bool) =
            match self.order(query.sort_column) {
                Some(perm) if query.descending => (Box::new(perm.iter().rev().copied()), true),
                Some(perm) => (Box::new(perm.iter().copied()), false),
                None => (Box::new(0..self.lineages.len() as u32), false),
            };

        // `NULLS LAST` regardless of direction: the ascending permutation puts
        // the null medians at the end, so a descending walk would otherwise
        // surface them first. They are dropped from the ordered walk and
        // appended after everything else instead.
        let nulls_last = descending && query.sort_column == Some(SortColumn::MedianInterNightDt);

        let mut ordered: Vec<u32> = Vec::new();
        let mut trailing_nulls: Vec<u32> = Vec::new();
        for idx in order {
            if !matches(&idx) {
                continue;
            }
            if nulls_last {
                let entry = &self.lineages[idx as usize];
                if self.branches[entry.best as usize]
                    .median_inter_night_dt_days
                    .is_none()
                {
                    trailing_nulls.push(idx);
                    continue;
                }
            }
            ordered.push(idx);
        }
        ordered.extend(trailing_nulls);

        let total = ordered.len() as i64;
        let page = ordered
            .into_iter()
            .skip(query.offset)
            .take(query.limit)
            .collect();

        (page, total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Representative `(semi_major_axis, eccentricity)` for a family label,
    /// chosen so `DynamicalFamily::classify` maps it back to that exact
    /// family — lets test call sites keep naming families by label (readable
    /// intent) even though `family` is now always re-derived from (a, e) via
    /// [`crate::best_orbit::resolve_best_orbit`] rather than trusted off a
    /// stored column. Covers only the labels this test module actually uses.
    fn elements_for_family(family: &str) -> (f64, f64) {
        match family {
            "MB>Inner" => (2.2, 0.1),
            "Trojan" => (5.0, 0.05),
            "Centaur" => (10.0, 0.1),
            "NEA>Apollo" => (1.5, 0.5),
            "KBO>SDO" => (50.0, 0.5),
            other => panic!("elements_for_family: no fixture for {other:?}"),
        }
    }

    /// Builds a one-branch-per-lineage snapshot; `median` carries the only
    /// nullable column, which is where the ordering rules are subtle.
    fn row(
        branch_id: i64,
        lineage_id: i64,
        llr: f64,
        family: &str,
        median: Option<f64>,
    ) -> SnapshotRow {
        let (semi_major_axis, eccentricity) = elements_for_family(family);
        SnapshotRow {
            branch_id,
            lineage_id,
            designation: format!("B{branch_id}"),
            lineage_designation: format!("L{lineage_id}"),
            cumulative_llr: llr,
            n_real_updates: 0,
            arc_length_days: 0.0,
            n_nights: if median.is_some() { 3 } else { 1 },
            median_inter_night_dt_days: median,
            well_sampled_nights: 0,
            semi_major_axis,
            eccentricity,
            // Not exercised by any test in this module (those all go through
            // the (a, e) plot / listing paths) — the attributable-state ->
            // Keplerian conversion that reads these lives in, and is tested
            // by, `orbit3d::server_fns`.
            ra: 0.0,
            dec: 0.0,
            ra_dot: 0.0,
            dec_dot: 0.0,
            rho: 0.0,
            rho_dot: 0.0,
            epoch: 60_000.0,
            r_obs_x: 0.0,
            r_obs_y: 0.0,
            r_obs_z: 0.0,
            v_obs_x: 0.0,
            v_obs_y: 0.0,
            v_obs_z: 0.0,
        }
    }

    fn query<'a>(
        sort_column: Option<SortColumn>,
        descending: bool,
        hidden: &'a [DynamicalFamily],
        search: &'a str,
    ) -> PageQuery<'a> {
        PageQuery {
            search,
            hidden_families: hidden,
            hidden_tiers: &[],
            sort_column,
            descending,
            offset: 0,
            limit: 100,
        }
    }

    /// Lineage ids of the returned page, so assertions read in domain terms
    /// rather than in internal indices.
    fn ids(snap: &Snapshot, q: &PageQuery<'_>) -> Vec<i64> {
        let (page, _) = snap.page(q);
        page.into_iter()
            .map(|i| snap.lineages[i as usize].lineage_id)
            .collect()
    }

    /// Every branch marked eligible, with no recorded fit or failure — i.e.
    /// every branch resolves to [`QualityTier::NotFitted`]. What every test
    /// not specifically exercising the quality-tier cascade wants: the
    /// family/sort/search/paging behaviour under test shouldn't have to
    /// depend on the (unrelated) fit/failure/eligibility tables.
    fn all_not_fitted(rows: &[SnapshotRow]) -> QualityIndex {
        QualityIndex {
            eligible: rows.iter().map(|r| r.branch_id).collect(),
            latest_fit: HashMap::new(),
            latest_failure_at: HashMap::new(),
            cross_match: HashSet::new(),
        }
    }

    fn test_marker() -> DataMarker {
        DataMarker {
            branches_filenode: 1,
            best_state_filenode: 2,
            overlay_changes: 0,
        }
    }

    /// [`assemble`] with every branch defaulted to [`QualityTier::NotFitted`]
    /// (see [`all_not_fitted`]) — the call every pre-existing test (family
    /// grouping, sorting, search, paging) uses.
    fn assemble_default(rows: Vec<SnapshotRow>, n_hypotheses: i64, n_archived: i64) -> Snapshot {
        let quality = all_not_fitted(&rows);
        assemble(
            rows,
            n_hypotheses,
            n_archived,
            &quality,
            &HashMap::new(),
            test_marker(),
        )
        .expect("assembles")
    }

    #[test]
    fn best_branch_per_lineage_is_the_highest_llr_one() {
        let snap = assemble_default(
            vec![
                row(1, 10, 1.0, "MB>Inner", None),
                row(2, 10, 5.0, "Trojan", None),
                row(3, 10, 3.0, "Centaur", None),
            ],
            0,
            0,
        );

        assert_eq!(snap.lineages.len(), 1);
        let entry = &snap.lineages[0];
        assert_eq!(snap.branches[entry.best as usize].branch_id, 2);

        // The others keep descending-LLR order, as the old
        // `ORDER BY lineage_id, cumulative_llr DESC` produced.
        let others: Vec<i64> = snap.others_idx
            [entry.others.start as usize..entry.others.end as usize]
            .iter()
            .map(|&i| snap.branches[i as usize].branch_id)
            .collect();
        assert_eq!(others, vec![3, 1]);
    }

    #[test]
    fn non_finite_llr_is_neutralized_rather_than_sorting_first() {
        // Postgres ranks NaN above every float, so an unsanitized NaN branch
        // would win "best of lineage" and head a descending sort.
        let snap = assemble_default(
            vec![
                row(1, 10, f64::NAN, "MB>Inner", None),
                row(2, 10, 2.0, "MB>Inner", None),
                row(3, 20, f64::INFINITY, "MB>Inner", None),
                row(4, 20, -1.0, "MB>Inner", None),
            ],
            0,
            0,
        );

        // Best of lineage 10 is the real 2.0, not the NaN.
        assert_eq!(snap.branches[snap.lineages[0].best as usize].branch_id, 2);
        // Best of lineage 20 is the sanitized Infinity (0.0), which still
        // beats -1.0 — the point is that it is no longer treated as huge.
        assert_eq!(
            snap.branches[snap.lineages[1].best as usize].cumulative_llr,
            0.0
        );

        let desc = ids(
            &snap,
            &query(Some(SortColumn::CumulativeLlr), true, &[], ""),
        );
        assert_eq!(desc, vec![10, 20]);
    }

    #[test]
    fn null_medians_sort_last_in_both_directions() {
        let snap = assemble_default(
            vec![
                row(1, 10, 0.0, "MB>Inner", Some(5.0)),
                row(2, 20, 0.0, "MB>Inner", None),
                row(3, 30, 0.0, "MB>Inner", Some(1.0)),
                row(4, 40, 0.0, "MB>Inner", None),
            ],
            0,
            0,
        );

        let asc = ids(
            &snap,
            &query(Some(SortColumn::MedianInterNightDt), false, &[], ""),
        );
        assert_eq!(asc, vec![30, 10, 20, 40]);

        // Reversing the permutation would otherwise surface the nulls first;
        // this is the `NULLS LAST` the SQL applied regardless of direction.
        // The nulls are all ties, so their `lineage_id` tiebreak reverses with
        // everything else — consistent with how ties behave in the non-null
        // part of the same descending walk.
        let desc = ids(
            &snap,
            &query(Some(SortColumn::MedianInterNightDt), true, &[], ""),
        );
        assert_eq!(desc, vec![10, 30, 40, 20]);
    }

    #[test]
    fn family_sorts_by_heliocentric_distance_not_alphabetically() {
        let snap = assemble_default(
            vec![
                row(1, 10, 0.0, "Trojan", None),
                row(2, 20, 0.0, "NEA>Apollo", None),
                row(3, 30, 0.0, "KBO>SDO", None),
                row(4, 40, 0.0, "MB>Inner", None),
            ],
            0,
            0,
        );

        let asc = ids(&snap, &query(Some(SortColumn::Family), false, &[], ""));
        assert_eq!(asc, vec![20, 40, 10, 30]);
    }

    #[test]
    fn hidden_families_are_matched_against_the_best_branch_only() {
        // Lineage 10's best branch is the Trojan; its hidden MB>Inner sibling
        // must not drag the whole lineage out of the listing.
        let snap = assemble_default(
            vec![
                row(1, 10, 5.0, "Trojan", None),
                row(2, 10, 1.0, "MB>Inner", None),
                row(3, 20, 5.0, "MB>Inner", None),
            ],
            0,
            0,
        );

        let hidden = [DynamicalFamily::MbInner];
        let visible = ids(&snap, &query(None, false, &hidden, ""));
        assert_eq!(visible, vec![10]);

        let (_, total) = snap.page(&query(None, false, &hidden, ""));
        assert_eq!(total, 1, "the count must agree with the listing");
    }

    #[test]
    fn search_is_a_case_insensitive_substring_match() {
        let mut rows = vec![row(1, 10, 0.0, "MB>Inner", None)];
        rows[0].lineage_designation = "FF25aBcD".to_string();
        rows.push(row(2, 20, 0.0, "MB>Inner", None));

        let snap = assemble_default(rows, 0, 0);

        assert_eq!(ids(&snap, &query(None, false, &[], "abc")), vec![10]);
        assert_eq!(ids(&snap, &query(None, false, &[], "l20")), vec![20]);
        assert!(ids(&snap, &query(None, false, &[], "nope")).is_empty());
        // Empty search matches everything, as the NULL `ILIKE` pattern did.
        assert_eq!(ids(&snap, &query(None, false, &[], "")), vec![10, 20]);
    }

    #[test]
    fn default_order_is_ascending_lineage_id_and_ignores_direction() {
        let snap = assemble_default(
            vec![
                row(1, 30, 0.0, "MB>Inner", None),
                row(2, 10, 0.0, "MB>Inner", None),
                row(3, 20, 0.0, "MB>Inner", None),
            ],
            0,
            0,
        );

        assert_eq!(ids(&snap, &query(None, false, &[], "")), vec![10, 20, 30]);
        assert_eq!(ids(&snap, &query(None, true, &[], "")), vec![10, 20, 30]);
    }

    #[test]
    fn paging_slices_the_filtered_order_and_reports_the_full_total() {
        let rows: Vec<SnapshotRow> = (0..10)
            .map(|i| row(i, i, i as f64, "MB>Inner", None))
            .collect();
        let snap = assemble_default(rows, 0, 0);

        let mut q = query(Some(SortColumn::CumulativeLlr), true, &[], "");
        q.offset = 3;
        q.limit = 4;

        let (page, total) = snap.page(&q);
        let page_ids: Vec<i64> = page
            .into_iter()
            .map(|i| snap.lineages[i as usize].lineage_id)
            .collect();

        assert_eq!(page_ids, vec![6, 5, 4, 3]);
        assert_eq!(total, 10, "total counts every match, not just the page");
    }

    #[test]
    fn plot_series_are_grouped_by_family_in_legend_order() {
        let snap = assemble_default(
            vec![
                row(1, 10, 0.0, "Trojan", None),
                row(2, 20, 0.0, "NEA>Apollo", None),
                row(3, 30, 0.0, "NEA>Apollo", None),
            ],
            0,
            0,
        );

        let series = crate::homepage::series_wire::decode_series(&snap.series_payload)
            .expect("snapshot payload decodes");
        let families: Vec<DynamicalFamily> = series.iter().map(|s| s.family).collect();
        assert_eq!(
            families,
            vec![DynamicalFamily::NeaApollo, DynamicalFamily::Trojan]
        );
        assert_eq!(series[0].a.len(), 2);
        assert_eq!(series[1].a.len(), 1);
        assert_eq!(
            series[0].lineage_designations,
            vec![Box::from("L20"), Box::from("L30")]
        );
    }

    /// Mirrors `hidden_families_are_matched_against_the_best_branch_only`:
    /// a hidden tier on a non-best branch must not drag its lineage out.
    #[test]
    fn hidden_tiers_are_matched_against_the_best_branch_only() {
        let rows = vec![
            row(1, 10, 5.0, "Trojan", None),   // best of lineage 10 — eligible
            row(2, 10, 1.0, "MB>Inner", None), // sibling — left ineligible
            row(3, 20, 5.0, "MB>Inner", None), // best of lineage 20 — left ineligible
        ];
        let quality = QualityIndex {
            // Branches 2 and 3 are left out of the eligible set on purpose.
            eligible: [1].into_iter().collect(),
            latest_fit: HashMap::new(),
            latest_failure_at: HashMap::new(),
            cross_match: HashSet::new(),
        };
        let snap =
            assemble(rows, 0, 0, &quality, &HashMap::new(), test_marker()).expect("assembles");

        assert_eq!(
            snap.branches[snap.lineages[0].best as usize].quality_tier,
            QualityTier::NotFitted
        );
        assert_eq!(
            snap.branches[snap.others_idx[snap.lineages[0].others.start as usize] as usize]
                .quality_tier,
            QualityTier::Ineligible
        );

        let hidden = [QualityTier::Ineligible];
        let visible = ids(
            &snap,
            &PageQuery {
                search: "",
                hidden_families: &[],
                hidden_tiers: &hidden,
                sort_column: None,
                descending: false,
                offset: 0,
                limit: 100,
            },
        );
        // Lineage 20's best branch is Ineligible (hidden), so it drops out;
        // lineage 10's best branch is NotFitted (not hidden) even though its
        // sibling is Ineligible — the sibling must not hide lineage 10.
        assert_eq!(visible, vec![10]);
    }

    #[test]
    fn quality_tier_sorts_best_first_when_descending() {
        let rows = vec![
            row(1, 10, 0.0, "MB>Inner", None),
            row(2, 20, 0.0, "MB>Inner", None),
            row(3, 30, 0.0, "MB>Inner", None),
        ];
        // Lineage 10 -> Ineligible, 20 -> NotFitted (default), 30 -> eligible
        // but also NotFitted (no fit/failure recorded) — distinguished from
        // 10 only by eligibility.
        let quality = QualityIndex {
            eligible: [2, 3].into_iter().collect(),
            latest_fit: HashMap::new(),
            latest_failure_at: HashMap::new(),
            cross_match: HashSet::new(),
        };
        let snap =
            assemble(rows, 0, 0, &quality, &HashMap::new(), test_marker()).expect("assembles");

        // Default click direction (`SortDirection::Desc`) must surface the
        // best tier first, exactly like every other column's "larger first"
        // convention — see the comment on `SortColumn::QualityTier` in
        // `build_orders`. Lineages 20 and 30 tie on tier (both `NotFitted`);
        // descending reverses their ascending (lineage_id-ascending) tiebreak
        // too, same as every other column's ties, so 30 comes before 20.
        let desc = ids(&snap, &query(Some(SortColumn::QualityTier), true, &[], ""));
        assert_eq!(desc, vec![30, 20, 10]);
    }
}
