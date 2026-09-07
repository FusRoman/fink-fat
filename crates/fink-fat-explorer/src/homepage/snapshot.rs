//! In-RAM snapshot of everything the homepage displays.
//!
//! The homepage used to answer every user interaction — a legend toggle, a
//! column sort, a page change, a keystroke in the search box — with fresh SQL.
//! The expensive part was never `branches` (a few hundred thousand narrow
//! rows) but `kf_state`: `dynamic_family` lives there, one row per hypothesis,
//! and reaching it means a `LATERAL` best-hypothesis lookup plus a primary-key
//! probe *per branch*. On a loaded instance that is ~214k random reads into a
//! ~20 GB table, and `list_lineages` paid for it twice per call (once for the
//! count, once for the page).
//!
//! Nothing in this data changes while the server runs: `fink-fat convert`
//! rewrites these tables wholesale (`TRUNCATE ... CASCADE` then `COPY`), and
//! the explorer only ever writes to `orbit_fits`. So the whole working set is
//! loaded once into an immutable [`Snapshot`] — about 35 MB at 214k branches —
//! and every homepage server function becomes an in-memory scan.
//!
//! Server-only: `main.rs` gates the module on the `server` feature, so none of
//! this reaches the wasm bundle.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use tokio::sync::OnceCell;

use crate::get_pool;
use crate::homepage::dynamic_pop_plot::FamilySeries;
use crate::homepage::family::DynamicalFamily;
use crate::homepage::interaction::SortColumn;

/// How long a snapshot is served before a rebuild is triggered in the
/// background. The data only changes on a `fink-fat convert` re-run, so this
/// is a safety net rather than a freshness guarantee — the navbar's refresh
/// button is the intended way to pick up a new conversion immediately.
const SNAPSHOT_TTL: Duration = Duration::from_secs(15 * 60);

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
    dynamic_family: String,
    semi_major_axis: f64,
    eccentricity: f64,
}

/// The one query the whole homepage is built from: every branch, joined to its
/// best hypothesis' Kalman state for the precomputed family and orbital
/// elements. Served by `idx_hypotheses_branch_log_weight` and `kf_state`'s
/// primary key.
const SNAPSHOT_QUERY: &str = "
    SELECT b.branch_id, b.lineage_id, b.designation, b.lineage_designation,
           b.cumulative_llr, b.n_real_updates,
           b.arc_length_days, b.n_nights, b.median_inter_night_dt_days,
           ks.dynamic_family, ks.semi_major_axis, ks.eccentricity
    FROM branches b
    CROSS JOIN LATERAL (
        SELECT hypothesis_id
        FROM hypotheses h
        WHERE h.branch_id = b.branch_id
        ORDER BY h.log_weight DESC
        LIMIT 1
    ) bh
    JOIN kf_state ks ON ks.hypothesis_id = bh.hypothesis_id
";

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
    /// Ready-to-serialize plot series, grouped by family in
    /// `DynamicalFamily`'s `Ord` (increasing heliocentric distance), which is
    /// the order the legend lists them in. Grouping once here, rather than in
    /// the browser on every data load, is also what lets the plot ship two
    /// flat coordinate arrays per family instead of one JSON object per point.
    pub series: Vec<FamilySeries>,
    pub n_branches: i64,
    pub n_hypotheses: i64,
    pub n_lineages: i64,
    pub n_archived: i64,
    /// Increments on every successful build. The homepage's refresh button
    /// watches it to know when a rebuild it asked for has actually landed —
    /// until then the previous snapshot is still what gets served.
    pub version: u64,
    built_at: Instant,
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
static SNAPSHOT: OnceCell<RwLock<Option<Arc<Snapshot>>>> = OnceCell::const_new();

/// Guards against concurrent rebuilds: several requests can notice the same
/// expired snapshot at once, and the build is minutes of work on a large
/// instance.
static REBUILDING: AtomicBool = AtomicBool::new(false);

async fn cell() -> &'static RwLock<Option<Arc<Snapshot>>> {
    SNAPSHOT.get_or_init(|| async { RwLock::new(None) }).await
}

/// The current snapshot, or `None` while the first build is still running.
///
/// Never blocks on a build: if the snapshot has aged past [`SNAPSHOT_TTL`] the
/// stale one is returned and a rebuild is spawned, so a TTL expiry is invisible
/// to whoever happens to make the next request.
pub async fn snapshot() -> Option<Arc<Snapshot>> {
    let current = {
        // Cloning an Arc is the whole critical section — no await is held
        // across the lock, so a plain std RwLock is the right primitive.
        let guard = cell().await.read().expect("snapshot lock poisoned");
        guard.clone()
    };

    match &current {
        Some(snap) if snap.built_at.elapsed() >= SNAPSHOT_TTL => spawn_rebuild(),
        Some(_) => {}
        // First build: kick it off, and let the caller show a loading state.
        None => spawn_rebuild(),
    }

    current
}

/// Force a rebuild regardless of age — what the navbar's refresh button calls
/// after a `fink-fat convert`. Returns immediately; the old snapshot keeps
/// being served until the new one is ready.
pub fn request_refresh() {
    spawn_rebuild();
}

/// Clears [`REBUILDING`] on drop, so a panicking build cannot wedge the flag
/// at `true` and block every future rebuild for the life of the process.
struct RebuildGuard;

impl Drop for RebuildGuard {
    fn drop(&mut self) {
        REBUILDING.store(false, Ordering::SeqCst);
    }
}

fn spawn_rebuild() {
    // `swap` rather than a load-then-store: two requests hitting an expired
    // snapshot simultaneously must not both start a multi-minute build.
    if REBUILDING.swap(true, Ordering::SeqCst) {
        return;
    }

    tokio::spawn(async move {
        let _guard = RebuildGuard;

        match build().await {
            Ok(snap) => {
                let mut guard = cell().await.write().expect("snapshot lock poisoned");
                *guard = Some(Arc::new(snap));
            }
            Err(e) => {
                // A failed rebuild leaves the previous snapshot in place; the
                // next request simply retries.
                tracing::error!("homepage snapshot build failed: {e}");
            }
        }
    });
}

async fn build() -> Result<Snapshot, sqlx::Error> {
    let pool = get_pool().await;

    let rows: Vec<SnapshotRow> = sqlx::query_as(SNAPSHOT_QUERY).fetch_all(pool).await?;

    // The two counts the snapshot cannot derive from `branches`. `n_branches`
    // and `n_lineages` come out of the rows themselves.
    let (n_hypotheses,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM hypotheses")
        .fetch_one(pool)
        .await?;
    let (n_archived,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM archived_trajectories")
        .fetch_one(pool)
        .await?;

    Ok(assemble(rows, n_hypotheses, n_archived))
}

/// Turns raw rows into the indexed, pre-sorted structure the server functions
/// read. Split out from [`build`] so it is exercisable without a database.
fn assemble(rows: Vec<SnapshotRow>, n_hypotheses: i64, n_archived: i64) -> Snapshot {
    let mut branches: Vec<BranchRow> = rows
        .into_iter()
        .map(|row| BranchRow {
            branch_id: row.branch_id,
            lineage_id: row.lineage_id,
            designation: row.designation.into_boxed_str(),
            lineage_designation: row.lineage_designation.into_boxed_str(),
            cumulative_llr: sanitize_llr(row.cumulative_llr),
            n_real_updates: row.n_real_updates,
            arc_length_days: row.arc_length_days,
            n_nights: row.n_nights,
            median_inter_night_dt_days: row.median_inter_night_dt_days,
            family: DynamicalFamily::from_label(&row.dynamic_family),
            semi_major_axis: row.semi_major_axis as f32,
            eccentricity: row.eccentricity as f32,
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
    let series = build_series(&branches);

    let n_branches = branches.len() as i64;
    let n_lineages = lineages.len() as i64;

    Snapshot {
        branches,
        lineages,
        others_idx,
        sorted,
        series,
        n_branches,
        n_hypotheses,
        n_lineages,
        n_archived,
        version: NEXT_VERSION.fetch_add(1, Ordering::SeqCst),
        built_at: Instant::now(),
    }
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

/// Groups every branch's (a, e) by family for the population plot.
fn build_series(branches: &[BranchRow]) -> Vec<FamilySeries> {
    let mut grouped: HashMap<DynamicalFamily, (Vec<f32>, Vec<f32>)> = HashMap::new();
    for branch in branches {
        let entry = grouped.entry(branch.family).or_default();
        entry.0.push(branch.semi_major_axis);
        entry.1.push(branch.eccentricity);
    }

    let mut series: Vec<FamilySeries> = grouped
        .into_iter()
        .map(|(family, (a, e))| FamilySeries { family, a, e })
        .collect();
    // Increasing heliocentric distance, the order the legend renders in.
    series.sort_by_key(|s| s.family);
    series
}

/// Which lineages a listing request selects, and in what order.
pub struct PageQuery<'a> {
    /// Already lowercased by the caller; empty means "match everything".
    pub search: &'a str,
    pub hidden_families: &'a [DynamicalFamily],
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

    /// Builds a one-branch-per-lineage snapshot; `median` carries the only
    /// nullable column, which is where the ordering rules are subtle.
    fn row(
        branch_id: i64,
        lineage_id: i64,
        llr: f64,
        family: &str,
        median: Option<f64>,
    ) -> SnapshotRow {
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
            dynamic_family: family.to_string(),
            semi_major_axis: 2.5,
            eccentricity: 0.1,
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

    #[test]
    fn best_branch_per_lineage_is_the_highest_llr_one() {
        let snap = assemble(
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
        let snap = assemble(
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
        let snap = assemble(
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
        let snap = assemble(
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
        let snap = assemble(
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

        let snap = assemble(rows, 0, 0);

        assert_eq!(ids(&snap, &query(None, false, &[], "abc")), vec![10]);
        assert_eq!(ids(&snap, &query(None, false, &[], "l20")), vec![20]);
        assert!(ids(&snap, &query(None, false, &[], "nope")).is_empty());
        // Empty search matches everything, as the NULL `ILIKE` pattern did.
        assert_eq!(ids(&snap, &query(None, false, &[], "")), vec![10, 20]);
    }

    #[test]
    fn default_order_is_ascending_lineage_id_and_ignores_direction() {
        let snap = assemble(
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
        let snap = assemble(rows, 0, 0);

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
        let snap = assemble(
            vec![
                row(1, 10, 0.0, "Trojan", None),
                row(2, 20, 0.0, "NEA>Apollo", None),
                row(3, 30, 0.0, "NEA>Apollo", None),
            ],
            0,
            0,
        );

        let families: Vec<DynamicalFamily> = snap.series.iter().map(|s| s.family).collect();
        assert_eq!(
            families,
            vec![DynamicalFamily::NeaApollo, DynamicalFamily::Trojan]
        );
        assert_eq!(snap.series[0].a.len(), 2);
        assert_eq!(snap.series[1].a.len(), 1);
    }
}
