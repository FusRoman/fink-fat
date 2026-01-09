use fink_fat_engine::{
    alerts::AlertStore, engine_config::score_config::ScoreConfig, graph::score::ScoredEdge, night_id::NightId, seeding::{pairs::Pairs, seed_id::SeedId, seed_node::SeedNode, triplets::Triplets}
};

/// Build a unified list of [`SeedNode`] from pre-generated pairs and triplets.
///
/// This helper is meant for evaluation workloads where you already have:
/// - `pairs: Pairs` (ordered `(a, b)`),
/// - `triplets: Triplets` (ordered `(a, b, c)`),
/// and you want a single `Vec<SeedNode>` containing all seeds.
///
/// The function guarantees that `SeedId`s are **unique** across the returned
/// vector by assigning them sequentially:
/// - pairs first,
/// - then triplets with an offset.
///
/// Arguments
/// ---------
/// * `store` – Engine alert store used to resolve `AlertId -> Alert`.
/// * `night_id` – Night identifier attached to all returned seeds.
/// * `pairs` – Pair list used to build linear tangent-plane seeds.
/// * `triplets` – Triplet list used to build quadratic tangent-plane seeds.
/// * `max_speed_rad_per_day` – Optional max angular speed filter applied to
///   pair seeds only (triplet seeds are always kept).
///
/// Return
/// ------
/// * `Vec<SeedNode>` – Concatenated list of all successfully built seeds.
///   Pair seeds that fail the optional speed filter are skipped.
pub fn build_seed_nodes_from_pairs_and_triplets(
    store: &AlertStore,
    night_id: NightId,
    pairs: &Pairs,
    triplets: &Triplets,
    max_speed_rad_per_day: Option<f64>,
) -> Vec<SeedNode> {
    // Worst-case capacity: all pairs pass + all triplets.
    let mut out: Vec<SeedNode> = Vec::with_capacity(pairs.len() + triplets.len());

    // ---------------------------------------------------------------------
    // 1) Pair seeds (may be filtered by max_speed_rad_per_day)
    // ---------------------------------------------------------------------
    for &p in pairs.iter() {
        let alert_a = &store.alerts[p.a.idx()];
        let alert_b = &store.alerts[p.b.idx()];

        let seed_id = SeedId::new(out.len() as u64);

        if let Some(seed) =
            SeedNode::from_pair(seed_id, night_id, alert_a, alert_b, max_speed_rad_per_day)
        {
            out.push(seed);
        }
    }

    // ---------------------------------------------------------------------
    // 2) Triplet seeds (no filtering here, always built)
    // ---------------------------------------------------------------------
    for &t in triplets.iter() {
        let alert_a = &store.alerts[t.a.idx()];
        let alert_b = &store.alerts[t.b.idx()];
        let alert_c = &store.alerts[t.c.idx()];

        let seed_id = SeedId::new(out.len() as u64);

        let seed = SeedNode::from_triplet(seed_id, night_id, alert_a, alert_b, alert_c);
        out.push(seed);
    }

    out
}

/// Score all directed edges `i -> j` among a set of seeds.
///
/// This function enumerates all ordered pairs `(i, j)` with `i != j`,
/// applies [`ScoredEdge::score`], and returns only the edges that pass
/// the hard gates (i.e. `Some(ScoredEdge)`).
///
/// Arguments
/// ---------
/// * `seeds` – Seed nodes to connect (can span multiple nights/revisits).
/// * `cfg` – Inter-night scoring configuration (gates + weights).
/// * `delta_revisit_fn` – Closure used to compute the revisit gap Δ between
///   `i` and `j` (e.g. 1 for adjacent revisits, or derived from night ids).
///
/// Return
/// ------
/// * `Vec<ScoredEdge>` – All accepted scored edges, in deterministic order:
///   increasing `i` index, then increasing `j` index.
pub fn score_all_seed_edges<F>(
    seeds: &[SeedNode],
    cfg: &ScoreConfig,
    mut delta_revisit_fn: F,
) -> Vec<ScoredEdge>
where
    F: FnMut(&SeedNode, &SeedNode) -> u32,
{
    let n = seeds.len();
    // Rough heuristic capacity: assume only a fraction passes gating.
    let mut out = Vec::with_capacity(n.saturating_mul(8));

    for a in 0..n {
        let i = &seeds[a];
        for b in 0..n {
            if a == b {
                continue;
            }
            let j = &seeds[b];
            let delta = delta_revisit_fn(i, j);

            if let Some(edge) = ScoredEdge::score(i, j, cfg, delta) {
                out.push(edge);
            }
        }
    }
    out
}
