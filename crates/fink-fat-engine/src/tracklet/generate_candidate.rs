use ahash::{HashMap, HashMapExt, HashSet};
use photom::{
    NightId,
    coordinates::ecliptic::EclipticCoordCov,
    observation_dataset::{ObsDataset, iter::MemLayoutObservations, observation::Observation},
};

use crate::{
    engine_config::EngineConfig,
    error::EngineError,
    pipeline::PipelineStage,
    propagator::index::NightIndex,
    tracklet::{ScoredObservation, Tracklet, track_storage::TrackId},
};

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// A tracklet paired with its scored candidate observations.
///
/// Each [`ScoredObservation`] carries the squared Mahalanobis distance $d^2$
/// computed during the spatial association gate. This distance is reused in
/// downstream steps (magnitude filter, deduplication) without recomputation.
///
/// Lifetimes
/// ---------
/// * `'t` – Lifetime of the tracklet references (active tracklet slice).
/// * `'o` – Lifetime of the observation references (night observation slice).
type ScoredAssociation<'t, 'o> = (&'t Tracklet, Vec<ScoredObservation<'o>>);

/// A list of scored associations, one entry per tracklet that survived the
/// spatial association gate.
type ScoredAssociations<'t, 'o> = Vec<ScoredAssociation<'t, 'o>>;

/// A tracklet paired with its final deduplicated candidate observations.
///
/// Produced after the magnitude filter and exposure-time deduplication steps.
/// The [`ScoredObservation`] wrapper is dropped at the deduplication stage;
/// only the observation reference is retained.
///
/// Lifetimes
/// ---------
/// * `'t` – Lifetime of the tracklet references.
/// * `'o` – Lifetime of the observation references.
pub type Association<'t, 'o> = (&'t Tracklet, Vec<&'o Observation>);

/// A list of final associations, one entry per tracklet that survived all
/// pipeline filters.
pub type Associations<'o> = Vec<(TrackId, Vec<&'o Observation>)>;

// ---------------------------------------------------------------------------
// Night materialization
// ---------------------------------------------------------------------------

/// Materialize a night's observations from the dataset as a contiguous slice.
///
/// The observation dataset supports two internal memory layouts:
/// `Contiguous` (a single flat slice) and `Split` (multiple disjoint buffers).
/// This function requires the `Contiguous` layout because downstream code
/// borrows `&[Observation]` directly — no copy is made.
///
/// Arguments
/// ---------
/// * `obs_dataset` – Source observation dataset.
/// * `night`       – Identifier of the night to materialize.
///
/// Return
/// ------
/// * `Ok(&[Observation])` – Borrowed slice of the night's observations.
/// * `Err(EngineError::StageFailed)` – If the night is absent from the dataset
///   or its memory layout is `Split`.
pub fn materialize_contiguous_night<'o>(
    obs_dataset: &'o ObsDataset,
    night: &NightId,
) -> Result<&'o [Observation], EngineError> {
    match obs_dataset
        .materialize_night(night)
        .ok_or_else(|| EngineError::StageFailed {
            stage: PipelineStage::BuildSeeds,
            message: format!("night {night} not found in observation dataset"),
        })? {
        MemLayoutObservations::Contiguous(slice) => Ok(slice),
        MemLayoutObservations::Split(_) => Err(EngineError::StageFailed {
            stage: PipelineStage::BuildSeeds,
            message: format!(
                "night {night} is in split layout, expected contiguous; \
                 cannot borrow &[Observation] slice"
            ),
        }),
    }
}

// ---------------------------------------------------------------------------
// HEALPix prefilter
// ---------------------------------------------------------------------------

/// Map an ecliptic coordinate to its HEALPix nested-scheme pixel index.
///
/// HEALPix partitions the sphere into $12 \times 4^{\text{depth}}$ equal-area
/// pixels. At `depth = 5` each pixel subtends roughly $1.8°$, which is
/// appropriate for a coarse sky-coverage prefilter.
///
/// Arguments
/// ---------
/// * `ecl`   – Ecliptic coordinate (longitude and latitude in radians).
/// * `depth` – HEALPix resolution parameter (order).
///
/// Return
/// ------
/// * `u64` – Nested pixel index containing `ecl`.
fn ecl_to_healpix_pixel(ecl: &EclipticCoordCov, depth: u8) -> u64 {
    cdshealpix::nested::hash(depth, ecl.coord.lon, ecl.coord.lat)
}

/// Return the HEALPix pixel that contains a tracklet's current predicted
/// position.
///
/// The position is taken from the tracklet's internal state, which represents
/// the propagated sky position at the tracklet's reference epoch.
///
/// Arguments
/// ---------
/// * `tracklet` – Source tracklet.
/// * `depth`    – HEALPix resolution parameter.
///
/// Return
/// ------
/// * `u64` – Nested pixel index of the tracklet's predicted sky position.
fn tracklet_healpix_pixel(tracklet: &Tracklet, depth: u8) -> u64 {
    let position = tracklet
        .state()
        .expect(&format!(
            "trying to get eclipticstate from an orbit state: \ntracklet_id: {:?}",
            tracklet.key()
        ))
        .position();
    cdshealpix::nested::hash(depth, position.lon, position.lat)
}

/// Test whether a tracklet's predicted position overlaps a night's sky
/// coverage.
///
/// A tracklet is considered to overlap if its HEALPix pixel either:
/// - is directly present in `occupied_pixels`, or
/// - is a neighbour (including diagonal) of at least one occupied pixel.
///
/// The one-pixel margin absorbs positional uncertainty near cell boundaries
/// and avoids discarding tracklets whose prediction falls just outside an
/// occupied cell.
///
/// Arguments
/// ---------
/// * `tracklet`        – Tracklet whose predicted position is tested.
/// * `occupied_pixels` – Set of HEALPix pixels containing at least one alert
///   from the target night.
/// * `depth`           – HEALPix resolution parameter (must match the depth
///   used to build `occupied_pixels`).
///
/// Return
/// ------
/// * `true`  – The tracklet's predicted position is in or adjacent to an
///   occupied pixel.
/// * `false` – The tracklet's predicted position is fully outside the night's
///   footprint.
fn tracklet_in_occupied_region(
    tracklet: &Tracklet,
    occupied_pixels: &HashSet<u64>,
    depth: u8,
) -> bool {
    let pix = tracklet_healpix_pixel(tracklet, depth);

    // Direct hit — tracklet lands in an occupied cell.
    if occupied_pixels.contains(&pix) {
        return true;
    }

    // One-pixel margin — check all 8 neighbours (diagonal included).
    cdshealpix::nested::neighbours(depth, pix, true)
        .values_vec()
        .iter()
        .any(|&p| occupied_pixels.contains(&p))
}

/// Build the set of HEALPix pixels covered by a night's alerts.
///
/// The resulting set is used as a fast sky-footprint index: a tracklet whose
/// predicted position falls outside every pixel in this set (and all their
/// neighbours) cannot possibly match any alert from that night, and is
/// discarded before the more expensive spatial association step.
///
/// Ecliptic coordinates are taken from the [`NightIndex`] to avoid
/// recomputing equatorial-to-ecliptic conversions.
///
/// Arguments
/// ---------
/// * `ecl_coords` – Precomputed ecliptic coordinates of the night's alerts
///   (same order as the observation slice stored in the index).
/// * `depth`      – HEALPix resolution parameter.
///
/// Return
/// ------
/// * `HashSet<u64>` – Pixel indices (nested scheme) occupied by at least one
///   alert.
fn occupied_healpix_pixels(ecl_coords: &[EclipticCoordCov], depth: u8) -> HashSet<u64> {
    ecl_coords
        .iter()
        .map(|ecl| ecl_to_healpix_pixel(ecl, depth))
        .collect()
}

// ---------------------------------------------------------------------------
// Association
// ---------------------------------------------------------------------------

/// Attempt to associate a single tracklet with observations from the target
/// night.
///
/// Delegates to [`Tracklet::associate_tracklet_to_night`] and wraps the result
/// as a [`ScoredAssociation`]. Each candidate observation is paired with its
/// squared Mahalanobis distance $d^2$ computed during the association gate,
/// avoiding any recomputation in downstream steps.
///
/// Arguments
/// ---------
/// * `tracklet`          – Tracklet to associate.
/// * `index`             – Spatial and temporal index of the night's alerts.
/// * `association_sigma` – Angular search radius in units of $\sigma$.
/// * `chi2_threshold`    – Maximum allowed $\chi^2$ residual.
///
/// Return
/// ------
/// * `Some(ScoredAssociation)` – The tracklet paired with its scored
///   candidates (at least one match found).
/// * `None` – No observation passed the association criteria.
fn associate_tracklet<'t, 'o>(
    tracklet: &'t Tracklet,
    index: &NightIndex<'o>,
    association_sigma: f64,
    chi2_threshold: f64,
) -> Option<ScoredAssociation<'t, 'o>> {
    let matches = tracklet.associate_tracklet_to_night(index, association_sigma, chi2_threshold);
    (!matches.is_empty()).then_some((tracklet, matches))
}

// ---------------------------------------------------------------------------
// Magnitude filter
// ---------------------------------------------------------------------------

/// Filter a list of scored candidate observations by magnitude proximity.
///
/// Only observations whose magnitude lies within `max_diff` magnitudes of the
/// reference value `ref_mag` are retained:
///
/// $$|m_{\text{obs}} - m_{\text{ref}}| \leq \Delta m_{\max}$$
///
/// The [`ScoredObservation`] wrapper is preserved so that the precomputed
/// Mahalanobis distance remains available in downstream steps.
///
/// Arguments
/// ---------
/// * `candidates` – Scored candidate observations to filter.
/// * `ref_mag`    – Reference magnitude (mean magnitude of the tracklet).
/// * `max_diff`   – Maximum allowed magnitude difference $\Delta m_{\max}$.
///
/// Return
/// ------
/// * `Vec<ScoredObservation>` – Candidates that satisfy the magnitude
///   criterion.
fn filter_scored_by_magnitude<'o>(
    candidates: Vec<ScoredObservation<'o>>,
    ref_mag: f64,
    max_diff: f64,
) -> Vec<ScoredObservation<'o>> {
    candidates
        .into_iter()
        .filter(|s| (s.obs.photometry().magnitude - ref_mag).abs() <= max_diff)
        .collect()
}

/// Apply the magnitude filter to a single tracklet association.
///
/// The reference magnitude is computed as the mean magnitude of the tracklet's
/// existing observations. If the mean cannot be computed (e.g. all photometric
/// values are missing), the association is dropped entirely.
///
/// Arguments
/// ---------
/// * `association`        – Scored association to filter.
/// * `obs_dataset`        – Dataset used to retrieve photometric values.
/// * `max_magnitude_diff` – Maximum allowed magnitude difference
///   $\Delta m_{\max}$.
///
/// Return
/// ------
/// * `Some(ScoredAssociation)` – Filtered association (at least one candidate
///   survived).
/// * `None` – Reference magnitude unavailable, or no candidate survived the
///   filter.
fn apply_magnitude_filter<'t, 'o>(
    (tracklet, candidates): ScoredAssociation<'t, 'o>,
    max_magnitude_diff: f64,
) -> Option<ScoredAssociation<'t, 'o>> {
    // Compute the tracklet's mean magnitude as photometric reference.
    let (ref_mag, _) = tracklet.get_ref_mag();
    let filtered = filter_scored_by_magnitude(candidates, ref_mag, max_magnitude_diff);
    (!filtered.is_empty()).then_some((tracklet, filtered))
}

/// Apply the magnitude filter to all tracklet associations.
///
/// Wraps [`apply_magnitude_filter`] over the full association list, dropping
/// tracklets whose candidate set becomes empty after filtering.
///
/// Arguments
/// ---------
/// * `associations`       – Scored associations from the spatial association
///   step.
/// * `obs_dataset`        – Dataset used to retrieve photometric values.
/// * `max_magnitude_diff` – Maximum allowed magnitude difference
///   $\Delta m_{\max}$.
///
/// Return
/// ------
/// * [`ScoredAssociations`] – Associations that survived the magnitude filter.
fn filter_by_magnitude<'t, 'o>(
    associations: ScoredAssociations<'t, 'o>,
    max_magnitude_diff: f64,
) -> ScoredAssociations<'t, 'o> {
    associations
        .into_iter()
        .filter_map(|assoc| apply_magnitude_filter(assoc, max_magnitude_diff))
        .collect()
}

// ---------------------------------------------------------------------------
// Deduplication by exposure time
// ---------------------------------------------------------------------------

/// Group scored candidate observations by exposure time.
///
/// The exposure time (MJDTT) is represented as its raw `u64` bit pattern to
/// allow exact equality comparison without floating-point ambiguity.
///
/// Arguments
/// ---------
/// * `candidates` – Flat list of scored candidate observations.
///
/// Return
/// ------
/// * `HashMap<u64, Vec<ScoredObservation>>` – Candidates grouped by their
///   MJDTT bit pattern.
fn group_scored_by_exposure_time<'o>(
    candidates: Vec<ScoredObservation<'o>>,
) -> HashMap<u64, Vec<ScoredObservation<'o>>> {
    let mut by_time: HashMap<u64, Vec<ScoredObservation<'o>>> = HashMap::new();
    for scored in candidates {
        by_time
            .entry(scored.obs.mjd_tt().to_bits())
            .or_default()
            .push(scored);
    }
    by_time
}

/// Select the best candidate from a group sharing the same exposure time.
///
/// The best candidate is the one with the smallest precomputed squared
/// Mahalanobis distance $d^2$. If the group contains a single element, it is
/// returned immediately. Candidates with $d^2 = \infty$ (e.g. arising from a
/// failed covariance inversion) are always outranked by finite-distance
/// candidates but can still win if no better option exists.
///
/// Arguments
/// ---------
/// * `group` – One or more scored candidates at the same exposure time.
///
/// Return
/// ------
/// * `&Observation` – The observation with the smallest $d^2$.
fn best_in_group<'o>(group: Vec<ScoredObservation<'o>>) -> &'o Observation {
    // Single candidate — no comparison needed.
    if group.len() == 1 {
        return group.into_iter().next().unwrap().obs;
    }

    // Select by precomputed d² — no propagation needed here.
    group
        .into_iter()
        .min_by(|a, b| {
            a.mahalanobis_sq
                .partial_cmp(&b.mahalanobis_sq)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap()
        .obs
}

/// Deduplicate a tracklet's scored candidates, keeping at most one per
/// exposure time.
///
/// Groups candidates by exposure time, then selects the best observation in
/// each group using the precomputed Mahalanobis distance (see
/// [`best_in_group`]). No additional propagation is performed at this stage.
///
/// Arguments
/// ---------
/// * `candidates` – All scored candidates for a single tracklet (may contain
///   multiple observations per exposure time).
///
/// Return
/// ------
/// * `Vec<&Observation>` – At most one observation per unique exposure time.
fn dedup_scored_candidates<'o>(candidates: Vec<ScoredObservation<'o>>) -> Vec<&'o Observation> {
    group_scored_by_exposure_time(candidates)
        .into_values()
        .map(best_in_group)
        .collect()
}

/// For each tracklet, retain at most one observation per unique exposure time.
///
/// Several alerts can share the exact same exposure time (e.g. duplicate
/// detections on overlapping CCD chips). Keeping all of them would introduce
/// spurious constraints in the orbit solver. This step resolves ambiguities by
/// selecting the single best candidate per time slot using the precomputed
/// squared Mahalanobis distance $d^2$ — no additional propagation is required.
///
/// Tracklets whose candidate list becomes empty after deduplication are
/// dropped.
///
/// Arguments
/// ---------
/// * `associations` – Scored associations from the spatial association and
///   magnitude-filter steps.
///
/// Return
/// ------
/// * [`Associations`] – Final associations with at most one observation per
///   exposure time per tracklet. The [`ScoredObservation`] wrapper is dropped
///   at this point; only the observation reference is retained.
fn deduplicate_by_exposure_time<'t, 'o>(
    associations: ScoredAssociations<'t, 'o>,
) -> Associations<'o> {
    associations
        .into_iter()
        .filter_map(|(tracklet, candidates)| {
            let deduped = dedup_scored_candidates(candidates);
            (!deduped.is_empty()).then_some((tracklet.key(), deduped))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Build the list of candidate associations between active tracklets and the
/// observations of the next night.
///
/// This is the main entry point for the inter-night linking step. Given a set
/// of active tracklets and the full observation dataset, it returns — for each
/// tracklet that has at least one surviving candidate — the pairing of that
/// tracklet with its filtered and deduplicated candidate observations.
///
/// Pipeline
/// --------
/// The function applies four successive filters:
///
/// 1. **HEALPix prefilter** — rejects tracklets whose predicted position falls
///    entirely outside the night's sky footprint (coarse $\sim 1.8°$ cells at
///    depth 5). This is a cheap $O(N_{\text{tracklets}})$ guard before the
///    spatial KD-tree query.
///
/// 2. **Spatial association** — for each surviving tracklet, queries the
///    [`NightIndex`] to find observations within `association_sigma` $\sigma$
///    of the propagated prediction and below `chi2_threshold`. Each matching
///    observation is wrapped in a [`ScoredObservation`] carrying its
///    precomputed $d^2$, which is reused in step 4 without recomputation.
///
/// 3. **Magnitude filter** — discards candidates whose magnitude deviates from
///    the tracklet's mean magnitude by more than `max_magnitude_diff`:
///    $$|m_{\text{obs}} - \bar{m}_{\text{tracklet}}| \leq \Delta m_{\max}$$
///
/// 4. **Exposure-time deduplication** — when multiple candidates share the
///    same exposure time (e.g. overlapping chips), keeps only the one with the
///    smallest precomputed $d^2$. No additional propagation is performed here.
///
/// Arguments
/// ---------
/// * `obs_dataset`     – Full observation dataset (all nights).
/// * `next_night`      – Identifier of the target night.
/// * `engine_config`   – Association parameters:
///   - `association_sigma`   – Positional search radius in units of $\sigma$.
///   - `chi2_threshold`      – Maximum $\chi^2$ residual for acceptance.
///   - `max_magnitude_diff`  – Maximum magnitude difference $\Delta m_{\max}$.
/// * `night_tracklets` – Active tracklets to be linked to the next night.
///
/// Return
/// ------
/// * `Ok(Associations)` – One entry per tracklet that survived all filters,
///   paired with its candidate observations.
/// * `Err(EngineError::StageFailed)` – If the target night is absent from the
///   dataset or uses a non-contiguous memory layout.
pub(crate) fn generate_next_night_candidate<'t, 'o>(
    observations: &'o [Observation],
    engine_config: &EngineConfig,
    night_tracklets: impl Iterator<Item = &'t Tracklet>,
) -> Result<Associations<'o>, EngineError> {
    const PREFILTER_HEALPIX_DEPTH: u8 = 5;

    // --- Step 1: build the spatial + temporal index for the target night.
    let index = NightIndex::build_night_index(observations);

    // --- Step 2: HEALPix prefilter — discard tracklets outside the footprint.
    let occupied_pixels = occupied_healpix_pixels(&index.ecl_coords, PREFILTER_HEALPIX_DEPTH);

    // --- Step 3: spatial association — each match is scored with its d²
    // computed once during the χ² gate; no recomputation downstream.
    let associations: ScoredAssociations = night_tracklets
        .filter(|t| {
            !t.is_orbit()
                && tracklet_in_occupied_region(t, &occupied_pixels, PREFILTER_HEALPIX_DEPTH)
        })
        .filter_map(|tracklet| {
            associate_tracklet(
                tracklet,
                &index,
                engine_config.association_sigma,
                engine_config.chi2_threshold,
            )
        })
        .collect();

    // --- Step 4: magnitude filter — reject photometrically inconsistent candidates.
    let associations = filter_by_magnitude(associations, engine_config.max_magnitude_diff);

    // --- Step 5: deduplication — one observation per exposure time per tracklet,
    // selected by precomputed d² (no additional propagation).
    Ok(deduplicate_by_exposure_time(associations))
}
