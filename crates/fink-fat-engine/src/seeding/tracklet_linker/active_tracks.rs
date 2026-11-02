//! Spatial bookkeeping for the tracklet candidates still open for extension
//! during [`super::link_tracklets`]'s time-ordered sweep.
//!
//! Tracks are indexed by their most recent observation's spatial cell, so a
//! newly-arrived observation only has to compare against tracks in its own
//! sky neighborhood rather than the whole active population — this keeps
//! each sweep step close to `O(1)` amortized instead of `O(active tracks)`,
//! which matters at LSST-era observation volumes (see the module doc of
//! [`super`] for the pair-count blowup this whole linker exists to fix).
//!
//! This module owns no motion-fit math (see [`super::track_fit::TrackFit`])
//! and makes no gating decisions (see [`super::evaluate_candidate`]) — its
//! only responsibility is "which tracks are still near this sky position
//! and recent enough to extend."

use ahash::AHashMap;
use photom::observation_dataset::observation::Observation;

use crate::spacetime_bucket::spatial_binner::SpatialKey;

use super::track_fit::TrackFit;

/// A single tracklet candidate: the observation indices linked into it so
/// far, plus its incremental motion fit.
pub(crate) struct Track {
    /// Indices into the caller's time-sorted observation slice, in the
    /// order they were linked — which is also time order, since
    /// [`super::link_tracklets`] sweeps chronologically.
    pub(crate) member_indices: Vec<usize>,
    /// This track's incremental linear-motion fit.
    pub(crate) fit: TrackFit,
    /// Spatial cell of the track's most recent observation, kept in sync
    /// with `fit`/`member_indices` so it can be located in
    /// [`ActiveTracks::cell_index`] without recomputing it.
    last_cell: SpatialKey,
}

/// Pool of tracks indexed by the spatial cell of their most recent
/// observation.
///
/// Closed (stale) tracks are dropped from the spatial index as soon as
/// they're identified in [`Self::candidates_near`] but keep their storage
/// slot in `tracks` — every track, active or closed, is needed at the end
/// of the sweep to extract its endpoint pair.
#[derive(Default)]
pub(crate) struct ActiveTracks {
    /// All tracks ever created, active or closed, indexed by their position
    /// in this vector (used as a lightweight `TrackId`).
    tracks: Vec<Track>,
    /// Spatial cell → ids of tracks currently active in that cell.
    cell_index: AHashMap<SpatialKey, Vec<usize>>,
}

impl ActiveTracks {
    /// Start a brand-new track from an observation with no matching active
    /// track nearby.
    pub(crate) fn start_track(
        &mut self,
        observation_index: usize,
        fit: TrackFit,
        cell: SpatialKey,
    ) {
        let track_id = self.tracks.len();
        self.tracks.push(Track {
            member_indices: vec![observation_index],
            fit,
            last_cell: cell,
        });
        self.cell_index.entry(cell).or_default().push(track_id);
    }

    /// Extend an existing track with a newly-linked observation, folding it
    /// into the track's fit and re-indexing the track under its new
    /// spatial cell (motion between successive observations can cross a
    /// cell boundary).
    pub(crate) fn extend_track(
        &mut self,
        track_id: usize,
        observation_index: usize,
        observation: &Observation,
        new_cell: SpatialKey,
    ) {
        let old_cell = self.tracks[track_id].last_cell;
        if let Some(ids) = self.cell_index.get_mut(&old_cell) {
            ids.retain(|&id| id != track_id);
        }

        let track = &mut self.tracks[track_id];
        track.member_indices.push(observation_index);
        track.fit.push(observation);
        track.last_cell = new_cell;

        self.cell_index.entry(new_cell).or_default().push(track_id);
    }

    /// Read-only access to a track by id.
    pub(crate) fn track(&self, track_id: usize) -> &Track {
        &self.tracks[track_id]
    }

    /// Ids of the tracks currently active in any of `cells` and still
    /// within `max_dt` of `current_epoch`.
    ///
    /// As a side effect, tracks found to be stale (last observation more
    /// than `max_dt` behind `current_epoch`) are evicted from the spatial
    /// index — the sweep is chronological, so a stale track can never
    /// become extendable again.
    pub(crate) fn candidates_near(
        &mut self,
        cells: &[SpatialKey],
        current_epoch: f64,
        max_dt: f64,
    ) -> Vec<usize> {
        let mut candidates = Vec::new();
        for &cell in cells {
            let Some(ids) = self.cell_index.get_mut(&cell) else {
                continue;
            };
            let mut i = 0;
            while i < ids.len() {
                let track_id = ids[i];
                let is_within_horizon =
                    current_epoch - self.tracks[track_id].fit.last_epoch() <= max_dt;
                if is_within_horizon {
                    candidates.push(track_id);
                    i += 1;
                } else {
                    // Stale: never extendable again, drop from the active index.
                    ids.swap_remove(i);
                }
            }
        }
        candidates
    }

    /// Consume `self`, returning every track ever created (active or
    /// closed) for final endpoint-pair extraction.
    pub(crate) fn into_tracks(self) -> Vec<Track> {
        self.tracks
    }
}
