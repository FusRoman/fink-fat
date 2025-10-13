# python/fink_fat/rollinglinkstate.pyi
from __future__ import annotations

from typing import List, TypedDict, Dict
import numpy as np
import numpy.typing as npt

from fink_fat import FinkFatParams  # global config used by run_nightly_step


"""
High-level inter-night linking (rolling state) — Python stubs.

This module exposes the `RollingLinkState` façade used to:
- ingest nights one by one (NumPy columns → AlertStore → NightSnapshot),
- link night N → night N+1 with the engine (candidate retrieval, scoring, assignment),
- update the internal TrackRegistry according to a conflict policy,
- export convenient summaries and DataFrame-ready dicts.

Units & Conventions
-------------------
- Angles in **radians**; times in **days** (MJD TT).
- `night_id` is an integer identifier for a night (dense or calendar-based).
- Exports use deterministic ordering for reproducibility.
"""


# ---------------------------------------------------------------------
# Conflict policy (track_registry.rs)
# ---------------------------------------------------------------------

class DetectConflictPolicy:
    """
    Policy for handling detection-to-track conflicts in the registry.

    Members
    -------
    KeepFirst : int
        First assignment wins; later conflicting assignments are ignored.
    Overwrite : int
        Latest assignment overrides the previous one (one-to-one mapping).
    Combinatorial : int
        Allow multiple trajectories per detection (deduped set).
    """
    KeepFirst: int
    Overwrite: int
    Combinatorial: int


# ---------------------------------------------------------------------
# Typed dicts for RollingLinkState.stats()
# ---------------------------------------------------------------------

class PairCostSummary(TypedDict):
    """
    Per-pair or global cost distribution summary.
    """
    count: int
    min: float
    max: float
    mean: float
    median: float
    p90: float
    p95: float
    p99: float


class EdgesKeptSummary(TypedDict):
    """
    Summary of kept candidate edges after hard gates & top-K per pairwise link.
    """
    total: int
    mean_per_pair: float
    median_per_pair: float
    max_per_pair: float


class PairStats(TypedDict):
    """
    One entry per pairwise link result (night_left → night_right).
    """
    night_left: int
    night_right: int
    n_matches: int
    edges_kept: int
    cost: PairCostSummary


class RollingStats(TypedDict):
    """
    Output structure of RollingLinkState.stats().
    """
    total_pairs: int
    total_matches: int
    pairs: List[PairStats]
    cost: PairCostSummary          # global cost summary over all pair results
    edges_kept: EdgesKeptSummary   # global edges-kept summary


# ---------------------------------------------------------------------
# Typed dict for export_linked_detections_dict()
# ---------------------------------------------------------------------

class LinkedDetectionsCols(TypedDict):
    """
    Columnar export of all linked detections (DataFrame-ready).

    Keys
    ----
    candid : list[int]
        DIA candidate/row identifier per detection (64-bit upstream).
    ra : list[float]
        Right ascension (rad).
    dec : list[float]
        Declination (rad).
    jd : list[float]
        Julian Date (days; convenience field).
    mjd_tt : list[float]
        Modified Julian Date in TT (days).
    trajectory_id : list[int]
        Canonical trajectory id assigned by the registry (DSU representative).
    """
    candid: List[int]
    ra: List[float]
    dec: List[float]
    jd: List[float]
    mjd_tt: List[float]
    trajectory_id: List[int]


# ---------------------------------------------------------------------
# RollingLinkState façade
# ---------------------------------------------------------------------

class RollingLinkState:
    """
    Rolling state machine for inter-night linking & registry updates.

    Notes
    -----
    - Keeps the last NightSnapshot to link against the next night.
    - Accumulates pairwise link results for statistics.
    - Maintains an internal TrackRegistry with the chosen conflict policy.
    - Optionally caches per-night AlertStores for exports.
    """

    def __init__(self, conflict_policy: DetectConflictPolicy) -> None:
        """
        Create an empty rolling state with the given conflict policy.

        Parameters
        ----------
        conflict_policy : DetectConflictPolicy
            Strategy for resolving detection↔trajectory conflicts
            during nightly updates of the registry.
        """
        ...

    def run_nightly_step(
        self,
        dia_source_id: npt.NDArray[np.uint64],
        ra: npt.NDArray[np.float64],
        ra_err: npt.NDArray[np.float64],
        dec: npt.NDArray[np.float64],
        dec_err: npt.NDArray[np.float64],
        mjd_tt: npt.NDArray[np.float64],
        flux: npt.NDArray[np.float32],
        flux_err: npt.NDArray[np.float32],
        band: npt.NDArray[np.uint8],
        night_id: int,
        params: FinkFatParams,
    ) -> None:
        """
        Ingest one night: build store & snapshot, link to previous if present,
        update the registry, and cache the store for downstream exports.

        Parameters
        ----------
        dia_source_id, ra, ra_err, dec, dec_err, mjd_tt, flux, flux_err, band : ndarray
            LSST DIA columns (1D, aligned, correct dtypes).
        night_id : int
            Identifier for this night.
        params : FinkFatParams
            Global configuration (seeding, features, linking).

        Notes
        -----
        - On the **first** call, seeds are bootstrapped to fresh tracks so all
          member detections immediately carry a stable `trajectory_id`.
        - On subsequent calls, the method links previous→current, merges
          trajectories if chains join, and assigns all member detections to the
          canonical (representative) trajectory id.
        """
        ...

    def stats(self) -> RollingStats:
        """
        Summarize pairwise link results accumulated so far.

        Returns
        -------
        RollingStats
            Global counters, per-pair breakdown, cost distributions,
            and edges-kept summary.
        """
        ...

    def export_linked_detections_dict(self) -> LinkedDetectionsCols:
        """
        Export detection-level rows associated with linked tracks.

        Returns
        -------
        LinkedDetectionsCols
            Dict of parallel lists (DataFrame-ready). Uses the internal
            TrackRegistry and the cached per-night AlertStores.
        """
        ...
