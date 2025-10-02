# python/fink_fat/alerts.pyi
from __future__ import annotations

from fink_fat import PyFinkFatParams

"""
Fink-FAT Python bindings: alert model and intra-night seeding surface.

Overview
--------
This module exposes a compact alert data model (`Alert`, `AlertStore`) and a
bridge to the Rust seeding pipeline that generates **pairs** and **triplets**
of detections within (typically) a single night.

Units & Conventions
-------------------
- Right ascension/declination (`ra`, `dec`) are **radians** (ICRS/J2000).
- Times are **MJD (TT)**, i.e., Modified Julian Date in Terrestrial Time.
- `flux` is a **PSF difference flux** (units as provided upstream, commonly nJy).
- `band` is an **integer** photometric band code (instrument-specific).
- `AlertId` is a **dense 0-based index** into an `AlertStore.alerts` list.

Quick Start
-----------
>>> import numpy as np
>>> from fink_fat import AlertStore, FinkFatParams
>>> store = AlertStore.from_numpy(
...     dia_source_id=np.array([11, 12, 13], dtype=np.uint64),
...     ra=np.array([1.0, 1.001, 1.002], dtype=np.float64),
...     dec=np.array([0.1, 0.101, 0.102], dtype=np.float64),
...     mjd_tt=np.array([60000.1, 60000.12, 60000.15], dtype=np.float64),
...     flux=np.array([1000.0, 980.0, 960.0], dtype=np.float32),
...     flux_err=np.array([10.0, 10.0, 10.0], dtype=np.float32),
...     band=np.array([1, 1, 1], dtype=np.uint8),
... )
>>> params = FinkFatParams.default()  # LSST-like defaults
>>> pairs, trips = store.generate_seeds(params)
>>> uids = store.build_link_uids_dict(pairs, trips)
>>> sorted(uids["pairs"]["pair_uid"])[0].startswith("P|")
True

Notes
-----
- Construction from NumPy is **zero-copy** into Rust slices, then a single
  copy into a contiguous Rust vector for performance and safety.
- Seeding can optionally emit progress bars if enabled in `FinkFatParams`.
"""

from typing import List, Tuple, Dict, Any, TypedDict
import numpy as np
import numpy.typing as npt

# -------------------------------------------------------------------
# Type aliases (public API)
# -------------------------------------------------------------------

AlertId = int
"""
Dense, 0-based index into an `AlertStore`.

Notes
-----
This is not the LSST `diaSourceId`. It is a transient index assigned at
ingestion time so that `AlertStore.alerts[id]` is O(1).
"""

Pair = Tuple[AlertId, AlertId]
"""
Candidate same-object **pair** of alerts within one night.

Notes
-----
The order `(a, b)` follows the generation order and is **not** guaranteed to be
time-sorted unless the seeding configuration enforces it.
"""

Triplet = Tuple[AlertId, AlertId, AlertId]
"""
Candidate same-object **triplet** of alerts within one night.

Notes
-----
The `(a, b, c)` order follows generation and is **not** necessarily chronological
unless `enforce_time_order=True` in the underlying configuration.
"""

Pairs = List[Pair]
"""List of candidate pairs."""

Triplets = List[Triplet]
"""List of candidate triplets."""

# -------------------------------------------------------------------
# Typed dicts for build_link_uids_dict()
# -------------------------------------------------------------------

class PairCols(TypedDict):
    """
    Columnar representation of pair seeds.

    Keys
    ----
    pair_uid : list[str]
        Stable, human-readable UID with sorted DIA ids: ``"P|{min_dia}|{max_dia}"``.
    a_alert_id, b_alert_id : list[int]
        Store indices of the two alerts (dense `AlertId`).
    a_dia_source_id, b_dia_source_id : list[int]
        The corresponding LSST `diaSourceId`s (64-bit).
    """

    pair_uid: List[str]
    a_alert_id: List[int]
    b_alert_id: List[int]
    a_dia_source_id: List[int]
    b_dia_source_id: List[int]

class TripletCols(TypedDict):
    """
    Columnar representation of triplet seeds.

    Keys
    ----
    trip_uid : list[str]
        Stable, human-readable UID with DIA ids sorted ascending:
        ``"T|{dia1}|{dia2}|{dia3}"``.
    a_alert_id, b_alert_id, c_alert_id : list[int]
        Store indices of the three alerts (dense `AlertId`).
    a_dia_source_id, b_dia_source_id, c_dia_source_id : list[int]
        The corresponding LSST `diaSourceId`s (64-bit).
    """

    trip_uid: List[str]
    a_alert_id: List[int]
    b_alert_id: List[int]
    c_alert_id: List[int]
    a_dia_source_id: List[int]
    b_dia_source_id: List[int]
    c_dia_source_id: List[int]

class LinkUIDs(TypedDict):
    """
    Output structure of :meth:`AlertStore.build_link_uids_dict`.

    Keys
    ----
    pairs : PairCols
        Columnar table for pair seeds.
    triplets : TripletCols
        Columnar table for triplet seeds.

    Notes
    -----
    The lists within each sub-dict are **parallel arrays** (same length and
    aligned by index), making it straightforward to build a pandas DataFrame:

    >>> import pandas as pd
    >>> u = store.build_link_uids_dict(pairs, trips)
    >>> df_pairs = pd.DataFrame(u["pairs"])
    >>> df_trips = pd.DataFrame(u["triplets"])
    """

    pairs: PairCols
    triplets: TripletCols

# -------------------------------------------------------------------
# Core data model
# -------------------------------------------------------------------

class Alert:
    """
    Single detection from the alert stream.

    Attributes
    ----------
    id : int
        Dense, 0-based `AlertId` assigned at ingestion. This indexes into the
        owning :class:`AlertStore` (``store.alerts[id]``).
    dia_source_id : int
        LSST **diaSourceId** (64-bit, stable across processes/files).
    ra : float
        Right ascension in **radians** (ICRS/J2000).
    dec : float
        Declination in **radians** (ICRS/J2000).
    mjd_tt : float
        Detection time as **Modified Julian Date in TT** (days).
    flux : float
        **PSF difference flux**. Positive values typically indicate that the
        science image is brighter than the template at this position.
        Common unit: **nJy** (depending on upstream calibration).
    flux_err : float
        1-sigma uncertainty on `flux` (same unit as `flux`).
    band : int
        Integer photometric band code (instrument-specific).

    Notes
    -----
    - The data model is intentionally compact and immutable to enable safe
      sharing across threads and fast iteration in Rust.
    """

    @property
    def id(self) -> int: ...
    @property
    def dia_source_id(self) -> int: ...
    @property
    def ra(self) -> float: ...
    @property
    def dec(self) -> float: ...
    @property
    def mjd_tt(self) -> float: ...
    @property
    def flux(self) -> float: ...
    @property
    def flux_err(self) -> float: ...
    @property
    def band(self) -> int: ...
    def __str__(self) -> str:
        """
        Return a compact single-line summary.

        Returns
        -------
        str
            Example:
            ``"Alert(id=7, dia_source_id=..., ra=..., dec=..., mjd_tt=..., flux=...±..., band=...)"``.
        """
        ...

    def __repr__(self) -> str:
        """
        Return a detailed representation including all scalar fields.

        Returns
        -------
        str
        """
        ...

class AlertStore:
    """
    Contiguous alert storage for (typically) a single night.

    Attributes
    ----------
    start_mjd : float
        Floor of the minimum `mjd_tt` in the store (TT). Used as origin for
        uniform time binning in the seeding pipeline.
    alerts : list[Alert]
        Dense vector of alerts, indexable by `AlertId`.

    Notes
    -----
    The store is designed for cache-friendly iteration and neighborhood
    queries used by the geometrical seeding stage.
    """

    start_mjd: float
    alerts: List[Alert]

    @staticmethod
    def from_numpy(
        dia_source_id: npt.NDArray[np.uint64],
        ra: npt.NDArray[np.float64],
        dec: npt.NDArray[np.float64],
        mjd_tt: npt.NDArray[np.float64],
        flux: npt.NDArray[np.float32],
        flux_err: npt.NDArray[np.float32],
        band: npt.NDArray[np.uint8],
    ) -> AlertStore:
        """
        Build an :class:`AlertStore` from aligned 1-D NumPy arrays.

        Parameters
        ----------
        dia_source_id : ndarray[uint64], shape (N,)
            LSST `diaSourceId` per alert (64-bit).
        ra, dec : ndarray[float64], shape (N,)
            ICRS right ascension/declination **in radians**.
        mjd_tt : ndarray[float64], shape (N,)
            Detection times as **MJD (TT)** in days.
        flux, flux_err : ndarray[float32], shape (N,)
            PSF **difference flux** and its 1-sigma uncertainty.
            Common unit: **nJy** (depending on upstream calibration).
        band : ndarray[uint8], shape (N,)
            Integer photometric band code.

        Returns
        -------
        AlertStore
            Store with `N` alerts and `start_mjd = floor(min(mjd_tt))`.

        Raises
        ------
        ValueError
            If any array is not 1-D, not C-contiguous, or has an incompatible dtype.
        RuntimeError
            If array lengths differ (the Rust layer asserts equal lengths).

        Notes
        -----
        - The function borrows zero-copy slices from the input arrays, then
          **copies once** into a contiguous Rust vector for performance.
        - Units are not converted; callers must provide radians and TT.
        - `AlertId` for the i-th alert is exactly `i`.
        """
        ...

    def get(self, id: int) -> Alert:
        """
        Retrieve an :class:`Alert` by its dense `AlertId`.

        Parameters
        ----------
        id : int
            Dense alert identifier (``0 <= id < len(store)``).

        Returns
        -------
        Alert

        Raises
        ------
        IndexError
            If `id` is out of range.

        Examples
        --------
        >>> a = store.get(0)      # first alert
        >>> a.dia_source_id > 0
        True
        """
        ...

    def __len__(self) -> int:
        """
        Return the number of alerts in the store.

        Returns
        -------
        int
        """
        ...

    def __getitem__(self, idx: int) -> Alert:
        """
        Return the alert at position `idx`.

        Parameters
        ----------
        idx : int
            Dense positional index (equivalent to `AlertId`).

        Returns
        -------
        Alert

        Raises
        ------
        IndexError
            If `idx` is out of range.
        """
        ...

    def __str__(self) -> str:
        """
        Return a compact string summary.

        Returns
        -------
        str
            Example: ``"AlertStore(n_alerts=12345, start_mjd=60000.0)"``.
        """
        ...

    def __repr__(self) -> str:
        """
        Return a verbose summary including time span and band set.

        Returns
        -------
        str
            Example:
            ``"AlertStore(n_alerts=..., start_mjd=..., time_span=[tmin, tmax], bands={...})"``.
        """
        ...

    def generate_seeds(self, params: PyFinkFatParams) -> Tuple[Pairs, Triplets]:
        """
        Generate intra-night **pairs** and **triplets** of alerts.

        Parameters
        ----------
        params : PyFinkFatParams
            Seeding configuration. The following fields are consumed:
            - Spatial/temporal bucketing: **HEALPix depth**, **time bin width (days, TT)**.
            - Pair thresholds (e.g., **max Δt (days, TT)**, **max angular separation (rad)**,
              optional photometric constraints).
            - Triplet thresholds (e.g., **max Δt between legs (days, TT)**,
              **max pair separation (rad)**, **max predicted residual (rad)**,
              **enforce_time_order**).
            - UI: **show_progress** (progress bars on/off).

        Returns
        -------
        (Pairs, Triplets)
            Two lists of dense alert index tuples representing candidate
            same-object motion seeds.

        Notes
        -----
        - Complexity is approximately O(N) after bucketing, with small
          neighborhood searches per bin (exact scaling depends on cadence and
          thresholds).
        - If progress is enabled, three bars are emitted: **buckets → pairs → triplets**.
        - Output order is deterministic for a fixed input ordering and parameters.

        Examples
        --------
        >>> pairs, trips = store.generate_seeds(FinkFatParams.default())
        >>> len(pairs) >= 0 and len(trips) >= 0
        True
        """
        ...

    def build_link_uids_dict(self, pairs: Pairs, triplets: Triplets) -> LinkUIDs:
        """
        Build deterministic, human-readable **UIDs** for pairs and triplets.

        Parameters
        ----------
        pairs : list[tuple[int, int]]
            Pair seeds as dense `AlertId`s (typically from `generate_seeds`).
        triplets : list[tuple[int, int, int]]
            Triplet seeds as dense `AlertId`s.

        Returns
        -------
        dict
            Nested dict of column lists with keys **"pairs"** and **"triplets"**.
            See the :class:`PairCols` and :class:`TripletCols` schemas.

        Notes
        -----
        - Pair UID format: ``"P|{min_dia}|{max_dia}"`` – order-invariant.
        - Triplet UID format: ``"T|{d1}|{d2}|{d3}"`` where `d1 <= d2 <= d3`.
        - Intended for **joining across nights**, deduplication, and auditability.

        Examples
        --------
        >>> u = store.build_link_uids_dict(pairs, trips)
        >>> "pairs" in u and "triplets" in u
        True
        >>> # Convert to pandas DataFrame
        >>> import pandas as pd
        >>> df_pairs = pd.DataFrame(u["pairs"])
        >>> df_pairs.columns.tolist()[:3]  # doctest: +ELLIPSIS
        ['pair_uid', 'a_alert_id', 'b_alert_id']
        """
        ...
