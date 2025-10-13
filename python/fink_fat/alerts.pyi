# python/fink_fat/alerts.pyi
from __future__ import annotations

from typing import List, Tuple, Dict, Any, TypedDict
import numpy as np
import numpy.typing as npt

from fink_fat import (
    FinkFatParams,
)  # <- exposed name via #[pyclass(name = "FinkFatParams")]

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
...     ra_err=np.full(3, np.deg2rad(0.5/3600), dtype=np.float64),
...     dec=np.array([0.1, 0.101, 0.102], dtype=np.float64),
...     dec_err=np.full(3, np.deg2rad(0.5/3600), dtype=np.float64),
...     mjd_tt=np.array([60000.1, 60000.12, 60000.15], dtype=np.float64),
...     flux=np.array([1000.0, 980.0, 960.0], dtype=np.float32),
...     flux_err=np.array([10.0, 10.0, 10.0], dtype=np.float32),
...     band=np.array([1, 1, 1], dtype=np.uint8),
... )
>>> params = FinkFatParams.default()  # LSST-like defaults
>>> pairs, trips = store.generate_seeds(params)  # lists of opaque Pair/Triplet objects
>>> uids = store.build_link_uids_dict(pairs, trips)
>>> sorted(uids["pairs"]["pair_uid"])[0].startswith("P|")
True

Notes
-----
- Construction from NumPy is **zero-copy** into Rust slices, then a single
  copy into a contiguous Rust vector for performance and safety.
- Seeding can optionally emit progress bars if enabled in `FinkFatParams`.
"""

# -------------------------------------------------------------------
# Public opaque classes returned by the bindings
# -------------------------------------------------------------------

class AlertId:
    """
    Opaque wrapper for a dense 0-based alert index (PyO3 class).

    Notes
    -----
    - This is **not** the LSST `diaSourceId`.
    - Some builds may expose utility methods (e.g., `idx()`), but consumers
      should treat this as an opaque handle.
    """

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class Pair:
    """
    Opaque intra-night pair seed (PyO3 class).

    Notes
    -----
    - Use :meth:`AlertStore.build_link_uids_dict` to obtain concrete integer
      indices (`a_alert_id`, `b_alert_id`) for dataframe/export workflows.
    """

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class Triplet:
    """
    Opaque intra-night triplet seed (PyO3 class).

    Notes
    -----
    - Use :meth:`AlertStore.build_link_uids_dict` to obtain concrete integer
      indices (`a_alert_id`, `b_alert_id`, `c_alert_id`).
    """

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

# For readability in type hints
Pairs = List[Pair]
Triplets = List[Triplet]

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
        Store indices of the two alerts (dense `AlertId` as Python ints).
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
        Store indices of the three alerts (dense `AlertId` as Python ints).
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
    id : AlertId
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
        **PSF difference flux**.
    flux_err : float
        1-sigma uncertainty on `flux`.
    band : int
        Integer photometric band code (instrument-specific).
    """

    @property
    def id(self) -> AlertId: ...
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
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

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
    """

    start_mjd: float
    alerts: List[Alert]

    @staticmethod
    def from_numpy(
        dia_source_id: npt.NDArray[np.uint64],
        ra: npt.NDArray[np.float64],
        ra_err: npt.NDArray[np.float64],
        dec: npt.NDArray[np.float64],
        dec_err: npt.NDArray[np.float64],
        mjd_tt: npt.NDArray[np.float64],
        flux: npt.NDArray[np.float32],
        flux_err: npt.NDArray[np.float32],
        band: npt.NDArray[np.uint8],
    ) -> AlertStore:
        """
        Build an :class:`AlertStore` from aligned 1-D NumPy arrays.

        Returns
        -------
        AlertStore
            Store with `N` alerts and `start_mjd = floor(min(mjd_tt))`.
        """
        ...

    def get_py(self, id: AlertId, /) -> Alert:
        """
        Retrieve an :class:`Alert` by its dense `AlertId`.

        Notes
        -----
        This is the explicit Python entry-point exposed by the bindings.
        Prefer `store[idx]` or :meth:`build_link_uids_dict` for most workflows.
        """
        ...

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Alert: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def generate_seeds(self, params: FinkFatParams) -> Tuple[Pairs, Triplets]:
        """
        Generate intra-night **pairs** and **triplets** of alerts.

        Returns
        -------
        (Pairs, Triplets)
            Lists of opaque PyO3 objects (:class:`Pair`, :class:`Triplet`).
        """
        ...

    def build_link_uids_dict(self, pairs: Pairs, triplets: Triplets) -> LinkUIDs:
        """
        Build deterministic, human-readable UIDs for pairs and triplets.

        Returns
        -------
        LinkUIDs
            Dict-of-lists ready for pandas DataFrame construction.
        """
        ...
