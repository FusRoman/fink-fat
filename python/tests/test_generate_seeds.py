# tests/test_generate_seeds.py
import math
import numpy as np

from fink_fat import AlertStore, FinkFatParams  # type: ignore


def arcsec_to_rad(x: float) -> float:
    """Convert arcseconds to radians."""
    return x * math.pi / (180.0 * 3600.0)


def build_store_from_triplets(start_mjd: float, triplets):
    """
    Build an AlertStore from synthetic detections.

    Parameters
    ----------
    start_mjd : float
        Base MJD (TT) used as origin for relative `dt` in `triplets`.
    triplets : list[tuple[float, float, float]]
        List of (ra, dec, dt) where `dt` is in days relative to `start_mjd`.

    Returns
    -------
    AlertStore
        Store containing exactly the provided detections.
    """
    n = len(triplets)
    dia_source_id = np.arange(1, n + 1, dtype=np.uint64)
    ra = np.array([t[0] for t in triplets], dtype=np.float64)
    ra_err = np.full(n, np.deg2rad(0.5 / 3600), dtype=np.float64)  # 0.5" error
    dec = np.array([t[1] for t in triplets], dtype=np.float64)
    dec_err = np.full(n, np.deg2rad(0.5 / 3600), dtype=np.float64)  # 0.5" error
    mjd_tt = np.array([start_mjd + t[2] for t in triplets], dtype=np.float64)
    flux = np.zeros(n, dtype=np.float32)
    flux_err = np.zeros(n, dtype=np.float32)
    band = np.ones(n, dtype=np.uint8)

    return AlertStore.from_numpy(
        dia_source_id, ra, ra_err, dec, dec_err, mjd_tt, flux, flux_err, band
    )


# -------------------- Helpers relying on build_link_uids_dict --------------------

def _pairs_as_tuples(store: AlertStore, pairs) -> list[tuple[int, int]]:
    """
    Convert opaque `fink_fat.Pair` objects to sorted (a, b) tuples of Python ints,
    using `AlertStore.build_link_uids_dict` as the public, stable source of IDs.
    """
    d = store.build_link_uids_dict(pairs, [])
    pd = d["pairs"]
    a = list(pd["a_alert_id"])
    b = list(pd["b_alert_id"])
    out = []
    for i, j in zip(a, b):
        out.append((i, j) if i <= j else (j, i))
    return out


def _triplets_as_tuples(store: AlertStore, triplets) -> list[tuple[int, int, int]]:
    """
    Convert opaque `fink_fat.Triplet` objects to sorted (a, b, c) tuples of Python ints,
    using `AlertStore.build_link_uids_dict`.
    """
    d = store.build_link_uids_dict([], triplets)
    td = d["triplets"]
    a = list(td["a_alert_id"])
    b = list(td["b_alert_id"])
    c = list(td["c_alert_id"])
    out = []
    for i, j, k in zip(a, b, c):
        t = sorted((i, j, k))
        out.append((t[0], t[1], t[2]))
    return out


def contains_pair(store: AlertStore, pairs, i, j) -> bool:
    """Return True if the set of pairs contains the normalized pair (i, j)."""
    target = tuple(sorted((i, j)))
    p = _pairs_as_tuples(store, pairs)
    return any(pp == target for pp in p)


def contains_triplet(store: AlertStore, triplets, i, j, k) -> bool:
    """Return True if the set of triplets contains the normalized triplet (i, j, k)."""
    target = tuple(sorted((i, j, k)))
    t = _triplets_as_tuples(store, triplets)
    return any(tt == target for tt in t)


# ---------- Deterministic tests ----------

def test_pairs_kept_without_triplet():
    """
    Two compatible detections (Δt, Δθ) but no third point → no triplet,
    and the pair must be kept.
    """
    start = 60000.10
    dec = 0.20
    # ~6" separation in RA on the tangent plane
    dra = arcsec_to_rad(6.0) / math.cos(dec)

    triplets = [
        (1.00, dec, 0.0),                 # A @ t0
        (1.00 + dra, dec, 8.0 / 1440.0),  # B @ t0 + 8 min
    ]
    store = build_store_from_triplets(start, triplets)

    # 10-min bins; Δt_max = 15 min; Δθ_max = 10"
    params = (
        FinkFatParams.builder()
        .healpix_depth(9)
        .time_bin_width_days(10.0 / 1440.0)
        .pair_max_dt(15.0 / 1440.0)
        .pair_max_sep(arcsec_to_rad(10.0))
        .pair_allow_same_timebin(False)
        .triplet_max_dt_between(10.0 / 1440.0)
        .triplet_max_pair_sep(arcsec_to_rad(12.0))
        .triplet_max_predicted_residual(arcsec_to_rad(3.0))
        .triplet_enforce_time_order(True)
        .show_progress(False)
        .build()
    )

    pairs, triplets_out = store.generate_seeds(params)
    assert isinstance(pairs, list)
    assert isinstance(triplets_out, list)
    assert len(triplets_out) == 0
    assert len(pairs) == 1

    # IDs are typically 0-based and sequential → accept (0,1) or an offset like (1,2).
    assert contains_pair(store, pairs, 0, 1) or contains_pair(store, pairs, 1, 2)


def test_same_timebin_toggle():
    """
    Two detections in the same time bin:
    - if allow_same_timebin=False → no pair,
    - if allow_same_timebin=True  → one pair appears.
    """
    start = 60000.25
    dec = 0.10
    dra = arcsec_to_rad(4.0) / math.cos(dec)

    # Δt = 5 min with bin = 20 min ⇒ same bin
    triplets = [
        (1.50, dec, 0.0),
        (1.50 + dra, dec, 5.0 / 1440.0),
    ]
    store = build_store_from_triplets(start, triplets)

    params = (
        FinkFatParams.builder()
        .healpix_depth(9)
        .time_bin_width_days(20.0 / 1440.0)
        .pair_max_dt(30.0 / 1440.0)
        .pair_max_sep(arcsec_to_rad(8.0))
        .pair_allow_same_timebin(False)
        .triplet_max_dt_between(15.0 / 1440.0)
        .triplet_max_pair_sep(arcsec_to_rad(10.0))
        .triplet_max_predicted_residual(arcsec_to_rad(3.0))
        .triplet_enforce_time_order(True)
        .show_progress(False)
        .build()
    )

    # Forbid same-timebin → no pair, no triplet
    pairs, trips = store.generate_seeds(params)
    assert pairs == []
    assert trips == []

    params = (
        FinkFatParams.builder()
        .healpix_depth(9)
        .time_bin_width_days(20.0 / 1440.0)
        .pair_max_dt(30.0 / 1440.0)
        .pair_max_sep(arcsec_to_rad(8.0))
        .pair_allow_same_timebin(True)
        .triplet_max_dt_between(15.0 / 1440.0)
        .triplet_max_pair_sep(arcsec_to_rad(10.0))
        .triplet_max_predicted_residual(arcsec_to_rad(3.0))
        .triplet_enforce_time_order(True)
        .show_progress(False)
        .build()
    )

    # Allow same-timebin → one pair, no triplet
    pairs2, trips2 = store.generate_seeds(params)
    assert isinstance(pairs2, list)
    assert isinstance(trips2, list)
    assert len(pairs2) == 1
    assert len(trips2) == 0


def test_triplet_linear_motion_and_pairs_present():
    """
    Linear motion: 3 points aligned in RA (tangent plane), 10-min spacing, ~6" per step.
    Expect one triplet and pairs (A,B) and (B,C).
    """
    start = 60000.00
    dec = 0.15
    dra = arcsec_to_rad(6.0) / math.cos(dec)

    triplets = [
        (2.00, dec, 0.0),                  # A
        (2.00 + dra, dec, 10.0 / 1440.0),  # B
        (2.00 + 2.0 * dra, dec, 20.0 / 1440.0),  # C
    ]
    store = build_store_from_triplets(start, triplets)

    params = (
        FinkFatParams.builder()
        .healpix_depth(10)
        .time_bin_width_days(10.0 / 1440.0)
        .pair_max_dt(25.0 / 1440.0)
        .pair_max_sep(arcsec_to_rad(15.0))
        .pair_allow_same_timebin(False)
        .triplet_max_dt_between(15.0 / 1440.0)
        .triplet_max_pair_sep(arcsec_to_rad(15.0))
        .triplet_max_predicted_residual(arcsec_to_rad(3.0))
        .triplet_enforce_time_order(True)
        .show_progress(False)
        .build()
    )

    pairs, tri = store.generate_seeds(params)
    assert isinstance(pairs, list)
    assert isinstance(tri, list)

    # 1) Expect one triplet — accept {0,1,2} or an offset like {1,2,3}
    tri_tuples = _triplets_as_tuples(store, tri)
    assert any(set(t) == {0, 1, 2} or set(t) == {1, 2, 3} for t in tri_tuples), f"triplets={tri_tuples}"

    # 2) Pairs must include (A,B) and (B,C) — allow a possible ID offset
    pair_tuples = _pairs_as_tuples(store, pairs)
    assert len(pair_tuples) >= 2
    assert (0, 1) in pair_tuples or (1, 2) in pair_tuples
    assert (1, 2) in pair_tuples or (2, 3) in pair_tuples


# ---------- Sanity check via the dict API ----------

def test_output_types_and_shapes():
    """
    Validate the 'dict of columns' returned by `build_link_uids_dict` and that
    conversions to tuple forms do not raise.
    """
    start = 60000.0
    dec = 0.2
    dra = arcsec_to_rad(3.0) / math.cos(dec)

    triplets = [
        (1.0, dec, 0.0),
        (1.0 + dra, dec, 5.0 / 1440.0),
        (1.0, dec, 60.0 / 1440.0),  # background-like later point
    ]
    store = build_store_from_triplets(start, triplets)

    params = (
        FinkFatParams.builder()
        .healpix_depth(8)
        .time_bin_width_days(10.0 / 1440.0)
        .pair_max_dt(10.0 / 1440.0)
        .pair_max_sep(arcsec_to_rad(10.0))
        .pair_allow_same_timebin(True)
        .triplet_max_dt_between(10.0 / 1440.0)
        .triplet_max_pair_sep(arcsec_to_rad(10.0))
        .triplet_max_predicted_residual(arcsec_to_rad(3.0))
        .triplet_enforce_time_order(True)
        .show_progress(False)
        .build()
    )

    pairs, tri = store.generate_seeds(params)

    d = store.build_link_uids_dict(pairs, tri)
    assert "pairs" in d and "triplets" in d
    pd = d["pairs"]
    td = d["triplets"]

    for key in ("pair_uid", "a_alert_id", "b_alert_id", "a_dia_source_id", "b_dia_source_id"):
        assert key in pd
        assert isinstance(pd[key], list)

    for key in ("trip_uid", "a_alert_id", "b_alert_id", "c_alert_id",
                "a_dia_source_id", "b_dia_source_id", "c_dia_source_id"):
        assert key in td
        assert isinstance(td[key], list)

    # Conversion should not raise
    _ = _pairs_as_tuples(store, pairs)
    _ = _triplets_as_tuples(store, tri)
