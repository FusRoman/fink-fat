# tests/test_generate_seeds.py
import math
import numpy as np

# Le module PyO3 exposé par ta crate (d'après #[pyclass(module = "fink_fat")])
from fink_fat import AlertStore, PyFinkFatParams  # type: ignore


def arcsec_to_rad(x: float) -> float:
    return x * math.pi / (180.0 * 3600.0)


def build_store_from_triplets(start_mjd: float, triplets):
    """
    triplets: iterable of (ra, dec, mjd_tt_offset_days)
      - ra/dec en radians
      - mjd_tt = start_mjd + offset
    Construit un AlertStore via AlertStore.from_numpy(...).
    """
    n = len(triplets)
    dia_source_id = np.arange(1, n + 1, dtype=np.uint64)
    ra = np.array([t[0] for t in triplets], dtype=np.float64)
    dec = np.array([t[1] for t in triplets], dtype=np.float64)
    mjd_tt = np.array([start_mjd + t[2] for t in triplets], dtype=np.float64)
    flux = np.zeros(n, dtype=np.float32)
    flux_err = np.zeros(n, dtype=np.float32)
    band = np.ones(n, dtype=np.uint8)  # n'importe quelle bande

    # On suppose que tu as bien un @staticmethod from_numpy(...)
    return AlertStore.from_numpy(dia_source_id, ra, dec, mjd_tt, flux, flux_err, band)


def normalize_pair(a, b):
    return (a, b) if a <= b else (b, a)


def contains_pair(pairs, i, j):
    target = normalize_pair(i, j)
    return any(normalize_pair(a, b) == target for (a, b) in pairs)


def contains_triplet(triplets, i, j, k):
    target = tuple(sorted((i, j, k)))
    return any(tuple(sorted(t)) == target for t in triplets)


# ---------- Tests déterministes ----------


def test_pairs_kept_without_triplet():
    """
    Deux alertes compatibles (Δt, Δθ) mais pas de 3e point -> pas de triplet,
    la paire doit être conservée.
    """
    start = 60000.10
    dec = 0.20
    # Séparation ~6" en RA sur le plan tangent
    dra = arcsec_to_rad(6.0) / math.cos(dec)

    triplets = [
        (1.00, dec, 0.0),  # A @ t0
        (1.00 + dra, dec, 8.0 / 1440.0),  # B @ t0 + 8 min
    ]
    store = build_store_from_triplets(start, triplets)

    # Bins de 10 min ; Δt_max = 15 min ; Δθ_max = 10"
    params = (
        PyFinkFatParams.builder()
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
    assert len(triplets_out) == 0
    assert len(pairs) == 1
    # Les ids sont séquentiels (0-based) si ton Rust les assigne ainsi ; le plus fréquent est [0,1].
    # Si ton impl démarre à 1, cet assert restera valide grâce à la normalisation ci-dessous.
    assert (
        contains_pair(pairs, 0, 1)
        or contains_pair(pairs, 1, 2)
        or contains_pair(pairs, 1, 0)
    )


def test_same_timebin_toggle():
    """
    Deux alertes dans le même bin temporel: si allow_same_timebin=False, on ne forme pas la paire;
    si True, la paire apparaît.
    """
    start = 60000.25
    dec = 0.10
    dra = arcsec_to_rad(4.0) / math.cos(dec)

    # Δt = 5 min, bin = 20 min ⇒ même bin
    triplets = [
        (1.50, dec, 0.0),
        (1.50 + dra, dec, 5.0 / 1440.0),
    ]
    store = build_store_from_triplets(start, triplets)

    params = (
        PyFinkFatParams.builder()
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

    # Interdit same-timebin
    pairs, trips = store.generate_seeds(params)
    assert pairs == []
    assert trips == []

    params = (
        PyFinkFatParams.builder()
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

    # Autorise same-timebin
    pairs2, trips2 = store.generate_seeds(params)
    assert len(pairs2) == 1
    assert len(trips2) == 0


def test_triplet_linear_motion_and_pairs_present():
    """
    Mouvement linéaire: 3 points alignés en RA (plan tangent), 10 min d'intervalle, ~6" par step.
    On attend un triplet et des paires (A,B) et (B,C).
    """
    start = 60000.00
    dec = 0.15
    dra = arcsec_to_rad(6.0) / math.cos(dec)

    triplets = [
        (2.00, dec, 0.0),  # A
        (2.00 + dra, dec, 10.0 / 1440.0),  # B
        (2.00 + 2.0 * dra, dec, 20.0 / 1440.0),  # C
    ]
    store = build_store_from_triplets(start, triplets)

    params = (
        PyFinkFatParams.builder()
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

    # 1) Un triplet (ids 0,1,2 selon impl la plus commune)
    # On ne dépend pas de l'offset d'id exact: on accepte toute permutation triée de {0,1,2}
    assert any(
        set(t) == {0, 1, 2} or set(t) == {1, 2, 3} for t in tri
    ), f"triplets={tri}"

    # 2) Les paires contiennent au moins (A,B) et (B,C)
    assert len(pairs) >= 2
    assert contains_pair(pairs, 0, 1) or contains_pair(
        pairs, 1, 2
    )  # selon indexation réelle
    assert contains_pair(pairs, 1, 2) or contains_pair(pairs, 2, 3)


# ---------- (Optionnel) petit sanity check sur les types ----------


def test_output_types_and_shapes():
    start = 60000.0
    dec = 0.2
    dra = arcsec_to_rad(3.0) / math.cos(dec)

    triplets = [
        (1.0, dec, 0.0),
        (1.0 + dra, dec, 5.0 / 1440.0),
        (1.0, dec, 60.0 / 1440.0),  # bruit plus tard
    ]
    store = build_store_from_triplets(start, triplets)

    params = (
        PyFinkFatParams.builder()
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

    # types
    assert isinstance(pairs, list)
    assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)
    assert isinstance(tri, list)
    assert all(isinstance(t, tuple) and len(t) == 3 for t in tri)
