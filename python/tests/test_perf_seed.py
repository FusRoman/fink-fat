# tests/perf/test_perf_generate_seeds_10m.py
import os
import time
import math
import numpy as np
import pytest

from fink_fat import AlertStore  # PyO3 module

# -------------- helpers --------------

def arcsec_to_rad(x: float) -> float:
    return x * math.pi / (180.0 * 3600.0)

def wrap_0_2pi(x: np.ndarray) -> np.ndarray:
    y = np.fmod(x, 2.0 * math.pi)
    y[y < 0.0] += 2.0 * math.pi
    return y

def uniform_sphere_dec(n: int, rng: np.random.Generator) -> np.ndarray:
    # dec ~ asin(u), u ~ U[-1,1] ⇒ uniforme sur la sphère
    u = rng.uniform(-1.0, 1.0, size=n)
    return np.arcsin(u)

def build_alert_store_from_numpy(
    start_mjd: float,
    dia_source_id: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    mjd_tt: np.ndarray,
    band: np.ndarray,
):
    n = len(ra)
    flux = np.zeros(n, dtype=np.float32)
    flux_err = np.zeros(n, dtype=np.float32)
    return AlertStore.from_numpy(dia_source_id, ra, dec, mjd_tt, flux, flux_err, band)


def synthesize_alerts(
    n_total: int,
    start_mjd: float,
    pair_frac: float = 0.20,       # fraction des alertes appartenant à des paires
    triplet_frac: float = 0.05,    # fraction des alertes appartenant à des triplets
    pair_dt_range=(5.0/1440.0, 25.0/1440.0),   # 5–25 min
    trip_dt: float = 10.0/1440.0,              # 10 min entre A→B et B→C
    step_arcsec: float = 6.0,     # ~6" par step (cohérent avec tes seuils)
    seed: int = 42,
):
    """
    Génère N alertes :
      - ~pair_frac * N alertes organisées en **paires**
      - ~triplet_frac * N alertes organisées en **triplets** (groupes de 3)
      - le reste = bruit aléatoire
    Le placement est vectorisé pour éviter les boucles Python.
    """
    assert 0.0 <= pair_frac < 1.0 and 0.0 <= triplet_frac < 1.0
    rng = np.random.default_rng(seed)

    # Calcule le nombre de groupes
    n_pairs_alerts = int(n_total * pair_frac)
    n_trip_alerts  = int(n_total * triplet_frac)
    # groupes (2 alertes par paire, 3 par triplet)
    n_pair_groups   = n_pairs_alerts // 2
    n_trip_groups   = n_trip_alerts  // 3

    n_pairs_alerts = 2 * n_pair_groups
    n_trip_alerts  = 3 * n_trip_groups
    n_noise = n_total - (n_pairs_alerts + n_trip_alerts)

    # Allocations finales
    ra  = np.empty(n_total, dtype=np.float64)
    dec = np.empty(n_total, dtype=np.float64)
    t   = np.empty(n_total, dtype=np.float64)
    band = np.ones(n_total, dtype=np.uint8)
    dia = np.arange(1, n_total + 1, dtype=np.int64)

    # --- Triplets A,B,C (vectorisé) ---
    idx0 = 0
    if n_trip_groups > 0:
        g = n_trip_groups
        # bases
        raA  = rng.uniform(0.0, 2.0 * math.pi, size=g)
        decA = uniform_sphere_dec(g, rng)
        tA   = rng.uniform(0.0, 0.2, size=g)  # 0–4.8 h dans la nuit (fenêtre arbitraire)

        # déplacements (~6" par step le long de RA dans le plan tangent)
        dra = arcsec_to_rad(step_arcsec) / np.cos(decA)

        raB = raA + dra
        raC = raA + 2.0 * dra
        decB = decA.copy()
        decC = decA.copy()
        tB  = tA + trip_dt
        tC  = tA + 2.0 * trip_dt

        # interleave A,B,C dans le buffer final
        ra[idx0:idx0 + 3*g:3] = raA
        ra[idx0+1:idx0 + 3*g:3] = raB
        ra[idx0+2:idx0 + 3*g:3] = raC
        dec[idx0:idx0 + 3*g:3] = decA
        dec[idx0+1:idx0 + 3*g:3] = decB
        dec[idx0+2:idx0 + 3*g:3] = decC
        t[idx0:idx0 + 3*g:3] = tA
        t[idx0+1:idx0 + 3*g:3] = tB
        t[idx0+2:idx0 + 3*g:3] = tC
        idx0 += 3 * g

    # --- Paires A,B (vectorisé) ---
    if n_pair_groups > 0:
        g = n_pair_groups
        raA  = rng.uniform(0.0, 2.0 * math.pi, size=g)
        decA = uniform_sphere_dec(g, rng)
        tA   = rng.uniform(0.0, 0.2, size=g)

        dra = arcsec_to_rad(step_arcsec) / np.cos(decA)
        dt  = rng.uniform(pair_dt_range[0], pair_dt_range[1], size=g)

        raB = raA + dra
        decB = decA.copy()
        tB  = tA + dt

        ra[idx0:idx0 + 2*g:2] = raA
        ra[idx0+1:idx0 + 2*g:2] = raB
        dec[idx0:idx0 + 2*g:2] = decA
        dec[idx0+1:idx0 + 2*g:2] = decB
        t[idx0:idx0 + 2*g:2] = tA
        t[idx0+1:idx0 + 2*g:2] = tB
        idx0 += 2 * g

    # --- Bruit (reste) ---
    if n_noise > 0:
        ra[idx0:]  = rng.uniform(0.0, 2.0 * math.pi, size=n_noise)
        dec[idx0:] = uniform_sphere_dec(n_noise, rng)
        t[idx0:]   = rng.uniform(0.0, 0.2, size=n_noise)

    # normalisation RA et temps absolu
    ra = wrap_0_2pi(ra)
    mjd = start_mjd + t

    store = build_alert_store_from_numpy(start_mjd, dia, ra, dec, mjd, band)
    return store


# -------------- perf test --------------

def test_perf_generate_seeds_10m():
    """
    Benchmark "end-to-end" sur 10 millions d'alertes synthétiques.
    - Génération : vectorisée NumPy (pairs + triplets + bruit)
    - generate_seeds : via le binding PyO3 (Rust en release)
    Affiche les temps de build + seeds, et le volume de seeds produits.
    """
    N = int(os.getenv("N_ALERTS", "10000000"))
    start_mjd = 62000.0

    # Paramétrage LSST-like (ajuste au besoin)
    healpix_depth = 10                     # NSIDE=256 (~13.7" pixel → proche d'un rayon de 10")
    time_bin_width_days = 10.0 / 1440.0   # 10 min
    pair_max_dt = 30.0 / 1440.0           # 30 min
    pair_max_sep = arcsec_to_rad(15.0)    # 15"
    allow_same_timebin = False

    trip_max_dt_between = 15.0 / 1440.0   # 15 min
    trip_max_pair_sep = arcsec_to_rad(15.0)
    trip_max_pred_resid = arcsec_to_rad(4.0)
    enforce_time_order = True

    t0 = time.perf_counter()
    store = synthesize_alerts(
        N,
        start_mjd,
        pair_frac=0.20,
        triplet_frac=0.05,
        pair_dt_range=(5.0/1440.0, 25.0/1440.0),
        trip_dt=10.0/1440.0,
        step_arcsec=6.0,
        seed=1234,
    )
    t1 = time.perf_counter()
    print(f"[perf] generated {N:,} alerts in {t1 - t0:.2f}s")

    pairs, triplets = store.generate_seeds(
        healpix_depth=healpix_depth,
        time_bin_width_days=time_bin_width_days,
        pair_max_dt=pair_max_dt,
        pair_max_sep=pair_max_sep,
        allow_same_timebin=allow_same_timebin,
        trip_max_dt_between=trip_max_dt_between,
        trip_max_pair_sep=trip_max_pair_sep,
        trip_max_pred_resid=trip_max_pred_resid,
        enforce_time_order=enforce_time_order,
    )
    t2 = time.perf_counter()

    # stats
    print(f"[perf] seeds generated in {t2 - t1:.2f}s")
    print(f"[perf] total pairs:    {len(pairs):,}")
    print(f"[perf] total triplets: {len(triplets):,}")
    print(f"[perf] total runtime:  {t2 - t0:.2f}s")

    # Sanity-check minimal (évite 'test vide')
    assert isinstance(pairs, list) and isinstance(triplets, list)
    # avec nos paramètres synthétiques, on s'attend à quelques seeds non nuls
    assert len(pairs) > 0

    print(pairs[:5])

    print(triplets[:5])

    print(store[pairs[0][0]], store[pairs[0][1]])
    print(store[triplets[0][0]], store[triplets[0][1]], store[triplets[0][2]])
    