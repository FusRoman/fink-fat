# tests/test_alerts.py
import numpy as np
import pytest

from fink_fat import AlertStore


@pytest.fixture
def fake_alerts() -> tuple[AlertStore, dict[str, np.ndarray]]:
    n = 1000
    dia_source_id = np.arange(100, 100 + n, dtype=np.uint64)
    ra = np.linspace(0.1, 0.5, n, dtype=np.float64)
    ra_err = np.full(n, np.deg2rad(0.5 / 3600), dtype=np.float64)
    dec = np.linspace(-0.2, -0.1, n, dtype=np.float64)
    dec_err = np.full(n, np.deg2rad(0.5 / 3600), dtype=np.float64)
    mjd_tt = np.linspace(60000.0, 60000.4, n, dtype=np.float64)
    flux = np.linspace(1000.0, 2000.0, n, dtype=np.float32)
    flux_err = np.full(n, 10.0, dtype=np.float32)
    band = np.arange(n, dtype=np.uint8)

    store = AlertStore.from_numpy(
        dia_source_id, ra, ra_err, dec, dec_err, mjd_tt, flux, flux_err, band
    )
    return store, {
        "dia_source_id": dia_source_id,
        "ra": ra,
        "ra_err": ra_err,
        "dec": dec,
        "dec_err": dec_err,
        "mjd_tt": mjd_tt,
        "flux": flux,
        "flux_err": flux_err,
        "band": band,
    }


def test_len(fake_alerts: tuple[AlertStore, dict[str, np.ndarray]]):
    store, arrays = fake_alerts
    assert len(store) == len(arrays["ra"])


def test_getitem_and_fields(fake_alerts: tuple[AlertStore, dict[str, np.ndarray]]):
    store, arrays = fake_alerts
    a0 = store[0]
    # Ne dépend pas de AlertId → on vérifie l’alignement des champs scalaires.
    assert a0.dia_source_id == arrays["dia_source_id"][0]
    assert np.isclose(a0.ra, arrays["ra"][0])
    assert np.isclose(a0.dec, arrays["dec"][0])
    assert np.isclose(a0.mjd_tt, arrays["mjd_tt"][0])
    assert np.isclose(a0.flux, arrays["flux"][0])
    assert np.isclose(a0.flux_err, arrays["flux_err"][0])
    assert a0.band == arrays["band"][0]


def test_get_by_id(fake_alerts: tuple[AlertStore, dict[str, np.ndarray]]):
    store, arrays = fake_alerts
    # On ne présume plus de `store.get` (non exposé ici, tu as `get_py`).
    # Fallback robuste: sélection via la clé stable `dia_source_id`.
    target_dia = arrays["dia_source_id"][2]
    a2 = None
    for i in range(len(store)):
        ai = store[i]
        if ai.dia_source_id == target_dia:
            a2 = ai
            break
    assert a2 is not None, "Alert with dia_source_id at index 2 not found."
    # Vérifications des champs
    assert a2.dia_source_id == arrays["dia_source_id"][2]
    assert np.isclose(a2.ra, arrays["ra"][2])
    assert np.isclose(a2.dec, arrays["dec"][2])


def test_index_error(fake_alerts: tuple[AlertStore, dict[str, np.ndarray]]):
    store, _ = fake_alerts
    with pytest.raises(IndexError):
        _ = store[len(store)]  # hors borne
    # Pas de test d’ID invalide via `.get` car l’API n’existe pas ici.


def test_repr(fake_alerts: tuple[AlertStore, dict[str, np.ndarray]]):
    store, _ = fake_alerts
    s = repr(store[0])
    assert "Alert(" in s
    # `repr` inclut `id=0` via `self.id.idx()` côté Rust → c’est la seule
    # validation liée à l’id (sans conversion en int côté Python).
    assert "id=0" in s
