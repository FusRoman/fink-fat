# python/tests/test_fink_fat_params.py
import math
import pytest

from fink_fat import PyFinkFatParams

def B():
    """Helper: return a fresh builder from the classmethod."""
    return PyFinkFatParams.builder()


def assert_float_close(a: float, b: float, rel=1e-12, abs_=0.0):
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a, b, rel_tol=rel, abs_tol=abs_), f"{a} != {b}"


def test_defaults_exist_and_dict_has_all_keys():
    p = PyFinkFatParams.default()
    d = p.to_dict()
    expected_keys = {
        "show_progress",
        "healpix_depth",
        "time_bin_width_days",
        "pair_max_dt",
        "pair_max_sep",
        "pair_max_flux_difference",
        "pair_allow_same_timebin",
        "triplet_max_dt_between",
        "triplet_max_pair_sep",
        "triplet_max_predicted_residual",
        "triplet_enforce_time_order",
        "triplet_max_flux_difference",
    }
    assert set(d.keys()) == expected_keys

    # Types plausibles
    assert isinstance(d["show_progress"], bool)
    assert isinstance(d["healpix_depth"], int)
    for k in [
        "time_bin_width_days",
        "pair_max_dt",
        "pair_max_sep",
        "triplet_max_dt_between",
        "triplet_max_pair_sep",
        "triplet_max_predicted_residual",
    ]:
        assert isinstance(d[k], float)
    for k in ["pair_max_flux_difference", "triplet_max_flux_difference"]:
        assert isinstance(d[k], float)
    assert isinstance(d["pair_allow_same_timebin"], bool)

    # Cohérence getters <-> dict
    assert d["show_progress"] == p.show_progress
    assert d["healpix_depth"] == p.healpix_depth
    assert_float_close(d["time_bin_width_days"], p.time_bin_width_days)
    assert_float_close(d["pair_max_dt"], p.pair_max_dt)
    assert_float_close(d["pair_max_sep"], p.pair_max_sep)
    assert d["pair_allow_same_timebin"] == p.pair_allow_same_timebin
    assert_float_close(d["triplet_max_dt_between"], p.triplet_max_dt_between)
    assert_float_close(d["triplet_max_pair_sep"], p.triplet_max_pair_sep)
    assert_float_close(
        d["triplet_max_predicted_residual"], p.triplet_max_predicted_residual
    )
    assert d["triplet_enforce_time_order"] == p.triplet_enforce_time_order


def test_builder_flat_setters_and_build_success():
    b = (
        B()
        .show_progress(True)
        .healpix_depth(12)
        .time_bin_width_days(0.03)
        .pair_max_dt(0.05)
        .pair_max_sep(0.0025)
        .pair_max_flux_difference(3.0)
        .pair_allow_same_timebin(False)
        .triplet_max_dt_between(0.02)
        .triplet_max_pair_sep(0.0018)
        .triplet_max_predicted_residual(5e-4)
        .triplet_enforce_time_order(True)
        .triplet_max_flux_difference(4.0)
    )
    p = b.build()

    assert p.show_progress is True
    assert p.healpix_depth == 12
    assert_float_close(p.time_bin_width_days, 0.03)
    assert_float_close(p.pair_max_dt, 0.05)
    assert_float_close(p.pair_max_sep, 0.0025)
    assert_float_close(p.pair_max_flux_difference, 3.0)
    assert p.pair_allow_same_timebin is False
    assert_float_close(p.triplet_max_dt_between, 0.02)
    assert_float_close(p.triplet_max_pair_sep, 0.0018)
    assert_float_close(p.triplet_max_predicted_residual, 5e-4)
    assert p.triplet_enforce_time_order is True
    assert_float_close(p.triplet_max_flux_difference, 4.0)


@pytest.mark.parametrize("depth", [30, 255])
def test_invalid_healpix_depth_out_of_range_build_raises(depth):
    # Profondeurs hors plage [0,29] => ValueError à la build (validation Rust)
    with pytest.raises(ValueError):
        B().healpix_depth(depth).build()


def test_invalid_healpix_depth_negative_setter_raises():
    # Valeur négative : Pyo3 peut lever OverflowError/TypeError au setter,
    # avant même la validation Rust sur build().
    with pytest.raises((OverflowError, TypeError, ValueError)):
        # setter lève déjà, donc on ne va probablement pas jusqu'à build()
        B().healpix_depth(-1)  # type: ignore[arg-type]


@pytest.mark.parametrize("dt", [0.0, -0.01, float("nan")])
def test_invalid_time_bin_width_days_raises(dt):
    with pytest.raises(ValueError):
        B().time_bin_width_days(dt).build()


@pytest.mark.parametrize("val", [-1.0, float("nan")])
def test_invalid_angles_and_photometry_raise(val):
    with pytest.raises(ValueError):
        B().pair_max_sep(val).build()
    with pytest.raises(ValueError):
        B().triplet_max_dt_between(val).build()
    with pytest.raises(ValueError):
        B().pair_max_flux_difference(val).build()
    with pytest.raises(ValueError):
        B().triplet_max_flux_difference(val).build()


def test_triplet_inconsistency_residual_greater_than_pair_sep_raises():
    with pytest.raises(ValueError):
        (
            B()
            .triplet_max_pair_sep(1.0e-3)
            .triplet_max_predicted_residual(2.0e-3)  # > pair sep => invalide
            .build()
        )


def test_properties_are_read_only():
    p = PyFinkFatParams.default()
    with pytest.raises(AttributeError):
        setattr(p, "healpix_depth", 9)
    with pytest.raises(AttributeError):
        setattr(p, "time_bin_width_days", 0.01)
    with pytest.raises(AttributeError):
        setattr(p, "show_progress", True)


def test_repr_is_informative_and_contains_key_fields():
    p = PyFinkFatParams.default()
    s = repr(p)
    assert isinstance(s, str) and len(s) > 0
    assert "depth" in s
    assert "pair_max_dt" in s
    assert "trip_max_pair_sep" in s


def test_to_dict_is_copy_not_live_view():
    p = PyFinkFatParams.default()
    d = p.to_dict()
    d["healpix_depth"] = 12345
    assert p.healpix_depth != 12345


def test_builder_is_fluent_and_reusable_from_scratch():
    p1 = B().healpix_depth(10).build()
    assert p1.healpix_depth == 10

    # Nouveau builder indépendant
    p2 = B().healpix_depth(11).build()
    assert p2.healpix_depth == 11
