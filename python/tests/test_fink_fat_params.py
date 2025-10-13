# python/tests/test_fink_fat_params.py
import math
import pytest
from pathlib import Path

import pytest

try:  # Python 3.11+
    import tomllib as _toml  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    try:
        import tomli as _toml  # type: ignore[assignment]
    except Exception:  # pragma: no cover
        _toml = None  # Only used for optional cross-checks

from fink_fat import FinkFatParams


def B():
    """Helper: return a fresh builder from the classmethod."""
    return FinkFatParams.builder()


def assert_float_close(a: float, b: float, rel=1e-12, abs_=0.0):
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a, b, rel_tol=rel, abs_tol=abs_), f"{a} != {b}"


def test_defaults_exist_and_dict_has_all_keys():
    # Nouvelle API : constructeur direct (valeurs par défaut validées côté Rust).
    p = FinkFatParams()
    d = p.to_dict()

    expected_keys = {
        # Global
        "show_progress",
        # Binning
        "healpix_depth",
        "time_bin_width_days",
        # Pairs
        "pair_max_dt",
        "pair_max_sep",
        "pair_max_flux_difference",
        "pair_allow_same_timebin",
        # Triplets
        "triplet_max_dt_between",
        "triplet_max_pair_sep",
        "triplet_max_predicted_residual",
        "triplet_enforce_time_order",
        "triplet_max_flux_difference",
        # Linking – predictor
        "link_k_sigma",
        "link_pad_cell_radius",
        "link_noise_q0",
        "link_noise_q1",
        "link_noise_q2",
        # Linking – weights
        "link_w_pos",
        "link_w_vel_dir",
        "link_w_vel_norm",
        "link_w_flux",
        "link_w_gap",
        "link_w_band_mismatch",
        # Linking – gates
        "link_max_d2_pos",
        "link_max_theta_vel",
        "link_max_speed_diff",
        # Linking – scales
        "link_theta0",
        "link_v0",
        "link_flux_sigma_floor",
        "link_gap_rho",
        "link_vel_eps_days",
        # Linking – limits & caps
        "link_top_k_per_left",
        "link_max_total_edges",
        "link_max_cost",
        "link_max_speed_rad_per_day",
    }

    # Dictionnaire complet
    assert set(d.keys()) == expected_keys

    # Types plausibles (optionnels autorisent None)
    assert isinstance(d["show_progress"], bool)
    assert isinstance(d["healpix_depth"], int)
    for k in [
        "time_bin_width_days",
        "pair_max_dt",
        "pair_max_sep",
        "triplet_max_dt_between",
        "triplet_max_pair_sep",
        "triplet_max_predicted_residual",
        "link_k_sigma",
        "link_noise_q0",
        "link_noise_q1",
        "link_noise_q2",
        "link_w_pos",
        "link_w_vel_dir",
        "link_w_vel_norm",
        "link_w_flux",
        "link_w_gap",
        "link_w_band_mismatch",
        "link_max_d2_pos",
        "link_max_theta_vel",
        "link_max_speed_diff",
        "link_theta0",
        "link_v0",
        "link_flux_sigma_floor",
        "link_gap_rho",
        "link_vel_eps_days",
    ]:
        assert isinstance(d[k], float)
    for k in ["pair_max_flux_difference", "triplet_max_flux_difference"]:
        assert isinstance(d[k], float)
    assert isinstance(d["pair_allow_same_timebin"], bool)
    assert (d["link_max_total_edges"] is None) or isinstance(
        d["link_max_total_edges"], int
    )
    assert (d["link_max_cost"] is None) or isinstance(d["link_max_cost"], float)
    assert (d["link_max_speed_rad_per_day"] is None) or isinstance(
        d["link_max_speed_rad_per_day"], float
    )

    # Cohérence getters <-> dict (échantillon représentatif)
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

    # Linking samples
    assert_float_close(d["link_k_sigma"], p.link_k_sigma)
    assert d["link_pad_cell_radius"] == p.link_pad_cell_radius
    assert_float_close(d["link_theta0"], p.link_theta0)


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
        # Linking – small sample of setters to verify surface
        .link_k_sigma(3.7)
        .link_theta0(2.5e-4)
        .link_top_k_per_left(32)
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
    # Linking assertions
    assert_float_close(p.link_k_sigma, 3.7)
    assert_float_close(p.link_theta0, 2.5e-4)
    assert p.link_top_k_per_left == 32


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
    p = FinkFatParams()
    with pytest.raises(AttributeError):
        setattr(p, "healpix_depth", 9)
    with pytest.raises(AttributeError):
        setattr(p, "time_bin_width_days", 0.01)
    with pytest.raises(AttributeError):
        setattr(p, "show_progress", True)


def test_repr_is_informative_and_contains_key_fields():
    p = FinkFatParams()
    s = repr(p)
    assert isinstance(s, str) and len(s) > 0
    # Noms mis à jour pour coller au __repr__ actuel
    assert "healpix_depth" in s
    assert "pair_max_dt" in s
    assert "triplet_max_dt_between" in s


def test_to_dict_is_copy_not_live_view():
    p = FinkFatParams()
    d = p.to_dict()
    d["healpix_depth"] = 12345
    assert p.healpix_depth != 12345


def test_builder_is_fluent_and_reusable_from_scratch():
    p1 = B().healpix_depth(10).build()
    assert p1.healpix_depth == 10

    # Nouveau builder indépendant
    p2 = B().healpix_depth(11).build()
    assert p2.healpix_depth == 11


# --- Paths --------------------------------------------------------------------

# Project-level tests/data/
DATA_DIR = Path(__file__).parent.parent.parent / "tests" / "data"
FULL_TOML = DATA_DIR / "params_full.toml"
MIN_TOML = DATA_DIR / "params_minimal.toml"
INVALID_TOML = DATA_DIR / "params_invalid.toml"


def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:  # pragma: no cover
        return False


# --- Valid TOML tests ----------------------------------------------------------


@pytest.mark.skipif(not _exists(FULL_TOML), reason="params_full.toml not found")
def test_from_toml_full_validate_roundtrip():
    """Load params_full.toml, validate, and round-trip via to_toml_str()."""
    s_in = FULL_TOML.read_text(encoding="utf-8")

    cfg1 = FinkFatParams.from_toml_str(s_in)
    cfg1.validate()  # should not raise

    # sanity on a few fields
    assert isinstance(cfg1.healpix_depth, int)
    assert cfg1.time_bin_width_days > 0
    assert cfg1.pair_max_dt >= 0
    assert cfg1.link_k_sigma > 0

    # round-trip via TOML
    s_out = cfg1.to_toml_str()
    cfg2 = FinkFatParams.from_toml_str(s_out)
    assert cfg1.to_dict() == cfg2.to_dict()

    # optional: parseable by Python TOML (loose check)
    if _toml is not None:
        doc = _toml.loads(s_in)
        assert isinstance(doc, dict)


@pytest.mark.skipif(not _exists(MIN_TOML), reason="params_minimal.toml not found")
def test_from_toml_minimal_validate_roundtrip():
    """
    Load params_minimal.toml (only version), relying on Rust defaults
    for missing fields; validate and round-trip.
    """
    s_in = MIN_TOML.read_text(encoding="utf-8")

    cfg = FinkFatParams.from_toml_str(s_in)
    cfg.validate()  # should not raise with defaulted values

    # round-trip preserves semantics
    cfg2 = FinkFatParams.from_toml_str(cfg.to_toml_str())
    assert cfg.to_dict() == cfg2.to_dict()


@pytest.mark.skipif(
    not (_exists(FULL_TOML) and _exists(MIN_TOML)),
    reason="params_full.toml or params_minimal.toml not found",
)
def test_distinct_objects_on_multiple_loads():
    """Ensure each load returns a distinct, validated object."""
    cfg1 = FinkFatParams.from_toml_str(FULL_TOML.read_text(encoding="utf-8"))
    cfg2 = FinkFatParams.from_toml_str(MIN_TOML.read_text(encoding="utf-8"))
    cfg1.validate()
    cfg2.validate()
    assert id(cfg1) != id(cfg2)


# --- Invalid TOML tests --------------------------------------------------------


@pytest.mark.skipif(not _exists(INVALID_TOML), reason="params_invalid.toml not found")
def test_params_invalid_toml_raises_value_error():
    """
    Semantically/structurally invalid file (params_invalid.toml)
    must raise ValueError from from_toml_str().
    """
    s_bad = INVALID_TOML.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        FinkFatParams.from_toml_str(s_bad)


def test_malformed_inline_toml_raises_value_error():
    """Malformed TOML string must raise ValueError."""
    bad = """
    [this is not valid toml
    x = 1
    """
    with pytest.raises(ValueError):
        FinkFatParams.from_toml_str(bad)


# --- Builder tests -------------------------------------------------------------


def test_builder_and_clear_optionals():
    """Exercise builder optionals and 'clear' helpers."""
    b = FinkFatParams.builder()
    b = (
        b.link_max_total_edges(1234)
        .link_max_cost(10.5)
        .link_max_speed_rad_per_day(0.05)
    )
    cfg = b.build()
    d = cfg.to_dict()
    assert d["link_max_total_edges"] == 1234
    assert d["link_max_cost"] == 10.5
    assert d["link_max_speed_rad_per_day"] == 0.05

    # Clear some optionals and ensure they become None
    b = (
        FinkFatParams.builder()
        .link_max_total_edges(500)
        .link_clear_max_total_edges()
        .link_max_cost(3.14)
        .link_clear_max_cost()
    )
    cfg2 = b.build()
    d2 = cfg2.to_dict()
    assert d2["link_max_total_edges"] is None
    assert d2["link_max_cost"] is None


def test_minimal_defaults_are_serializable_and_reparse_without_files():
    """
    Default constructor should serialize and reparse to identical dict.
    This does not depend on tests/data presence.
    """
    cfg = FinkFatParams()
    s = cfg.to_toml_str()
    cfg2 = FinkFatParams.from_toml_str(s)
    assert cfg.to_dict() == cfg2.to_dict()


@pytest.mark.parametrize(
    "depth,bin_days,k_sigma",
    [
        (6, 0.02, 2.5),
        (8, 0.05, 3.0),
        (10, 0.10, 4.0),
    ],
)
def test_builder_custom_values_roundtrip(depth, bin_days, k_sigma):
    """Set a few common fields with the builder and round-trip."""
    cfg = (
        FinkFatParams.builder()
        .healpix_depth(depth)
        .time_bin_width_days(bin_days)
        .link_k_sigma(k_sigma)
        .build()
    )
    assert cfg.healpix_depth == depth
    assert cfg.time_bin_width_days == pytest.approx(bin_days)
    assert cfg.link_k_sigma == pytest.approx(k_sigma)

    # round-trip through TOML
    cfg2 = FinkFatParams.from_toml_str(cfg.to_toml_str())
    assert cfg.to_dict() == cfg2.to_dict()


def test_semantically_invalid_via_builder_raises():
    """Demonstrate typical validation failures on the builder."""
    # Example: zero time bin width (must be > 0)
    b = FinkFatParams.builder().time_bin_width_days(0.0)
    with pytest.raises(ValueError):
        _ = b.build()

    # Example: negative weight (must be ≥ 0)
    b = FinkFatParams.builder().link_w_pos(-1.0)
    with pytest.raises(ValueError):
        _ = b.build()
