# fink_fat/params.pyi
# Type stubs for Fink-FAT Python parameter bindings (auto-completion & mypy).
# Docstrings follow the NumPy docstring convention.

from __future__ import annotations

from typing import TypedDict, Union

Number = Union[int, float]


class PyFinkFatParamsDict(TypedDict):
    """Dictionary view of scalar configuration fields.

    Keys
    ----
    show_progress : bool
        Whether progress bars are enabled.
    healpix_depth : int
        HEALPix depth (NSIDE = 2^depth). Valid range: 0..=29.
    time_bin_width_days : float
        Temporal bucket width (days, TT). Must be > 0.
    pair_max_dt : float
        Pair: maximum Δt between alerts (days, TT). ≥ 0.
    pair_max_sep : float
        Pair: maximum angular separation (radians). ≥ 0.
    pair_max_flux_difference : float
        Pair: maximum photometric difference (dimensionless). ≥ 0.
    pair_allow_same_timebin : bool
        Pair: whether same time-bin pairs are allowed.
    triplet_max_dt_between : float
        Triplet: maximum Δt between neighbors (days, TT). ≥ 0.
    triplet_max_pair_sep : float
        Triplet: maximum neighbor angular separation (radians). ≥ 0.
    triplet_max_predicted_residual : float
        Triplet: maximum predicted residual at `c` (radians). ≥ 0.
    triplet_enforce_time_order : bool
        Triplet: enforce strict ordering t(a) < t(b) < t(c).
    triplet_max_flux_difference : float
        Triplet: maximum photometric difference (dimensionless). ≥ 0.
    """


class PyFinkFatParams:
    """
    Validated global configuration for Fink-FAT (Python wrapper).

    This object owns the validated Rust configuration (`FinkFatParams`) and
    exposes **read-only** Python properties for all scalar settings, together
    with :meth:`validate` and :meth:`to_dict`.

    Notes
    -----
    Instances are produced by :meth:`PyFinkFatParams.default` or via the
    builder :meth:`PyFinkFatParams.builder`. Fields are read-only in Python;
    to change values, construct a new instance through the builder.

    See Also
    --------
    PyFinkFatParamsBuilder : Fluent builder (flat setters only).
    """

    # ---------------- Constructors ----------------

    @staticmethod
    def default() -> PyFinkFatParams:
        """
        Construct with LSST-like defaults.

        Returns
        -------
        PyFinkFatParams
            Parameter object initialized to project defaults.

        Notes
        -----
        Defaults roughly correspond to:
        - Binning: healpix_depth = 10, time_bin_width_days = 0.02 (≈ 28.8 min)
        - Pairs: max_dt = 0.06 d, max_sep = 0.003 rad, max_flux_difference = 5.0
        - Triplets: max_dt_between = 0.04 d, max_pair_sep = 0.0025 rad,
          max_predicted_residual = 8e-4 rad
        - show_progress = False
        """
        ...

    @staticmethod
    def builder() -> PyFinkFatParamsBuilder:
        """
        Create a fluent builder (flat setters only).

        Returns
        -------
        PyFinkFatParamsBuilder
            Builder allowing chained configuration of fields.

        Examples
        --------
        >>> b = PyFinkFatParams.builder()
        >>> p = (b.show_progress(True)
        ...       .healpix_depth(12)
        ...       .time_bin_width_days(0.03)
        ...       .pair_max_dt(0.05)
        ...       .triplet_max_predicted_residual(6e-4)
        ...       .build())
        """
        ...

    # ---------------- Methods ----------------

    def to_dict(self) -> PyFinkFatParamsDict:
        """
        Return a scalar-only dictionary representation.

        Returns
        -------
        PyFinkFatParamsDict
            Mapping of key configuration fields to scalar values.

        Notes
        -----
        This helper excludes non-scalar or nested structures (if any are
        added in the future).
        """
        ...

    def __repr__(self) -> str: ...  # pragma: no cover

    # ---------------- Read-only properties ----------------

    @property
    def show_progress(self) -> bool:
        """
        Whether progress bars are enabled.

        Returns
        -------
        bool
        """
        ...

    @property
    def healpix_depth(self) -> int:
        """
        HEALPix depth (NSIDE = 2^depth).

        Returns
        -------
        int
            Valid range is 0..=29.
        """
        ...

    @property
    def time_bin_width_days(self) -> float:
        """
        Temporal bucket width (days, TT).

        Returns
        -------
        float
            Must be strictly > 0.
        """
        ...

    @property
    def pair_max_dt(self) -> float:
        """
        Pair: maximum Δt between alerts (days, TT).

        Returns
        -------
        float
            Must be ≥ 0 and finite.
        """
        ...

    @property
    def pair_max_sep(self) -> float:
        """
        Pair: maximum angular separation (radians).

        Returns
        -------
        float
            Must be ≥ 0 and finite.
        """
        ...

    @property
    def pair_max_flux_difference(self) -> float:
        """
        Pair: maximum photometric difference (dimensionless).

        Returns
        -------
        float
            Must be ≥ 0 and finite.
        """
        ...

    @property
    def pair_allow_same_timebin(self) -> bool:
        """
        Pair: allow pairing within the same time bin.

        Returns
        -------
        bool
        """
        ...

    @property
    def triplet_max_dt_between(self) -> float:
        """
        Triplet: maximum Δt between consecutive neighbors (days, TT).

        Returns
        -------
        float
            Must be ≥ 0 and finite.
        """
        ...

    @property
    def triplet_max_pair_sep(self) -> float:
        """
        Triplet: maximum neighbor angular separation (radians).

        Returns
        -------
        float
            Must be ≥ 0 and finite.
        """
        ...

    @property
    def triplet_max_predicted_residual(self) -> float:
        """
        Triplet: maximum predicted residual at `c` (radians),
        when extrapolating linear motion from `a→b`.

        Returns
        -------
        float
            Must be ≥ 0 and finite.
        """
        ...

    @property
    def triplet_enforce_time_order(self) -> bool:
        """
        Triplet: enforce strict ordering t(a) < t(b) < t(c).

        Returns
        -------
        bool
        """
        ...

    @property
    def triplet_max_flux_difference(self) -> float:
        """
        Triplet: maximum photometric difference (dimensionless).

        Returns
        -------
        float
            Must be ≥ 0 and finite.
        """
        ...


class PyFinkFatParamsBuilder:
    """
    Fluent builder for :class:`PyFinkFatParams` (flat setters only).

    The builder mirrors the Rust `FinkFatParamsBuilder` flat API and performs
    Rust-side validation upon :meth:`build`.

    Notes
    -----
    The Python binding intentionally **does not** expose nested closure-based
    setters. All fields are configurable via flat setters.

    See Also
    --------
    PyFinkFatParams : Validated parameter object.
    """

    # ---------------- Global ----------------

    def show_progress(self, v: bool) -> PyFinkFatParamsBuilder:
        """
        Enable/disable progress bars.

        Parameters
        ----------
        v : bool
            Whether to show progress bars.

        Returns
        -------
        PyFinkFatParamsBuilder
            The same builder for chaining.
        """
        ...

    # ---------------- Binning setters ----------------

    def healpix_depth(self, v: int) -> PyFinkFatParamsBuilder:
        """
        Set HEALPix depth (NSIDE = 2^depth).

        Parameters
        ----------
        v : int
            Depth in the inclusive range [0, 29].

        Returns
        -------
        PyFinkFatParamsBuilder
        """
        ...

    def time_bin_width_days(self, v: float) -> PyFinkFatParamsBuilder:
        """
        Set temporal bucket width (days, TT).

        Parameters
        ----------
        v : float
            Strictly positive bin width in days.

        Returns
        -------
        PyFinkFatParamsBuilder
        """
        ...

    # ---------------- Pair setters ----------------

    def pair_max_dt(self, v: float) -> PyFinkFatParamsBuilder:
        """
        Set pair maximum Δt (days, TT).

        Parameters
        ----------
        v : float
            Non-negative, finite time difference.

        Returns
        -------
        PyFinkFatParamsBuilder
        """
        ...

    def pair_max_sep(self, v: float) -> PyFinkFatParamsBuilder:
        """
        Set pair maximum angular separation (radians).

        Parameters
        ----------
        v : float
            Non-negative, finite angle in radians.

        Returns
        -------
        PyFinkFatParamsBuilder
        """
        ...

    def pair_max_flux_difference(self, v: float) -> PyFinkFatParamsBuilder:
        """
        Set pair maximum photometric difference (dimensionless).

        Parameters
        ----------
        v : float
            Non-negative, finite threshold.

        Returns
        -------
        PyFinkFatParamsBuilder
        """
        ...

    def pair_allow_same_timebin(self, v: bool) -> PyFinkFatParamsBuilder:
        """
        Allow/disallow pairing within the same time bin.

        Parameters
        ----------
        v : bool
            If True, allow same time-bin pairs.

        Returns
        -------
        PyFinkFatParamsBuilder
        """
        ...

    # ---------------- Triplet setters ----------------

    def triplet_max_dt_between(self, v: float) -> PyFinkFatParamsBuilder:
        """
        Set maximum Δt between consecutive neighbors (days, TT).

        Parameters
        ----------
        v : float
            Non-negative, finite time difference.

        Returns
        -------
        PyFinkFatParamsBuilder
        """
        ...

    def triplet_max_pair_sep(self, v: float) -> PyFinkFatParamsBuilder:
        """
        Set maximum neighbor angular separation (radians).

        Parameters
        ----------
        v : float
            Non-negative, finite angle in radians.

        Returns
        -------
        PyFinkFatParamsBuilder
        """
        ...

    def triplet_max_predicted_residual(self, v: float) -> PyFinkFatParamsBuilder:
        """
        Set maximum predicted residual at `c` (radians), extrapolating `a→b`.

        Parameters
        ----------
        v : float
            Non-negative, finite residual in radians.

        Returns
        -------
        PyFinkFatParamsBuilder
        """
        ...

    def triplet_enforce_time_order(self, v: bool) -> PyFinkFatParamsBuilder:
        """
        Enforce strict time ordering (t(a) < t(b) < t(c)).

        Parameters
        ----------
        v : bool
            If True, enforce strict ordering.

        Returns
        -------
        PyFinkFatParamsBuilder
        """
        ...

    def triplet_max_flux_difference(self, v: float) -> PyFinkFatParamsBuilder:
        """
        Set triplet maximum photometric difference (dimensionless).

        Parameters
        ----------
        v : float
            Non-negative, finite threshold.

        Returns
        -------
        PyFinkFatParamsBuilder
        """
        ...

    # ---------------- Build ----------------

    def build(self) -> PyFinkFatParams:
        """
        Build a validated :class:`PyFinkFatParams`.

        Returns
        -------
        PyFinkFatParams
            Validated parameter object.

        Raises
        ------
        ValueError
            If validation fails (invalid range or inconsistent thresholds).
        """
        ...

    def __repr__(self) -> str: ...  # pragma: no cover

