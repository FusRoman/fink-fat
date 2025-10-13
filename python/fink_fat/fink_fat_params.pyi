# fink_fat/fink_fat_params.pyi
# Type stubs for Fink-FAT Python parameter bindings (auto-completion & mypy).
# Docstrings follow the NumPy docstring convention.

from __future__ import annotations
from typing import Any, Dict, Optional, TypedDict, Union

Number = Union[int, float]


class FinkFatParamsDict(TypedDict):
    """Dictionary view of scalar configuration fields.

    This mapping exposes a **flat** copy of all scalar parameters used by
    Fink-FAT. It is designed for quick inspection and serialization in Python.
    Mutating this dictionary does **not** affect the underlying Rust instance.

    Notes
    -----
    * **Angles** are in **radians**.
    * **Times** are in **days**, on the **TT** time scale.
    * Parameters with `Optional[...]` accept `None` to **disable** the related
      constraint or cap.

    Keys
    ----
    show_progress : bool
        Whether to enable progress bars in long-running operations.
    healpix_depth : int
        HEALPix order (K). NSIDE = 2**K. Typical range [0, 29].
    time_bin_width_days : float
        Temporal bucket width (days, TT). Must be > 0.

    pair_max_dt : float
        Pair — maximum time separation (days, TT). ≥ 0.
    pair_max_sep : float
        Pair — maximum great-circle separation (radians). ≥ 0.
    pair_max_flux_difference : float
        Pair — maximum photometric difference (dimensionless). ≥ 0.
    pair_allow_same_timebin : bool
        Pair — whether to allow within-bin pairing.

    triplet_max_dt_between : float
        Triplet — maximum Δt between neighboring detections (days, TT). ≥ 0.
    triplet_max_pair_sep : float
        Triplet — maximum neighbor separation (radians). ≥ 0.
    triplet_max_predicted_residual : float
        Triplet — max predicted residual at `c` from `a→b` (radians). ≥ 0.
    triplet_enforce_time_order : bool
        Triplet — enforce strict temporal ordering t(a) < t(b) < t(c).
    triplet_max_flux_difference : float
        Triplet — maximum photometric difference (dimensionless). ≥ 0.

    # Linking — predictor
    link_k_sigma : float
        Cone inflation factor (dimensionless). > 0.
    link_pad_cell_radius : bool
        If True, pad prediction cone by spatial cell radius.
    link_noise_q0 : float
        Additive model noise — variance floor (rad²). ≥ 0.
    link_noise_q1 : float
        Additive model noise — linear term per day (rad²/day). ≥ 0.
    link_noise_q2 : float
        Additive model noise — quadratic term per day² (rad²/day²). ≥ 0.

    # Linking — weights
    link_w_pos : float
        Weight for positional Mahalanobis term (dimensionless). ≥ 0.
    link_w_vel_dir : float
        Weight for velocity **direction** consistency (dimensionless). ≥ 0.
    link_w_vel_norm : float
        Weight for velocity **norm** (speed) consistency (dimensionless). ≥ 0.
    link_w_flux : float
        Weight for photometric consistency (dimensionless). ≥ 0.
    link_w_gap : float
        Weight for revisit-gap penalty (dimensionless). ≥ 0.
    link_w_band_mismatch : float
        Weight for band mismatch penalty (dimensionless). ≥ 0.

    # Linking — gates
    link_max_d2_pos : float
        Hard gate on positional Mahalanobis distance d². ≥ 0.
    link_max_theta_vel : float
        Hard gate on velocity direction mismatch (radians). ≥ 0.
    link_max_speed_diff : float
        Hard gate on absolute speed difference (rad/day). ≥ 0.

    # Linking — scales
    link_theta0 : float
        Angular scale used to normalize angular costs (radians). > 0.
    link_v0 : float
        Speed scale used to normalize speed costs (rad/day). > 0.
    link_flux_sigma_floor : float
        Sigma floor for flux cost (dimensionless). ≥ 0.
    link_gap_rho : float
        Exponent ρ for Δ>1 gap penalty (dimensionless). ≥ 0.
    link_vel_eps_days : float
        Finite-difference step to estimate plane-velocity at j (days). > 0.

    # Linking — limits & caps
    link_top_k_per_left : int
        Keep at most K candidate edges per left node (per seed). ≥ 1.
    link_max_total_edges : Optional[int]
        Global cap on edges after Top-K; None disables the cap. ≥ 1 if set.
    link_max_cost : Optional[float]
        Hard cutoff on edge cost; None disables the cutoff. ≥ 0 if set.
    link_max_speed_rad_per_day : Optional[float]
        Per-seed absolute speed cap (rad/day); None disables. ≥ 0 if set.
    """


class FinkFatParams:
    """
    Validated, read-only parameter object for Fink-FAT.

    This configuration drives:
      * **Binning** (HEALPix depth, time-bin width),
      * **Seeding** (pair/triplet thresholds),
      * **Inter-night linking** (prediction noise, scores, gates, and caps).

    Notes
    -----
    * All **angles** are in **radians**.
    * All **times** are in **days** on the **TT** time scale.
    * Use the fluent :class:`FinkFatParamsBuilder` to construct a validated instance.

    Examples
    --------
    >>> p = FinkFatParams()
    >>> p2 = (FinkFatParams.builder()
    ...        .healpix_depth(7)
    ...        .link_k_sigma(3.0)
    ...        .build())

    Docstrings explicitly state **units** and **constraints**; validation
    occurs on build and when calling :meth:`validate`.
    """

    # ----- Construction & I/O -----

    def __init__(self) -> None:
        """
        Initialize with Rust-side defaults (validated).

        Notes
        -----
        Defaults are chosen to be conservative and robust for LSST-like data
        volumes. Exact values are defined on the Rust side and may evolve
        across releases.
        """

    @staticmethod
    def builder() -> FinkFatParamsBuilder:
        """
        Return a new fluent builder.

        Returns
        -------
        FinkFatParamsBuilder
            Builder pre-populated with the same defaults as :class:`FinkFatParams`.
        """

    @staticmethod
    def from_toml_str(s: str) -> FinkFatParams:
        """
        Build a validated parameter object from a TOML string.

        Parameters
        ----------
        s : str
            TOML content defining a (possibly partial) configuration.

        Returns
        -------
        FinkFatParams
            Validated parameter object.

        Raises
        ------
        ValueError
            If parsing fails or any parameter violates its constraints.
        """

    def to_toml_str(self) -> str:
        """
        Serialize this configuration to a pretty TOML string.

        Returns
        -------
        str
            TOML content suitable for writing to disk or version control.
        """

    # ----- Global -----

    @property
    def show_progress(self) -> bool:
        """Whether to enable progress bars in CPU-bound routines."""

    # ----- Binning -----

    @property
    def healpix_depth(self) -> int:
        """
        HEALPix order (K).

        Notes
        -----
        NSIDE = 2**K. Higher values mean smaller sky pixels and more memory.
        Typical range is [6, 12] for LSST-like intra-night pre-binning.
        """

    @property
    def time_bin_width_days(self) -> float:
        """
        Temporal bucket width (days, TT). Must be > 0.

        Notes
        -----
        Controls the coarseness of the time discretization for seeding and
        initial candidate generation. Typical values are minutes to hours
        (e.g., 0.01–0.1 days).
        """

    # ----- Pairs -----

    @property
    def pair_max_dt(self) -> float:
        """Pair — maximum time separation (days, TT). Constraint: ≥ 0."""

    @property
    def pair_max_sep(self) -> float:
        """Pair — maximum great-circle separation (radians). Constraint: ≥ 0."""

    @property
    def pair_max_flux_difference(self) -> float:
        """Pair — maximum photometric difference (dimensionless). Constraint: ≥ 0."""

    @property
    def pair_allow_same_timebin(self) -> bool:
        """
        Pair — whether to allow pairing inside the **same** time bin.

        Notes
        -----
        Keeping this `False` can reduce spurious pairs when exposures are
        closely spaced within a bin.
        """

    # ----- Triplets -----

    @property
    def triplet_max_dt_between(self) -> float:
        """Triplet — maximum Δt between neighbors (days, TT). Constraint: ≥ 0."""

    @property
    def triplet_max_pair_sep(self) -> float:
        """Triplet — maximum neighbor separation (radians). Constraint: ≥ 0."""

    @property
    def triplet_max_predicted_residual(self) -> float:
        """
        Triplet — max predicted residual at `c` from `a→b` (radians).

        Notes
        -----
        Uses a constant-velocity prediction on the tangent plane from the `a→b`
        segment to estimate the expected location of `c`.
        """

    @property
    def triplet_enforce_time_order(self) -> bool:
        """Triplet — enforce strict ordering t(a) < t(b) < t(c)."""

    @property
    def triplet_max_flux_difference(self) -> float:
        """Triplet — maximum photometric difference (dimensionless). Constraint: ≥ 0."""

    # ----- Linking — predictor -----

    @property
    def link_k_sigma(self) -> float:
        """
        Cone inflation factor (dimensionless). Constraint: > 0.

        Notes
        -----
        Multiplies the predicted covariance to enlarge the search cone, trading
        recall vs. precision. Typical values: 2.0–4.0.
        """

    @property
    def link_pad_cell_radius(self) -> bool:
        """
        Whether to pad the search cone by the spatial cell radius.

        Notes
        -----
        When using HEALPix-based bucketing, enabling this reduces boundary
        effects at the cost of more candidates.
        """

    @property
    def link_noise_q0(self) -> float:
        """Additive model-noise variance floor (rad²). Constraint: ≥ 0."""

    @property
    def link_noise_q1(self) -> float:
        """Additive model-noise linear term per day (rad²/day). Constraint: ≥ 0."""

    @property
    def link_noise_q2(self) -> float:
        """Additive model-noise quadratic term per day² (rad²/day²). Constraint: ≥ 0."""

    # ----- Linking — weights -----

    @property
    def link_w_pos(self) -> float:
        """Weight for positional Mahalanobis term (dimensionless). Constraint: ≥ 0."""

    @property
    def link_w_vel_dir(self) -> float:
        """Weight for velocity direction consistency (dimensionless). Constraint: ≥ 0."""

    @property
    def link_w_vel_norm(self) -> float:
        """Weight for speed consistency (dimensionless). Constraint: ≥ 0."""

    @property
    def link_w_flux(self) -> float:
        """Weight for photometric consistency (dimensionless). Constraint: ≥ 0."""

    @property
    def link_w_gap(self) -> float:
        """Weight for revisit-gap penalty (dimensionless). Constraint: ≥ 0."""

    @property
    def link_w_band_mismatch(self) -> float:
        """Weight for band mismatch penalty (dimensionless). Constraint: ≥ 0."""

    # ----- Linking — gates -----

    @property
    def link_max_d2_pos(self) -> float:
        """Hard gate on positional Mahalanobis distance d². Constraint: ≥ 0."""

    @property
    def link_max_theta_vel(self) -> float:
        """Hard gate on velocity direction mismatch (radians). Constraint: ≥ 0."""

    @property
    def link_max_speed_diff(self) -> float:
        """Hard gate on absolute speed difference (rad/day). Constraint: ≥ 0."""

    # ----- Linking — scales -----

    @property
    def link_theta0(self) -> float:
        """Angular normalization scale for angular costs (radians). Constraint: > 0."""

    @property
    def link_v0(self) -> float:
        """Speed normalization scale for speed costs (rad/day). Constraint: > 0."""

    @property
    def link_flux_sigma_floor(self) -> float:
        """Sigma floor for flux-based cost (dimensionless). Constraint: ≥ 0."""

    @property
    def link_gap_rho(self) -> float:
        """Exponent ρ for Δ>1 gap penalty (dimensionless). Constraint: ≥ 0."""

    @property
    def link_vel_eps_days(self) -> float:
        """
        Finite-difference step at target time to estimate plane-velocity (days).

        Notes
        -----
        A small positive step is used to compute numerical derivatives for
        velocity-consistency terms.
        """

    # ----- Linking — limits & caps -----

    @property
    def link_top_k_per_left(self) -> int:
        """Max candidate edges kept per **left** node (per seed). Constraint: ≥ 1."""

    @property
    def link_max_total_edges(self) -> Optional[int]:
        """
        Global cap on edges after Top-K.

        Notes
        -----
        Use `None` to disable. When set, must be ≥ 1.
        """

    @property
    def link_max_cost(self) -> Optional[float]:
        """
        Hard cutoff on edge cost.

        Notes
        -----
        Use `None` to disable. When set, must be ≥ 0.
        """

    @property
    def link_max_speed_rad_per_day(self) -> Optional[float]:
        """
        Per-seed absolute speed cap (rad/day).

        Notes
        -----
        Use `None` to disable. When set, must be ≥ 0.
        """

    # ----- Utilities -----

    def validate(self) -> None:
        """
        Validate the entire parameter set.

        Raises
        ------
        ValueError
            If any parameter is out of bounds, ill-formed, or inconsistent.
        """

    def to_dict(self) -> FinkFatParamsDict:
        """
        Return a dictionary copy of the scalar fields.

        Returns
        -------
        FinkFatParamsDict
            Shallow copy of all scalar parameters. Mutating the returned
            dict does not affect the underlying Rust configuration.
        """

    def __repr__(self) -> str: ...  # pragma: no cover


class FinkFatParamsBuilder:
    """
    Fluent Python builder mirroring the Rust-side `FinkFatParamsBuilder`.

    All setters **mutate in place** and return the **same** builder instance
    (chainable). Call :meth:`build` to obtain a validated :class:`FinkFatParams`.

    Examples
    --------
    >>> cfg = (FinkFatParams.builder()
    ...        .show_progress(True)
    ...        .healpix_depth(8)
    ...        .pair_max_sep(0.002)
    ...        .link_k_sigma(3.0)
    ...        .build())
    """

    def __init__(self) -> None:
        """Create a builder pre-filled with Rust-side defaults."""

    # ----- Global -----

    def show_progress(self, v: bool) -> FinkFatParamsBuilder:
        """
        Enable or disable progress bars.

        Parameters
        ----------
        v : bool
            True to enable progress bars.

        Returns
        -------
        FinkFatParamsBuilder
            This builder (for chaining).
        """

    # ----- Binning -----

    def healpix_depth(self, v: int) -> FinkFatParamsBuilder:
        """
        Set HEALPix order (K).

        Parameters
        ----------
        v : int
            Depth K (NSIDE = 2**K). Typical range [6, 12] for LSST-like data.

        Returns
        -------
        FinkFatParamsBuilder
        """

    def time_bin_width_days(self, v: float) -> FinkFatParamsBuilder:
        """
        Set temporal bucket width.

        Parameters
        ----------
        v : float
            Width in **days (TT)**. Must be > 0.

        Returns
        -------
        FinkFatParamsBuilder
        """

    # ----- Pairs -----

    def pair_max_dt(self, v: float) -> FinkFatParamsBuilder:
        """
        Set maximum time separation for pairs.

        Parameters
        ----------
        v : float
            Days (TT). Constraint: ≥ 0.

        Returns
        -------
        FinkFatParamsBuilder
        """

    def pair_max_sep(self, v: float) -> FinkFatParamsBuilder:
        """
        Set maximum great-circle separation for pairs.

        Parameters
        ----------
        v : float
            Radians. Constraint: ≥ 0.

        Returns
        -------
        FinkFatParamsBuilder
        """

    def pair_max_flux_difference(self, v: float) -> FinkFatParamsBuilder:
        """
        Set maximum photometric difference for pairs.

        Parameters
        ----------
        v : float
            Dimensionless. Constraint: ≥ 0.

        Returns
        -------
        FinkFatParamsBuilder
        """

    def pair_allow_same_timebin(self, yes: bool) -> FinkFatParamsBuilder:
        """
        Allow/disallow pairing within the same time bin.

        Parameters
        ----------
        yes : bool
            If True, same-bin pairs are allowed.

        Returns
        -------
        FinkFatParamsBuilder
        """

    # ----- Triplets -----

    def triplet_max_dt_between(self, v: float) -> FinkFatParamsBuilder:
        """
        Set maximum Δt between neighboring detections for triplets.

        Parameters
        ----------
        v : float
            Days (TT). Constraint: ≥ 0.

        Returns
        -------
        FinkFatParamsBuilder
        """

    def triplet_max_pair_sep(self, v: float) -> FinkFatParamsBuilder:
        """
        Set maximum neighbor separation inside a triplet.

        Parameters
        ----------
        v : float
            Radians. Constraint: ≥ 0.

        Returns
        -------
        FinkFatParamsBuilder
        """

    def triplet_max_predicted_residual(self, v: float) -> FinkFatParamsBuilder:
        """
        Set max allowed predicted residual at `c` from the `a→b` segment.

        Parameters
        ----------
        v : float
            Radians. Constraint: ≥ 0.

        Returns
        -------
        FinkFatParamsBuilder
        """

    def triplet_enforce_time_order(self, yes: bool) -> FinkFatParamsBuilder:
        """
        Enforce strict temporal order t(a) < t(b) < t(c).

        Parameters
        ----------
        yes : bool
            If True, impose strict time ordering.

        Returns
        -------
        FinkFatParamsBuilder
        """

    def triplet_max_flux_difference(self, v: float) -> FinkFatParamsBuilder:
        """
        Set maximum photometric difference within a triplet.

        Parameters
        ----------
        v : float
            Dimensionless. Constraint: ≥ 0.

        Returns
        -------
        FinkFatParamsBuilder
        """

    # ----- Linking — predictor -----

    def link_k_sigma(self, v: float) -> FinkFatParamsBuilder:
        """
        Set cone inflation factor.

        Parameters
        ----------
        v : float
            Dimensionless multiplier (> 0) applied to the predicted covariance.

        Returns
        -------
        FinkFatParamsBuilder
        """

    def link_pad_cell_radius(self, yes: bool) -> FinkFatParamsBuilder:
        """
        Pad the prediction cone by the spatial cell radius.

        Parameters
        ----------
        yes : bool
            If True, expand the cone by the HEALPix cell radius.

        Returns
        -------
        FinkFatParamsBuilder
        """

    def link_noise_q0(self, v: float) -> FinkFatParamsBuilder:
        """
        Set additive model-noise variance floor.

        Parameters
        ----------
        v : float
            rad². Constraint: ≥ 0.

        Returns
        -------
        FinkFatParamsBuilder
        """

    def link_noise_q1(self, v: float) -> FinkFatParamsBuilder:
        """
        Set additive model-noise linear term.

        Parameters
        ----------
        v : float
            rad²/day. Constraint: ≥ 0.

        Returns
        -------
        FinkFatParamsBuilder
        """

    def link_noise_q2(self, v: float) -> FinkFatParamsBuilder:
        """
        Set additive model-noise quadratic term.

        Parameters
        ----------
        v : float
            rad²/day². Constraint: ≥ 0.

        Returns
        -------
        FinkFatParamsBuilder
        """

    # ----- Linking — weights -----

    def link_w_pos(self, v: float) -> FinkFatParamsBuilder:
        """
        Weight for positional Mahalanobis term.

        Parameters
        ----------
        v : float
            Dimensionless (≥ 0).

        Returns
        -------
        FinkFatParamsBuilder
        """

    def link_w_vel_dir(self, v: float) -> FinkFatParamsBuilder:
        """
        Weight for velocity **direction** consistency.

        Parameters
        ----------
        v : float
            Dimensionless (≥ 0).

        Returns
        -------
        FinkFatParamsBuilder
        """

    def link_w_vel_norm(self, v: float) -> FinkFatParamsBuilder:
        """
        Weight for velocity **norm** (speed) consistency.

        Parameters
        ----------
        v : float
            Dimensionless (≥ 0).

        Returns
        -------
        FinkFatParamsBuilder
        """

    def link_w_flux(self, v: float) -> FinkFatParamsBuilder:
        """
        Weight for photometric consistency.

        Parameters
        ----------
        v : float
            Dimensionless (≥ 0).

        Returns
        -------
        FinkFatParamsBuilder
        """

    def link_w_gap(self, v: float) -> FinkFatParamsBuilder:
        """
        Weight for revisit-gap penalty (Δ>1).

        Parameters
        ----------
        v : float
            Dimensionless (≥ 0).

        Returns
        -------
        FinkFatParamsBuilder
        """

    def link_w_band_mismatch(self, v: float) -> FinkFatParamsBuilder:
        """
        Weight for band mismatch penalty.

        Parameters
        ----------
        v : float
            Dimensionless (≥ 0).

        Returns
        -------
        FinkFatParamsBuilder
        """

    # ----- Linking — gates -----

    def link_max_d2_pos(self, v: float) -> FinkFatParamsBuilder:
        """
        Hard gate on positional Mahalanobis distance d².

        Parameters
        ----------
        v : float
            Dimensionless (≥ 0).

        Returns
        -------
        FinkFatParamsBuilder
        """

    def link_max_theta_vel(self, v: float) -> FinkFatParamsBuilder:
        """
        Hard gate on velocity direction mismatch.

        Parameters
        ----------
        v : float
            Radians (≥ 0).

        Returns
        -------
        FinkFatParamsBuilder
        """

    def link_max_speed_diff(self, v: float) -> FinkFatParamsBuilder:
        """
        Hard gate on absolute speed difference.

        Parameters
        ----------
        v : float
            rad/day (≥ 0).

        Returns
        -------
        FinkFatParamsBuilder
        """

    # ----- Linking — scales -----

    def link_theta0(self, v: float) -> FinkFatParamsBuilder:
        """
        Set angular normalization scale for angular costs.

        Parameters
        ----------
        v : float
            Radians (> 0).

        Returns
        -------
        FinkFatParamsBuilder
        """

    def link_v0(self, v: float) -> FinkFatParamsBuilder:
        """
        Set speed normalization scale for speed costs.

        Parameters
        ----------
        v : float
            rad/day (> 0).

        Returns
        -------
        FinkFatParamsBuilder
        """

    def link_flux_sigma_floor(self, v: float) -> FinkFatParamsBuilder:
        """
        Set sigma floor for flux-based cost.

        Parameters
        ----------
        v : float
            Dimensionless (≥ 0).

        Returns
        -------
        FinkFatParamsBuilder
        """

    def link_gap_rho(self, v: float) -> FinkFatParamsBuilder:
        """
        Set exponent ρ for Δ>1 gap penalty.

        Parameters
        ----------
        v : float
            Dimensionless (≥ 0).

        Returns
        -------
        FinkFatParamsBuilder
        """

    def link_vel_eps_days(self, v: float) -> FinkFatParamsBuilder:
        """
        Set finite-difference step at target time to estimate plane-velocity.

        Parameters
        ----------
        v : float
            Days (> 0).

        Returns
        -------
        FinkFatParamsBuilder
        """

    # ----- Linking — limits & caps -----

    def link_top_k_per_left(self, v: int) -> FinkFatParamsBuilder:
        """
        Keep at most K candidate edges per left node.

        Parameters
        ----------
        v : int
            K (≥ 1).

        Returns
        -------
        FinkFatParamsBuilder
        """

    def link_max_total_edges(self, v: Optional[int]) -> FinkFatParamsBuilder:
        """
        Set a global cap on edges after Top-K.

        Parameters
        ----------
        v : Optional[int]
            ≥ 1 to enable the cap; `None` to disable.

        Returns
        -------
        FinkFatParamsBuilder
        """

    def link_clear_max_total_edges(self) -> FinkFatParamsBuilder:
        """
        Disable the global edge cap (equivalent to `link_max_total_edges(None)`).

        Returns
        -------
        FinkFatParamsBuilder
        """

    def link_max_cost(self, v: Optional[float]) -> FinkFatParamsBuilder:
        """
        Set a hard cutoff on edge cost.

        Parameters
        ----------
        v : Optional[float]
            ≥ 0 to enable; `None` to disable.

        Returns
        -------
        FinkFatParamsBuilder
        """

    def link_clear_max_cost(self) -> FinkFatParamsBuilder:
        """
        Disable the hard cutoff on edge cost (equivalent to `link_max_cost(None)`).

        Returns
        -------
        FinkFatParamsBuilder
        """

    def link_max_speed_rad_per_day(self, v: Optional[float]) -> FinkFatParamsBuilder:
        """
        Set a per-seed absolute speed cap.

        Parameters
        ----------
        v : Optional[float]
            rad/day ≥ 0 to enable; `None` to disable.

        Returns
        -------
        FinkFatParamsBuilder
        """

    # ----- Build -----

    def build(self) -> FinkFatParams:
        """
        Validate and construct the parameter object.

        Returns
        -------
        FinkFatParams
            Validated parameter object.

        Raises
        ------
        ValueError
            If validation fails (invalid range or inconsistent thresholds).
        """
