//! Per-branch orbit-fit quality tier, ported from the `quality_flags.py`
//! cascade of the sibling `lsst_cross_fink_fat_analysis` project — minus the
//! tiers that depended on its SBN cross-match (`completion`/`contaminated`),
//! which has no equivalent in `fink-fat-explorer`.
//!
//! Two tiers have no counterpart in the Python version at all:
//! [`QualityTier::NotFitted`] and [`QualityTier::Ineligible`]. The Python
//! script ran once, exhaustively, over every branch, so "no orbit_fits row"
//! unambiguously meant "the Gauss IOD failed". The Rust bulk fit
//! (`crate::bulk_orbit_fit::run`) is instead run on demand and repeatedly, so
//! a branch with no row could mean it was never eligible, was eligible but
//! never included in a run, or was attempted and failed — three states that
//! only became distinguishable once failures started being recorded in
//! `orbit_fit_failures` (see `src/converter/sql.rs`).

use serde::{Deserialize, Serialize};

#[cfg(target_arch = "wasm32")]
use crate::homepage::family::DynamicalFamily;
#[cfg(target_arch = "wasm32")]
use plotly::common::{DashType, Line, Marker, MarkerSymbol};

#[cfg(feature = "server")]
pub use crate::fit_pipeline::fit::FitMethod;

/// The fields of a branch's latest `orbit_fits` row needed to place it on the
/// quality-tier cascade. `dof` (degrees of freedom) is derived here rather
/// than stored, exactly as `orbit_fit::latest::get_latest_orbit_fit_result`
/// already does for the single-lineage fit page: `num_measurements - 6`.
///
/// Server-only: it carries a `chrono::DateTime`, and `chrono` is only pulled
/// in by the `server` feature (see this crate's `Cargo.toml`) — the wasm
/// client never builds this type, only the plain [`QualityTier`] it resolves
/// to.
#[cfg(feature = "server")]
pub struct LatestFit {
    pub fit_method: FitMethod,
    pub num_measurements: i32,
    pub fitted_at: chrono::DateTime<chrono::Utc>,
}

#[cfg(feature = "server")]
impl LatestFit {
    fn degrees_of_freedom(&self) -> i32 {
        self.num_measurements - 6
    }
}

/// A fit is only considered "constrained" from this many nights onward — the
/// point where two nights stop being enough to pin down both distance and
/// radial velocity (investigation `docs/investigation_fit_orbital.md`,
/// section 3, in the sibling `lsst_cross_fink_fat_analysis` project).
pub const MIN_NIGHTS_FOR_CONSTRAINT: i64 = 3;

/// [`QualityTier::PrimeDiscovery`] additionally requires this many nights
/// with at least two observations each: two same-night points pin down that
/// night's angular rate directly, a stronger geometric signal than the arc
/// across nights alone.
pub const MIN_WELL_SAMPLED_NIGHTS: i64 = 5;

/// Submission-worthiness tier of one branch's latest orbit-fit attempt,
/// ordered best (submission-ready) to worst (structurally out of reach).
/// `Ord`/`PartialOrd` follow declaration order, which is what the "Quality"
/// column sort and the plot's trace grouping both rely on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum QualityTier {
    /// Converged differential correction, constrained (`dof > 0`, enough
    /// nights), and at least [`MIN_WELL_SAMPLED_NIGHTS`] nights with two or
    /// more observations each — the best-constrained candidates.
    PrimeDiscovery,
    /// Same fit/constraint conditions as `PrimeDiscovery`, but without the
    /// extra well-sampled-nights bar.
    Discovery,
    /// Converged differential correction, but not constrained: either an
    /// exact 3-observation interpolation (`dof <= 0`) or fewer than
    /// [`MIN_NIGHTS_FOR_CONSTRAINT`] nights.
    Unconstrained,
    /// The differential correction never converged; only the preliminary
    /// Gauss IOD orbit is available.
    IodOnly,
    /// The branch's most recent bulk-fit event was a failure (no orbit at
    /// all), more recent than any successful fit it might also have.
    Failed,
    /// Eligible for a bulk fit right now, but neither a fit nor a failure has
    /// been recorded for it yet — the state of every branch before its first
    /// bulk fit run.
    NotFitted,
    /// Does not meet the bulk fit's eligibility criteria
    /// (`orbit_fit::{MIN_OBSERVATIONS, MIN_BASELINE_DAYS}`) and so will never
    /// be submitted to it as things stand, regardless of run history.
    Ineligible,
}

/// Placed on the quality-tier cascade: mirrors `quality_flags.py`'s
/// `assign_quality_tier`, minus the SBN-linking-dependent tiers it also
/// assigned (`completion`/`contaminated`), which have no equivalent input
/// here.
///
/// `eligible` takes precedence over everything else — an ineligible branch
/// cannot have a meaningful `latest_fit`/`latest_failure_at` in the first
/// place. Between a fit and a failure, whichever is more recent wins, so a
/// branch that failed once and later succeeded (or the reverse) reflects its
/// current state rather than its history.
#[cfg(feature = "server")]
pub fn assign_quality_tier(
    eligible: bool,
    latest_fit: Option<&LatestFit>,
    latest_failure_at: Option<chrono::DateTime<chrono::Utc>>,
    n_nights: i64,
    well_sampled_nights: i64,
) -> QualityTier {
    if !eligible {
        return QualityTier::Ineligible;
    }

    let failure_is_latest = match (latest_fit, latest_failure_at) {
        (Some(fit), Some(failed_at)) => failed_at > fit.fitted_at,
        (None, Some(_)) => true,
        (_, None) => false,
    };
    if failure_is_latest {
        return QualityTier::Failed;
    }

    let Some(fit) = latest_fit else {
        return QualityTier::NotFitted;
    };

    if fit.fit_method == FitMethod::IodOnly {
        return QualityTier::IodOnly;
    }

    let constrained = fit.degrees_of_freedom() > 0 && n_nights >= MIN_NIGHTS_FOR_CONSTRAINT;
    if !constrained {
        return QualityTier::Unconstrained;
    }

    if well_sampled_nights >= MIN_WELL_SAMPLED_NIGHTS {
        QualityTier::PrimeDiscovery
    } else {
        QualityTier::Discovery
    }
}

impl QualityTier {
    pub fn label(self) -> &'static str {
        match self {
            Self::PrimeDiscovery => "Prime discovery",
            Self::Discovery => "Discovery",
            Self::Unconstrained => "Unconstrained",
            Self::IodOnly => "IOD only",
            Self::Failed => "Failed",
            Self::NotFitted => "Not fitted",
            Self::Ineligible => "Ineligible",
        }
    }

    /// A one-glyph stand-in for the tier's plot marker, used identically in
    /// the table badge and the plot's tier legend so the two stay visually
    /// linked. Not the actual SVG path plotly draws — just a readable hint.
    pub fn glyph(self) -> &'static str {
        match self {
            Self::PrimeDiscovery => "★",
            Self::Discovery => "◆",
            Self::Unconstrained => "▲",
            Self::IodOnly => "■",
            Self::Failed => "✕",
            Self::NotFitted => "●",
            Self::Ineligible => "○",
        }
    }

    /// daisyUI badge color class, roughly tracking severity.
    pub fn badge_class(self) -> &'static str {
        match self {
            Self::PrimeDiscovery => "badge-success",
            Self::Discovery => "badge-info",
            Self::Unconstrained | Self::IodOnly => "badge-warning",
            Self::Failed => "badge-error",
            Self::NotFitted | Self::Ineligible => "badge-ghost",
        }
    }

    /// Every variant, in the same best-to-worst order as the type's `Ord` —
    /// used to build the plot's tier legend and to enumerate `SortColumn`'s
    /// quality-tier permutation.
    pub const ALL: [QualityTier; 7] = [
        Self::PrimeDiscovery,
        Self::Discovery,
        Self::Unconstrained,
        Self::IodOnly,
        Self::Failed,
        Self::NotFitted,
        Self::Ineligible,
    ];
}

impl std::fmt::Display for QualityTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Builds the marker for one (family, tier) plot trace: color always comes
/// from the family, shape/opacity/border from the tier. Kept as a pure
/// function, separate from the trace-building loop in `dynamic_pop_plot`, so
/// the (family, tier) -> `Marker` mapping has one definition. Only compiled
/// for wasm since `plotly::common::Marker` is a client-side type; there is no
/// non-wasm test target for it.
#[cfg(target_arch = "wasm32")]
pub fn marker_for(family: DynamicalFamily, tier: QualityTier) -> Marker {
    let base = Marker::new().color(family.color());

    match tier {
        // No border: a dark outline on a small star marker reads as a solid
        // black shape and hides the family color underneath it. The star
        // shape (vs. `Discovery`'s diamond) is already enough to set this
        // tier apart.
        QualityTier::PrimeDiscovery => base.symbol(MarkerSymbol::Star).opacity(1.0),
        QualityTier::Discovery => base.symbol(MarkerSymbol::Diamond).opacity(1.0),
        QualityTier::Unconstrained => base.symbol(MarkerSymbol::TriangleUp).opacity(1.0),
        QualityTier::IodOnly => base.symbol(MarkerSymbol::Square).opacity(1.0),
        // Unchanged from the plot's pre-quality-tier look: a plain circle at
        // full opacity, so a branch that has simply never been bulk-fitted
        // yet renders exactly as it always has.
        QualityTier::NotFitted => base.symbol(MarkerSymbol::Circle).opacity(1.0),
        QualityTier::Ineligible => base.symbol(MarkerSymbol::Circle).opacity(0.55).line(
            Line::new()
                .color(family.color())
                .width(1.0)
                .dash(DashType::Dash),
        ),
        QualityTier::Failed => base.symbol(MarkerSymbol::X).opacity(0.15),
    }
}

#[cfg(all(test, feature = "server"))]
mod tests {
    use super::*;

    fn fit(
        method: FitMethod,
        num_measurements: i32,
        fitted_at: chrono::DateTime<chrono::Utc>,
    ) -> LatestFit {
        LatestFit {
            fit_method: method,
            num_measurements,
            fitted_at,
        }
    }

    fn t(seconds: i64) -> chrono::DateTime<chrono::Utc> {
        chrono::DateTime::from_timestamp(seconds, 0).unwrap()
    }

    #[test]
    fn ineligible_overrides_everything() {
        let f = fit(FitMethod::DifferentialCorrection, 20, t(100));
        assert_eq!(
            assign_quality_tier(false, Some(&f), None, 10, 10),
            QualityTier::Ineligible
        );
    }

    #[test]
    fn not_fitted_when_nothing_recorded() {
        assert_eq!(
            assign_quality_tier(true, None, None, 0, 0),
            QualityTier::NotFitted
        );
    }

    #[test]
    fn failed_when_only_a_failure_is_recorded() {
        assert_eq!(
            assign_quality_tier(true, None, Some(t(50)), 0, 0),
            QualityTier::Failed
        );
    }

    #[test]
    fn failed_when_failure_is_more_recent_than_a_stale_success() {
        let f = fit(FitMethod::DifferentialCorrection, 20, t(100));
        assert_eq!(
            assign_quality_tier(true, Some(&f), Some(t(200)), 5, 5),
            QualityTier::Failed
        );
    }

    #[test]
    fn fit_wins_when_more_recent_than_a_stale_failure() {
        let f = fit(FitMethod::DifferentialCorrection, 20, t(200));
        assert_eq!(
            assign_quality_tier(true, Some(&f), Some(t(100)), 5, 5),
            QualityTier::PrimeDiscovery
        );
    }

    #[test]
    fn iod_only_when_correction_never_converged() {
        let f = fit(FitMethod::IodOnly, 6, t(100));
        assert_eq!(
            assign_quality_tier(true, Some(&f), None, 5, 5),
            QualityTier::IodOnly
        );
    }

    #[test]
    fn unconstrained_on_zero_dof() {
        // 3 observations -> 6 measurements -> dof = 0.
        let f = fit(FitMethod::DifferentialCorrection, 6, t(100));
        assert_eq!(
            assign_quality_tier(true, Some(&f), None, 5, 5),
            QualityTier::Unconstrained
        );
    }

    #[test]
    fn unconstrained_on_too_few_nights_despite_positive_dof() {
        let f = fit(FitMethod::DifferentialCorrection, 20, t(100));
        assert_eq!(
            assign_quality_tier(true, Some(&f), None, 2, 2),
            QualityTier::Unconstrained
        );
    }

    #[test]
    fn discovery_below_the_well_sampled_nights_bar() {
        let f = fit(FitMethod::DifferentialCorrection, 20, t(100));
        assert_eq!(
            assign_quality_tier(true, Some(&f), None, 6, 4),
            QualityTier::Discovery
        );
    }

    #[test]
    fn prime_discovery_at_the_well_sampled_nights_bar() {
        let f = fit(FitMethod::DifferentialCorrection, 20, t(100));
        assert_eq!(
            assign_quality_tier(true, Some(&f), None, 6, 5),
            QualityTier::PrimeDiscovery
        );
    }

    #[test]
    fn ordering_is_best_to_worst() {
        assert!(QualityTier::PrimeDiscovery < QualityTier::Discovery);
        assert!(QualityTier::Discovery < QualityTier::Unconstrained);
        assert!(QualityTier::Unconstrained < QualityTier::IodOnly);
        assert!(QualityTier::IodOnly < QualityTier::Failed);
        assert!(QualityTier::Failed < QualityTier::NotFitted);
        assert!(QualityTier::NotFitted < QualityTier::Ineligible);
    }
}
