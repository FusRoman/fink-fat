//! Per-branch orbit-fit quality tier, ported from the `quality_flags.py`
//! cascade of the sibling `lsst_cross_fink_fat_analysis` project.
//! [`QualityTier::WellSampledIdentified`]/[`QualityTier::Identified`] play a
//! similar role to that script's SBN-cross-match-dependent `completion`
//! tier (a numerically good fit that turns out to match a known object, not
//! a new find) but are driven by `fink-fat-explorer`'s own CND/Skybot
//! cross-match results rather than a static SBN cache, and — unlike the
//! Python `completion`, which merges both novelty levels into one tier —
//! keep the well-sampled-nights distinction even once matched. The Python
//! `contaminated` tier (matched to *more than one* distinct object) has no
//! equivalent here: this cascade only distinguishes has-a-match from
//! has-none.
//!
//! Two tiers have no counterpart in the Python version at all:
//! [`QualityTier::NotFitted`] and [`QualityTier::Ineligible`]. The Python
//! script ran once, exhaustively, over every branch, so "no `orbit_fits`
//! row" unambiguously meant "the Gauss IOD failed". `fink-fat-explorer`'s
//! bulk fit is instead run on demand and repeatedly, so a branch with no row
//! could mean it was never eligible, was eligible but never included in a
//! run, or was attempted and failed — three states that only became
//! distinguishable once failures started being recorded separately.
//!
//! Shared with `fink-fat`'s `submit` CLI (this crate has no dependency on
//! `fink-fat-explorer`'s Postgres client or web framework), together with
//! [`WELL_SAMPLED_NIGHTS_QUERY`] and [`CROSS_MATCH_LINEAGES_QUERY`] — the raw
//! SQL text both the explorer's async `sqlx` queries and the CLI's
//! synchronous `postgres` queries run, so "is this lineage eligible for MPC
//! submission" can never silently drift between the web UI's badge and the
//! CLI's actual gate.

use serde::{Deserialize, Serialize};

/// How a stored orbit fit was actually obtained — the `orbit_fits.fit_method`
/// column.
///
/// `outfit`'s `differential_correction` never reports a diverged correction
/// as an error: it falls back to the preliminary Gauss orbit and returns it
/// as a success. That fallback is what [`Self::IodOnly`] records, and it is
/// the difference between "this orbit was least-squares fitted" and "this
/// orbit is a preliminary estimate the correction could not improve on".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FitMethod {
    DifferentialCorrection,
    IodOnly,
}

impl FitMethod {
    /// Parses the stored `orbit_fits.fit_method` value. Anything unrecognised
    /// reads as [`Self::IodOnly`]: the column is free-form `TEXT`, so a
    /// defensive fallback beats failing a whole query over a value that can
    /// only be one of two strings.
    ///
    /// # Arguments
    /// * `s` — the raw `orbit_fits.fit_method` column value.
    ///
    /// # Return
    /// The parsed [`FitMethod`].
    pub fn from_column(s: &str) -> Self {
        match s {
            "differential_correction" => Self::DifferentialCorrection,
            _ => Self::IodOnly,
        }
    }

    /// The value written to `orbit_fits.fit_method` — the single definition
    /// of those two strings.
    pub fn as_column(self) -> &'static str {
        match self {
            Self::DifferentialCorrection => "differential_correction",
            Self::IodOnly => "iod_only",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::DifferentialCorrection => "Differential correction",
            Self::IodOnly => "Gauss IOD only",
        }
    }
}

/// The fields of a branch's latest `orbit_fits` row needed to place it on the
/// quality-tier cascade. `dof` (degrees of freedom) is derived here rather
/// than stored: `num_measurements - 6`.
pub struct LatestFit {
    pub fit_method: FitMethod,
    pub num_measurements: i32,
    pub fitted_at: chrono::DateTime<chrono::Utc>,
}

impl LatestFit {
    fn degrees_of_freedom(&self) -> i32 {
        self.num_measurements - 6
    }
}

/// A fit is only considered "constrained" from this many nights onward — the
/// point where two nights stop being enough to pin down both distance and
/// radial velocity.
pub const MIN_NIGHTS_FOR_CONSTRAINT: i64 = 3;

/// [`QualityTier::WellSampledDiscovery`]/[`QualityTier::WellSampledIdentified`]
/// additionally require this many nights with at least two observations
/// each: two same-night points pin down that night's angular rate directly,
/// a stronger geometric signal than the arc across nights alone.
pub const MIN_WELL_SAMPLED_NIGHTS: i64 = 5;

/// Submission-worthiness tier of one branch's latest orbit-fit attempt,
/// ordered best (submission-ready) to worst (structurally out of reach).
/// `Ord`/`PartialOrd` follow declaration order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum QualityTier {
    /// Converged differential correction, constrained (`dof > 0`, enough
    /// nights), at least [`MIN_WELL_SAMPLED_NIGHTS`] nights with two or more
    /// observations each, and no active cross-match hit (CND or Skybot) —
    /// the best-constrained, genuinely novel candidates. Eligible for MPC
    /// submission.
    WellSampledDiscovery,
    /// Same fit/constraint conditions as `WellSampledDiscovery`, but without
    /// the extra well-sampled-nights bar. Eligible for MPC submission.
    Discovery,
    /// Same conditions as [`Self::WellSampledDiscovery`], but the lineage
    /// has an active cross-match hit (CND or Skybot) — numerically as
    /// well-constrained, but not a new find: it matches a known object.
    WellSampledIdentified,
    /// Same conditions as [`Self::Discovery`], but the lineage has an
    /// active cross-match hit (CND or Skybot).
    Identified,
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
    /// Does not meet the bulk fit's eligibility criteria and so will never
    /// be submitted to it as things stand, regardless of run history.
    Ineligible,
}

impl QualityTier {
    /// Whether this tier is one of the two MPC-submission-eligible tiers
    /// (novel, sufficiently well-constrained). Used by `fink-fat submit`'s
    /// step-1 eligibility gate and by the explorer's "Prepare a submission"
    /// candidate list — the single place that decision is spelled out.
    pub fn is_submission_eligible(self) -> bool {
        matches!(self, Self::WellSampledDiscovery | Self::Discovery)
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::WellSampledDiscovery => "Well-sampled discovery",
            Self::Discovery => "Discovery",
            Self::WellSampledIdentified => "Well-sampled identified",
            Self::Identified => "Identified",
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
            Self::WellSampledDiscovery => "★",
            Self::Discovery => "◆",
            Self::WellSampledIdentified => "✦",
            Self::Identified => "◈",
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
            Self::WellSampledDiscovery => "badge-success",
            Self::Discovery => "badge-info",
            Self::WellSampledIdentified => "badge-accent",
            Self::Identified => "badge-secondary",
            Self::Unconstrained | Self::IodOnly => "badge-warning",
            Self::Failed => "badge-error",
            Self::NotFitted | Self::Ineligible => "badge-ghost",
        }
    }

    /// Every variant, in the same best-to-worst order as the type's `Ord`.
    pub const ALL: [QualityTier; 9] = [
        Self::WellSampledDiscovery,
        Self::Discovery,
        Self::WellSampledIdentified,
        Self::Identified,
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

/// Placed on the quality-tier cascade: `eligible` takes precedence over
/// everything else — an ineligible branch cannot have a meaningful
/// `latest_fit`/`latest_failure_at` in the first place. Between a fit and a
/// failure, whichever is more recent wins, so a branch that failed once and
/// later succeeded (or the reverse) reflects its current state rather than
/// its history. `has_cross_match` only affects the outcome when the branch
/// would otherwise land in [`QualityTier::WellSampledDiscovery`] or
/// [`QualityTier::Discovery`] — it is ignored by every other branch of the
/// cascade, so a lineage with an active cross-match hit but an
/// unconverged/unconstrained/failed fit still gets the same tier it would
/// without one.
///
/// # Arguments
///
/// * `eligible` — whether the branch meets the bulk fit's eligibility
///   criteria at all.
/// * `latest_fit` — the branch's most recent `orbit_fits` row, if any.
/// * `latest_failure_at` — the branch's most recent recorded fit failure, if
///   any.
/// * `n_nights` — total distinct nights the branch has observations on.
/// * `well_sampled_nights` — nights with two or more observations each.
/// * `has_cross_match` — whether the lineage has an active CND or Skybot
///   cross-match hit.
///
/// # Return
///
/// The resolved [`QualityTier`].
pub fn assign_quality_tier(
    eligible: bool,
    latest_fit: Option<&LatestFit>,
    latest_failure_at: Option<chrono::DateTime<chrono::Utc>>,
    n_nights: i64,
    well_sampled_nights: i64,
    has_cross_match: bool,
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

    let well_sampled = well_sampled_nights >= MIN_WELL_SAMPLED_NIGHTS;
    match (well_sampled, has_cross_match) {
        (true, false) => QualityTier::WellSampledDiscovery,
        (false, false) => QualityTier::Discovery,
        (true, true) => QualityTier::WellSampledIdentified,
        (false, true) => QualityTier::Identified,
    }
}

/// Minimum number of observations a branch needs to be eligible for an
/// orbit fit attempt at all — see [`ELIGIBLE_BRANCH_QUERY`]. Mirrors
/// `fink-fat-explorer::fit_pipeline::params::MIN_OBSERVATIONS`; kept as an
/// independent copy here (a plain numeric literal, not `pub(crate)`-scoped
/// in that module) rather than a cross-crate dependency, since
/// `fink-fat-explorer` is not something this crate can depend on. If that
/// constant ever changes, update this one to match.
pub const MIN_OBSERVATIONS: i64 = 3;

/// Minimum time baseline (days) across a branch's observations for it to be
/// fit-eligible — see [`ELIGIBLE_BRANCH_QUERY`]. Mirrors
/// `fink-fat-explorer::fit_pipeline::params::MIN_BASELINE_DAYS` (see
/// [`MIN_OBSERVATIONS`]'s doc for why this is a separate copy).
pub const MIN_BASELINE_DAYS: f64 = 0.25;

/// Every branch the bulk orbit fit would attempt right now: at least
/// [`MIN_OBSERVATIONS`] observations spanning at least [`MIN_BASELINE_DAYS`].
/// Bind `$1 = MIN_OBSERVATIONS`, `$2 = MIN_BASELINE_DAYS`. Mirrors
/// `fink-fat-explorer::fit_pipeline::params::ELIGIBLE_BRANCH_QUERY` (see
/// [`MIN_OBSERVATIONS`]'s doc for why this is a separate copy) — this is the
/// `eligible` input [`assign_quality_tier`] needs.
pub const ELIGIBLE_BRANCH_QUERY: &str = "
    SELECT bo.branch_id, b.lineage_designation
    FROM branch_observations bo
    JOIN branches b ON b.branch_id = bo.branch_id
    JOIN observations o ON o.id = bo.obs_id
    GROUP BY bo.branch_id, b.lineage_designation
    HAVING count(*) >= $1 AND (max(o.mjd_tt) - min(o.mjd_tt)) >= $2
";

/// A branch's latest `orbit_fits` row, just the columns [`assign_quality_tier`]
/// needs — the `latest_fit` input.
pub const LATEST_ORBIT_FIT_QUERY: &str = "
    SELECT DISTINCT ON (branch_id) branch_id, fit_method, num_measurements, fitted_at
    FROM orbit_fits
    WHERE branch_id IS NOT NULL
    ORDER BY branch_id, fitted_at DESC
";

/// A branch's latest recorded failed bulk-fit attempt, if any — the
/// `latest_failure_at` input.
pub const LATEST_ORBIT_FIT_FAILURE_QUERY: &str = "
    SELECT DISTINCT ON (branch_id) branch_id, attempted_at
    FROM orbit_fit_failures
    ORDER BY branch_id, attempted_at DESC
";

/// Number of distinct nights (`observations.night_id`) on which each branch
/// has two or more observations — the geometric bar
/// [`QualityTier::WellSampledDiscovery`] adds on top of [`QualityTier::Discovery`].
/// Shared verbatim by `fink-fat-explorer`'s homepage snapshot (async `sqlx`)
/// and `fink-fat submit`'s eligibility check (synchronous `postgres`).
pub const WELL_SAMPLED_NIGHTS_QUERY: &str = "
    SELECT branch_id, COUNT(*) AS well_sampled_nights
    FROM (
        SELECT bo.branch_id, o.night_id, COUNT(*) AS n
        FROM branch_observations bo
        JOIN observations o ON o.id = bo.obs_id
        GROUP BY bo.branch_id, o.night_id
    ) per_night
    WHERE n >= 2
    GROUP BY branch_id
";

/// Every `lineage_designation` with an active CND or Skybot cross-match hit
/// — Skybot counts any observation ever positively matched; CND only counts
/// the lineage's *latest* check attempt (an older positive hit superseded by
/// a clean re-check no longer counts), mirroring
/// `fink-fat-explorer::cross_match_status::build_cross_match_index`'s fold
/// semantics exactly. Shared so `fink-fat submit`'s eligibility check can
/// reproduce the same `has_cross_match` gate
/// [`assign_quality_tier`] needs without depending on the explorer's own
/// `sqlx`-based module.
pub const CROSS_MATCH_LINEAGES_QUERY: &str = "
    SELECT DISTINCT lineage_designation
    FROM (
        SELECT lineage_designation
        FROM skybot_obs_status
        WHERE jsonb_array_length(hits) > 0
        UNION ALL
        SELECT lineage_designation
        FROM (
            SELECT DISTINCT ON (lineage_designation) lineage_designation, hits
            FROM cnd_queries
            ORDER BY lineage_designation, queried_at DESC
        ) latest_cnd
        WHERE jsonb_array_length(hits) > 0
    ) matched
";

#[cfg(test)]
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
    fn fit_method_from_column_round_trips() {
        assert_eq!(
            FitMethod::from_column(FitMethod::DifferentialCorrection.as_column()),
            FitMethod::DifferentialCorrection
        );
        assert_eq!(
            FitMethod::from_column(FitMethod::IodOnly.as_column()),
            FitMethod::IodOnly
        );
    }

    #[test]
    fn fit_method_from_column_falls_back_to_iod_only_on_unrecognized_value() {
        assert_eq!(FitMethod::from_column("garbage"), FitMethod::IodOnly);
    }

    #[test]
    fn ineligible_overrides_everything() {
        let f = fit(FitMethod::DifferentialCorrection, 20, t(100));
        assert_eq!(
            assign_quality_tier(false, Some(&f), None, 10, 10, true),
            QualityTier::Ineligible
        );
    }

    #[test]
    fn not_fitted_when_nothing_recorded() {
        assert_eq!(
            assign_quality_tier(true, None, None, 0, 0, false),
            QualityTier::NotFitted
        );
    }

    #[test]
    fn failed_when_only_a_failure_is_recorded() {
        assert_eq!(
            assign_quality_tier(true, None, Some(t(50)), 0, 0, false),
            QualityTier::Failed
        );
    }

    #[test]
    fn failed_when_failure_is_more_recent_than_a_stale_success() {
        let f = fit(FitMethod::DifferentialCorrection, 20, t(100));
        assert_eq!(
            assign_quality_tier(true, Some(&f), Some(t(200)), 5, 5, false),
            QualityTier::Failed
        );
    }

    #[test]
    fn fit_wins_when_more_recent_than_a_stale_failure() {
        let f = fit(FitMethod::DifferentialCorrection, 20, t(200));
        assert_eq!(
            assign_quality_tier(true, Some(&f), Some(t(100)), 5, 5, false),
            QualityTier::WellSampledDiscovery
        );
    }

    #[test]
    fn iod_only_when_correction_never_converged() {
        let f = fit(FitMethod::IodOnly, 6, t(100));
        assert_eq!(
            assign_quality_tier(true, Some(&f), None, 5, 5, false),
            QualityTier::IodOnly
        );
    }

    #[test]
    fn unconstrained_on_zero_dof() {
        // 3 observations -> 6 measurements -> dof = 0.
        let f = fit(FitMethod::DifferentialCorrection, 6, t(100));
        assert_eq!(
            assign_quality_tier(true, Some(&f), None, 5, 5, false),
            QualityTier::Unconstrained
        );
    }

    #[test]
    fn unconstrained_on_too_few_nights_despite_positive_dof() {
        let f = fit(FitMethod::DifferentialCorrection, 20, t(100));
        assert_eq!(
            assign_quality_tier(true, Some(&f), None, 2, 2, false),
            QualityTier::Unconstrained
        );
    }

    #[test]
    fn discovery_below_the_well_sampled_nights_bar() {
        let f = fit(FitMethod::DifferentialCorrection, 20, t(100));
        assert_eq!(
            assign_quality_tier(true, Some(&f), None, 6, 4, false),
            QualityTier::Discovery
        );
    }

    #[test]
    fn well_sampled_discovery_at_the_well_sampled_nights_bar() {
        let f = fit(FitMethod::DifferentialCorrection, 20, t(100));
        assert_eq!(
            assign_quality_tier(true, Some(&f), None, 6, 5, false),
            QualityTier::WellSampledDiscovery
        );
    }

    #[test]
    fn identified_when_discovery_conditions_have_a_cross_match() {
        let f = fit(FitMethod::DifferentialCorrection, 20, t(100));
        assert_eq!(
            assign_quality_tier(true, Some(&f), None, 6, 4, true),
            QualityTier::Identified
        );
    }

    #[test]
    fn well_sampled_identified_when_well_sampled_discovery_conditions_have_a_cross_match() {
        let f = fit(FitMethod::DifferentialCorrection, 20, t(100));
        assert_eq!(
            assign_quality_tier(true, Some(&f), None, 6, 5, true),
            QualityTier::WellSampledIdentified
        );
    }

    #[test]
    fn cross_match_is_ignored_below_the_top_four_tiers() {
        // Unconstrained (zero dof), IOD-only, failed, not-fitted, and
        // ineligible must all stay unaffected by `has_cross_match=true` —
        // the split only matters once the fit is converged and constrained.
        let unconstrained_fit = fit(FitMethod::DifferentialCorrection, 6, t(100));
        assert_eq!(
            assign_quality_tier(true, Some(&unconstrained_fit), None, 5, 5, true),
            QualityTier::Unconstrained
        );

        let iod_fit = fit(FitMethod::IodOnly, 6, t(100));
        assert_eq!(
            assign_quality_tier(true, Some(&iod_fit), None, 5, 5, true),
            QualityTier::IodOnly
        );

        assert_eq!(
            assign_quality_tier(true, None, Some(t(50)), 0, 0, true),
            QualityTier::Failed
        );
        assert_eq!(
            assign_quality_tier(true, None, None, 0, 0, true),
            QualityTier::NotFitted
        );
        assert_eq!(
            assign_quality_tier(false, None, None, 0, 0, true),
            QualityTier::Ineligible
        );
    }

    #[test]
    fn ordering_is_best_to_worst() {
        assert!(QualityTier::WellSampledDiscovery < QualityTier::Discovery);
        assert!(QualityTier::Discovery < QualityTier::WellSampledIdentified);
        assert!(QualityTier::WellSampledIdentified < QualityTier::Identified);
        assert!(QualityTier::Identified < QualityTier::Unconstrained);
        assert!(QualityTier::Unconstrained < QualityTier::IodOnly);
        assert!(QualityTier::IodOnly < QualityTier::Failed);
        assert!(QualityTier::Failed < QualityTier::NotFitted);
        assert!(QualityTier::NotFitted < QualityTier::Ineligible);
    }

    #[test]
    fn is_submission_eligible_is_true_only_for_the_top_two_tiers() {
        assert!(QualityTier::WellSampledDiscovery.is_submission_eligible());
        assert!(QualityTier::Discovery.is_submission_eligible());
        for tier in QualityTier::ALL {
            if !matches!(
                tier,
                QualityTier::WellSampledDiscovery | QualityTier::Discovery
            ) {
                assert!(
                    !tier.is_submission_eligible(),
                    "{tier:?} should not be eligible"
                );
            }
        }
    }
}
