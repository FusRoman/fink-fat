//! Homepage-specific rendering for [`QualityTier`]: the plot marker mapping.
//! The tier enum itself, its `label`/`glyph`/`badge_class`, and the
//! [`assign_quality_tier`] cascade now live in the shared
//! [`fink_fat_ades::quality_tier`] crate (re-exported below under this
//! module's original path, so every existing `crate::homepage::quality_tier`
//! call site needed no changes) — shared with the `fink-fat submit` CLI's
//! eligibility gate so the two can never disagree on what "eligible for MPC
//! submission" means. See that crate's docs for the full cascade rationale.

pub use fink_fat_ades::quality_tier::{
    assign_quality_tier, FitMethod, LatestFit, QualityTier, MIN_NIGHTS_FOR_CONSTRAINT,
    MIN_WELL_SAMPLED_NIGHTS,
};

#[cfg(target_arch = "wasm32")]
use crate::homepage::family::DynamicalFamily;
#[cfg(target_arch = "wasm32")]
use plotly::common::{DashType, Line, Marker, MarkerSymbol};

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
        QualityTier::WellSampledDiscovery => base.symbol(MarkerSymbol::Star).opacity(1.0),
        QualityTier::Discovery => base.symbol(MarkerSymbol::Diamond).opacity(1.0),
        // Hollow variant of the matching unmatched tier's shape: a matched
        // lineage visually pairs with its unmatched sibling (same shape)
        // while staying distinguishable (open vs. filled).
        QualityTier::WellSampledIdentified => base.symbol(MarkerSymbol::StarOpen).opacity(1.0),
        QualityTier::Identified => base.symbol(MarkerSymbol::DiamondOpen).opacity(1.0),
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
