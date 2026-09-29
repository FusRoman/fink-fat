//! The homepage's (a, e) plot's glossary: a "?" icon whose popover explains
//! how to read the plot and what its colors/markers mean.
//!
//! Uses the shared [`crate::help_popover::HelpPopover`] shell for
//! the hover/pin/outside-click/Escape mechanic. Also home to
//! [`FAMILY_GROUPS`]/[`FamilyReferenceList`] and
//! [`TIER_ENTRIES`]/[`TierReferenceList`] — the [`DynamicalFamily`]/
//! [`QualityTier`] reference lists shown here, and reused as-is by
//! [`crate::homepage::orbit3d_population_glossary`] and
//! [`crate::homepage::lineage_table_glossary`], since all three views share
//! the same family/tier vocabulary.

use dioxus::prelude::*;

use crate::help_popover::HelpPopover;
use crate::homepage::family::DynamicalFamily;
use crate::homepage::quality_tier::QualityTier;

/// A short description of the plot, shown above the glossary.
pub(super) const PLOT_DESCRIPTION: &str = "Every point is one lineage's best-fit orbit: x is its \
    semi-major axis (AU, log scale), y its eccentricity. Color marks its dynamical family, and the \
    marker's shape/opacity marks its orbit-fit quality tier — see the legends below the view \
    switch to toggle either on or off. Hover a point for its exact values, and click a point to \
    open that lineage's page.";

/// Core plot-reading concepts, shown before the family/tier reference lists.
pub(super) const CONCEPTS: &[(&str, &str)] = &[
    (
        "a — semi-major axis",
        "Half the orbital ellipse's long axis — roughly the object's average distance from the \
         Sun. The plot's x axis, in AU, on a log scale so the inner and outer solar system are \
         both readable at once.",
    ),
    (
        "e — eccentricity",
        "How elongated the orbit is: 0 is a circle, close to 1 is a very stretched ellipse. The \
         plot's y axis.",
    ),
    (
        "AU — astronomical unit",
        "Mean Earth-Sun distance, 149,597,870.7 km. Every axis of the plot is in AU.",
    ),
    (
        "Dynamical family",
        "Classified from (a, e) alone, following IMCCE's SkyBoT boundaries \
         (https://ssp.imcce.fr/webservices/skybot/). Drives the point's color — see the family \
         reference below.",
    ),
    (
        "Quality tier",
        "How reliable the underlying orbit fit is, from a well-constrained differential correction \
         down to a Gauss-IOD-only estimate or a failed fit. Drives the point's marker shape and \
         opacity — see the tier reference below.",
    ),
    (
        "Click a point",
        "Opens the page for that point's lineage — its full observation history, fit diagnostics \
         and orbit.",
    ),
];

/// One dynamical-family reference entry: the variant (for its color swatch
/// and label) plus a plain-language description of what it physically means.
type FamilyEntry = (DynamicalFamily, &'static str);

/// [`DynamicalFamily`] reference, grouped by region for scannability instead
/// of one flat list of 22. Descriptions are deliberately non-technical —
/// [`DynamicalFamily::classify`] in `homepage::family` is the source of
/// truth for the exact (a, e) boundaries.
pub(super) const FAMILY_GROUPS: &[(&str, &[FamilyEntry])] = &[
    (
        "Near-Earth objects",
        &[
            (
                DynamicalFamily::NeaAtira,
                "Entirely inside Earth's orbit — never gets close enough to cross it.",
            ),
            (
                DynamicalFamily::NeaAten,
                "Crosses Earth's orbit, spending most of its time inside it.",
            ),
            (
                DynamicalFamily::NeaApollo,
                "Crosses Earth's orbit, with its closest approach (perihelion) inside it.",
            ),
            (
                DynamicalFamily::NeaAmor,
                "Approaches Earth's orbit from outside without crossing it.",
            ),
        ],
    ),
    (
        "Mars-crossers & Hungaria",
        &[
            (
                DynamicalFamily::MarsCrosserDeep,
                "Crosses Mars' orbit well inside it.",
            ),
            (
                DynamicalFamily::MarsCrosserShallow,
                "Crosses Mars' orbit, but only slightly.",
            ),
            (
                DynamicalFamily::Hungaria,
                "Tight group just inside the main belt, beyond Mars' orbit.",
            ),
        ],
    ),
    (
        "Main Belt",
        &[
            (
                DynamicalFamily::MbInner,
                "Main belt, inner sub-zone (between Mars and the belt's middle).",
            ),
            (
                DynamicalFamily::MbMiddle,
                "Main belt, middle sub-zone.",
            ),
            (
                DynamicalFamily::MbOuter,
                "Main belt, outer sub-zone (closer to Jupiter).",
            ),
            (
                DynamicalFamily::MbCybele,
                "Outer main-belt group just inside Jupiter's 2:1 orbital resonance.",
            ),
            (
                DynamicalFamily::MbHilda,
                "Group locked in Jupiter's 3:2 orbital resonance, just inside its orbit.",
            ),
        ],
    ),
    (
        "Trojan & Centaur",
        &[
            (
                DynamicalFamily::Trojan,
                "Shares Jupiter's orbit, clustered at its leading/trailing Lagrange points (L4/L5).",
            ),
            (
                DynamicalFamily::Centaur,
                "Between Jupiter and Neptune — an unstable, transitional population between the \
                 outer belt and the Kuiper belt.",
            ),
        ],
    ),
    (
        "Kuiper Belt objects",
        &[
            (
                DynamicalFamily::KboSdo,
                "Scattered disc object: beyond Neptune, but with a perihelion still close enough \
                 to be gravitationally perturbed by it.",
            ),
            (
                DynamicalFamily::KboDetached,
                "Beyond Neptune, eccentric, but too far for Neptune to still perturb it — decoupled \
                 from its influence.",
            ),
            (
                DynamicalFamily::KboClassicalInner,
                "Beyond Neptune on a low-eccentricity \"classical\" orbit, inner sub-zone.",
            ),
            (
                DynamicalFamily::KboClassicalMain,
                "Beyond Neptune on a low-eccentricity \"classical\" orbit, main sub-zone.",
            ),
            (
                DynamicalFamily::KboClassicalOuter,
                "Beyond Neptune on a low-eccentricity \"classical\" orbit, outer sub-zone.",
            ),
        ],
    ),
    (
        "Other",
        &[
            (
                DynamicalFamily::Unknown,
                "Semi-major axis too small to classify (below 0.08 AU) — most likely a fit \
                 artifact rather than a real orbit.",
            ),
            (
                DynamicalFamily::Vulcanoid,
                "Hypothetical population orbiting very close to the Sun, inside Mercury's orbit.",
            ),
            (
                DynamicalFamily::Ioc,
                "Very distant orbit (semi-major axis beyond roughly 2000 AU) — the Inner Oort \
                 Cloud region.",
            ),
        ],
    ),
];

/// [`QualityTier`] reference, condensed from the tier's own doc comments in
/// `fink_fat_ades::quality_tier`. Ordered best to worst, same as
/// [`QualityTier::ALL`].
pub(super) const TIER_ENTRIES: &[(QualityTier, &str)] = &[
    (
        QualityTier::WellSampledDiscovery,
        "Well-constrained fit (converged, enough nights, at least 5 nights with 2+ observations \
         each) and no known cross-match — the strongest, genuinely novel candidates.",
    ),
    (
        QualityTier::Discovery,
        "Well-constrained fit and no known cross-match, without the extra well-sampled-nights bar \
         of the tier above.",
    ),
    (
        QualityTier::WellSampledIdentified,
        "Same fit quality as well-sampled discovery, but matches a known object (an active CND or \
         Skybot cross-match hit).",
    ),
    (
        QualityTier::Identified,
        "Same fit quality as discovery, but matches a known object.",
    ),
    (
        QualityTier::Unconstrained,
        "The fit converged but isn't well-constrained — too few nights, or an exact \
         3-observation interpolation.",
    ),
    (
        QualityTier::IodOnly,
        "The least-squares correction never converged; only the preliminary Gauss IOD orbit is \
         available.",
    ),
    (
        QualityTier::Failed,
        "The most recent bulk-fit attempt for this branch failed outright.",
    ),
    (
        QualityTier::NotFitted,
        "Eligible for a bulk fit, but hasn't been attempted yet.",
    ),
    (
        QualityTier::Ineligible,
        "Doesn't currently meet the bulk fit's eligibility criteria.",
    ),
];

/// A "?" icon that shows how to read the (a, e) plot on hover, and keeps it
/// open (and scrollable) once clicked. See [`HelpPopover`] for the
/// interaction mechanic.
#[component]
pub fn PopulationPlotGlossary() -> Element {
    rsx! {
        HelpPopover { aria_label: "Help for the (a, e) plot",
            div { class: "mb-3",
                div { class: "font-semibold mb-1", "About this plot" }
                p { class: "opacity-80", "{PLOT_DESCRIPTION}" }
            }
            div { class: "font-semibold mb-2", "Core concepts" }
            dl { style: "columns: 2 16rem; column-gap: 1.5rem;",
                for (term , definition) in CONCEPTS {
                    div { style: "break-inside: avoid; margin-bottom: 0.75rem;",
                        dt { class: "font-semibold", "{term}" }
                        dd { class: "opacity-80", "{definition}" }
                    }
                }
            }
            div { class: "font-semibold mb-2 mt-3", "Dynamical families" }
            FamilyReferenceList {}
            div { class: "font-semibold mb-2 mt-3", "Quality tiers" }
            TierReferenceList {}
        }
    }
}

/// [`FAMILY_GROUPS`], rendered as one color-swatched `dl` per group. Shared
/// by every homepage glossary that mentions dynamical family (the (a, e)
/// plot, the 3D view, the lineage table).
#[component]
pub(super) fn FamilyReferenceList() -> Element {
    rsx! {
        for (group_name , entries) in FAMILY_GROUPS {
            div { class: "mb-2",
                div { class: "opacity-60 font-semibold mb-1", "{group_name}" }
                dl { style: "columns: 2 16rem; column-gap: 1.5rem;",
                    for (family , definition) in *entries {
                        div { style: "break-inside: avoid; margin-bottom: 0.5rem; display: flex; gap: 0.375rem;",
                            span {
                                style: "display: inline-block; width: 0.6rem; height: 0.6rem; border-radius: 9999px; \
                                        margin-top: 0.2rem; flex-shrink: 0; background-color: {family.color()};",
                            }
                            div {
                                dt { class: "font-semibold", "{family.label()}" }
                                dd { class: "opacity-80", "{definition}" }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// [`TIER_ENTRIES`], rendered as a glyph-prefixed `dl`. Shared by every
/// homepage glossary that mentions quality tier (the (a, e) plot, the
/// lineage table).
#[component]
pub(super) fn TierReferenceList() -> Element {
    rsx! {
        dl { style: "columns: 2 16rem; column-gap: 1.5rem;",
            for (tier , definition) in TIER_ENTRIES {
                div { style: "break-inside: avoid; margin-bottom: 0.5rem;",
                    dt { class: "font-semibold", "{tier.glyph()} {tier.label()}" }
                    dd { class: "opacity-80", "{definition}" }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::homepage::family::ORDERED_LABELS;

    fn flat_family_entries() -> Vec<FamilyEntry> {
        FAMILY_GROUPS
            .iter()
            .flat_map(|(_, entries)| entries.iter().copied())
            .collect()
    }

    /// Every [`DynamicalFamily`] variant is documented exactly once —
    /// walking `ORDERED_LABELS` (the enum's own exhaustive label list)
    /// rather than a second hardcoded array, so a future new variant makes
    /// this test fail instead of silently missing from the popover.
    #[test]
    fn every_dynamical_family_variant_is_documented_exactly_once() {
        let documented = flat_family_entries();
        assert_eq!(documented.len(), ORDERED_LABELS.len());
        for label in ORDERED_LABELS {
            let family = DynamicalFamily::from_label(label);
            let count = documented.iter().filter(|(f, _)| *f == family).count();
            assert_eq!(count, 1, "{label} should have exactly one glossary entry");
        }
    }

    /// Same idea for [`QualityTier`], walking its own [`QualityTier::ALL`].
    #[test]
    fn every_quality_tier_variant_is_documented_exactly_once() {
        assert_eq!(TIER_ENTRIES.len(), QualityTier::ALL.len());
        for tier in QualityTier::ALL {
            let count = TIER_ENTRIES.iter().filter(|(t, _)| *t == tier).count();
            assert_eq!(count, 1, "{tier} should have exactly one glossary entry");
        }
    }

    #[test]
    fn no_entry_is_empty() {
        for (term, definition) in CONCEPTS {
            assert!(!term.trim().is_empty() && !definition.trim().is_empty());
        }
        for (family, definition) in flat_family_entries() {
            assert!(!family.label().trim().is_empty() && !definition.trim().is_empty());
        }
        for (tier, definition) in TIER_ENTRIES {
            assert!(!tier.label().trim().is_empty() && !definition.trim().is_empty());
        }
    }

    /// The click-to-navigate behavior must be documented, since it isn't
    /// otherwise discoverable from the plot itself.
    #[test]
    fn the_description_mentions_clicking_a_point() {
        assert!(PLOT_DESCRIPTION.to_lowercase().contains("click"));
        assert!(CONCEPTS.iter().any(|(term, _)| term.contains("Click")));
    }
}
