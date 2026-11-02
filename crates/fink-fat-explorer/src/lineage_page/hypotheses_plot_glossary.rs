//! The lineage page's "Hypothesis bank" tab's glossary: a "?" icon whose
//! popover explains what a bank hypothesis is and this tab's plots.
//!
//! Uses the shared [`crate::help_popover::HelpPopover`] shell.
//!
//! The default numeric values quoted in [`CONCEPTS`] (`gate_chi2 = 23.0`,
//! `search_region_chi2 = 400.0`, `min_hypotheses = 5`) come from
//! `fink-fat-engine`'s `KFBankConfig::default()`
//! (`crates/fink-fat-engine/src/engine_config/kf_bank_config.rs`) — they
//! aren't standalone `pub const`s importable from this crate, so a test
//! guards them by literal comparison instead of a cross-crate import; update
//! both together if the engine's defaults ever change.

use dioxus::prelude::*;

use super::x_axis::X_AXIS_UNIT_CONCEPT;

/// A short description of the tab, shown above the glossary.
pub(super) const PLOT_DESCRIPTION: &str = "Bank-level diagnostics of the multi-hypothesis \
    replay: how many range/range-rate hypotheses survive at each real observation, and how \
    large the region the tracker searched for the next one was. Both should shrink as the \
    range/range-rate ambiguity collapses. \"Hypothesis\" here means a single candidate state in \
    the range-finding Kalman bank, not the seeding stage's multi-night track hypothesis of the \
    same name.";

/// Core plot-reading concepts, shown before the glossary.
pub(super) const CONCEPTS: &[(&str, &str)] = &[
    (
        "Hypothesis (this tab)",
        "One weighted (ρ, ρ̇) candidate carried by the Kalman bank — not the same concept as a \
         seeding-stage track hypothesis (an ordered multi-night sequence of seeds) elsewhere in \
         this app, despite sharing the name.",
    ),
    (
        "Live hypotheses",
        "How many hypotheses are still in the bank after each step. The bank keeps at least 5 \
         (its configured floor) except right at the very end.",
    ),
    (
        "Effective sample size",
        "1 / Σwᵢ², from each hypothesis's normalized weight wᵢ. Close to 1 means one hypothesis \
         dominates the bank; close to the live-hypothesis count means the weights are still \
         near-uniform (the bank hasn't converged on a winner yet).",
    ),
    (
        "Search region radius",
        "Conservative bounding radius (arcsec, log scale) of the pre-update search region — the \
         \"error box\" the production candidate search would have queried before absorbing that \
         observation. Deliberately sized much looser (χ² = 400, ~20σ) than the bank's own \
         per-hypothesis update gate (χ² = 23, ~99.999%), so the search stays generous even while \
         the update itself stays strict.",
    ),
    X_AXIS_UNIT_CONCEPT,
];

/// A "?" icon that shows how to read this tab's plots on hover, and keeps
/// it open (and scrollable) once clicked. See
/// [`crate::help_popover::HelpPopover`] for the interaction mechanic.
#[component]
pub fn HypothesesPlotGlossary() -> Element {
    rsx! {
        crate::help_popover::HelpPopover { aria_label: "Help for the hypothesis bank",
            div { class: "mb-3",
                div { class: "font-semibold mb-1", "About this tab" }
                p { class: "opacity-80", "{PLOT_DESCRIPTION}" }
            }
            div { class: "font-semibold mb-2", "Concepts" }
            dl { style: "columns: 2 16rem; column-gap: 1.5rem;",
                for (term , definition) in CONCEPTS {
                    div { style: "break-inside: avoid; margin-bottom: 0.75rem;",
                        dt { class: "font-semibold", "{term}" }
                        dd { class: "opacity-80", "{definition}" }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_entry_is_empty() {
        for (term, definition) in CONCEPTS {
            assert!(!term.trim().is_empty() && !definition.trim().is_empty());
        }
    }

    #[test]
    fn no_duplicate_terms() {
        let mut terms: Vec<&str> = CONCEPTS.iter().map(|(t, _)| *t).collect();
        terms.sort_unstable();
        terms.dedup();
        assert_eq!(terms.len(), CONCEPTS.len(), "duplicate glossary term");
    }

    /// The hypothesis-vs-TrackHypothesis disambiguation is the entire point
    /// of this glossary existing — it must not silently regress.
    #[test]
    fn the_description_disambiguates_from_track_hypothesis() {
        assert!(PLOT_DESCRIPTION.contains("seeding stage"));
    }

    /// Numeric claims must match `fink-fat-engine`'s `KFBankConfig::default()`
    /// (see module doc for the exact source).
    #[test]
    fn the_quoted_bank_defaults_are_correct() {
        let search_region = CONCEPTS
            .iter()
            .find(|(term, _)| term.contains("Search region"))
            .expect("a search-region entry");
        assert!(search_region.1.contains("400"));
        assert!(search_region.1.contains("23"));

        let live = CONCEPTS
            .iter()
            .find(|(term, _)| term.contains("Live hypotheses"))
            .expect("a live-hypotheses entry");
        assert!(live.1.contains('5'));
    }
}
