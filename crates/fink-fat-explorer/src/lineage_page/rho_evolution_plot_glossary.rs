//! The lineage page's "ρ / ρ̇ evolution" tab's glossary: a "?" icon whose
//! popover explains topocentric range/range-rate and this tab's plots.
//!
//! Uses the shared [`crate::help_popover::HelpPopover`] shell.

use dioxus::prelude::*;

use super::x_axis::X_AXIS_UNIT_CONCEPT;

/// A short description of the tab, shown above the glossary.
pub(super) const PLOT_DESCRIPTION: &str = "Evolution of the range and range-rate to this branch \
    as real observations are absorbed. The top two plots track the single (MAP) filter's own \
    estimate; the bottom two show *every* hypothesis still alive in the bank at each step, to \
    see the range ambiguity narrow down over time rather than only its eventual winner.";

/// Core plot-reading concepts, shown before the glossary. ρ/ρ̇ are the last
/// two components of the attributable-coordinates state vector
/// (α, δ, α̇, δ̇, ρ, ρ̇) — see `fink-fat-engine`'s `KFState` doc comment
/// (Milani et al. 2007).
pub(super) const CONCEPTS: &[(&str, &str)] = &[
    (
        "ρ — topocentric range",
        "Straight-line distance from the observer to the object, in AU. Unlike sky position \
         (RA/Dec), range can't be measured directly from a single observation — it's inferred by \
         the filter from how the object's apparent motion curves over time.",
    ),
    (
        "ρ̇ — topocentric range-rate",
        "How fast that distance is changing, in AU/day.",
    ),
    (
        "Posterior (top two plots)",
        "The single (MAP) filter's own ρ/ρ̇ estimate after absorbing each observation, with a 1σ \
         error bar from its posterior covariance.",
    ),
    (
        "All hypotheses (bottom two plots)",
        "A semi-transparent point per surviving hypothesis in the bank, at its *pre-update* \
         (predicted, not yet fitted) ρ/ρ̇ for that step. Watch the cloud narrow as more \
         observations rule out incompatible ranges.",
    ),
    X_AXIS_UNIT_CONCEPT,
];

/// A "?" icon that shows how to read this tab's plots on hover, and keeps
/// it open (and scrollable) once clicked. See
/// [`crate::help_popover::HelpPopover`] for the interaction mechanic.
#[component]
pub fn RhoEvolutionPlotGlossary() -> Element {
    rsx! {
        crate::help_popover::HelpPopover { aria_label: "Help for rho / rho-dot evolution",
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
}
