//! The lineage page's trajectory plot's glossary: a "?" icon whose popover
//! explains the sky-plane view and its traces.
//!
//! Uses the shared [`crate::help_popover::HelpPopover`] shell.

use dioxus::prelude::*;

use super::x_axis::X_AXIS_UNIT_CONCEPT;

/// A short description of the plot, shown above the glossary.
pub(super) const PLOT_DESCRIPTION: &str = "Sky-plane view (right ascension vs. declination) of \
    this branch's real observations and the Kalman filter's predictions for them. Marker size \
    grows with time, standing in for a direction arrow. Skybot/MPC cross-match results, once \
    searched for, are overlaid as extra markers.";

/// Core plot-reading concepts, shown before the glossary.
pub(super) const CONCEPTS: &[(&str, &str)] = &[
    (
        "Observations",
        "The branch's real observed positions (blue), each with its 1σ astrometric error bar in \
         RA and Dec.",
    ),
    (
        "Kalman prediction (pre-update)",
        "The filter's predicted sky position (red-orange) at each observation, computed \
         *before* that observation was absorbed — so it's a genuine prediction, not a fit to the \
         point it's next to. Error bars come from the propagated sky covariance at that same \
         pre-update instant.",
    ),
    (
        "Skybot matches",
        "Diamond markers, one stable color per matched object, shown once a Skybot cone search \
         has been run from the controls above the plot (see the results panel, ☰).",
    ),
    (
        "MPC near-duplicates",
        "Open-circle markers: existing MPC observations close in time and position to this \
         branch's own, shown once a near-duplicate search has been run from the controls above.",
    ),
    X_AXIS_UNIT_CONCEPT,
];

/// A "?" icon that shows how to read the trajectory plot on hover, and keeps
/// it open (and scrollable) once clicked. See
/// [`crate::help_popover::HelpPopover`] for the interaction mechanic.
#[component]
pub fn TrajectoryPlotGlossary() -> Element {
    rsx! {
        crate::help_popover::HelpPopover { aria_label: "Help for the trajectory plot",
            div { class: "mb-3",
                div { class: "font-semibold mb-1", "About this plot" }
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
