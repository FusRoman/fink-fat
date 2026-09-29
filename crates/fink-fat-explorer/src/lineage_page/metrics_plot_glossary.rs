//! The lineage page's "Filter consistency metrics" tab's glossary: a "?"
//! icon whose popover explains its three sub-plots.
//!
//! Uses the shared [`crate::help_popover::HelpPopover`] shell.

use dioxus::prelude::*;

use super::x_axis::X_AXIS_UNIT_CONCEPT;

/// A short description of the tab, shown above the glossary.
pub(super) const PLOT_DESCRIPTION: &str = "How consistently the single (MAP) Kalman filter's \
    predictions matched each real observation as they were absorbed, one point per observation: \
    a statistical goodness-of-fit (χ²), a physical distance (angular separation), and a running \
    evidence score (log-likelihood).";

/// Core plot-reading concepts, shown before the glossary. The χ² gate value
/// is asserted against `metrics_plot::CHI2_GATE_95` by a test, so this text
/// can never silently drift from the real constant.
pub(super) const CONCEPTS: &[(&str, &str)] = &[
    (
        "χ² (NIS)",
        "Normalized Innovation Squared: how far the observation fell from the filter's \
         prediction, in units of the predicted uncertainty (2 degrees of freedom, since a sky \
         position has 2 components). Low and roughly constant is a healthy fit.",
    ),
    (
        "Covariance-inflation threshold (dashed line)",
        "Set at 5.991 (the 95% χ² value for 2 d.o.f.). When the smoothed χ² crosses it, the \
         single filter inflates its own covariance to stay consistent with the data — this is \
         *not* the multi-hypothesis bank's discard gate (a separate, much looser threshold used \
         only on the Hypothesis bank tab).",
    ),
    (
        "Separation",
        "Raw angular distance (arcsec) between the observed and predicted sky position at each \
         step — the same mismatch χ² measures, but in a physical unit rather than a \
         statistical one.",
    ),
    (
        "Cumulative log-likelihood",
        "Running sum of the single-hypothesis Gaussian log-likelihood of each observation's \
         innovation. A proxy for the production Cumulative LLR score (which additionally scores \
         against a clutter background and photometry) — useful for spotting which observations \
         drove the fit, not a reproduction of the stored value.",
    ),
    X_AXIS_UNIT_CONCEPT,
];

/// A "?" icon that shows how to read this tab's plots on hover, and keeps
/// it open (and scrollable) once clicked. See
/// [`crate::help_popover::HelpPopover`] for the interaction mechanic.
#[component]
pub fn MetricsPlotGlossary() -> Element {
    rsx! {
        crate::help_popover::HelpPopover { aria_label: "Help for the filter consistency metrics",
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
    use super::super::metrics_plot::CHI2_GATE_95;
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

    /// The quoted gate value must match the real constant plotted on the
    /// chart, not an independently-typed copy that could drift.
    #[test]
    fn the_quoted_threshold_matches_the_real_constant() {
        assert_eq!(CHI2_GATE_95, 5.991);
        let (_, definition) = CONCEPTS
            .iter()
            .find(|(term, _)| term.contains("inflation"))
            .expect("an inflation-threshold entry");
        assert!(definition.contains("5.991"));
    }
}
