//! The lineage page's light curve plot's glossary: a "?" icon whose popover
//! explains the magnitude-vs-time view.
//!
//! Uses the shared [`crate::help_popover::HelpPopover`] shell.

use dioxus::prelude::*;

use super::x_axis::X_AXIS_UNIT_CONCEPT;

/// A short description of the plot, shown above the glossary.
pub(super) const PLOT_DESCRIPTION: &str = "Magnitude vs. time for the branch's real \
    observations, one trace per photometric band (the standard LSST ugrizy palette). The y axis \
    is flipped so brighter (smaller magnitude) sits at the top, matching how brightness is \
    normally read.";

/// Core plot-reading concepts, shown before the glossary.
pub(super) const CONCEPTS: &[(&str, &str)] = &[
    (
        "Magnitude",
        "Logarithmic brightness — lower is brighter. Each point has a 1σ error bar from the \
         observation's own magnitude error.",
    ),
    (
        "Band",
        "Photometric filter the observation was taken through (LSST's u, g, r, i, z, y, \
         shortest to longest wavelength), color-coded per the legend.",
    ),
    X_AXIS_UNIT_CONCEPT,
];

/// A "?" icon that shows how to read the light curve on hover, and keeps it
/// open (and scrollable) once clicked. See
/// [`crate::help_popover::HelpPopover`] for the interaction mechanic.
#[component]
pub fn LightCurvePlotGlossary() -> Element {
    rsx! {
        crate::help_popover::HelpPopover { aria_label: "Help for the light curve",
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
