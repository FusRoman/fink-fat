//! The homepage's 3D orbit view's glossary: a "?" icon whose popover
//! explains what the view shows.
//!
//! Uses the shared [`crate::help_popover::HelpPopover`] shell, and
//! [`crate::homepage::population_plot_glossary::FamilyReferenceList`] for
//! the dynamical-family reference — this view uses the same family colors
//! as the (a, e) plot, so the reference list is identical; it has no
//! quality-tier encoding of its own, so no tier reference is shown here.

use dioxus::prelude::*;

use crate::help_popover::HelpPopover;
use crate::homepage::population_plot_glossary::FamilyReferenceList;

/// A short description of the view, shown above the glossary.
///
/// Deliberately says this view has no per-point hover/click-through yet —
/// unlike the (a, e) plot, [`crate::orbit3d::plot3d::Scatter3dPlot`] builds
/// tracked-object traces with no `hover_text`/`custom_data`, so a point only
/// shows plotly's bare default hover (trace name and x/y/z).
pub(super) const PLOT_DESCRIPTION: &str = "Current heliocentric positions — at the moment the \
    page's data was loaded — of every tracked lineage's best-fit orbit, one point per lineage, \
    colored by dynamical family (the same colors as the (a, e) plot, but without its \
    quality-tier marker distinction). The Sun sits at the origin; the planets and a few tracked \
    perturbers are drawn both as their current position and their full orbital ellipse, for \
    scale. Unlike the (a, e) plot, this view has no per-point click-through to the lineage page \
    yet.";

/// Core view-reading concepts, shown before the family reference list.
pub(super) const CONCEPTS: &[(&str, &str)] = &[
    (
        "Current position",
        "Where each object is at the moment the page's data was loaded — not at the date of its \
         last observation. Advanced from its fitted orbit with two-body (Keplerian) motion.",
    ),
    (
        "AU — astronomical unit",
        "Mean Earth-Sun distance, 149,597,870.7 km. Every axis of the view is in AU.",
    ),
    (
        "Dynamical family",
        "Classified from (a, e) alone, following IMCCE's SkyBoT boundaries \
         (https://ssp.imcce.fr/webservices/skybot/). Drives each point's color — see the \
         reference below.",
    ),
    (
        "Planets & tracked perturbers",
        "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto, Ceres, Pallas and \
         Vesta — each drawn as a full orbital ellipse plus its current position, for scale.",
    ),
];

/// A "?" icon that shows how to read the 3D view on hover, and keeps it open
/// (and scrollable) once clicked. See [`HelpPopover`] for the interaction
/// mechanic.
#[component]
pub fn Orbit3DPopulationGlossary() -> Element {
    rsx! {
        HelpPopover { aria_label: "Help for the 3D orbit view",
            div { class: "mb-3",
                div { class: "font-semibold mb-1", "About this view" }
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

    /// This view genuinely has no click-through yet — the description must
    /// say so honestly rather than imply the (a, e) plot's behavior applies
    /// here too.
    #[test]
    fn the_description_says_there_is_no_click_through_yet() {
        assert!(PLOT_DESCRIPTION
            .to_lowercase()
            .contains("no per-point click-through"));
    }
}
