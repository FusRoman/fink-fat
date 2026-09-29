//! The lineage page's observations table's glossary: a "?" icon whose
//! popover explains each column.
//!
//! Uses the shared [`crate::help_popover::HelpPopover`] shell. Column
//! meanings are drawn from [`fink_fat_ades::model::ObservationRow`]'s own
//! doc comments — the canonical, DB-client-agnostic definition shared with
//! the `fink-fat submit` CLI's ADES export, not a paraphrase.

use dioxus::prelude::*;

/// A short description of the table, shown above the glossary.
pub(super) const PLOT_DESCRIPTION: &str = "Every real observation used by this lineage's best \
    branch, in track order (its position column). Rows are pre-sorted; there is no column sort \
    or filter here.";

/// Column reference, in table order.
pub(super) const COLUMNS: &[(&str, &str)] = &[
    (
        "#",
        "Position of this observation within the branch's track order.",
    ),
    (
        "ObjectId",
        "The alert's persistent object identifier at its survey (ZTF's `objectId`; for LSST this \
         equals the observation's own `diaSourceId`). Links out to the Fink portal page for that \
         object when the observatory code is recognized.",
    ),
    (
        "MJD (TT) / ISO (UTC)",
        "Toggles the epoch display between Modified Julian Date, Terrestrial Time (the \
         pipeline's native timescale) and a calendar date/time in UTC.",
    ),
    (
        "RA (deg) / Dec (deg)",
        "Right ascension / declination, each with its own 1σ astrometric error in arcsec.",
    ),
    (
        "Magnitude",
        "Photometric magnitude and its 1σ error — already-resolved photometry, not the raw \
         alert-packet fields (e.g. ZTF's magpsf/sigmapsf): fink-fat stores only this resolved \
         value.",
    ),
    (
        "Filter",
        "fink-fat's internal photometric band index (an LSST-sourced 0-5 → u,g,r,i,z,y \
         convention), shown here as the raw number rather than translated to a letter.",
    ),
    (
        "Observatory",
        "Raw MPC observatory code the observation was submitted under (e.g. I41 for ZTF/Palomar, \
         X05 for LSST/Rubin) — shown as-is, not resolved to a survey name in this table.",
    ),
];

/// A "?" icon that shows how to read the observations table on hover, and
/// keeps it open (and scrollable) once clicked. See
/// [`crate::help_popover::HelpPopover`] for the interaction mechanic.
#[component]
pub fn ObservationsTableGlossary() -> Element {
    rsx! {
        crate::help_popover::HelpPopover { aria_label: "Help for the observations table",
            div { class: "mb-3",
                div { class: "font-semibold mb-1", "About this table" }
                p { class: "opacity-80", "{PLOT_DESCRIPTION}" }
            }
            div { class: "font-semibold mb-2", "Columns" }
            dl { style: "columns: 2 16rem; column-gap: 1.5rem;",
                for (term , definition) in COLUMNS {
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
        for (term, definition) in COLUMNS {
            assert!(!term.trim().is_empty() && !definition.trim().is_empty());
        }
    }

    #[test]
    fn no_duplicate_terms() {
        let mut terms: Vec<&str> = COLUMNS.iter().map(|(t, _)| *t).collect();
        terms.sort_unstable();
        terms.dedup();
        assert_eq!(terms.len(), COLUMNS.len(), "duplicate glossary term");
    }

    /// This table stores already-resolved photometry, not raw alert-packet
    /// field names — the glossary must say so rather than imply otherwise.
    #[test]
    fn the_magnitude_entry_disclaims_raw_alert_fields() {
        let (_, definition) = COLUMNS
            .iter()
            .find(|(term, _)| *term == "Magnitude")
            .expect("a Magnitude entry");
        assert!(definition.contains("magpsf"));
    }
}
