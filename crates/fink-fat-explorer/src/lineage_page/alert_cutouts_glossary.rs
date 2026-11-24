//! The lineage page's alert-images carousel's glossary: a "?" icon whose
//! popover explains the cutout stamps and how to browse them.
//!
//! Uses the shared [`crate::help_popover::HelpPopover`] shell. Cutout-kind
//! terminology matches this codebase's own `CutoutKind` (Science/Template/
//! Difference — the same three-stamp convention documented by the Fink
//! broker for both surveys this app tracks, ZTF at doc.ztf.fink-broker.org
//! and LSST at doc.lsst.fink-broker.org) and `Survey` (`crate::survey`).

use dioxus::prelude::*;

/// A short description of the carousel, shown above the glossary.
pub(super) const PLOT_DESCRIPTION: &str = "One alert at a time, its three cutout stamps side by \
    side. Previous/Next steps through the branch's real observations — the three images always \
    move together. Which survey (ZTF or LSST) an observation came from is inferred from its MPC \
    observatory code.";

/// Core concepts, shown before the glossary.
pub(super) const CONCEPTS: &[(&str, &str)] = &[
    (
        "Science",
        "The new image, taken at the alert's own epoch — what the sky actually looked like at \
         that observation.",
    ),
    (
        "Template",
        "The reference image the survey compares against — a stack of earlier, deeper images of \
         the same field with no transient present.",
    ),
    (
        "Difference",
        "Science minus Template: what changed. This is the image the survey's detection pipeline \
         actually triggers on — a moving or transient source stands out even against a crowded \
         field.",
    ),
    (
        "Compare slider",
        "Click a Science or Template stamp (once loaded) to open a full-screen slider: drag to \
         reveal more of one image or the other, to spot what moved between them.",
    ),
    (
        "MJD (TT) / ISO (UTC)",
        "Toggles how each stamp's observation epoch is displayed: Modified Julian Date, \
         Terrestrial Time (the pipeline's native timescale) or a calendar date/time in UTC.",
    ),
    (
        "SNR",
        "Signal-to-noise ratio of the detection. Labelled \"(fetched)\" when read live from the \
         survey's own source catalog (LSST only), or \"(Pogson)\" when approximated from the \
         magnitude error (1.0857 / mag_err) because no catalog value is available.",
    ),
];

/// A "?" icon that shows how to read the alert images on hover, and keeps
/// it open (and scrollable) once clicked. See
/// [`crate::help_popover::HelpPopover`] for the interaction mechanic.
#[component]
pub fn AlertCutoutsGlossary() -> Element {
    rsx! {
        crate::help_popover::HelpPopover { aria_label: "Help for the alert images",
            div { class: "mb-3",
                div { class: "font-semibold mb-1", "About this carousel" }
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

    /// The three cutout-kind entries must match `CutoutKind::label()`
    /// exactly, not a paraphrase that could drift from the real UI labels.
    #[test]
    fn the_cutout_kinds_match_the_real_labels() {
        for label in ["Science", "Template", "Difference"] {
            assert!(CONCEPTS.iter().any(|(term, _)| *term == label), "{label}");
        }
    }
}
