//! The homepage's lineage table's glossary: a "?" icon whose popover
//! explains the columns and how to interact with the table.
//!
//! Uses the shared [`crate::help_popover::HelpPopover`] shell, and
//! reuses [`crate::homepage::population_plot_glossary::FamilyReferenceList`]/
//! [`crate::homepage::population_plot_glossary::TierReferenceList`] for the
//! Family/Quality columns' reference lists, since they carry the exact same
//! meaning here as on the (a, e) plot.

use dioxus::prelude::*;

use crate::help_popover::HelpPopover;
use crate::homepage::population_plot_glossary::{FamilyReferenceList, TierReferenceList};

/// A short description of the table, shown above the glossary.
pub(super) const TABLE_DESCRIPTION: &str = "One row per lineage — its best branch's own summary. \
    Click a sortable column header to sort by it, click again to flip direction. Click the \
    Lineage link to open that lineage's page; click elsewhere on a row to expand it and see the \
    lineage's other branches, if it has any. The search box above matches a lineage's \
    designation, case-insensitively, anywhere in the string. The family/tier legend above the \
    view switch filters this table the same way it filters the (a, e) plot.";

/// Column reference, in table order.
pub(super) const COLUMNS: &[(&str, &str)] = &[
    (
        "Designation",
        "The best branch's own designation. Not sortable.",
    ),
    (
        "Lineage",
        "The lineage's designation — click it to open that lineage's page. Not sortable.",
    ),
    (
        "Family",
        "Dynamical family badge — see the reference below.",
    ),
    (
        "Cumulative LLR",
        "Running log-likelihood-ratio score accumulated across the branch's updates — the \
         tracker's internal measure of how well the observations fit a single moving object. \
         Higher is stronger evidence; used to rank and prune candidate tracks.",
    ),
    (
        "Updates",
        "How many real observations the branch has actually absorbed (a missed/skipped update \
         doesn't count) — the evidence volume behind its Cumulative LLR.",
    ),
    (
        "Arc (days)",
        "Time span between the branch's first and last observation.",
    ),
    (
        "Nights",
        "Number of distinct observation nights the branch spans.",
    ),
    (
        "Median Δt (days)",
        "Median gap between consecutive observation nights. Shown as \"—\" for a branch with only \
         one night, which has no such gap to measure.",
    ),
    (
        "Quality",
        "Orbit-fit reliability tier badge — see the reference below.",
    ),
];

/// A "?" icon that shows how to read the lineage table on hover, and keeps
/// it open (and scrollable) once clicked. See [`HelpPopover`] for the
/// interaction mechanic.
#[component]
pub fn LineageTableGlossary() -> Element {
    rsx! {
        HelpPopover { aria_label: "Help for the lineage table",
            div { class: "mb-3",
                div { class: "font-semibold mb-1", "About this table" }
                p { class: "opacity-80", "{TABLE_DESCRIPTION}" }
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
            div { class: "font-semibold mb-2 mt-3", "Dynamical families" }
            FamilyReferenceList {}
            div { class: "font-semibold mb-2 mt-3", "Quality tiers" }
            TierReferenceList {}
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

    /// Both row-interaction behaviors must be documented, since neither is
    /// otherwise discoverable from the table itself.
    #[test]
    fn the_description_mentions_the_link_and_the_expand_behavior() {
        assert!(TABLE_DESCRIPTION.contains("Lineage link"));
        assert!(TABLE_DESCRIPTION.to_lowercase().contains("expand"));
    }

    /// The description must not claim the "Active"/"Archived" dropdown
    /// filters anything — it currently isn't wired to any signal or query.
    #[test]
    fn the_description_does_not_claim_the_status_filter_works() {
        assert!(!TABLE_DESCRIPTION.to_lowercase().contains("archived"));
        assert!(!TABLE_DESCRIPTION.to_lowercase().contains("active"));
    }
}
