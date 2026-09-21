//! The lineage 3D plot's glossary: a "?" icon whose popover explains the
//! acronyms and concepts the plot's legend, hovers and switches use (MOID,
//! LD, AU, phase angle, ...).
//!
//! The popover opens on hover and can be *pinned* open by clicking the icon:
//! pinned, it stays put while the pointer leaves and can be scrolled, and a
//! click anywhere else (or Escape) closes it. It is driven by Dioxus state
//! rather than a CSS-only tooltip (daisyUI's `tooltip`), which closes as
//! soon as the pointer crosses the gap between the icon and the bubble — so
//! its content could neither be reached nor scrolled — and whose size limits
//! are inline styles here so they do not depend on Tailwind having generated
//! a utility class.

use dioxus::prelude::*;

/// The glossary, in reading order: units and frames first, then the orbit's
/// own quantities, then what the plot draws, then how to read the
/// uncertainty.
///
/// Numbers quoted here (AU and LD in km) are kept in step with
/// `orbit3d::geometry::KM_PER_AU` / `KM_PER_LUNAR_DISTANCE` by a test.
pub(super) const GLOSSARY: &[(&str, &str)] = &[
    (
        "AU — astronomical unit",
        "Mean Earth–Sun distance, 149,597,870.7 km. Every axis of the plot is in AU.",
    ),
    (
        "LD — lunar distance",
        "Mean Earth–Moon distance, 384,400 km (about 0.00257 AU). Used for very small distances, \
         such as a close MOID.",
    ),
    (
        "Ecliptic",
        "The plane of the Earth's orbit around the Sun. The plot uses the heliocentric ecliptic \
         J2000 frame: the Sun at the origin, the ecliptic as the z = 0 plane.",
    ),
    (
        "Inclination (i)",
        "Angle between the object's orbital plane and the ecliptic — the violet slice at the Sun. \
         0° is an orbit in the ecliptic, 90° a polar one.",
    ),
    (
        "Ascending / descending node",
        "The two points where the orbit crosses the ecliptic, going north (ascending) or south \
         (descending). The dashed line joining them is the line of nodes, which passes through \
         the Sun.",
    ),
    (
        "a, e, P",
        "Semi-major axis (half the ellipse's long axis), eccentricity (0 = circle, close to 1 = \
         very elongated) and orbital period.",
    ),
    (
        "Perihelion (q) / aphelion (Q)",
        "The orbit's closest and farthest points from the Sun: q = a(1 − e), Q = a(1 + e).",
    ),
    (
        "Osculating orbit",
        "The Kepler ellipse that matches the object's position and velocity at one instant. It \
         ignores the planets' perturbations, so it drifts away from the true path over time.",
    ),
    (
        "MOID — minimum orbit intersection distance",
        "The smallest distance between two orbits taken as fixed ellipses, wherever the bodies \
         actually are. A small MOID with a planet means the orbits pass close to each other, not \
         that the two bodies will meet: that also depends on timing. Computed from the osculating \
         orbits, so it is indicative.",
    ),
    (
        "Heliocentric / topocentric distance",
        "Distance from the Sun / from the observer's position at the time of the observation.",
    ),
    (
        "Observation crosses",
        "A sky position (ra/dec) gives only a direction. Each cross sits on the measured line of \
         sight, at the distance where that line passes closest to the position the fitted orbit \
         predicts for that date — so its offset from the orbit shows the fit's residual.",
    ),
    (
        "Sight lines",
        "Dashed lines Sun → observation → observer, one pair per observation.",
    ),
    (
        "Phase angle",
        "Angle at the object between the directions to the Sun and to the observer. 0° means the \
         Sun is behind the observer (fully lit face); large values mean a crescent-like view.",
    ),
    (
        "Solar elongation",
        "Angle at the observer between the Sun and the object. Small values mean the object was \
         close to the Sun on the sky, hard to observe.",
    ),
    (
        "H — absolute magnitude",
        "The brightness the object would have 1 AU from both the Sun and the observer at zero \
         phase angle. Estimated per observation with the H,G phase function (G = 0.15), in the \
         observation's own band, with no colour correction: indicative only.",
    ),
    (
        "Uncertainty clones",
        "Many orbits drawn at random from the N-body fit's covariance — each one consistent with \
         the observations. Their scatter shows where the object could be: tight at the last \
         observation, spread out at \"now\". Only shown for a converged fit with a usable \
         covariance.",
    ),
    (
        "Deviations exaggerated ×N",
        "Clones are usually too close to the best orbit to see at the plot's scale, so each \
         clone's deviation from the best orbit is multiplied by N. ×1 is the true scale.",
    ),
    (
        "Outer solar system",
        "Jupiter and beyond (semi-major axis above 4 AU). Hidden by default for an object whose \
         semi-major axis is below Jupiter's, to keep the plot's scale on the inner solar system.",
    ),
];

/// Width of the glossary popover: two columns of definitions, but never
/// wider than the window.
const POPOVER_WIDTH: &str = "min(40rem, calc(100vw - 2rem))";
/// Height limit of the glossary popover: most of the window, so the content
/// scrolls inside the popover instead of running off the screen.
const POPOVER_MAX_HEIGHT: &str = "min(75vh, 36rem)";

/// A "?" icon that shows [`GLOSSARY`] on hover, and keeps it open (and
/// scrollable) once clicked.
///
/// The popover is glued to the icon with a small transparent padding, not a
/// gap, so the pointer can travel from the icon into the popover without
/// the hover being lost.
#[component]
pub fn PlotGlossary() -> Element {
    let mut hovered = use_signal(|| false);
    let mut pinned = use_signal(|| false);
    let open = hovered() || pinned();
    let icon_opacity = if open { "1" } else { "0.6" };

    rsx! {
        div {
            style: "position: relative; display: inline-block;",
            onmouseenter: move |_| hovered.set(true),
            onmouseleave: move |_| hovered.set(false),
            onkeydown: move |evt| {
                if evt.key() == Key::Escape {
                    pinned.set(false);
                    hovered.set(false);
                }
            },
            // While pinned, a click anywhere outside the popover closes it.
            if pinned() {
                div {
                    style: "position: fixed; inset: 0; z-index: 40;",
                    onclick: move |_| {
                        pinned.set(false);
                        hovered.set(false);
                    },
                }
            }
            button {
                r#type: "button",
                class: "inline-flex items-center justify-center w-5 h-5 rounded-full border border-current text-xs leading-none cursor-help",
                style: "position: relative; z-index: 50; opacity: {icon_opacity};",
                aria_label: "Glossary of the terms used in this plot",
                aria_expanded: "{open}",
                onclick: move |_| pinned.toggle(),
                "?"
            }
            if open {
                div { style: "position: absolute; top: 100%; left: 0; z-index: 50; padding-top: 0.375rem;",
                    div {
                        class: "bg-base-100 rounded-box p-3 text-xs",
                        style: "width: {POPOVER_WIDTH}; max-height: {POPOVER_MAX_HEIGHT}; overflow-y: auto; \
                                border: 1px solid rgba(128, 128, 128, 0.35); \
                                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.25);",
                        div { class: "flex items-baseline justify-between mb-2",
                            span { class: "font-semibold", "Glossary" }
                            span { class: "opacity-60",
                                if pinned() {
                                    "click outside or press Esc to close"
                                } else {
                                    "click the ? to keep it open and scroll"
                                }
                            }
                        }
                        dl { style: "columns: 2 16rem; column-gap: 1.5rem;",
                            for (term , definition) in GLOSSARY {
                                div { style: "break-inside: avoid; margin-bottom: 0.75rem;",
                                    dt { class: "font-semibold", "{term}" }
                                    dd { class: "opacity-80", "{definition}" }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lineage_page::orbit3d_tab::{with_thousands_separators, OUTER_SOLAR_SYSTEM_MIN_AU};
    use crate::orbit3d::geometry;

    fn definition_of(term_start: &str) -> &'static str {
        GLOSSARY
            .iter()
            .find(|(term, _)| term.starts_with(term_start))
            .unwrap_or_else(|| panic!("no glossary entry for {term_start:?}"))
            .1
    }

    /// The acronyms the plot's hovers and legend actually use.
    #[test]
    fn the_glossary_covers_the_terms_the_plot_uses() {
        for term in [
            "AU",
            "LD",
            "MOID",
            "Ecliptic",
            "Inclination",
            "Ascending",
            "Perihelion",
            "Osculating",
            "Phase angle",
            "Solar elongation",
            "H ",
            "Uncertainty clones",
            "Deviations exaggerated",
            "Outer solar system",
        ] {
            definition_of(term);
        }
    }

    #[test]
    fn no_entry_is_empty_or_repeated() {
        for (term, definition) in GLOSSARY {
            assert!(!term.trim().is_empty() && !definition.trim().is_empty());
        }
        let mut terms: Vec<&str> = GLOSSARY.iter().map(|(t, _)| *t).collect();
        terms.sort_unstable();
        terms.dedup();
        assert_eq!(terms.len(), GLOSSARY.len(), "duplicate glossary term");
    }

    /// The kilometre values quoted for AU and LD are the ones the hovers
    /// convert with.
    #[test]
    fn the_quoted_unit_values_match_the_conversion_constants() {
        assert_eq!(format!("{:.1}", geometry::KM_PER_AU), "149597870.7");
        assert!(definition_of("AU").contains("149,597,870.7"));

        let ld_km = with_thousands_separators(geometry::KM_PER_LUNAR_DISTANCE as u64);
        assert!(definition_of("LD").contains(&ld_km), "{ld_km}");

        // About 0.00257 AU is one lunar distance.
        let ld_au = geometry::KM_PER_LUNAR_DISTANCE / geometry::KM_PER_AU;
        assert!(definition_of("LD").contains(&format!("{ld_au:.5}")));
    }

    /// The popover must fit in the window: its size limits are relative to
    /// the viewport, never fixed pixel sizes that a small window could not
    /// hold.
    #[test]
    fn the_popover_is_sized_relative_to_the_viewport() {
        assert!(POPOVER_WIDTH.contains("100vw"), "{POPOVER_WIDTH}");
        assert!(POPOVER_MAX_HEIGHT.contains("vh"), "{POPOVER_MAX_HEIGHT}");
    }

    /// The threshold quoted for the outer solar system is the one the
    /// switch uses.
    #[test]
    fn the_outer_solar_system_entry_states_the_switch_threshold() {
        let threshold = format!("{} AU", OUTER_SOLAR_SYSTEM_MIN_AU);
        assert!(definition_of("Outer solar system").contains(&threshold));
    }
}
