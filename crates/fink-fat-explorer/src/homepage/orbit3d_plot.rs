//! The homepage's 3D view: current heliocentric positions of every tracked
//! lineage, grouped and colored by dynamical family — plus the planets and
//! tracked perturbers for scale and context.
//!
//! Deliberately positions-only for tracked objects, unlike the lineage
//! page's single-object view: tracing a full orbital ellipse per lineage
//! here would mean drawing on the order of `n_lineages` curves at once,
//! which is both unreadable (an opaque tangle at this population size) and
//! far more data to ship than one point per lineage.

use std::collections::HashMap;

use dioxus::prelude::*;

use crate::homepage::family::DynamicalFamily;
use crate::orbit3d::plot3d::{planet_traces, Scatter3dPlot, Trace3D, TraceSource, TraceStyle};
use crate::orbit3d::server_fns::{get_homepage_orbit3d, get_planets_3d};
use crate::orbit3d::types::ObjectPoint3D;

/// Same rationale as `homepage::dynamic_pop_plot`'s `WARMUP_POLL_MS`: how
/// often to re-check whether the homepage snapshot has finished building.
const WARMUP_POLL_MS: u64 = 1000;

/// Groups tracked-object points by [`DynamicalFamily`] into one
/// [`Trace3D`] per family present, colored by
/// [`DynamicalFamily::color`] — the same grouping
/// `homepage::snapshot::build_series` does for the (a, e) plot, just without
/// the quality-tier split (a 3D scatter has no equivalent to the (a, e)
/// plot's marker-shape-per-tier convention, and tier is still available in
/// [`ObjectPoint3D`] for a future increment to use).
fn family_traces(points: &[ObjectPoint3D]) -> Vec<Trace3D> {
    let mut grouped: HashMap<DynamicalFamily, Vec<[f64; 3]>> = HashMap::new();
    for point in points {
        grouped
            .entry(point.family)
            .or_default()
            .push(point.position);
    }

    let mut families: Vec<DynamicalFamily> = grouped.keys().copied().collect();
    families.sort();

    families
        .into_iter()
        .map(|family| Trace3D {
            name: family.label().to_string(),
            color: family.color().to_string(),
            style: TraceStyle::Markers,
            source: TraceSource::TrackedObject,
            points: grouped.remove(&family).unwrap_or_default(),
            hover_text: Vec::new(),
        })
        .collect()
}

/// The subtitle under the plot's title: how many objects are drawn, and — when
/// some lineages could not be — how many are missing and why.
///
/// # Arguments
///
/// * `n_plotted` — objects drawn.
/// * `n_excluded` — lineages left out because their best solution is not a
///   closed ellipse.
///
/// # Returns
///
/// `"16338 objects plotted"`, or, if any were excluded, `"16338 objects
/// plotted (509 more not shown: no closed elliptical orbit)"` — the (a, e)
/// plot counts those at `a = e = 0`, which is why its total is higher.
fn objects_status_text(n_plotted: usize, n_excluded: usize) -> String {
    if n_excluded == 0 {
        format!("{n_plotted} objects plotted")
    } else {
        format!(
            "{n_plotted} objects plotted ({n_excluded} more not shown: no closed elliptical orbit)"
        )
    }
}

#[component]
pub fn Orbit3DPlot() -> Element {
    let mut objects = use_resource(move || async move { get_homepage_orbit3d().await });
    let planets = use_resource(move || async move { get_planets_3d().await });

    // `Ok(None)` means the homepage snapshot is still building — poll for
    // it, exactly like `dynamic_pop_plot::DynamicPopPlot`.
    use_effect(move || {
        let warming = matches!(&*objects.read(), Some(Ok(None)));
        if warming {
            spawn(async move {
                crate::sleep_ms(WARMUP_POLL_MS).await;
                objects.restart();
            });
        }
    });

    // Held back until the planets request has settled (success or failure),
    // so the plot is drawn once with everything in it rather than first with
    // the objects alone and then redrawn when the planets arrive.
    let traces = use_memo(move || {
        let planets = planets.read();
        if planets.is_none() {
            return Vec::new();
        }
        let mut traces = match &*objects.read() {
            Some(Ok(Some(objects))) => family_traces(&objects.points),
            _ => Vec::new(),
        };
        if let Some(Ok(bodies)) = &*planets {
            traces.extend(planet_traces(bodies));
        }
        traces
    });

    let status_text = match &*objects.read() {
        Some(Ok(Some(objects))) => objects_status_text(objects.points.len(), objects.n_excluded),
        Some(Ok(None)) => "Building the population index...".to_string(),
        Some(Err(e)) => format!("Error: {e}"),
        None => String::new(),
    };
    let is_loading = traces.read().is_empty() && !matches!(&*objects.read(), Some(Err(_)));

    rsx! {
        div { class: "card bg-base-100 shadow-sm",
            div { class: "card-body",
                div { class: "text-center mb-1",
                    h2 { class: "text-2xl font-bold tracking-tight", "The Solar System, in 3D" }
                    p { class: "text-sm opacity-60", "{status_text}" }
                }
                div { class: "relative",
                    Scatter3dPlot { plot_id: "homepage-orbit3d-plot-div", traces: traces() }
                    if is_loading {
                        div { class: "absolute inset-0 flex items-center justify-center",
                            span { class: "loading loading-dots loading-lg" }
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

    #[test]
    fn the_status_text_is_plain_when_every_lineage_is_plotted() {
        assert_eq!(objects_status_text(16_847, 0), "16847 objects plotted");
    }

    #[test]
    fn the_status_text_reports_the_lineages_without_a_closed_orbit() {
        let text = objects_status_text(16_338, 509);
        assert!(text.starts_with("16338 objects plotted"), "{text}");
        assert!(text.contains("509 more not shown"), "{text}");
        assert!(text.contains("no closed elliptical orbit"), "{text}");
    }
}
