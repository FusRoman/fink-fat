//! The shared 3D scatter/line plot component both the homepage's population
//! view and the lineage page's single-object view render into.

use dioxus::prelude::*;

#[cfg(target_arch = "wasm32")]
use plotly::{
    common::{Line, Marker, MarkerSymbol, Mode, Title},
    layout::{AspectMode, Axis, Layout, LayoutScene, Margin},
    Plot, Scatter3D,
};

use serde::{Deserialize, Serialize};

use crate::orbit3d::types::Body3D;

/// How a [`Trace3D`]'s points should be connected — markers for a position
/// (or a handful of positions), a line for an orbital ellipse.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceStyle {
    /// Unconnected markers — a body's current position.
    Markers,
    /// A connected polyline with no markers — an orbital ellipse.
    Line,
}

/// Which kind of body a [`Trace3D`] represents, driving its marker size in
/// [`Scatter3dPlot`]: fink-fat's own tracked objects vastly outnumber the
/// dozen planets/perturbers they share the scene with (thousands on the
/// homepage view), so they need a visibly smaller marker to stay
/// distinguishable from the planets rather than drowning them out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceSource {
    /// One of `orbit3d::ephem_provider::TRACKED_BODIES`.
    Planet,
    /// A fink-fat tracked object — its current position, or its own orbit.
    TrackedObject,
}

/// One named, colored set of points to draw in a [`Scatter3dPlot`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Trace3D {
    pub name: String,
    /// CSS color (e.g. a hex string like `"#d29b2d"`), reused as-is from
    /// [`crate::homepage::family::DynamicalFamily::color`] for tracked
    /// objects, or a fixed palette entry for planets/perturbers.
    pub color: String,
    pub style: TraceStyle,
    pub source: TraceSource,
    /// Heliocentric ecliptic points, AU (see
    /// [`crate::orbit3d::geometry::point_at_true_anomaly`] for the exact
    /// frame).
    pub points: Vec<[f64; 3]>,
}

/// Marker size for a planet/perturber's [`TraceStyle::Markers`] trace —
/// large and, per [`PLANET_MARKER_BORDER_COLOR`]/[`PLANET_MARKER_SYMBOL`],
/// outlined and diamond-shaped, so the dozen planets stay identifiable at a
/// glance against a cloud of thousands of small round tracked-object
/// markers (see [`TraceSource`]'s doc) — size alone wasn't enough once the
/// tracked-object count got into the tens of thousands.
#[cfg(target_arch = "wasm32")]
const PLANET_MARKER_SIZE: usize = 9;
/// Marker shape for planets/perturbers — a diamond reads as visually
/// distinct from every tracked-object trace's plain circle even before
/// color or size are taken into account.
#[cfg(target_arch = "wasm32")]
const PLANET_MARKER_SYMBOL: MarkerSymbol = MarkerSymbol::Diamond;
/// Border color/width around a planet marker: several planet colors
/// ([`body_color`]'s grays/tans, chosen for real-world accuracy) are close
/// in value to the tracked-object palette, so a dark outline is what
/// actually guarantees contrast regardless of which family colors happen to
/// surround a given planet.
#[cfg(target_arch = "wasm32")]
const PLANET_MARKER_BORDER_COLOR: &str = "#1a1a1a";
#[cfg(target_arch = "wasm32")]
const PLANET_MARKER_BORDER_WIDTH: f64 = 1.5;
/// Marker size for a fink-fat tracked object's [`TraceStyle::Markers`]
/// trace — smaller than [`PLANET_MARKER_SIZE`] so a scene with thousands of
/// tracked objects still reads the dozen planet markers clearly (see
/// [`TraceSource`]'s doc).
#[cfg(target_arch = "wasm32")]
const OBJECT_MARKER_SIZE: usize = 1;
/// Marker size plotly still draws at each vertex of a [`TraceStyle::Line`]
/// trace — kept small so the curve itself reads as a smooth line rather than
/// a dotted one. Shared by every line trace regardless of [`TraceSource`]:
/// the crowding [`OBJECT_MARKER_SIZE`] addresses is a marker-count problem
/// (thousands of position dots), not a line-count one (at most a few lines
/// per page).
#[cfg(target_arch = "wasm32")]
const LINE_VERTEX_SIZE: usize = 2;
/// Highest opacity [`population_opacity`] returns — for a small
/// tracked-object trace (at or below [`OBJECT_OPACITY_LOW_POPULATION`]
/// points), which has no clutter of its own to cut through.
const OBJECT_OPACITY_MAX: f64 = 0.95;
/// Lowest opacity [`population_opacity`] returns — for a large
/// tracked-object trace (at or above [`OBJECT_OPACITY_HIGH_POPULATION`]
/// points), so a dense cluster reads as a soft cloud instead of a solid
/// mass.
const OBJECT_OPACITY_MIN: f64 = 0.35;
/// Population size at or below which [`population_opacity`] returns
/// [`OBJECT_OPACITY_MAX`] — e.g. a rare dynamical family.
const OBJECT_OPACITY_LOW_POPULATION: f64 = 10.0;
/// Population size at or above which [`population_opacity`] returns
/// [`OBJECT_OPACITY_MIN`] — e.g. the main belt.
const OBJECT_OPACITY_HIGH_POPULATION: f64 = 10_000.0;

/// Marker opacity for a tracked-object trace of `n` points, interpolated on
/// a log scale between [`OBJECT_OPACITY_MAX`] and [`OBJECT_OPACITY_MIN`].
///
/// Log, not linear: dynamical family sizes span several orders of magnitude
/// (a handful of Trojans/Hungarias next to tens of thousands of main-belt
/// objects), so a linear scale would put almost every family at
/// indistinguishably-near-[`OBJECT_OPACITY_MAX`] and let only the single
/// largest one register any transparency at all.
///
/// # Arguments
///
/// * `n` — number of points the trace carries.
///
/// # Returns
///
/// Opacity in `[OBJECT_OPACITY_MIN, OBJECT_OPACITY_MAX]`: `n <=
/// OBJECT_OPACITY_LOW_POPULATION` maps to the former, `n >=
/// OBJECT_OPACITY_HIGH_POPULATION` to the latter, and every value in between
/// is interpolated linearly in `log10(n)`.
fn population_opacity(n: usize) -> f64 {
    let log_n = (n.max(1) as f64).log10();
    let log_low = OBJECT_OPACITY_LOW_POPULATION.log10();
    let log_high = OBJECT_OPACITY_HIGH_POPULATION.log10();

    let t = ((log_n - log_low) / (log_high - log_low)).clamp(0.0, 1.0);
    OBJECT_OPACITY_MAX + t * (OBJECT_OPACITY_MIN - OBJECT_OPACITY_MAX)
}

/// Marker size for the fixed Sun marker at the origin — larger than every
/// other marker so it reads as the frame's center at a glance.
#[cfg(target_arch = "wasm32")]
const SUN_MARKER_SIZE: usize = 10;
#[cfg(target_arch = "wasm32")]
const SUN_COLOR: &str = "#ffcc33";

/// A fixed color for one of `orbit3d::ephem_provider::TRACKED_BODIES`, shared
/// by every page that plots planets/perturbers so the same body always reads
/// as the same color. Roughly the body's real-world hue; an unrecognized
/// name (should not happen — every caller builds names from
/// `TRACKED_BODIES`) falls back to a neutral gray rather than panicking.
pub fn body_color(name: &str) -> &'static str {
    match name {
        "Mercury" => "#9a9a9a",
        "Venus" => "#d9b98a",
        "Earth" => "#3a7bd5",
        "Mars" => "#c1440e",
        "Jupiter" => "#c88b3a",
        "Saturn" => "#e0c16c",
        "Uranus" => "#7fd4d9",
        "Neptune" => "#3f54ba",
        "Pluto" => "#b5a191",
        "Ceres" => "#8c8c8c",
        "Pallas" => "#a89f91",
        "Vesta" => "#c9c2b0",
        _ => "#888888",
    }
}

/// Builds two [`Trace3D`]s per body — its orbital ellipse and its current
/// position, same color — from a [`Body3D`] list. Shared by the homepage's
/// and the lineage page's 3D views so a planet/perturber always renders the
/// same way regardless of which page requested it.
///
/// # Arguments
///
/// * `bodies` — as returned by `orbit3d::server_fns::get_planets_3d`.
///
/// # Returns
///
/// `2 * bodies.len()` traces: for each body, one [`TraceStyle::Line`] trace
/// (the ellipse) and one [`TraceStyle::Markers`] trace (the current
/// position), in that order.
pub fn planet_traces(bodies: &[Body3D]) -> Vec<Trace3D> {
    bodies
        .iter()
        .flat_map(|body| {
            let color = body_color(&body.name).to_string();
            [
                Trace3D {
                    name: format!("{} (orbit)", body.name),
                    color: color.clone(),
                    style: TraceStyle::Line,
                    source: TraceSource::Planet,
                    points: body.orbit.clone(),
                },
                Trace3D {
                    name: body.name.clone(),
                    color,
                    style: TraceStyle::Markers,
                    source: TraceSource::Planet,
                    points: vec![body.position],
                },
            ]
        })
        .collect()
}

/// Heliocentric ecliptic 3D scatter/line plot: a fixed Sun marker at the
/// origin plus one trace per [`Trace3D`].
///
/// # Arguments
///
/// * `plot_id` — HTML id for the plot's mount div. Must be unique among
///   every `Scatter3dPlot` that could be mounted at once (the homepage and
///   the lineage page each use their own literal), since plotly's JS
///   bindings address the plot by this id.
/// * `traces` — the orbits/positions to draw, in addition to the Sun.
#[component]
pub fn Scatter3dPlot(plot_id: &'static str, traces: Vec<Trace3D>) -> Element {
    let mut is_mounted = use_signal(|| false);
    // Same rationale as `homepage::dynamic_pop_plot::DynamicPopPlot`: the
    // first draw needs `new_plot`, every later one is a cheaper `react`
    // diff, and this only matters on the wasm build (the only one that ever
    // draws).
    #[cfg(target_arch = "wasm32")]
    let mut drawn = use_signal(|| false);

    // `use_reactive!` (rather than a plain `move || {..}`) so the effect
    // re-runs when the `traces` *prop* changes: `traces` is an owned `Vec`,
    // not a `Signal` read inside the closure, so without this the effect's
    // only tracked dependency would be `is_mounted` — meaning it would draw
    // once (whatever `traces` happened to hold when the div first mounted,
    // often still empty while the data resource is loading) and then never
    // again, even as new data streamed in. Same pattern as
    // `lineage_page::trajectory_plot::TrajectoryPlot`.
    use_effect(use_reactive!(|(plot_id, traces)| {
        #[cfg(target_arch = "wasm32")]
        {
            if !is_mounted() {
                return;
            }

            let plot = {
                if traces.is_empty() {
                    return;
                }

                let mut plot = Plot::new();

                let sun = Scatter3D::new(vec![0.0], vec![0.0], vec![0.0])
                    .name("Sun")
                    .mode(Mode::Markers)
                    .marker(Marker::new().color(SUN_COLOR).size(SUN_MARKER_SIZE));
                plot.add_trace(sun);

                for trace in traces.iter() {
                    let xs: Vec<f64> = trace.points.iter().map(|p| p[0]).collect();
                    let ys: Vec<f64> = trace.points.iter().map(|p| p[1]).collect();
                    let zs: Vec<f64> = trace.points.iter().map(|p| p[2]).collect();

                    let (mode, marker) = match trace.style {
                        TraceStyle::Markers => {
                            let base = Marker::new().color(trace.color.clone());
                            let marker = match trace.source {
                                // Diamond + dark outline: color alone isn't
                                // reliable contrast against thousands of
                                // tracked-object hues, see the constants'
                                // doc comments.
                                TraceSource::Planet => base
                                    .size(PLANET_MARKER_SIZE)
                                    .symbol(PLANET_MARKER_SYMBOL)
                                    .line(
                                        Line::new()
                                            .color(PLANET_MARKER_BORDER_COLOR)
                                            .width(PLANET_MARKER_BORDER_WIDTH),
                                    ),
                                TraceSource::TrackedObject => base
                                    .size(OBJECT_MARKER_SIZE)
                                    .opacity(population_opacity(trace.points.len())),
                            };
                            (Mode::Markers, marker)
                        }
                        TraceStyle::Line => (
                            Mode::Lines,
                            Marker::new()
                                .color(trace.color.clone())
                                .size(LINE_VERTEX_SIZE),
                        ),
                    };

                    let t = Scatter3D::new(xs, ys, zs)
                        .name(trace.name.clone())
                        .mode(mode)
                        .marker(marker);
                    plot.add_trace(t);
                }

                let axis = |label: &str| Axis::new().title(Title::from(format!("{label} (AU)")));
                let scene = LayoutScene::new()
                    .x_axis(axis("x"))
                    .y_axis(axis("y"))
                    .z_axis(axis("z"))
                    // `Data` keeps AU on every axis at the same visual
                    // scale — the default `Auto` stretches each axis to
                    // fill the plot independently, which would draw every
                    // orbit as a near-circle regardless of its true shape.
                    .aspect_mode(AspectMode::Data);

                let layout = Layout::new()
                    .height(700)
                    .scene(scene)
                    .margin(Margin::new().top(20).right(20));

                plot.set_layout(layout);
                plot
            };

            spawn(async move {
                if *drawn.peek() {
                    plotly::bindings::react(plot_id, &plot).await;
                } else {
                    plotly::bindings::new_plot(plot_id, &plot).await;
                    drawn.set(true);
                }
            });
        }
    }));

    rsx! {
        div {
            id: "{plot_id}",
            style: "width: 100%; min-height: 60vh;",
            onmounted: move |_| is_mounted.set(true),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn population_opacity_is_max_at_or_below_the_low_population_threshold() {
        assert_eq!(population_opacity(1), OBJECT_OPACITY_MAX);
        assert_eq!(population_opacity(10), OBJECT_OPACITY_MAX);
    }

    #[test]
    fn population_opacity_is_min_at_or_above_the_high_population_threshold() {
        assert_eq!(population_opacity(10_000), OBJECT_OPACITY_MIN);
        assert_eq!(population_opacity(1_000_000), OBJECT_OPACITY_MIN);
    }

    #[test]
    fn population_opacity_decreases_monotonically_with_population() {
        let sizes = [1, 5, 10, 50, 200, 1_000, 5_000, 10_000, 50_000];
        let opacities: Vec<f64> = sizes.iter().map(|&n| population_opacity(n)).collect();
        for pair in opacities.windows(2) {
            assert!(
                pair[1] <= pair[0],
                "opacity should never increase with population size: {opacities:?}"
            );
        }
    }

    /// A tenfold population increase should have the same effect on opacity
    /// regardless of where it falls in the range — the defining property of
    /// a log scale.
    #[test]
    fn population_opacity_is_linear_in_log10_population() {
        let step_100_to_1000 = population_opacity(100) - population_opacity(1_000);
        let step_1000_to_10000 = population_opacity(1_000) - population_opacity(10_000);
        assert!(
            (step_100_to_1000 - step_1000_to_10000).abs() < 1e-9,
            "equal log-population steps should shift opacity by equal amounts: \
             {step_100_to_1000} vs {step_1000_to_10000}"
        );
    }
}
