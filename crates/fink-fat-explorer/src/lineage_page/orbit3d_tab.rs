//! The lineage page's 3D tab: one tracked object's full orbital ellipse and
//! current position, plotted alongside the planets and tracked perturbers.

use dioxus::prelude::*;

use crate::format_epoch::iso_utc;

use super::light_curve_plot::band_name;
use super::orbit3d_glossary::PlotGlossary;

use crate::orbit3d::geometry;
use crate::orbit3d::plot3d::{
    planet_traces, LandmarkKind, Scatter3dPlot, Trace3D, TraceSource, TraceStyle,
};
use crate::orbit3d::server_fns::get_lineage_orbit3d;
use crate::orbit3d::types::{
    Body3D, LineageOrbit3D, ObservationPoint3D, OrbitSummary3D, UncertaintyCloud3D,
    UNCERTAINTY_ORBIT_SAMPLES,
};

/// Scale applied to every sphere of the lineage view (the Sun, the planets
/// and the object), relative to the sizes the homepage uses. One object's
/// orbit is what this view is about, so the balls are kept small enough not
/// to hide it; the Sun stays larger than the planets, as they scale together.
const SPHERE_SCALE: f64 = 0.2;

/// Color for the tracked object's own orbit/position traces — distinct from
/// every planet/perturber color in `orbit3d::plot3d::body_color`.
const OBJECT_COLOR: &str = "#ff3b6f";
/// Color of the observation crosses — dark, so they stand out on plotly's
/// light scene background and against both [`OBJECT_COLOR`] and the
/// (mostly light-toned) planet palette.
const OBSERVATION_COLOR: &str = "#1f2937";
/// Color of the dashed sight lines — a lighter gray than
/// [`OBSERVATION_COLOR`] so the guides stay behind the crosses they connect.
const SIGHT_LINE_COLOR: &str = "#6b7280";

/// Color of the perihelion and aphelion markers.
const APSIDES_COLOR: &str = "#b91c4b";
/// Color of the node markers and of the line of nodes.
const NODES_COLOR: &str = "#0f766e";
/// Color of the MOID-with-Earth markers and segment.
const MOID_COLOR: &str = "#3a7bd5";

/// Color of the uncertainty cloud's clone dots — amber, so the clones read
/// apart from the pink orbit they scatter around.
const UNCERTAINTY_CLOUD_COLOR: &str = "rgba(245, 158, 11, 0.75)";
/// Color of the clone orbits.
const UNCERTAINTY_ORBIT_COLOR: &str = "rgba(245, 158, 11, 0.5)";
/// Whether the ecliptic plane wireframe is shown when the tab opens.
const SHOW_ECLIPTIC_DEFAULT: bool = true;
/// Whether the orbit uncertainty is shown when the tab opens.
const SHOW_UNCERTAINTY_DEFAULT: bool = true;
/// The exaggeration factors offered for the clones' deviations from the
/// best orbit. `1.0` is the true scale.
const EXAGGERATION_CHOICES: [f64; 4] = [1.0, 10.0, 100.0, 1000.0];
/// The exaggeration selected when the tab opens. At the true scale a
/// converged fit's clones sit within a few thousandths of an AU of the
/// orbit — under a pixel on a plot that spans the planets — so they would be
/// invisible; the legend and caption always state the factor in use.
const EXAGGERATION_DEFAULT: f64 = 10.0;
/// Color of the ecliptic plane wireframe.
const ECLIPTIC_COLOR: &str = "rgba(96, 165, 250, 0.7)";
/// Color of the inclination pie-slice and its label.
const INCLINATION_COLOR: &str = "#7c3aed";
/// Number of points of each ring of the ecliptic wireframe.
const ECLIPTIC_RING_SEGMENTS: usize = 72;
/// The ecliptic wireframe extends this far beyond the orbit's aphelion.
const ECLIPTIC_RADIUS_MARGIN: f64 = 1.1;
/// Smallest radius, AU, of the ecliptic wireframe — a near-circular inner
/// orbit would otherwise get one barely bigger than the Sun's sphere.
const ECLIPTIC_MIN_RADIUS_AU: f64 = 1.0;
/// Number of radial spokes of the ecliptic wireframe.
const ECLIPTIC_SPOKES: usize = 8;
/// The inclination pie-slice's radius as a fraction of the perihelion
/// distance, so it fits inside the orbit.
const INCLINATION_WEDGE_PERIHELION_FRACTION: f64 = 0.5;
/// Smallest radius, AU, of the inclination pie-slice — clear of the Sun's
/// sphere even for an orbit with a tiny perihelion.
const INCLINATION_WEDGE_MIN_RADIUS_AU: f64 = 0.6;
/// Bodies whose semi-major axis exceeds this, AU, belong to the outer solar
/// system — Jupiter and beyond; the main belt (and Ceres, Pallas, Vesta)
/// ends well inside it.
pub(super) const OUTER_SOLAR_SYSTEM_MIN_AU: f64 = 4.0;
/// Jupiter's semi-major axis, AU, used to pick the switch's automatic state
/// if Jupiter is somehow missing from the planet list.
const FALLBACK_JUPITER_SEMI_MAJOR_AXIS_AU: f64 = 5.2;
/// Number of segments of the inclination arc.
const INCLINATION_ARC_SEGMENTS: usize = 24;
/// Below this inclination, degrees, no pie-slice is drawn — it would be a
/// sliver.
const INCLINATION_MIN_DRAWN_DEG: f64 = 0.05;

/// The points of the dashed sight lines: for each observation, the Sun (the
/// heliocentric origin), then the observation's position, then the observer
/// — consecutive triples, one two-segment polyline per observation.
///
/// # Arguments
///
/// * `observations` — the observations as placed in 3D.
///
/// # Returns
///
/// `3 * observations.len()` points, to be drawn with
/// `TraceStyle::DashedPolylines(3)`.
fn sight_line_points(observations: &[ObservationPoint3D]) -> Vec<[f64; 3]> {
    observations
        .iter()
        .flat_map(|o| [[0.0; 3], o.position, o.observer_position])
        .collect()
}

/// The hover text of each observation's sight line — what a user is
/// pointing at when the line is the nearest thing under the cursor.
///
/// # Arguments
///
/// * `observations` — the observations as placed in 3D.
///
/// # Returns
///
/// One text per observation, in order (rank shown one-based).
fn sight_line_hover_texts(observations: &[ObservationPoint3D]) -> Vec<String> {
    observations
        .iter()
        .enumerate()
        .map(|(i, o)| {
            format!(
                "Sight line of observation #{}<br>Sun to object: {:.3} AU<br>Observer to object: {:.3} AU",
                i + 1,
                o.heliocentric_distance_au,
                o.topocentric_distance_au,
            )
        })
        .collect()
}

/// The hover text of one observation cross: its rank, its epoch in ISO-8601
/// UTC, its photometry and observing site, its heliocentric and topocentric
/// distances, its phase angle and solar elongation, and the absolute
/// magnitude it implies.
///
/// # Arguments
///
/// * `index` — zero-based rank of the observation in the lineage's
///   observation list (shown one-based).
/// * `point` — the observation as placed in 3D.
///
/// # Returns
///
/// Plotly hover HTML (`<br>` separated lines).
fn observation_hover_text(index: usize, point: &ObservationPoint3D) -> String {
    let band = band_name(point.filter);
    let absolute_magnitude = match point.absolute_magnitude {
        Some(h) => format!("H ≈ {h:.2} ({band} band, G = 0.15, no colour correction)"),
        None => "H: n/a (phase angle out of the H,G model's range)".to_string(),
    };
    format!(
        "Observation #{}<br>{}<br>Magnitude: {:.2} ± {:.2} ({band} band, MPC {})<br>\
         Heliocentric distance: {:.4} AU<br>Topocentric distance: {:.4} AU<br>\
         Phase angle: {:.2}°<br>Solar elongation: {:.1}°<br>{absolute_magnitude}",
        index + 1,
        iso_utc(point.mjd_tt),
        point.magnitude,
        point.mag_err,
        point.mpc_code,
        point.heliocentric_distance_au,
        point.topocentric_distance_au,
        point.phase_angle_deg,
        point.elongation_deg,
    )
}

/// Below this distance, AU, a MOID is also worth reading in lunar
/// distances (0.05 AU is about 19.5 LD).
const MOID_LUNAR_DISTANCE_BELOW_AU: f64 = 0.05;
/// Below this distance, one lunar distance, a MOID is read in kilometres.
const MOID_KILOMETRES_BELOW_LUNAR_DISTANCES: f64 = 1.0;

/// Writes an integer with a comma every three digits, e.g. `312400` →
/// `"312,400"`.
pub(super) fn with_thousands_separators(value: u64) -> String {
    let digits = value.to_string();
    let mut grouped = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, digit) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            grouped.push(',');
        }
        grouped.push(digit);
    }
    grouped
}

/// A minimum orbit intersection distance in the unit that reads best for
/// its size: astronomical units normally, lunar distances once it is small,
/// kilometres once it is under one lunar distance. The AU (or LD) value
/// stays alongside for reference.
///
/// # Arguments
///
/// * `distance_au` — the distance, AU.
///
/// # Returns
///
/// * `d >= 0.05 AU` → `"0.1234 AU"`;
/// * `1 LD <= d < 0.05 AU` → `"4.52 LD (0.0118 AU)"`;
/// * `d < 1 LD` → `"312,400 km (0.813 LD)"`.
fn format_moid(distance_au: f64) -> String {
    let lunar_distances = distance_au * geometry::KM_PER_AU / geometry::KM_PER_LUNAR_DISTANCE;
    if distance_au >= MOID_LUNAR_DISTANCE_BELOW_AU {
        format!("{distance_au:.4} AU")
    } else if lunar_distances >= MOID_KILOMETRES_BELOW_LUNAR_DISTANCES {
        format!("{lunar_distances:.2} LD ({distance_au:.4} AU)")
    } else {
        let km = with_thousands_separators((distance_au * geometry::KM_PER_AU).round() as u64);
        format!("{km} km ({lunar_distances:.3} LD)")
    }
}

/// The hover text of the object's sphere: its orbit's numbers, its
/// distances to the Sun and the Earth at the view's epoch, and its MOID with
/// every planet.
///
/// # Arguments
///
/// * `summary` — the object's orbit summary.
///
/// # Returns
///
/// Plotly hover HTML (`<br>` separated lines).
fn object_hover_text(summary: &OrbitSummary3D) -> String {
    let earth = match summary.earth_distance_au {
        Some(d) => format!("{d:.3} AU"),
        None => "n/a".to_string(),
    };
    let moids: String = summary
        .moids
        .iter()
        .map(|m| {
            format!(
                "<br>&nbsp;&nbsp;{}: {}",
                m.name,
                format_moid(m.moid.distance_au)
            )
        })
        .collect();
    format!(
        "Current position<br>a = {:.3} AU, e = {:.4}, i = {:.2}°<br>\
         q = {:.3} AU, Q = {:.3} AU, P = {:.2} yr<br>\
         Distance to the Sun: {:.3} AU<br>Distance to the Earth: {earth}<br>\
         MOID (osculating orbits):{moids}",
        summary.semi_major_axis_au,
        summary.eccentricity,
        summary.inclination_deg,
        summary.perihelion_au,
        summary.aphelion_au,
        summary.period_years,
        summary.sun_distance_au,
    )
}

/// The orbit landmarks: perihelion, aphelion, nodes with their line, and
/// the MOID with the Earth.
///
/// # Arguments
///
/// * `summary` — the object's orbit summary.
///
/// # Returns
///
/// Between five and seven traces: one marker trace each for the perihelion
/// and aphelion, one for the two nodes, the dashed line of nodes, and — if
/// the Earth's MOID is known — its two end markers and dashed segment.
fn landmark_traces(summary: &OrbitSummary3D) -> Vec<Trace3D> {
    let l = &summary.landmarks;
    let marker = |name: &str,
                  color: &str,
                  kind: LandmarkKind,
                  points: Vec<[f64; 3]>,
                  hover_text: Vec<String>| Trace3D {
        name: name.to_string(),
        color: color.to_string(),
        style: TraceStyle::Markers,
        source: TraceSource::Landmark(kind),
        points,
        hover_text,
    };

    let mut traces = vec![
        marker(
            "Perihelion",
            APSIDES_COLOR,
            LandmarkKind::Perihelion,
            vec![l.perihelion],
            vec![format!("Perihelion<br>q = {:.4} AU", summary.perihelion_au)],
        ),
        marker(
            "Aphelion",
            APSIDES_COLOR,
            LandmarkKind::Aphelion,
            vec![l.aphelion],
            vec![format!("Aphelion<br>Q = {:.4} AU", summary.aphelion_au)],
        ),
        marker(
            "Nodes",
            NODES_COLOR,
            LandmarkKind::Node,
            vec![l.ascending_node, l.descending_node],
            vec!["Ascending node".to_string(), "Descending node".to_string()],
        ),
        Trace3D {
            name: "Line of nodes".to_string(),
            color: NODES_COLOR.to_string(),
            style: TraceStyle::DashedPolylines(2),
            source: TraceSource::Landmark(LandmarkKind::Node),
            points: vec![l.ascending_node, l.descending_node],
            hover_text: vec![
                "Line of nodes<br>(where the orbital plane crosses the ecliptic)".to_string(),
            ],
        },
    ];

    if let Some(earth) = summary.moids.iter().find(|m| m.name == "Earth") {
        let m = &earth.moid;
        traces.push(marker(
            "MOID with Earth",
            MOID_COLOR,
            LandmarkKind::Moid,
            vec![m.point_a, m.point_b],
            vec![
                format!(
                    "MOID with Earth: {}<br>(on the object's orbit)",
                    format_moid(m.distance_au)
                ),
                format!(
                    "MOID with Earth: {}<br>(on the Earth's orbit)",
                    format_moid(m.distance_au)
                ),
            ],
        ));
        traces.push(Trace3D {
            name: "MOID with Earth (segment)".to_string(),
            color: MOID_COLOR.to_string(),
            style: TraceStyle::DashedPolylines(2),
            source: TraceSource::Landmark(LandmarkKind::Moid),
            points: vec![m.point_a, m.point_b],
            hover_text: vec![format!("MOID with Earth: {}", format_moid(m.distance_au))],
        });
    }
    traces
}

/// Scales points' deviations from a center by `factor`:
/// `center + factor · (point − center)`.
///
/// # Arguments
///
/// * `points` — the points to scale.
/// * `center` — the point deviations are measured from.
/// * `factor` — the exaggeration; `1.0` leaves the points unchanged.
///
/// # Returns
///
/// The scaled points, in order.
fn exaggerate_about(points: &[[f64; 3]], center: [f64; 3], factor: f64) -> Vec<[f64; 3]> {
    points
        .iter()
        .map(|p| [0, 1, 2].map(|i| center[i] + factor * (p[i] - center[i])))
        .collect()
}

/// Scales every clone orbit's deviation from the best orbit by `factor`,
/// point by point (both are sampled at the same true anomalies).
///
/// # Arguments
///
/// * `orbits` — the clone orbits, each as many points as `best`.
/// * `best` — the best orbit.
/// * `factor` — the exaggeration.
///
/// # Returns
///
/// The clone orbits' points, concatenated (`orbits.len() * best.len()`
/// points), ready for `TraceStyle::ClosedPolylines`.
fn exaggerate_orbits(orbits: &[Vec<[f64; 3]>], best: &[[f64; 3]], factor: f64) -> Vec<[f64; 3]> {
    orbits
        .iter()
        .flat_map(|orbit| {
            orbit.iter().enumerate().map(|(j, p)| {
                let center = best.get(j).copied().unwrap_or(*p);
                [0, 1, 2].map(|i| center[i] + factor * (p[i] - center[i]))
            })
        })
        .collect()
}

/// The traces of the orbit uncertainty: the faint clone orbits and the two
/// clouds of clone positions (at the last observation and at the view's
/// epoch), named after the N-body fit covariance they were drawn from and,
/// when exaggerated, the factor applied.
///
/// # Arguments
///
/// * `cloud` — the uncertainty cloud.
/// * `exaggeration` — factor applied to every clone's deviation from the
///   best solution (`1.0` = true scale).
///
/// # Returns
///
/// Up to three traces; empty ones are left out.
fn uncertainty_traces(cloud: &UncertaintyCloud3D, exaggeration: f64) -> Vec<Trace3D> {
    let scale = if exaggeration == 1.0 {
        String::new()
    } else {
        format!(", deviations ×{exaggeration}")
    };
    let dots = |name: String, points: Vec<[f64; 3]>| Trace3D {
        name,
        color: UNCERTAINTY_CLOUD_COLOR.to_string(),
        style: TraceStyle::Markers,
        source: TraceSource::Uncertainty,
        points,
        hover_text: Vec::new(),
    };

    let mut traces = Vec::new();
    if !cloud.orbits.is_empty() {
        traces.push(Trace3D {
            name: format!("Clone orbits ({}{scale})", cloud.orbits.len()),
            color: UNCERTAINTY_ORBIT_COLOR.to_string(),
            style: TraceStyle::ClosedPolylines(UNCERTAINTY_ORBIT_SAMPLES),
            source: TraceSource::Uncertainty,
            points: exaggerate_orbits(&cloud.orbits, &cloud.best_orbit, exaggeration),
            hover_text: (1..=cloud.orbits.len())
                .map(|k| format!("Clone orbit {k} of {}{scale}", cloud.orbits.len()))
                .collect(),
        });
    }
    if !cloud.at_last_observation.is_empty() {
        traces.push(dots(
            format!(
                "Uncertainty @ last observation ({}{scale})",
                cloud.at_last_observation.len()
            ),
            exaggerate_about(
                &cloud.at_last_observation,
                cloud.center_last_observation,
                exaggeration,
            ),
        ));
    }
    if !cloud.at_now.is_empty() {
        traces.push(dots(
            format!(
                "Uncertainty @ now ({}/{} kept{scale})",
                cloud.n_kept_now, cloud.n_clones
            ),
            exaggerate_about(&cloud.at_now, cloud.center_now, exaggeration),
        ));
    }
    traces
}

/// The caption shown under the uncertainty switch: which covariance the
/// cloud comes from, the exaggeration in use, and what is hidden.
///
/// # Arguments
///
/// * `cloud` — the uncertainty cloud.
/// * `exaggeration` — the factor currently applied.
///
/// # Returns
///
/// A one-paragraph plain-text caption.
fn uncertainty_caption(cloud: &UncertaintyCloud3D, exaggeration: f64) -> String {
    let scale = if exaggeration == 1.0 {
        "The clones are drawn at their true scale.".to_string()
    } else {
        format!(
            "The clones' deviations from the best orbit are exaggerated ×{exaggeration} so they \
             are visible: at the true scale they are far smaller than the planets' orbits."
        )
    };
    format!(
        "Orbit uncertainty drawn from the N-body fit covariance ({}); {} of {} sampled clones are \
         closed orbits. {scale} The \"@ now\" cloud keeps {} of {} clones — the far-flung ones \
         are hidden so they do not blow up the plot's scale.",
        cloud.detail, cloud.n_clones, cloud.n_sampled, cloud.n_kept_now, cloud.n_clones,
    )
}

/// The ecliptic plane, as a light wireframe: two concentric rings and a few
/// spokes around the Sun.
///
/// A wireframe and not a filled disc on purpose: in plotly's 3D scenes the
/// hovered element is the *nearest* drawn object, and a filled surface is
/// the nearest object under the cursor almost everywhere — it swallowed the
/// hover of every other trace. Thin lines only compete where they pass.
///
/// # Arguments
///
/// * `summary` — the object's orbit summary (sets the wireframe's radius).
///
/// # Returns
///
/// Two traces sharing one legend entry: the rings and the spokes.
fn ecliptic_plane_traces(summary: &OrbitSummary3D) -> Vec<Trace3D> {
    let radius = (summary.aphelion_au * ECLIPTIC_RADIUS_MARGIN).max(ECLIPTIC_MIN_RADIUS_AU);
    let rings = [radius, radius / 2.0];

    let spoke = |k: usize| {
        let angle = std::f64::consts::TAU * k as f64 / ECLIPTIC_SPOKES as f64;
        [[0.0; 3], [radius * angle.cos(), radius * angle.sin(), 0.0]]
    };

    vec![
        Trace3D {
            name: "Ecliptic plane".to_string(),
            color: ECLIPTIC_COLOR.to_string(),
            style: TraceStyle::ClosedPolylines(ECLIPTIC_RING_SEGMENTS),
            source: TraceSource::Landmark(LandmarkKind::EclipticPlane),
            points: rings
                .iter()
                .flat_map(|&r| geometry::ecliptic_disc_outline(r, ECLIPTIC_RING_SEGMENTS))
                .collect(),
            hover_text: rings
                .iter()
                .map(|r| format!("Ecliptic plane<br>(ring at {r:.2} AU from the Sun)"))
                .collect(),
        },
        Trace3D {
            name: "Ecliptic plane (spokes)".to_string(),
            color: ECLIPTIC_COLOR.to_string(),
            style: TraceStyle::SolidPolylines(2),
            source: TraceSource::Landmark(LandmarkKind::EclipticSpokes),
            points: (0..ECLIPTIC_SPOKES).flat_map(spoke).collect(),
            hover_text: vec!["Ecliptic plane".to_string(); ECLIPTIC_SPOKES],
        },
    ]
}

/// The orbit's inclination to the ecliptic: a pie-slice at the Sun spanning
/// the angle between the ecliptic and the orbital plane (measured
/// perpendicular to the line of nodes), whose value is in the hover of the
/// slice and of an icon on its arc — no text label cluttering the plot.
///
/// # Arguments
///
/// * `summary` — the object's orbit summary.
///
/// # Returns
///
/// The pie-slice outline and an icon on its arc — empty if the orbit is
/// (nearly) in the ecliptic.
fn inclination_traces(summary: &OrbitSummary3D) -> Vec<Trace3D> {
    if summary.inclination_deg < INCLINATION_MIN_DRAWN_DEG {
        return Vec::new();
    }

    let radius = (summary.perihelion_au * INCLINATION_WEDGE_PERIHELION_FRACTION)
        .max(INCLINATION_WEDGE_MIN_RADIUS_AU);
    let wedge = geometry::inclination_wedge(
        summary.ascending_node_longitude_deg,
        summary.inclination_deg,
        radius,
        INCLINATION_ARC_SEGMENTS,
    );
    let value = format!("i = {:.2}°", summary.inclination_deg);
    let hover =
        format!("Inclination<br>{value}<br>(angle between the ecliptic and the orbital plane)");

    vec![
        Trace3D {
            name: format!("Inclination ({value})"),
            color: INCLINATION_COLOR.to_string(),
            style: TraceStyle::SolidPolylines(wedge.outline.len()),
            source: TraceSource::Landmark(LandmarkKind::Inclination),
            points: wedge.outline,
            hover_text: vec![hover.clone()],
        },
        Trace3D {
            name: "Inclination icon".to_string(),
            color: INCLINATION_COLOR.to_string(),
            style: TraceStyle::Markers,
            source: TraceSource::Landmark(LandmarkKind::InclinationIcon),
            points: vec![wedge.arc_midpoint],
            hover_text: vec![hover],
        },
    ]
}

/// Whether a body belongs to the outer solar system (Jupiter and beyond),
/// judged by its semi-major axis so it does not depend on names.
///
/// # Arguments
///
/// * `body` — a planet or perturber.
///
/// # Returns
///
/// `true` when its semi-major axis exceeds [`OUTER_SOLAR_SYSTEM_MIN_AU`].
fn is_outer_solar_system(body: &Body3D) -> bool {
    body.elements.semi_major_axis_au > OUTER_SOLAR_SYSTEM_MIN_AU
}

/// Whether the outer solar system should be shown by default for an object:
/// only when the object's orbit reaches out to Jupiter's, i.e. its
/// semi-major axis exceeds Jupiter's — a main-belt asteroid has no use for
/// Pluto stretching the plot's scale.
///
/// # Arguments
///
/// * `object_semi_major_axis_au` — the object's semi-major axis, AU.
/// * `planets` — the planets/perturbers; Jupiter's semi-major axis is read
///   from it.
///
/// # Returns
///
/// `true` if the object's semi-major axis exceeds Jupiter's.
fn auto_show_outer_solar_system(object_semi_major_axis_au: f64, planets: &[Body3D]) -> bool {
    let jupiter = planets
        .iter()
        .find(|b| b.name == "Jupiter")
        .map_or(FALLBACK_JUPITER_SEMI_MAJOR_AXIS_AU, |b| {
            b.elements.semi_major_axis_au
        });
    object_semi_major_axis_au > jupiter
}

/// The planets to draw: all of them, or only the inner solar system and the
/// main-belt perturbers.
///
/// # Arguments
///
/// * `planets` — every planet/perturber.
/// * `show_outer` — whether to keep the outer solar system.
///
/// # Returns
///
/// The bodies to plot, in the original order.
fn visible_planets(planets: &[Body3D], show_outer: bool) -> Vec<Body3D> {
    planets
        .iter()
        .filter(|b| show_outer || !is_outer_solar_system(b))
        .cloned()
        .collect()
}

/// The small note next to the outer-solar-system switch explaining its
/// automatic state.
///
/// # Arguments
///
/// * `object_semi_major_axis_au` — the object's semi-major axis, AU.
/// * `planets` — the planets/perturbers.
///
/// # Returns
///
/// E.g. `"auto: a = 3.16 AU, inside Jupiter's orbit"`.
fn outer_switch_note(object_semi_major_axis_au: f64, planets: &[Body3D]) -> String {
    let side = if auto_show_outer_solar_system(object_semi_major_axis_au, planets) {
        "beyond"
    } else {
        "inside"
    };
    format!("auto: a = {object_semi_major_axis_au:.2} AU, {side} Jupiter's orbit")
}

/// Every trace of the lineage 3D view.
///
/// # Arguments
///
/// * `data` — the object's orbit, observations and the planets.
/// * `show_uncertainty` — whether to include the orbit uncertainty traces
///   (when `data` has an uncertainty cloud).
/// * `exaggeration` — factor applied to the uncertainty clones' deviations
///   (`1.0` = true scale); unused when the uncertainty is not shown.
/// * `show_outer` — whether to draw the outer solar system (Jupiter and
///   beyond) or only the inner planets and main-belt perturbers.
/// * `show_ecliptic` — whether to draw the ecliptic plane wireframe (the
///   inclination pie-slice is always drawn).
///
/// # Returns
///
/// The orbit, the object's sphere (with the orbit summary as its hover), the
/// observation crosses and sight lines (when there are observations), the
/// orbit landmarks, the ecliptic plane and inclination, the uncertainty
/// (when shown and available), and the planets.
fn lineage_traces(
    data: &LineageOrbit3D,
    show_uncertainty: bool,
    exaggeration: f64,
    show_outer: bool,
    show_ecliptic: bool,
) -> Vec<Trace3D> {
    let mut traces = vec![
        Trace3D {
            name: "Orbit".to_string(),
            color: OBJECT_COLOR.to_string(),
            style: TraceStyle::Line,
            source: TraceSource::TrackedObject,
            points: data.object_orbit.clone(),
            hover_text: Vec::new(),
        },
        Trace3D {
            name: "Current position".to_string(),
            color: OBJECT_COLOR.to_string(),
            style: TraceStyle::Markers,
            source: TraceSource::FocusObject,
            points: vec![data.object_position],
            hover_text: vec![object_hover_text(&data.summary)],
        },
    ];
    if !data.observation_points.is_empty() {
        traces.push(Trace3D {
            name: "Sight lines".to_string(),
            color: SIGHT_LINE_COLOR.to_string(),
            style: TraceStyle::DashedPolylines(3),
            source: TraceSource::Observation,
            points: sight_line_points(&data.observation_points),
            hover_text: sight_line_hover_texts(&data.observation_points),
        });
        traces.push(Trace3D {
            name: "Observations".to_string(),
            color: OBSERVATION_COLOR.to_string(),
            style: TraceStyle::Markers,
            source: TraceSource::Observation,
            points: data.observation_points.iter().map(|o| o.position).collect(),
            hover_text: data
                .observation_points
                .iter()
                .enumerate()
                .map(|(i, o)| observation_hover_text(i, o))
                .collect(),
        });
    }
    traces.extend(landmark_traces(&data.summary));
    if show_ecliptic {
        traces.extend(ecliptic_plane_traces(&data.summary));
    }
    traces.extend(inclination_traces(&data.summary));
    if let (true, Some(cloud)) = (show_uncertainty, &data.uncertainty) {
        traces.extend(uncertainty_traces(cloud, exaggeration));
    }
    traces.extend(planet_traces(&visible_planets(&data.planets, show_outer)));
    traces
}

#[component]
pub fn LineageOrbit3DTab(lineage_id: String) -> Element {
    let resource_lineage_id = lineage_id.clone();
    let orbit3d = use_resource(use_reactive!(|(resource_lineage_id,)| get_lineage_orbit3d(
        resource_lineage_id
    )));

    let mut show_uncertainty = use_signal(|| SHOW_UNCERTAINTY_DEFAULT);
    let mut exaggeration = use_signal(|| EXAGGERATION_DEFAULT);
    // `None` until the user flips the switch: the outer solar system then
    // follows the object's semi-major axis.
    let mut outer_override = use_signal(|| None::<bool>);
    let mut show_ecliptic = use_signal(|| SHOW_ECLIPTIC_DEFAULT);
    let show_outer = move |data: &LineageOrbit3D| {
        outer_override().unwrap_or_else(|| {
            auto_show_outer_solar_system(data.summary.semi_major_axis_au, &data.planets)
        })
    };

    // Toggling the switch only changes which traces are built — the data is
    // not reloaded, and the plot redraws through its usual `traces` diff.
    let traces = use_memo(move || match &*orbit3d.read() {
        Some(Ok(Some(data))) => lineage_traces(
            data,
            show_uncertainty(),
            exaggeration(),
            show_outer(data),
            show_ecliptic(),
        ),
        _ => Vec::new(),
    });

    rsx! {
        match &*orbit3d.read() {
            Some(Ok(Some(data))) => rsx! {
                div { class: "flex flex-col gap-1 mb-2",
                    div { class: "flex flex-wrap items-center gap-x-6 gap-y-1",
                        label { class: "flex items-center gap-2 cursor-pointer text-sm w-fit",
                            input {
                                r#type: "checkbox",
                                class: "toggle toggle-sm",
                                checked: show_outer(data),
                                onchange: move |evt| outer_override.set(Some(evt.checked())),
                            }
                            span { "Show outer solar system" }
                            if outer_override().is_none() {
                                span { class: "text-xs text-base-content/60",
                                    "({outer_switch_note(data.summary.semi_major_axis_au, &data.planets)})"
                                }
                            }
                        }
                        label { class: "flex items-center gap-2 cursor-pointer text-sm w-fit",
                            input {
                                r#type: "checkbox",
                                class: "toggle toggle-sm",
                                checked: show_ecliptic(),
                                onchange: move |evt| show_ecliptic.set(evt.checked()),
                            }
                            span { "Show ecliptic plane" }
                        }
                        PlotGlossary {}
                    }
                    match &data.uncertainty {
                        Some(cloud) => rsx! {
                            div { class: "flex flex-wrap items-center gap-x-6 gap-y-1",
                                label { class: "flex items-center gap-2 cursor-pointer text-sm w-fit",
                                    input {
                                        r#type: "checkbox",
                                        class: "toggle toggle-sm",
                                        checked: show_uncertainty(),
                                        onchange: move |evt| show_uncertainty.set(evt.checked()),
                                    }
                                    span { "Show orbit uncertainty" }
                                }
                                if show_uncertainty() {
                                    label { class: "flex items-center gap-2 text-sm",
                                        span { "Deviations exaggerated" }
                                        select {
                                            class: "select select-xs w-24",
                                            onchange: move |evt| {
                                                if let Ok(factor) = evt.value().parse::<f64>() {
                                                    exaggeration.set(factor);
                                                }
                                            },
                                            for factor in EXAGGERATION_CHOICES {
                                                option { value: "{factor}", selected: factor == exaggeration(), "×{factor}" }
                                            }
                                        }
                                    }
                                }
                            }
                            if show_uncertainty() {
                                p { class: "text-xs text-base-content/60", "{uncertainty_caption(cloud, exaggeration())}" }
                            }
                        },
                        None => rsx! {
                            p { class: "text-xs text-base-content/60",
                                "Orbit uncertainty is not available: "
                                "{data.uncertainty_unavailable_reason.clone().unwrap_or_default()}."
                            }
                        },
                    }
                }
                Scatter3dPlot {
                    plot_id: "lineage-orbit3d-plot-div",
                    traces: traces(),
                    sphere_scale: SPHERE_SCALE,
                }
            },
            Some(Ok(None)) => rsx! {
                div { class: "alert alert-info",
                    "This lineage has no orbit fit yet — run one from the fit page to see its 3D orbit."
                }
            },
            Some(Err(e)) => rsx! {
                div { class: "alert alert-error", "Failed to load the 3D orbit: {e}" }
            },
            None => rsx! {
                div { class: "flex justify-center py-12",
                    span { class: "loading loading-spinner loading-lg" }
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(position: [f64; 3], observer_position: [f64; 3]) -> ObservationPoint3D {
        ObservationPoint3D {
            position,
            observer_position,
            mjd_tt: 51_544.5,
            phase_angle_deg: 17.256,
            magnitude: 19.4213,
            mag_err: 0.0812,
            filter: 1,
            mpc_code: "X05".to_string(),
            elongation_deg: 112.34,
            absolute_magnitude: Some(17.9),
            heliocentric_distance_au: 2.345_678,
            topocentric_distance_au: 1.234_5,
        }
    }

    #[test]
    fn sight_lines_run_sun_to_observation_to_observer_per_observation() {
        let a = point([2.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let b = point([0.0, 3.0, 0.0], [0.0, 1.0, 0.0]);

        assert_eq!(
            sight_line_points(&[a, b]),
            [
                [0.0; 3],
                [2.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0; 3],
                [0.0, 3.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        );
    }

    #[test]
    fn observation_hover_text_shows_rank_utc_date_and_both_distances() {
        // `point`'s epoch, 2000-01-01 12:00:00 TT, is 11:58:55.816 UTC.
        let point = point([1.0, 2.0, 3.0], [0.0; 3]);

        let text = observation_hover_text(2, &point);

        assert!(text.contains("Observation #3"), "{text}");
        assert!(text.contains("2000-01-01T11:58:55"), "{text}");
        assert!(text.contains("Heliocentric distance: 2.3457 AU"), "{text}");
        assert!(text.contains("Topocentric distance: 1.2345 AU"), "{text}");
        assert!(text.contains("Phase angle: 17.26°"), "{text}");
        assert!(
            text.contains("Magnitude: 19.42 ± 0.08 (g band, MPC X05)"),
            "{text}"
        );
        assert!(text.contains("Solar elongation: 112.3°"), "{text}");
        assert!(text.contains("H ≈ 17.90 (g band, G = 0.15"), "{text}");
    }

    #[test]
    fn observation_hover_text_flags_an_unavailable_absolute_magnitude() {
        let mut p = point([1.0, 2.0, 3.0], [0.0; 3]);
        p.absolute_magnitude = None;
        assert!(observation_hover_text(0, &p).contains("H: n/a"));
    }

    fn summary() -> OrbitSummary3D {
        use crate::orbit3d::geometry::{Landmarks, Moid};
        use crate::orbit3d::types::PlanetMoid3D;

        OrbitSummary3D {
            semi_major_axis_au: 3.163,
            eccentricity: 0.2402,
            inclination_deg: 12.34,
            ascending_node_longitude_deg: 80.0,
            perihelion_au: 2.4032,
            aphelion_au: 3.9228,
            period_years: 5.62,
            sun_distance_au: 2.9,
            earth_distance_au: Some(2.1),
            landmarks: Landmarks {
                perihelion: [2.4, 0.0, 0.0],
                aphelion: [-3.9, 0.0, 0.0],
                ascending_node: [0.0, 2.5, 0.0],
                descending_node: [0.0, -3.0, 0.0],
            },
            moids: vec![
                PlanetMoid3D {
                    name: "Earth".to_string(),
                    moid: Moid {
                        distance_au: 1.234,
                        point_a: [1.0, 1.0, 0.0],
                        point_b: [1.0, 0.0, 0.0],
                    },
                },
                PlanetMoid3D {
                    name: "Mars".to_string(),
                    moid: Moid {
                        distance_au: 0.5,
                        point_a: [1.5, 1.0, 0.0],
                        point_b: [1.5, 0.5, 0.0],
                    },
                },
            ],
        }
    }

    #[test]
    fn object_hover_text_shows_the_orbit_numbers_distances_and_moids() {
        let text = object_hover_text(&summary());
        assert!(
            text.contains("a = 3.163 AU, e = 0.2402, i = 12.34°"),
            "{text}"
        );
        assert!(
            text.contains("q = 2.403 AU, Q = 3.923 AU, P = 5.62 yr"),
            "{text}"
        );
        assert!(text.contains("Distance to the Sun: 2.900 AU"), "{text}");
        assert!(text.contains("Distance to the Earth: 2.100 AU"), "{text}");
        assert!(text.contains("Earth: 1.2340 AU"), "{text}");
        assert!(text.contains("Mars: 0.5000 AU"), "{text}");
    }

    #[test]
    fn a_large_moid_is_written_in_astronomical_units() {
        assert_eq!(format_moid(1.234), "1.2340 AU");
        assert_eq!(format_moid(0.05), "0.0500 AU");
    }

    #[test]
    fn a_small_moid_is_written_in_lunar_distances_with_the_au_for_reference() {
        // 0.01 AU = 1_495_978.7 km = 3.89 LD.
        assert_eq!(format_moid(0.01), "3.89 LD (0.0100 AU)");
        // Just under the AU threshold (0.049 AU ≈ 19.07 LD).
        assert!(format_moid(0.049).starts_with("19.07 LD"));
    }

    #[test]
    fn a_moid_under_one_lunar_distance_is_written_in_kilometres() {
        // 0.002 AU = 299_195.7 km = 0.778 LD.
        assert_eq!(format_moid(0.002), "299,196 km (0.778 LD)");
        assert_eq!(format_moid(0.0), "0 km (0.000 LD)");
    }

    /// One lunar distance is the boundary: exactly 1 LD reads in LD, just
    /// under it in km.
    #[test]
    fn the_lunar_distance_and_kilometre_units_switch_at_one_lunar_distance() {
        let one_ld_au = geometry::KM_PER_LUNAR_DISTANCE / geometry::KM_PER_AU;
        assert!(format_moid(one_ld_au * 1.0001).contains(" LD ("));
        assert!(format_moid(one_ld_au * 0.9999).contains(" km ("));
    }

    #[test]
    fn thousands_separators_group_by_three_digits() {
        assert_eq!(with_thousands_separators(0), "0");
        assert_eq!(with_thousands_separators(999), "999");
        assert_eq!(with_thousands_separators(1_000), "1,000");
        assert_eq!(with_thousands_separators(384_400), "384,400");
        assert_eq!(with_thousands_separators(12_345_678), "12,345,678");
    }

    #[test]
    fn the_earth_moid_hover_uses_the_unit_that_fits_its_size() {
        let mut close = summary();
        close.moids[0].moid.distance_au = 0.002;
        let traces = landmark_traces(&close);

        let markers = traces.iter().find(|t| t.name == "MOID with Earth").unwrap();
        assert!(
            markers.hover_text[0].contains("299,196 km"),
            "{:?}",
            markers.hover_text
        );
        let segment = traces
            .iter()
            .find(|t| t.name == "MOID with Earth (segment)")
            .unwrap();
        assert!(segment.hover_text[0].contains("299,196 km"));
        assert!(object_hover_text(&close).contains("Earth: 299,196 km"));
    }

    #[test]
    fn landmark_traces_cover_apsides_nodes_line_of_nodes_and_earth_moid() {
        let traces = landmark_traces(&summary());
        let names: Vec<&str> = traces.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "Perihelion",
                "Aphelion",
                "Nodes",
                "Line of nodes",
                "MOID with Earth",
                "MOID with Earth (segment)"
            ]
        );
        // Every marker point has its own hover text.
        for t in traces.iter().filter(|t| t.style == TraceStyle::Markers) {
            assert_eq!(t.points.len(), t.hover_text.len(), "{}", t.name);
        }
    }

    #[test]
    fn landmark_traces_skip_the_moid_when_the_earth_is_missing() {
        let mut s = summary();
        s.moids.retain(|m| m.name != "Earth");
        assert_eq!(landmark_traces(&s).len(), 4);
    }

    fn cloud() -> UncertaintyCloud3D {
        UncertaintyCloud3D {
            detail: "differential correction, normalised RMS 0.87".to_string(),
            n_sampled: 300,
            n_clones: 290,
            n_kept_now: 271,
            center_last_observation: [1.0, 0.0, 0.0],
            center_now: [2.0, 0.0, 0.0],
            best_orbit: vec![[1.0, 0.0, 0.0]; UNCERTAINTY_ORBIT_SAMPLES],
            at_last_observation: vec![[1.1, 0.0, 0.0]; 290],
            at_now: vec![[2.1, 0.0, 0.0]; 271],
            orbits: vec![vec![[1.1, 0.0, 0.0]; UNCERTAINTY_ORBIT_SAMPLES]; 20],
        }
    }

    fn lineage(uncertainty: Option<UncertaintyCloud3D>) -> LineageOrbit3D {
        LineageOrbit3D {
            object_position: [1.0, 2.0, 0.0],
            object_orbit: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            summary: summary(),
            uncertainty_unavailable_reason: uncertainty
                .is_none()
                .then(|| "the N-body fit did not converge".to_string()),
            uncertainty,
            observation_points: Vec::new(),
            planets: Vec::new(),
        }
    }

    fn uncertainty_trace_names(traces: &[Trace3D]) -> Vec<&str> {
        traces
            .iter()
            .filter(|t| t.source == TraceSource::Uncertainty)
            .map(|t| t.name.as_str())
            .collect()
    }

    /// The switch off ⇒ no uncertainty trace at all (the whole point of the
    /// switch: a lighter plot).
    #[test]
    fn lineage_traces_omit_the_uncertainty_when_the_switch_is_off() {
        let data = lineage(Some(cloud()));
        assert!(
            uncertainty_trace_names(&lineage_traces(&data, false, 10.0, true, true)).is_empty()
        );
    }

    #[test]
    fn lineage_traces_include_orbits_and_both_clouds_when_the_switch_is_on() {
        let data = lineage(Some(cloud()));
        let traces = lineage_traces(&data, true, 1.0, true, true);

        let names = uncertainty_trace_names(&traces);
        assert_eq!(names.len(), 3, "{names:?}");
        assert!(names.iter().any(|n| n.contains("@ last observation (290)")));
        assert!(names.iter().any(|n| n.contains("@ now (271/290 kept)")));
        assert!(
            names.iter().all(|n| !n.contains('×')),
            "true scale needs no factor: {names:?}"
        );

        let orbits = traces
            .iter()
            .find(|t| matches!(t.style, TraceStyle::ClosedPolylines(_)))
            .unwrap();
        assert_eq!(orbits.points.len(), 20 * UNCERTAINTY_ORBIT_SAMPLES);
        assert_eq!(
            orbits.style,
            TraceStyle::ClosedPolylines(UNCERTAINTY_ORBIT_SAMPLES)
        );
    }

    #[test]
    fn lineage_traces_without_a_cloud_have_no_uncertainty_even_when_the_switch_is_on() {
        let data = lineage(None);
        assert!(uncertainty_trace_names(&lineage_traces(&data, true, 10.0, true, true)).is_empty());
    }

    #[test]
    fn the_uncertainty_switch_does_not_change_any_other_trace() {
        let data = lineage(Some(cloud()));
        let on = lineage_traces(&data, true, 10.0, true, true);
        let off = lineage_traces(&data, false, 10.0, true, true);
        let others = |t: &[Trace3D]| -> Vec<String> {
            t.iter()
                .filter(|t| t.source != TraceSource::Uncertainty)
                .map(|t| t.name.clone())
                .collect()
        };
        assert_eq!(others(&on), others(&off));
    }

    #[test]
    fn exaggeration_scales_deviations_from_the_center_and_states_the_factor() {
        let traces = uncertainty_traces(&cloud(), 10.0);

        let last = traces
            .iter()
            .find(|t| t.name.contains("@ last observation"))
            .unwrap();
        // A clone 0.1 AU from its center (1, 0, 0) lands 1 AU away.
        assert!(
            (last.points[0][0] - 2.0).abs() < 1e-12,
            "{:?}",
            last.points[0]
        );
        assert!(last.name.contains("deviations ×10"), "{}", last.name);

        let now = traces.iter().find(|t| t.name.contains("@ now")).unwrap();
        assert!(
            (now.points[0][0] - 3.0).abs() < 1e-12,
            "{:?}",
            now.points[0]
        );

        let orbits = traces
            .iter()
            .find(|t| matches!(t.style, TraceStyle::ClosedPolylines(_)))
            .unwrap();
        assert!((orbits.points[0][0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn exaggeration_of_one_leaves_positions_unchanged() {
        let points = [[1.5, -2.0, 0.3], [0.0, 1.0, 2.0]];
        assert_eq!(exaggerate_about(&points, [9.0, 9.0, 9.0], 1.0), points);
    }

    #[test]
    fn caption_states_the_fit_source_and_the_exaggeration() {
        let scaled = uncertainty_caption(&cloud(), 100.0);
        assert!(scaled.contains("N-body fit covariance"), "{scaled}");
        assert!(scaled.contains("290 of 300"), "{scaled}");
        assert!(scaled.contains("exaggerated ×100"), "{scaled}");
        assert!(scaled.contains("271 of 290"), "{scaled}");

        let true_scale = uncertainty_caption(&cloud(), 1.0);
        assert!(true_scale.contains("true scale"), "{true_scale}");
        assert!(!true_scale.contains("exaggerated"), "{true_scale}");
    }

    #[test]
    fn the_ecliptic_plane_is_a_light_wireframe_covering_the_orbit() {
        let traces = ecliptic_plane_traces(&summary());
        let names: Vec<&str> = traces.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["Ecliptic plane", "Ecliptic plane (spokes)"]);

        // Two rings, the outer one 10% beyond the 3.9228 AU aphelion.
        let rings = &traces[0];
        assert_eq!(
            rings.style,
            TraceStyle::ClosedPolylines(ECLIPTIC_RING_SEGMENTS)
        );
        assert_eq!(rings.points.len(), 2 * ECLIPTIC_RING_SEGMENTS);
        let outer = geometry::distance(rings.points[0], [0.0; 3]);
        let inner = geometry::distance(rings.points[ECLIPTIC_RING_SEGMENTS], [0.0; 3]);
        assert!(
            (outer - 3.9228 * ECLIPTIC_RADIUS_MARGIN).abs() < 1e-9,
            "{outer}"
        );
        assert!((inner - outer / 2.0).abs() < 1e-9);
        assert!(rings.points.iter().all(|p| p[2] == 0.0));

        // Spokes: two-point groups from the Sun to the outer ring.
        let spokes = &traces[1];
        assert_eq!(spokes.style, TraceStyle::SolidPolylines(2));
        assert_eq!(spokes.points.len(), 2 * ECLIPTIC_SPOKES);
        for spoke in spokes.points.chunks(2) {
            assert_eq!(spoke[0], [0.0; 3]);
            assert!((geometry::distance(spoke[1], [0.0; 3]) - outer).abs() < 1e-9);
        }
    }

    /// Every line of the wireframe says what it is on hover — the hovered
    /// element in a plotly 3D scene is the nearest drawn object, so a line
    /// with no hover would silence whatever it passes near.
    #[test]
    fn every_ecliptic_line_has_a_hover_text() {
        let traces = ecliptic_plane_traces(&summary());
        assert_eq!(traces[0].hover_text.len(), 2);
        assert_eq!(traces[1].hover_text.len(), ECLIPTIC_SPOKES);
        for text in traces.iter().flat_map(|t| &t.hover_text) {
            assert!(text.contains("Ecliptic plane"), "{text}");
        }
    }

    #[test]
    fn the_ecliptic_wireframe_has_a_minimum_radius() {
        let mut inner = summary();
        inner.aphelion_au = 0.4;
        let rings = &ecliptic_plane_traces(&inner)[0];
        let radius = geometry::distance(rings.points[0], [0.0; 3]);
        assert!((radius - ECLIPTIC_MIN_RADIUS_AU).abs() < 1e-9, "{radius}");
    }

    #[test]
    fn the_inclination_slice_and_its_icon_show_the_value_on_hover() {
        let traces = inclination_traces(&summary());
        let names: Vec<&str> = traces.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["Inclination (i = 12.34°)", "Inclination icon"]);

        assert!(
            matches!(traces[0].style, TraceStyle::SolidPolylines(n) if n == traces[0].points.len())
        );
        for trace in &traces {
            assert!(trace.hover_text[0].contains("i = 12.34°"), "{}", trace.name);
        }
        assert_eq!(
            traces[1].source,
            TraceSource::Landmark(LandmarkKind::InclinationIcon)
        );
        // The icon sits on the arc, at the slice's radius.
        let slice_radius = geometry::distance(traces[0].points[0], [0.0; 3]);
        let icon_radius = geometry::distance(traces[1].points[0], [0.0; 3]);
        assert!((icon_radius - slice_radius).abs() < 1e-9);
    }

    #[test]
    fn an_orbit_in_the_ecliptic_gets_no_inclination_slice() {
        let mut flat = summary();
        flat.inclination_deg = 0.0;
        assert!(inclination_traces(&flat).is_empty());
    }

    #[test]
    fn the_ecliptic_switch_only_controls_the_wireframe() {
        let data = lineage(Some(cloud()));
        let names = |show_ecliptic: bool| -> Vec<String> {
            lineage_traces(&data, false, 10.0, true, show_ecliptic)
                .into_iter()
                .map(|t| t.name)
                .collect()
        };
        let on = names(true);
        let off = names(false);

        assert!(on.iter().any(|n| n == "Ecliptic plane"));
        assert!(!off.iter().any(|n| n.starts_with("Ecliptic plane")));
        // The inclination slice stays either way.
        assert!(on.iter().any(|n| n.starts_with("Inclination (")));
        assert!(off.iter().any(|n| n.starts_with("Inclination (")));
    }

    #[test]
    fn no_guide_line_is_left_without_a_hover_text() {
        let mut data = lineage(Some(cloud()));
        data.observation_points = vec![
            ObservationPoint3D {
                position: [2.0, 0.0, 0.0],
                observer_position: [1.0, 0.0, 0.0],
                mjd_tt: 51_544.5,
                phase_angle_deg: 10.0,
                magnitude: 20.0,
                mag_err: 0.1,
                filter: 1,
                mpc_code: "X05".to_string(),
                elongation_deg: 100.0,
                absolute_magnitude: Some(18.0),
                heliocentric_distance_au: 2.0,
                topocentric_distance_au: 1.0,
            };
            3
        ];

        for trace in lineage_traces(&data, true, 10.0, true, true) {
            let groups = match trace.style {
                TraceStyle::DashedPolylines(n)
                | TraceStyle::ClosedPolylines(n)
                | TraceStyle::SolidPolylines(n) => trace.points.len().div_ceil(n),
                _ => continue,
            };
            assert_eq!(
                trace.hover_text.len(),
                groups,
                "{}: one hover text per line group",
                trace.name
            );
        }
    }

    #[test]
    fn sight_line_hover_texts_name_the_observation_and_its_two_ranges() {
        let point = |h: f64, t: f64| ObservationPoint3D {
            position: [h, 0.0, 0.0],
            observer_position: [h - t, 0.0, 0.0],
            mjd_tt: 51_544.5,
            phase_angle_deg: 10.0,
            magnitude: 20.0,
            mag_err: 0.1,
            filter: 1,
            mpc_code: "X05".to_string(),
            elongation_deg: 100.0,
            absolute_magnitude: None,
            heliocentric_distance_au: h,
            topocentric_distance_au: t,
        };
        let texts = sight_line_hover_texts(&[point(2.0, 1.0), point(3.0, 1.5)]);
        assert_eq!(texts.len(), 2);
        assert!(texts[0].contains("observation #1"));
        assert!(texts[0].contains("Sun to object: 2.000 AU"));
        assert!(texts[1].contains("observation #2"));
        assert!(texts[1].contains("Observer to object: 1.500 AU"));
    }

    fn planet(name: &str, semi_major_axis_au: f64) -> Body3D {
        use crate::orbit3d::geometry::Keplerian;
        use crate::orbit3d::types::BodyKind;
        Body3D {
            name: name.to_string(),
            kind: BodyKind::Planet,
            position: [semi_major_axis_au, 0.0, 0.0],
            orbit: Vec::new(),
            elements: Keplerian {
                epoch_mjd_tt: 60_000.0,
                semi_major_axis_au,
                eccentricity: 0.0,
                inclination_deg: 0.0,
                ascending_node_longitude_deg: 0.0,
                periapsis_argument_deg: 0.0,
                mean_anomaly_deg: 0.0,
            },
        }
    }

    fn solar_system() -> Vec<Body3D> {
        vec![
            planet("Mercury", 0.39),
            planet("Earth", 1.0),
            planet("Mars", 1.52),
            planet("Ceres", 2.77),
            planet("Jupiter", 5.2),
            planet("Saturn", 9.5),
            planet("Neptune", 30.1),
            planet("Pluto", 39.5),
        ]
    }

    #[test]
    fn the_outer_solar_system_is_jupiter_and_beyond_not_the_main_belt() {
        let outer: Vec<String> = solar_system()
            .iter()
            .filter(|b| is_outer_solar_system(b))
            .map(|b| b.name.clone())
            .collect();
        assert_eq!(outer, ["Jupiter", "Saturn", "Neptune", "Pluto"]);
    }

    #[test]
    fn hiding_the_outer_solar_system_keeps_the_inner_planets_and_belt_perturbers() {
        let names = |v: Vec<Body3D>| v.into_iter().map(|b| b.name).collect::<Vec<_>>();
        assert_eq!(
            names(visible_planets(&solar_system(), false)),
            ["Mercury", "Earth", "Mars", "Ceres"]
        );
        assert_eq!(names(visible_planets(&solar_system(), true)).len(), 8);
    }

    #[test]
    fn the_switch_defaults_to_the_outer_solar_system_only_beyond_jupiters_orbit() {
        let planets = solar_system();
        // A main-belt asteroid: inner solar system only.
        assert!(!auto_show_outer_solar_system(2.9, &planets));
        // Just inside Jupiter's semi-major axis: still inner only.
        assert!(!auto_show_outer_solar_system(5.2, &planets));
        // A Centaur / distant object: show everything.
        assert!(auto_show_outer_solar_system(13.0, &planets));
        assert!(auto_show_outer_solar_system(42.0, &planets));
    }

    #[test]
    fn the_switch_default_falls_back_to_a_nominal_jupiter_without_one_in_the_list() {
        assert!(!auto_show_outer_solar_system(3.0, &[]));
        assert!(auto_show_outer_solar_system(6.0, &[]));
    }

    #[test]
    fn the_switch_note_says_which_side_of_jupiter_the_orbit_is_on() {
        let planets = solar_system();
        assert_eq!(
            outer_switch_note(3.1630, &planets),
            "auto: a = 3.16 AU, inside Jupiter's orbit"
        );
        assert!(outer_switch_note(30.0, &planets).contains("beyond Jupiter's orbit"));
    }

    fn planet_trace_names(traces: &[Trace3D]) -> Vec<&str> {
        traces
            .iter()
            .filter(|t| t.source == TraceSource::Planet)
            .map(|t| t.name.as_str())
            .collect()
    }

    #[test]
    fn lineage_traces_drop_the_outer_planets_when_the_switch_is_off() {
        let mut data = lineage(None);
        data.planets = solar_system();

        let with_outer = lineage_traces(&data, false, 1.0, true, true);
        let inner_only = lineage_traces(&data, false, 1.0, false, true);

        let with_outer = planet_trace_names(&with_outer);
        let inner_only = planet_trace_names(&inner_only);
        assert!(with_outer.contains(&"Pluto") && with_outer.contains(&"Jupiter (orbit)"));
        assert!(inner_only.contains(&"Earth") && inner_only.contains(&"Ceres"));
        for outer in ["Jupiter", "Saturn", "Neptune", "Pluto"] {
            assert!(
                !inner_only.iter().any(|n| n.starts_with(outer)),
                "{outer} still drawn: {inner_only:?}"
            );
        }
    }
}
