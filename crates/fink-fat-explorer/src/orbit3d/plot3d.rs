//! The shared 3D scatter/line plot component both the homepage's population
//! view and the lineage page's single-object view render into.

use dioxus::prelude::*;

#[cfg(target_arch = "wasm32")]
use plotly::{
    common::{HoverInfo, Line, Marker, MarkerSymbol, Mode, Title},
    layout::{AspectMode, Axis, Layout, LayoutScene, Margin},
    mesh3d::Lighting,
    Mesh3D, Plot, Scatter3D,
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
    /// Several separate dashed polylines in one trace (one legend entry,
    /// one toggle): the trace's `points` are consumed in consecutive groups
    /// of this many points, each group drawn as its own polyline with no
    /// segment joining one group to the next — e.g. `3` for a series of
    /// two-segment sight lines. The dashes are cut geometrically
    /// ([`dash_polyline`]) rather than delegated to plotly's `line.dash`: in
    /// `scatter3d` that pattern is scaled by the trace's *total* arc length,
    /// so on a trace of many long lines each dash and gap spans several AU
    /// and every line stops short of its end point.
    DashedPolylines(usize),
    /// Several separate *closed* solid polylines in one trace: like
    /// [`Self::DashedPolylines`], `points` come in consecutive groups of
    /// this many points, but each group is joined back to its own first
    /// point (a full loop) and drawn as a thin solid line — e.g. a series of
    /// sampled orbits.
    ClosedPolylines(usize),
    /// Several separate *open* solid polylines in one trace: the same
    /// grouping as [`Self::ClosedPolylines`] but each group is drawn as
    /// given, not looped back — e.g. an arc, or a pie-slice outline that
    /// closes itself.
    SolidPolylines(usize),
}

/// Which kind of body a [`Trace3D`] represents, driving how its
/// [`TraceStyle::Markers`] points are drawn in [`Scatter3dPlot`]: fink-fat's
/// own tracked objects vastly outnumber the dozen planets/perturbers they
/// share the homepage scene with (thousands of them), so they get a tiny dot
/// while planets are drawn as shaded spheres ([`sphere_mesh`]) that stay
/// distinguishable rather than drowning in the cloud.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceSource {
    /// One of `orbit3d::ephem_provider::TRACKED_BODIES` — a sphere.
    Planet,
    /// One of many fink-fat tracked objects — its current position (a tiny
    /// dot), or its own orbit.
    TrackedObject,
    /// The single object a page is about (the lineage page's 3D tab). Alone
    /// in the scene, so it can afford to be drawn as a sphere like a planet.
    FocusObject,
    /// One real observation of the focus object — a cross.
    Observation,
    /// A characteristic point of the focus object's orbit — a marker whose
    /// symbol depends on its [`LandmarkKind`].
    Landmark(LandmarkKind),
    /// A clone of the focus object's orbit drawn from its covariance — a
    /// small translucent dot, or (with [`TraceStyle::ClosedPolylines`]) a
    /// faint orbit.
    Uncertainty,
}

/// Which characteristic point a [`TraceSource::Landmark`] trace holds; picks
/// the marker symbol so the four kinds read apart at a glance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LandmarkKind {
    /// The perihelion — a diamond.
    Perihelion,
    /// The aphelion — a square.
    Aphelion,
    /// An orbital node — a circle.
    Node,
    /// An end of a minimum-orbit-intersection-distance segment — an open
    /// diamond.
    Moid,
    /// The ecliptic plane's concentric rings.
    EclipticPlane,
    /// The ecliptic plane's radial spokes; shares the rings' legend entry.
    /// The plane is drawn as this light wireframe rather than a filled disc
    /// because in plotly's 3D scenes a filled surface is "nearest object"
    /// under the cursor everywhere and would swallow the hover of every
    /// other trace.
    EclipticSpokes,
    /// The inclination pie-slice between the ecliptic and the orbital
    /// plane.
    Inclination,
    /// An icon on the inclination pie-slice whose hover gives the
    /// inclination value; kept out of the legend.
    InclinationIcon,
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
    /// Custom hover text (plotly HTML: `<br>` breaks lines). Empty means
    /// plotly's default hover for a line/marker trace (trace name +
    /// coordinates) and no hover for the polyline styles.
    ///
    /// The entries follow the trace's unit: one per point for
    /// [`TraceStyle::Markers`]; one per *group* for the polyline styles
    /// ([`TraceStyle::DashedPolylines`], [`TraceStyle::ClosedPolylines`],
    /// [`TraceStyle::SolidPolylines`]), shown wherever that group's line is
    /// hovered; a sphere uses the first entry.
    ///
    /// Giving every guide line a meaningful hover matters: in plotly's 3D
    /// scenes the hovered element is simply the *nearest* drawn object, so a
    /// line without hover text (`hoverinfo: skip`) silences the hover of
    /// whatever it passes near — its own text at least tells the user what
    /// they are pointing at.
    #[serde(default)]
    pub hover_text: Vec<String>,
}

/// Radius, in AU, of the sphere drawn for each planet/perturber's
/// [`TraceStyle::Markers`] trace (see [`sphere_mesh`]).
///
/// A data-space radius rather than a screen-space marker size: plotly's
/// `scatter3d` has no sphere symbol, so a planet is a small shaded
/// `mesh3d` sphere instead. The plot uses `AspectMode::Data`, so the sphere
/// stays round and simply grows/shrinks with the camera zoom. Uniform for
/// every body — planets are told apart by color, not by (unrealistic at this
/// scale) relative size.
const PLANET_SPHERE_RADIUS_AU: f64 = 0.15;
/// Radius, in AU, of the sphere drawn for a page's single focus object
/// ([`TraceSource::FocusObject`]) — clearly bigger than the tiny dot every
/// tracked object otherwise gets, yet a bit smaller than a planet.
const FOCUS_SPHERE_RADIUS_AU: f64 = 0.10;
/// Stroke width of a dashed guide line ([`TraceStyle::DashedPolylines`]).
#[cfg(target_arch = "wasm32")]
const DASHED_LINE_WIDTH: f64 = 2.0;
/// Nominal dash length, AU, of a [`TraceStyle::DashedPolylines`] line — see
/// [`dash_segment`] for how it is adjusted per segment.
const DASH_LENGTH_AU: f64 = 0.10;
/// Gap between dashes, AU, of a [`TraceStyle::DashedPolylines`] line.
const DASH_GAP_AU: f64 = 0.06;
/// Marker size of an uncertainty-cloud clone ([`TraceSource::Uncertainty`]).
#[cfg(target_arch = "wasm32")]
const UNCERTAINTY_MARKER_SIZE: usize = 3;
/// Stroke width of a faint clone orbit ([`TraceStyle::ClosedPolylines`]).
#[cfg(target_arch = "wasm32")]
const CLOSED_POLYLINE_WIDTH: f64 = 1.5;
/// Stroke width of an open solid polyline ([`TraceStyle::SolidPolylines`]).
#[cfg(target_arch = "wasm32")]
const SOLID_POLYLINE_WIDTH: f64 = 3.0;
/// Marker size of an orbit landmark ([`TraceSource::Landmark`]).
#[cfg(target_arch = "wasm32")]
const LANDMARK_MARKER_SIZE: usize = 6;
/// Marker size of an observation cross ([`TraceSource::Observation`]).
#[cfg(target_arch = "wasm32")]
const OBSERVATION_MARKER_SIZE: usize = 5;
/// Stroke width of an observation cross — plotly draws `scatter3d`'s
/// `cross` symbol as lines, so this is what makes it legible.
#[cfg(target_arch = "wasm32")]
const OBSERVATION_MARKER_LINE_WIDTH: f64 = 2.0;

/// The sphere radius a [`TraceSource`]'s markers are drawn with, if that
/// source is drawn as spheres at all.
///
/// # Arguments
///
/// * `source` — which kind of body the trace represents.
///
/// # Returns
///
/// `Some(radius)` in AU for [`TraceSource::Planet`] and
/// [`TraceSource::FocusObject`]; `None` for the sources drawn as plain
/// `scatter3d` markers.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn sphere_radius_au(source: TraceSource) -> Option<f64> {
    match source {
        TraceSource::Planet => Some(PLANET_SPHERE_RADIUS_AU),
        TraceSource::FocusObject => Some(FOCUS_SPHERE_RADIUS_AU),
        TraceSource::TrackedObject
        | TraceSource::Observation
        | TraceSource::Landmark(_)
        | TraceSource::Uncertainty => None,
    }
}
/// Number of latitude bands of a planet sphere — see [`sphere_mesh`].
#[cfg(target_arch = "wasm32")]
const PLANET_SPHERE_LAT_BANDS: usize = 12;
/// Number of longitude segments of a planet sphere — see [`sphere_mesh`].
#[cfg(target_arch = "wasm32")]
const PLANET_SPHERE_LON_SEGMENTS: usize = 20;
/// Marker size for a fink-fat tracked object's [`TraceStyle::Markers`]
/// trace — tiny so a scene with thousands of tracked objects still reads the
/// dozen planet spheres clearly (see [`TraceSource`]'s doc).
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

/// Radius, in AU, of the Sun's sphere — deliberately a bit larger than
/// [`PLANET_SPHERE_RADIUS_AU`] so the Sun reads as the frame's center (and the
/// biggest body) when zoomed into the inner system.
#[cfg(target_arch = "wasm32")]
const SUN_SPHERE_RADIUS_AU: f64 = 0.25;
#[cfg(target_arch = "wasm32")]
const SUN_COLOR: &str = "#ffcc33";
/// Where the scene's single light sits, in the plot's data coordinates.
///
/// Plotly's default is `(1e5, 1e5, 0)` — a light in the ecliptic plane — which
/// leaves every sphere's top (`+z`) cap unlit, seen from the usual
/// above-the-ecliptic camera as a dark spot at each planet's north pole.
/// Placing it well above the plane, on the same side as the default camera,
/// lights the visible hemisphere evenly. Far enough away to act as a
/// directional light at any zoom.
const LIGHT_POSITION: [f64; 3] = [1e5, 1e5, 1e5];

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

/// A triangle mesh in the vertex/index layout `mesh3d` expects.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[derive(Clone, Debug, PartialEq)]
struct TriangleMesh {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    /// First vertex index of each triangle.
    i: Vec<usize>,
    /// Second vertex index of each triangle.
    j: Vec<usize>,
    /// Third vertex index of each triangle.
    k: Vec<usize>,
}

/// A `mesh3d` trace with an explicit scene light position.
///
/// `plotly::traces::mesh3d::LightPosition` serialises each coordinate as a
/// one-element array, which plotly.js's numeric `lightposition.{x,y,z}`
/// attributes reject (silently falling back to the default light), so the
/// position is serialised here as plain numbers instead.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[derive(Clone, serde::Serialize)]
struct LitMesh {
    #[serde(flatten)]
    mesh: plotly::Mesh3D<f64, f64, f64>,
    #[serde(rename = "lightposition")]
    light_position: LightPositionXyz,
}

/// A light position as plotly.js expects it: three plain numbers.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[derive(Clone, Copy, serde::Serialize)]
struct LightPositionXyz {
    x: f64,
    y: f64,
    z: f64,
}

impl plotly::Trace for LitMesh {
    fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

/// Wraps `mesh` with [`LIGHT_POSITION`] as its light.
///
/// # Arguments
///
/// * `mesh` — the mesh trace to light.
///
/// # Returns
///
/// A [`LitMesh`] ready to be added to a plot.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn lit(mesh: Box<plotly::Mesh3D<f64, f64, f64>>) -> LitMesh {
    LitMesh {
        mesh: *mesh,
        light_position: LightPositionXyz {
            x: LIGHT_POSITION[0],
            y: LIGHT_POSITION[1],
            z: LIGHT_POSITION[2],
        },
    }
}

/// Builds a UV sphere (latitude/longitude grid) as a triangle mesh.
///
/// Pure geometry, no plotting dependency, so it is unit-testable on the
/// native target even though only the wasm build draws with it. Each pole is a
/// single shared vertex (a triangle fan) rather than one duplicated vertex per
/// longitude segment: plotly derives vertex normals for lighting from the
/// adjacent faces, and the zero-area triangles a duplicated pole produces give
/// it degenerate normals — a dark spot at the pole.
///
/// # Arguments
///
/// * `center` — sphere center, in the same units as `radius`.
/// * `radius` — sphere radius.
/// * `lat_bands` — number of latitude bands (pole to pole); clamped to at
///   least 2.
/// * `lon_segments` — number of longitude segments around the axis; clamped
///   to at least 3.
///
/// # Returns
///
/// A [`TriangleMesh`] with `2 + (lat_bands - 1) * lon_segments` vertices — every
/// one at exactly `radius` from `center`, the north pole (`+z`) first and the
/// south pole last — and `2 * (lat_bands - 1) * lon_segments` non-degenerate
/// triangles wound counter-clockwise seen from outside, all indices in
/// bounds.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn sphere_mesh(
    center: [f64; 3],
    radius: f64,
    lat_bands: usize,
    lon_segments: usize,
) -> TriangleMesh {
    let lat_bands = lat_bands.max(2);
    let lon_segments = lon_segments.max(3);
    let n_rings = lat_bands - 1;

    let north = [center[0], center[1], center[2] + radius];
    let south = [center[0], center[1], center[2] - radius];
    let rings = (1..=n_rings).flat_map(|ring| {
        let polar = std::f64::consts::PI * ring as f64 / lat_bands as f64;
        (0..lon_segments).map(move |segment| {
            let azimuth = std::f64::consts::TAU * segment as f64 / lon_segments as f64;
            [
                center[0] + radius * polar.sin() * azimuth.cos(),
                center[1] + radius * polar.sin() * azimuth.sin(),
                center[2] + radius * polar.cos(),
            ]
        })
    });
    let vertices: Vec<[f64; 3]> = std::iter::once(north)
        .chain(rings)
        .chain(std::iter::once(south))
        .collect();

    let south_index = vertices.len() - 1;
    // Vertex `segment` (wrapping) of ring `ring`, `1 <= ring <= n_rings`.
    let ring_vertex =
        |ring: usize, segment: usize| 1 + (ring - 1) * lon_segments + segment % lon_segments;

    let north_fan = (0..lon_segments).map(|s| [0, ring_vertex(1, s), ring_vertex(1, s + 1)]);
    let south_fan = (0..lon_segments).map(|s| {
        [
            south_index,
            ring_vertex(n_rings, s + 1),
            ring_vertex(n_rings, s),
        ]
    });
    let bands = (1..n_rings).flat_map(|ring| {
        (0..lon_segments).flat_map(move |s| {
            [
                [
                    ring_vertex(ring, s),
                    ring_vertex(ring + 1, s),
                    ring_vertex(ring + 1, s + 1),
                ],
                [
                    ring_vertex(ring, s),
                    ring_vertex(ring + 1, s + 1),
                    ring_vertex(ring, s + 1),
                ],
            ]
        })
    });
    let triangles: Vec<[usize; 3]> = north_fan.chain(bands).chain(south_fan).collect();

    TriangleMesh {
        x: vertices.iter().map(|v| v[0]).collect(),
        y: vertices.iter().map(|v| v[1]).collect(),
        z: vertices.iter().map(|v| v[2]).collect(),
        i: triangles.iter().map(|t| t[0]).collect(),
        j: triangles.iter().map(|t| t[1]).collect(),
        k: triangles.iter().map(|t| t[2]).collect(),
    }
}

/// Builds a shaded sphere as a `mesh3d` trace.
///
/// # Arguments
///
/// * `name` — legend/hover name.
/// * `color` — CSS color of the sphere.
/// * `center` — sphere center, AU.
/// * `radius` — sphere radius, AU.
/// * `lighting` — material properties (ambient/diffuse/specular).
/// * `hover_text` — custom hover text (plotly HTML), or `None` for
///   plotly's default.
///
/// # Returns
///
/// The unlit-position mesh trace; pass it through [`lit`] to give it the
/// scene light.
#[cfg(target_arch = "wasm32")]
fn sphere_trace(
    name: &str,
    color: &str,
    center: [f64; 3],
    radius: f64,
    lighting: Lighting,
    hover_text: Option<&str>,
) -> Box<Mesh3D<f64, f64, f64>> {
    let sphere = sphere_mesh(
        center,
        radius,
        PLANET_SPHERE_LAT_BANDS,
        PLANET_SPHERE_LON_SEGMENTS,
    );
    let mesh = Mesh3D::new(
        sphere.x,
        sphere.y,
        sphere.z,
        Some(sphere.i),
        Some(sphere.j),
        Some(sphere.k),
    )
    .name(name)
    .color(color.to_string())
    .flat_shading(false)
    .show_legend(true)
    .lighting(lighting);
    match hover_text {
        Some(text) => mesh.hover_text(text).hover_info(HoverInfo::Text),
        None => mesh,
    }
}

/// The marker of an orbit landmark: a per-kind symbol with a dark outline,
/// so it stays legible over the orbit lines and the planet spheres.
///
/// # Arguments
///
/// * `kind` — which characteristic point the marker stands for.
/// * `color` — CSS fill color.
///
/// # Returns
///
/// The configured [`Marker`].
#[cfg(target_arch = "wasm32")]
fn landmark_marker(kind: LandmarkKind, color: &str) -> Marker {
    let symbol = match kind {
        LandmarkKind::Perihelion => MarkerSymbol::Diamond,
        LandmarkKind::Aphelion => MarkerSymbol::Square,
        LandmarkKind::Moid => MarkerSymbol::DiamondOpen,
        LandmarkKind::InclinationIcon => MarkerSymbol::SquareOpen,
        // Drawn as a disc and a line rather than as markers.
        LandmarkKind::Node
        | LandmarkKind::EclipticPlane
        | LandmarkKind::EclipticSpokes
        | LandmarkKind::Inclination => MarkerSymbol::Circle,
    };
    Marker::new()
        .color(color.to_string())
        .size(LANDMARK_MARKER_SIZE)
        .symbol(symbol)
        .line(Line::new().color("#1a1a1a").width(1.0))
}

/// Cuts the segment `a → b` into dashes.
///
/// The dash count is the one that best fits the nominal `dash`/`gap`
/// lengths, then the dashes are stretched to fill the segment exactly: the
/// first dash starts at `a` and the last one ends at `b`, so a dashed line
/// always visibly reaches both of its end points. Gaps keep their nominal
/// length.
///
/// # Arguments
///
/// * `a`, `b` — segment end points.
/// * `dash` — nominal dash length (same units as the points).
/// * `gap` — gap length.
///
/// # Returns
///
/// The dashes as `[start, end]` pairs, in order from `a` to `b`; a single
/// dash covering the whole segment if it is shorter than `dash`, and none if
/// `a == b`.
fn dash_segment(a: [f64; 3], b: [f64; 3], dash: f64, gap: f64) -> Vec<[[f64; 3]; 2]> {
    let length = crate::orbit3d::geometry::distance(a, b);
    if length <= 0.0 || dash <= 0.0 {
        return Vec::new();
    }

    let n = (((length + gap) / (dash + gap)).round() as usize).max(1);
    let dash = ((length - (n - 1) as f64 * gap) / n as f64).max(0.0);
    let at = |distance: f64| {
        let t = distance / length;
        [
            a[0] + t * (b[0] - a[0]),
            a[1] + t * (b[1] - a[1]),
            a[2] + t * (b[2] - a[2]),
        ]
    };

    (0..n)
        .map(|i| {
            let start = i as f64 * (dash + gap);
            [at(start), at(start + dash)]
        })
        .collect()
}

/// Cuts a polyline into dashes, segment by segment, each restarting its own
/// pattern (see [`dash_segment`]).
///
/// # Arguments
///
/// * `points` — the polyline's vertices.
/// * `dash`, `gap` — nominal dash and gap lengths, as in [`dash_segment`].
///
/// # Returns
///
/// Every dash of every segment as `[start, end]` pairs, in polyline order.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn dash_polyline(points: &[[f64; 3]], dash: f64, gap: f64) -> Vec<[[f64; 3]; 2]> {
    points
        .windows(2)
        .flat_map(|pair| dash_segment(pair[0], pair[1], dash, gap))
        .collect()
}

/// One coordinate axis of `points`, with a gap (`None`, serialised as JSON
/// `null`, which plotly leaves unconnected) after every `group_len` points
/// except the last group, so a single `scatter3d` trace can draw several
/// disjoint polylines.
///
/// # Arguments
///
/// * `points` — the points of every polyline, group after group.
/// * `group_len` — number of points per polyline; clamped to at least 1.
/// * `axis` — which coordinate to extract (`0` = x, `1` = y, `2` = z).
///
/// # Returns
///
/// `points.len()` coordinates plus one gap between each pair of
/// consecutive groups.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn coordinate_with_gaps(points: &[[f64; 3]], group_len: usize, axis: usize) -> Vec<Option<f64>> {
    let group_len = group_len.max(1);
    let n_groups = points.len().div_ceil(group_len);
    points
        .chunks(group_len)
        .enumerate()
        .flat_map(|(i, group)| {
            group
                .iter()
                .map(|p| Some(p[axis]))
                .chain((i + 1 < n_groups).then_some(None))
        })
        .collect()
}

/// Gives a polyline trace its hover: the per-vertex `texts` when there are
/// any, otherwise no hover at all.
///
/// # Arguments
///
/// * `trace` — the scatter trace.
/// * `texts` — per-vertex hover text (see [`texts_with_gaps`]), possibly
///   empty.
///
/// # Returns
///
/// The trace with `hoverinfo: text` and its texts, or `hoverinfo: skip`.
#[cfg(target_arch = "wasm32")]
fn with_hover<X, Y, Z>(
    trace: Box<Scatter3D<X, Y, Z>>,
    texts: Vec<String>,
) -> Box<Scatter3D<X, Y, Z>>
where
    X: serde::Serialize + Clone,
    Y: serde::Serialize + Clone,
    Z: serde::Serialize + Clone,
{
    if texts.is_empty() {
        trace.hover_info(HoverInfo::Skip)
    } else {
        trace.text_array(texts).hover_info(HoverInfo::Text)
    }
}

/// Per-vertex hover text for a trace drawn as consecutive pieces of
/// `piece_len` points separated by one gap vertex each — the layout
/// `coordinate_with_gaps` gives.
///
/// # Arguments
///
/// * `piece_texts` — one text per piece.
/// * `piece_len` — number of points per piece; clamped to at least 1.
///
/// # Returns
///
/// For each piece its text repeated `piece_len` times, followed by an empty
/// string for the gap vertex (none after the last piece).
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn texts_with_gaps(piece_texts: &[String], piece_len: usize) -> Vec<String> {
    let piece_len = piece_len.max(1);
    let last = piece_texts.len().saturating_sub(1);
    piece_texts
        .iter()
        .enumerate()
        .flat_map(|(i, text)| {
            std::iter::repeat_n(text.clone(), piece_len).chain((i < last).then(String::new))
        })
        .collect()
}

/// Closes an open polyline by re-appending its first point, so the drawn
/// curve is a full loop.
///
/// `orbit3d::geometry::ellipse_points` deliberately stops one step short of
/// a full revolution and leaves closing to the caller; without this a
/// large orbit shows a visible gap — about 1 AU per step for Neptune at 180
/// samples — between its last and first sample.
///
/// # Arguments
///
/// * `points` — the polyline's vertices.
///
/// # Returns
///
/// `points` followed by a copy of its first point, or an empty vector for an
/// empty input.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn closed_loop(points: &[[f64; 3]]) -> Vec<[f64; 3]> {
    points.iter().chain(points.first()).copied().collect()
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
                    hover_text: Vec::new(),
                },
                Trace3D {
                    name: body.name.clone(),
                    color,
                    style: TraceStyle::Markers,
                    source: TraceSource::Planet,
                    points: vec![body.position],
                    hover_text: Vec::new(),
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
/// * `sphere_scale` — multiplies the radius of every drawn sphere (the Sun,
///   the planets, the focus object); `1.0` by default. The lineage page
///   shrinks them so a single object's orbit reads without huge balls in
///   the way, while the homepage keeps the default.
#[component]
pub fn Scatter3dPlot(
    plot_id: &'static str,
    traces: Vec<Trace3D>,
    #[props(default = 1.0)] sphere_scale: f64,
) -> Element {
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
    use_effect(use_reactive!(|(plot_id, traces, sphere_scale)| {
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

                // Self-luminous: almost all ambient, so it stays a bright
                // yellow ball rather than being shaded like a planet.
                plot.add_trace(Box::new(lit(sphere_trace(
                    "Sun",
                    SUN_COLOR,
                    [0.0, 0.0, 0.0],
                    SUN_SPHERE_RADIUS_AU * sphere_scale,
                    Lighting::new().ambient(0.9).diffuse(0.3).specular(0.1),
                    None,
                ))));

                for trace in traces.iter() {
                    // Separate dashed polylines: gaps between groups keep them
                    // disjoint within a single legend entry. Hover skipped —
                    // they are guides, and would only shadow the crosses'.
                    if let TraceStyle::DashedPolylines(group_len) = trace.style {
                        let per_group: Vec<Vec<[[f64; 3]; 2]>> = trace
                            .points
                            .chunks(group_len.max(1))
                            .map(|polyline| dash_polyline(polyline, DASH_LENGTH_AU, DASH_GAP_AU))
                            .collect();
                        let dashes: Vec<[f64; 3]> =
                            per_group.iter().flatten().flatten().copied().collect();
                        // Every dash carries its group's hover text, so a
                        // guide line says what it is when it is the nearest
                        // thing under the cursor.
                        let dash_texts: Vec<String> = if trace.hover_text.is_empty() {
                            Vec::new()
                        } else {
                            per_group
                                .iter()
                                .enumerate()
                                .flat_map(|(k, group)| {
                                    let text = trace
                                        .hover_text
                                        .get(k)
                                        .unwrap_or(&trace.hover_text[0])
                                        .clone();
                                    std::iter::repeat_n(text, group.len())
                                })
                                .collect()
                        };
                        let coords = |axis| coordinate_with_gaps(&dashes, 2, axis);
                        let t = Scatter3D::new(coords(0), coords(1), coords(2))
                            .name(trace.name.clone())
                            .mode(Mode::Lines)
                            .line(
                                Line::new()
                                    .color(trace.color.clone())
                                    .width(DASHED_LINE_WIDTH),
                            );
                        plot.add_trace(with_hover(t, texts_with_gaps(&dash_texts, 2)));
                        continue;
                    }

                    // Separate solid polylines (e.g. clone orbits, or the
                    // inclination pie-slice): closed ones are looped back to
                    // their start; either way each group is separated from
                    // the next by a gap.
                    if let TraceStyle::ClosedPolylines(group_len)
                    | TraceStyle::SolidPolylines(group_len) = trace.style
                    {
                        let closed = matches!(trace.style, TraceStyle::ClosedPolylines(_));
                        let group_len = group_len.max(1);
                        let (flat, stride, width) = if closed {
                            let looped: Vec<[f64; 3]> = trace
                                .points
                                .chunks(group_len)
                                .flat_map(closed_loop)
                                .collect();
                            (looped, group_len + 1, CLOSED_POLYLINE_WIDTH)
                        } else {
                            (trace.points.clone(), group_len, SOLID_POLYLINE_WIDTH)
                        };
                        let coords = |axis| coordinate_with_gaps(&flat, stride, axis);
                        let t = Scatter3D::new(coords(0), coords(1), coords(2))
                            .name(trace.name.clone())
                            .mode(Mode::Lines)
                            .line(Line::new().color(trace.color.clone()).width(width));
                        // One hover text per group, wherever its line is
                        // hovered; none at all if the trace has no text.
                        let group_texts: Vec<String> = if trace.hover_text.is_empty() {
                            Vec::new()
                        } else {
                            (0..trace.points.len().div_ceil(group_len))
                                .map(|k| {
                                    trace
                                        .hover_text
                                        .get(k)
                                        .unwrap_or(&trace.hover_text[0])
                                        .clone()
                                })
                                .collect()
                        };
                        // The ecliptic wireframe's rings and spokes toggle
                        // together from a single legend entry.
                        let t = match trace.source {
                            TraceSource::Landmark(LandmarkKind::EclipticPlane) => {
                                t.legend_group("ecliptic-plane")
                            }
                            TraceSource::Landmark(LandmarkKind::EclipticSpokes) => {
                                t.legend_group("ecliptic-plane").show_legend(false)
                            }
                            _ => t,
                        };
                        plot.add_trace(with_hover(t, texts_with_gaps(&group_texts, stride)));
                        continue;
                    }

                    let points = match trace.style {
                        // `DashedPolylines`/`ClosedPolylines` were drawn and
                        // skipped above.
                        TraceStyle::Line
                        | TraceStyle::DashedPolylines(_)
                        | TraceStyle::ClosedPolylines(_)
                        | TraceStyle::SolidPolylines(_) => closed_loop(&trace.points),
                        TraceStyle::Markers => trace.points.clone(),
                    };
                    let xs: Vec<f64> = points.iter().map(|p| p[0]).collect();
                    let ys: Vec<f64> = points.iter().map(|p| p[1]).collect();
                    let zs: Vec<f64> = points.iter().map(|p| p[2]).collect();

                    // Planets and the focus object are shaded `mesh3d`
                    // spheres: plotly's `scatter3d` has no sphere symbol, and
                    // a lit sphere keeps them distinct from tracked-object
                    // dots.
                    if let (TraceStyle::Markers, Some(radius)) =
                        (trace.style, sphere_radius_au(trace.source))
                    {
                        if let Some(&center) = trace.points.first() {
                            plot.add_trace(Box::new(lit(sphere_trace(
                                &trace.name,
                                &trace.color,
                                center,
                                radius * sphere_scale,
                                Lighting::new().ambient(0.55).diffuse(0.8).specular(0.3),
                                trace.hover_text.first().map(String::as_str),
                            ))));
                        }
                        continue;
                    }

                    let (mode, marker) = match trace.style {
                        TraceStyle::Markers => (
                            Mode::Markers,
                            match trace.source {
                                TraceSource::Observation => Marker::new()
                                    .color(trace.color.clone())
                                    .size(OBSERVATION_MARKER_SIZE)
                                    .symbol(MarkerSymbol::Cross)
                                    .line(
                                        Line::new()
                                            .color(trace.color.clone())
                                            .width(OBSERVATION_MARKER_LINE_WIDTH),
                                    ),
                                TraceSource::Landmark(kind) => landmark_marker(kind, &trace.color),
                                TraceSource::Uncertainty => Marker::new()
                                    .color(trace.color.clone())
                                    .size(UNCERTAINTY_MARKER_SIZE),
                                // Planets and the focus object were drawn as
                                // spheres above; what is left is the tiny dot
                                // of a tracked object.
                                TraceSource::Planet
                                | TraceSource::FocusObject
                                | TraceSource::TrackedObject => Marker::new()
                                    .color(trace.color.clone())
                                    .size(OBJECT_MARKER_SIZE)
                                    .opacity(population_opacity(trace.points.len())),
                            },
                        ),
                        // The polyline styles were drawn and skipped above.
                        TraceStyle::Line
                        | TraceStyle::DashedPolylines(_)
                        | TraceStyle::ClosedPolylines(_)
                        | TraceStyle::SolidPolylines(_) => (
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
                    let t = if trace.hover_text.is_empty() {
                        t
                    } else {
                        t.text_array(trace.hover_text.clone())
                            .hover_info(HoverInfo::Text)
                    };
                    // The inclination icon rides on its slice's legend entry.
                    let t = if trace.source == TraceSource::Landmark(LandmarkKind::InclinationIcon)
                    {
                        t.show_legend(false)
                    } else {
                        t
                    };
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
    fn sphere_mesh_has_expected_vertex_and_triangle_counts() {
        let mesh = sphere_mesh([0.0; 3], 1.0, 8, 12);
        assert_eq!(mesh.x.len(), 2 + 7 * 12);
        assert_eq!(mesh.y.len(), mesh.x.len());
        assert_eq!(mesh.z.len(), mesh.x.len());
        assert_eq!(mesh.i.len(), 2 * 7 * 12);
        assert_eq!(mesh.j.len(), mesh.i.len());
        assert_eq!(mesh.k.len(), mesh.i.len());
    }

    #[test]
    fn sphere_mesh_vertices_lie_on_the_sphere() {
        let center = [1.5, -2.0, 0.25];
        let radius = 0.4;
        let mesh = sphere_mesh(center, radius, 10, 16);
        for ((x, y), z) in mesh.x.iter().zip(&mesh.y).zip(&mesh.z) {
            let d = ((x - center[0]).powi(2) + (y - center[1]).powi(2) + (z - center[2]).powi(2))
                .sqrt();
            assert!((d - radius).abs() < 1e-12, "vertex at distance {d}");
        }
    }

    #[test]
    fn sphere_mesh_triangle_indices_are_in_bounds() {
        let mesh = sphere_mesh([0.0; 3], 1.0, 6, 9);
        let n = mesh.x.len();
        for idx in mesh.i.iter().chain(&mesh.j).chain(&mesh.k) {
            assert!(*idx < n, "index {idx} out of bounds for {n} vertices");
        }
    }

    #[test]
    fn sphere_mesh_clamps_degenerate_resolution() {
        let mesh = sphere_mesh([0.0; 3], 1.0, 0, 0);
        assert_eq!(mesh.x.len(), 2 + 3);
        assert_eq!(mesh.i.len(), 2 * 3);
    }

    /// Zero-area triangles are what gave plotly degenerate pole normals.
    #[test]
    fn sphere_mesh_has_no_degenerate_triangles() {
        let mesh = sphere_mesh([0.0; 3], 1.0, 12, 20);
        for ((&a, &b), &c) in mesh.i.iter().zip(&mesh.j).zip(&mesh.k) {
            assert!(
                a != b && b != c && a != c,
                "degenerate triangle ({a}, {b}, {c})"
            );
            let ab = [
                mesh.x[b] - mesh.x[a],
                mesh.y[b] - mesh.y[a],
                mesh.z[b] - mesh.z[a],
            ];
            let ac = [
                mesh.x[c] - mesh.x[a],
                mesh.y[c] - mesh.y[a],
                mesh.z[c] - mesh.z[a],
            ];
            let cross = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            let area2 = (cross[0].powi(2) + cross[1].powi(2) + cross[2].powi(2)).sqrt();
            assert!(area2 > 1e-9, "zero-area triangle ({a}, {b}, {c})");
        }
    }

    /// Outward winding: every triangle's normal points away from the center.
    #[test]
    fn sphere_mesh_triangles_are_wound_outward() {
        let mesh = sphere_mesh([0.0; 3], 1.0, 12, 20);
        for ((&a, &b), &c) in mesh.i.iter().zip(&mesh.j).zip(&mesh.k) {
            let p = |i: usize| [mesh.x[i], mesh.y[i], mesh.z[i]];
            let (pa, pb, pc) = (p(a), p(b), p(c));
            let ab = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
            let ac = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
            let normal = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            let centroid = [
                (pa[0] + pb[0] + pc[0]) / 3.0,
                (pa[1] + pb[1] + pc[1]) / 3.0,
                (pa[2] + pb[2] + pc[2]) / 3.0,
            ];
            let dot = normal[0] * centroid[0] + normal[1] * centroid[1] + normal[2] * centroid[2];
            assert!(dot > 0.0, "inward-facing triangle ({a}, {b}, {c})");
        }
    }

    #[test]
    fn lit_mesh_serialises_light_position_as_plain_numbers() {
        let mesh = plotly::Mesh3D::new(
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0],
            Some(vec![0]),
            Some(vec![1]),
            Some(vec![2]),
        );
        let json: serde_json::Value =
            serde_json::from_str(&plotly::Trace::to_json(&lit(mesh))).unwrap();

        assert_eq!(json["type"], "mesh3d");
        assert_eq!(json["lightposition"]["x"], LIGHT_POSITION[0]);
        assert_eq!(json["lightposition"]["y"], LIGHT_POSITION[1]);
        assert_eq!(json["lightposition"]["z"], LIGHT_POSITION[2]);
        assert_eq!(json["x"].as_array().map(Vec::len), Some(3));
    }

    #[test]
    fn only_planets_and_the_focus_object_are_drawn_as_spheres() {
        assert_eq!(
            sphere_radius_au(TraceSource::Planet),
            Some(PLANET_SPHERE_RADIUS_AU)
        );
        assert_eq!(
            sphere_radius_au(TraceSource::FocusObject),
            Some(FOCUS_SPHERE_RADIUS_AU)
        );
        assert_eq!(sphere_radius_au(TraceSource::TrackedObject), None);
        assert_eq!(sphere_radius_au(TraceSource::Observation), None);
    }

    fn length(dash: &[[f64; 3]; 2]) -> f64 {
        crate::orbit3d::geometry::distance(dash[0], dash[1])
    }

    /// The whole point of cutting dashes ourselves: the line visibly starts
    /// at its first point and ends at its last.
    #[test]
    fn dash_segment_starts_at_a_and_ends_at_b() {
        let a = [1.0, -2.0, 0.5];
        let b = [4.0, 2.0, 0.5];
        let dashes = dash_segment(a, b, 0.1, 0.06);

        assert!(dashes.len() > 1);
        assert_eq!(dashes.first().unwrap()[0], a);
        let last = dashes.last().unwrap()[1];
        assert!(crate::orbit3d::geometry::distance(last, b) < 1e-9);
    }

    #[test]
    fn dash_segment_dashes_are_equal_and_separated_by_the_gap() {
        let a = [0.0; 3];
        let b = [3.0, 0.0, 0.0];
        let dashes = dash_segment(a, b, 0.1, 0.06);

        let first = length(&dashes[0]);
        for d in &dashes {
            assert!((length(d) - first).abs() < 1e-9);
        }
        for pair in dashes.windows(2) {
            let gap = crate::orbit3d::geometry::distance(pair[0][1], pair[1][0]);
            assert!((gap - 0.06).abs() < 1e-9, "gap {gap}");
        }
        // Stretched to fit, but close to the nominal length.
        assert!((first - 0.1).abs() < 0.02, "dash {first}");
    }

    #[test]
    fn dash_segment_shorter_than_a_dash_is_one_full_dash() {
        let dashes = dash_segment([0.0; 3], [0.04, 0.0, 0.0], 0.1, 0.06);
        assert_eq!(dashes.len(), 1);
        assert!((length(&dashes[0]) - 0.04).abs() < 1e-12);
    }

    #[test]
    fn dash_segment_of_a_zero_length_segment_is_empty() {
        assert!(dash_segment([1.0; 3], [1.0; 3], 0.1, 0.06).is_empty());
    }

    #[test]
    fn dash_polyline_dashes_every_segment() {
        let pts = [[0.0; 3], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]];
        let dashes = dash_polyline(&pts, 0.1, 0.06);
        let first_segment = dash_segment(pts[0], pts[1], 0.1, 0.06).len();
        let second_segment = dash_segment(pts[1], pts[2], 0.1, 0.06).len();
        assert_eq!(dashes.len(), first_segment + second_segment);
        assert_eq!(dashes[first_segment][0], pts[1]);
    }

    #[test]
    fn coordinate_with_gaps_separates_groups_with_a_gap() {
        let pts = [
            [1.0, 10.0, 0.0],
            [2.0, 20.0, 0.0],
            [3.0, 30.0, 0.0],
            [4.0, 40.0, 0.0],
            [5.0, 50.0, 0.0],
            [6.0, 60.0, 0.0],
        ];
        assert_eq!(
            coordinate_with_gaps(&pts, 3, 0),
            [
                Some(1.0),
                Some(2.0),
                Some(3.0),
                None,
                Some(4.0),
                Some(5.0),
                Some(6.0)
            ]
        );
        assert_eq!(coordinate_with_gaps(&pts, 3, 1)[4], Some(40.0));
    }

    #[test]
    fn coordinate_with_gaps_has_no_trailing_gap_and_handles_empty_input() {
        assert_eq!(
            coordinate_with_gaps(&[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], 2, 0),
            [Some(1.0), Some(2.0)]
        );
        assert!(coordinate_with_gaps(&[], 3, 0).is_empty());
    }

    /// The texts line up with `coordinate_with_gaps`'s vertices: each
    /// piece's text on all of its points, an empty one on the gap.
    #[test]
    fn texts_with_gaps_match_the_coordinate_layout() {
        let pieces = ["a".to_string(), "b".to_string(), "c".to_string()];
        let texts = texts_with_gaps(&pieces, 2);
        assert_eq!(texts, ["a", "a", "", "b", "b", "", "c", "c"]);

        let points: Vec<[f64; 3]> = (0..6).map(|i| [i as f64, 0.0, 0.0]).collect();
        assert_eq!(coordinate_with_gaps(&points, 2, 0).len(), texts.len());
    }

    #[test]
    fn texts_with_gaps_of_nothing_is_empty() {
        assert!(texts_with_gaps(&[], 3).is_empty());
        assert_eq!(texts_with_gaps(&["x".to_string()], 0), ["x"]);
    }

    #[test]
    fn closed_loop_appends_the_first_point() {
        let pts = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]];
        let closed = closed_loop(&pts);
        assert_eq!(closed.len(), 4);
        assert_eq!(closed.first(), closed.last());
        assert_eq!(&closed[..3], &pts);
    }

    #[test]
    fn closed_loop_of_nothing_is_empty() {
        assert!(closed_loop(&[]).is_empty());
    }

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
