//! The lineage page's 3D tab: one tracked object's full orbital ellipse and
//! current position, plotted alongside the planets and tracked perturbers.

use dioxus::prelude::*;

use crate::orbit3d::plot3d::{planet_traces, Scatter3dPlot, Trace3D, TraceSource, TraceStyle};
use crate::orbit3d::server_fns::get_lineage_orbit3d;

/// Color for the tracked object's own orbit/position traces — distinct from
/// every planet/perturber color in `orbit3d::plot3d::body_color`.
const OBJECT_COLOR: &str = "#ff3b6f";

#[component]
pub fn LineageOrbit3DTab(lineage_id: String) -> Element {
    let resource_lineage_id = lineage_id.clone();
    let orbit3d = use_resource(use_reactive!(|(resource_lineage_id,)| get_lineage_orbit3d(
        resource_lineage_id
    )));

    let traces = use_memo(move || match &*orbit3d.read() {
        Some(Ok(Some(data))) => {
            let mut traces = vec![
                Trace3D {
                    name: "Orbit".to_string(),
                    color: OBJECT_COLOR.to_string(),
                    style: TraceStyle::Line,
                    source: TraceSource::TrackedObject,
                    points: data.object_orbit.clone(),
                },
                Trace3D {
                    name: "Current position".to_string(),
                    color: OBJECT_COLOR.to_string(),
                    style: TraceStyle::Markers,
                    source: TraceSource::TrackedObject,
                    points: vec![data.object_position],
                },
            ];
            traces.extend(planet_traces(&data.planets));
            traces
        }
        _ => Vec::new(),
    });

    rsx! {
        match &*orbit3d.read() {
            Some(Ok(Some(_))) => rsx! {
                Scatter3dPlot { plot_id: "lineage-orbit3d-plot-div", traces: traces() }
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
