use dioxus::prelude::*;

#[cfg(target_arch = "wasm32")]
use plotly::{
    common::{Marker, Mode, Title},
    layout::{Axis, AxisType, Layout, Margin},
    Plot, Scatter,
};

use crate::orbit_fit::ObsResidual;

use super::x_axis::XAxisUnit;
#[cfg(target_arch = "wasm32")]
use super::x_axis::{x_values_for_residuals, XAxisValues};

/// Residual RA/Dec (arcsec) and per-observation χ vs epoch, stacked on a
/// single shared x-axis so panning/zooming one keeps the other aligned — the
/// diagnostic plots for judging fit quality and spotting outliers.
#[component]
pub fn ResidualsPlot(residuals: Vec<ObsResidual>) -> Element {
    let mut x_axis_unit = use_signal(|| XAxisUnit::IsoUtcDate);

    rsx! {
        div { class: "card bg-base-100 shadow-sm flex-1",
            div { class: "card-body gap-4",
                div { class: "flex flex-wrap items-center justify-between gap-2",
                    h2 { class: "card-title", "Residuals" }
                    div { class: "join",
                        for unit in XAxisUnit::ALL {
                            button {
                                key: "{unit.label()}",
                                class: if x_axis_unit() == unit { "join-item btn btn-sm btn-active" } else { "join-item btn btn-sm" },
                                onclick: move |_| x_axis_unit.set(unit),
                                "{unit.label()}"
                            }
                        }
                    }
                }
                CombinedResidualsPlot { residuals, x_axis_unit: x_axis_unit() }
            }
        }
    }
}

#[component]
fn CombinedResidualsPlot(residuals: Vec<ObsResidual>, x_axis_unit: XAxisUnit) -> Element {
    let mut is_mounted = use_signal(|| false);
    #[cfg(target_arch = "wasm32")]
    let mut drawn = use_signal(|| false);

    use_effect(use_reactive!(|(residuals, x_axis_unit)| {
        #[cfg(target_arch = "wasm32")]
        {
            if !is_mounted() {
                return;
            }

            let ra: Vec<f64> = residuals.iter().map(|r| r.residual_ra_arcsec).collect();
            let dec: Vec<f64> = residuals.iter().map(|r| r.residual_dec_arcsec).collect();
            let (kept_chi, kept_idx): (Vec<f64>, Vec<usize>) = residuals
                .iter()
                .enumerate()
                .filter(|(_, r)| r.selection == crate::orbit_fit::ObsSelectionView::Kept)
                .map(|(i, r)| (r.chi, i))
                .unzip();
            let (rejected_chi, rejected_idx): (Vec<f64>, Vec<usize>) = residuals
                .iter()
                .enumerate()
                .filter(|(_, r)| r.selection == crate::orbit_fit::ObsSelectionView::Rejected)
                .map(|(i, r)| (r.chi, i))
                .unzip();

            let mut plot = Plot::new();
            let mut x_axis = Axis::new()
                .title(Title::from(x_axis_unit.axis_title()))
                .anchor("y2");

            macro_rules! subset {
                ($values:expr, $idx:expr) => {
                    match &$values {
                        XAxisValues::Numeric(x) => {
                            XAxisValues::Numeric($idx.iter().map(|&i| x[i]).collect())
                        }
                        XAxisValues::Date(x) => {
                            XAxisValues::Date($idx.iter().map(|&i| x[i].clone()).collect())
                        }
                    }
                };
            }

            let all_x = x_values_for_residuals(x_axis_unit, &residuals);
            if let XAxisValues::Date(_) = all_x {
                x_axis = x_axis.type_(AxisType::Date);
            }
            let kept_x = subset!(all_x, kept_idx);
            let rejected_x = subset!(all_x, rejected_idx);

            match all_x {
                XAxisValues::Numeric(x) => {
                    plot.add_trace(
                        Scatter::new(x.clone(), ra)
                            .name("Δα cos δ (arcsec)")
                            .mode(Mode::Markers)
                            .marker(Marker::new().color("#8d2dd2").size(6)),
                    );
                    plot.add_trace(
                        Scatter::new(x, dec)
                            .name("Δδ (arcsec)")
                            .mode(Mode::Markers)
                            .marker(Marker::new().color("#d2642d").size(6)),
                    );
                }
                XAxisValues::Date(x) => {
                    plot.add_trace(
                        Scatter::new(x.clone(), ra)
                            .name("Δα cos δ (arcsec)")
                            .mode(Mode::Markers)
                            .marker(Marker::new().color("#8d2dd2").size(6)),
                    );
                    plot.add_trace(
                        Scatter::new(x, dec)
                            .name("Δδ (arcsec)")
                            .mode(Mode::Markers)
                            .marker(Marker::new().color("#d2642d").size(6)),
                    );
                }
            }

            match kept_x {
                XAxisValues::Numeric(x) => {
                    plot.add_trace(
                        Scatter::new(x, kept_chi)
                            .name("kept")
                            .mode(Mode::Markers)
                            .y_axis("y2")
                            .marker(Marker::new().color("#2d8d5a").size(6)),
                    );
                }
                XAxisValues::Date(x) => {
                    plot.add_trace(
                        Scatter::new(x, kept_chi)
                            .name("kept")
                            .mode(Mode::Markers)
                            .y_axis("y2")
                            .marker(Marker::new().color("#2d8d5a").size(6)),
                    );
                }
            }
            match rejected_x {
                XAxisValues::Numeric(x) => {
                    plot.add_trace(
                        Scatter::new(x, rejected_chi)
                            .name("rejected")
                            .mode(Mode::Markers)
                            .y_axis("y2")
                            .marker(Marker::new().color("#d23434").size(6)),
                    );
                }
                XAxisValues::Date(x) => {
                    plot.add_trace(
                        Scatter::new(x, rejected_chi)
                            .name("rejected")
                            .mode(Mode::Markers)
                            .y_axis("y2")
                            .marker(Marker::new().color("#d23434").size(6)),
                    );
                }
            }

            let layout = Layout::new()
                .height(380)
                .margin(Margin::new().top(10).right(10).bottom(40))
                .x_axis(x_axis)
                .y_axis(
                    Axis::new()
                        .title(Title::from("residual (arcsec)"))
                        .domain(&[0.55, 1.0]),
                )
                .y_axis2(
                    Axis::new()
                        .title(Title::from("χ"))
                        .domain(&[0.0, 0.42])
                        .anchor("x"),
                );
            plot.set_layout(layout);

            spawn(async move {
                if *drawn.peek() {
                    plotly::bindings::react("orbit-fit-residuals-plot", &plot).await;
                } else {
                    plotly::bindings::new_plot("orbit-fit-residuals-plot", &plot).await;
                    drawn.set(true);
                }
            });
        }
    }));

    rsx! {
        div {
            id: "orbit-fit-residuals-plot",
            style: "width: 100%;",
            onmounted: move |_| is_mounted.set(true),
        }
    }
}
