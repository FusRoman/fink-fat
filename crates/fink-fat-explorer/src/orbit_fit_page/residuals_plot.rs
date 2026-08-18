use dioxus::prelude::*;

#[cfg(target_arch = "wasm32")]
use plotly::{
    common::{Marker, Mode, Title},
    layout::{Axis, Layout, Margin},
    Plot, Scatter,
};

use crate::orbit_fit::ObsResidual;

/// Residual RA/Dec (arcsec) vs epoch, and per-observation chi — the
/// diagnostic plots for judging fit quality and spotting outliers.
#[component]
pub fn ResidualsPlot(residuals: Vec<ObsResidual>) -> Element {
    rsx! {
        div { class: "card bg-base-100 shadow-sm flex-1",
            div { class: "card-body gap-4",
                h2 { class: "card-title", "Residuals" }
                ResidualScatterPlot { residuals: residuals.clone() }
                ChiPlot { residuals }
            }
        }
    }
}

#[component]
fn ResidualScatterPlot(residuals: Vec<ObsResidual>) -> Element {
    let mut is_mounted = use_signal(|| false);
    #[cfg(target_arch = "wasm32")]
    let mut drawn = use_signal(|| false);

    use_effect(use_reactive!(|(residuals,)| {
        #[cfg(target_arch = "wasm32")]
        {
            if !is_mounted() {
                return;
            }

            let x: Vec<f64> = residuals.iter().map(|r| r.mjd_tt).collect();
            let ra: Vec<f64> = residuals.iter().map(|r| r.residual_ra_arcsec).collect();
            let dec: Vec<f64> = residuals.iter().map(|r| r.residual_dec_arcsec).collect();

            let mut plot = Plot::new();
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

            let layout = Layout::new()
                .height(220)
                .margin(Margin::new().top(10).right(10).bottom(30))
                .x_axis(Axis::new().title(Title::from("MJD (TT)")))
                .y_axis(Axis::new().title(Title::from("residual (arcsec)")));
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

#[component]
fn ChiPlot(residuals: Vec<ObsResidual>) -> Element {
    let mut is_mounted = use_signal(|| false);
    #[cfg(target_arch = "wasm32")]
    let mut drawn = use_signal(|| false);

    use_effect(use_reactive!(|(residuals,)| {
        #[cfg(target_arch = "wasm32")]
        {
            if !is_mounted() {
                return;
            }

            let x: Vec<f64> = residuals.iter().map(|r| r.mjd_tt).collect();
            let chi: Vec<f64> = residuals.iter().map(|r| r.chi).collect();
            let colors: Vec<&str> = residuals
                .iter()
                .map(|r| match r.selection {
                    crate::orbit_fit::ObsSelectionView::Kept => "#2d8d5a",
                    crate::orbit_fit::ObsSelectionView::Rejected => "#d23434",
                })
                .collect();

            let mut plot = Plot::new();
            plot.add_trace(
                Scatter::new(x, chi)
                    .name("χ per observation")
                    .mode(Mode::Markers)
                    .marker(Marker::new().color_array(colors).size(6)),
            );

            let layout = Layout::new()
                .height(200)
                .margin(Margin::new().top(10).right(10).bottom(30))
                .show_legend(false)
                .x_axis(Axis::new().title(Title::from("MJD (TT)")))
                .y_axis(Axis::new().title(Title::from("χ (green = kept, red = rejected)")));
            plot.set_layout(layout);

            spawn(async move {
                if *drawn.peek() {
                    plotly::bindings::react("orbit-fit-chi-plot", &plot).await;
                } else {
                    plotly::bindings::new_plot("orbit-fit-chi-plot", &plot).await;
                    drawn.set(true);
                }
            });
        }
    }));

    rsx! {
        div {
            id: "orbit-fit-chi-plot",
            style: "width: 100%;",
            onmounted: move |_| is_mounted.set(true),
        }
    }
}
