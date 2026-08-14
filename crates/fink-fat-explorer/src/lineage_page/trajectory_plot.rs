use dioxus::prelude::*;

#[cfg(target_arch = "wasm32")]
use plotly::{
    common::{ErrorData, ErrorType, Marker, Mode, Title},
    layout::{Axis, Layout, Margin},
    Plot, Scatter,
};

use super::kf_replay::KfStep;
use super::observations_table::ObservationRow;

const DEG_PER_ARCSEC: f64 = 1.0 / 3600.0;

/// Sky-plane trajectory: the observed track (with its own astrometric error
/// bars) overlaid with the Kalman filter's pre-update prediction at each
/// real observation — the "decision" the filter made, with its own error
/// box, before absorbing that point.
#[component]
pub fn TrajectoryPlot(observations: Vec<ObservationRow>, replay: Vec<KfStep>) -> Element {
    let mut is_mounted = use_signal(|| false);
    #[cfg(target_arch = "wasm32")]
    let mut drawn = use_signal(|| false);

    use_effect(use_reactive!(|(observations, replay)| {
        #[cfg(target_arch = "wasm32")]
        {
            if !is_mounted() {
                return;
            }

            let obs_ra: Vec<f64> = observations.iter().map(|o| o.ra.to_degrees()).collect();
            let obs_dec: Vec<f64> = observations.iter().map(|o| o.dec.to_degrees()).collect();
            let obs_ra_err: Vec<f64> = observations
                .iter()
                .map(|o| o.ra_err.to_degrees())
                .collect();
            let obs_dec_err: Vec<f64> = observations
                .iter()
                .map(|o| o.dec_err.to_degrees())
                .collect();

            let pred_ra: Vec<f64> = replay.iter().map(|s| s.predicted_ra_deg).collect();
            let pred_dec: Vec<f64> = replay.iter().map(|s| s.predicted_dec_deg).collect();
            let pred_ra_err: Vec<f64> = replay
                .iter()
                .map(|s| s.sigma_pred_ra_arcsec * DEG_PER_ARCSEC)
                .collect();
            let pred_dec_err: Vec<f64> = replay
                .iter()
                .map(|s| s.sigma_pred_dec_arcsec * DEG_PER_ARCSEC)
                .collect();

            let mut plot = Plot::new();

            plot.add_trace(
                Scatter::new(obs_ra, obs_dec)
                    .name("Observations")
                    .mode(Mode::Markers)
                    .marker(Marker::new().color("#2d7fd2").size(7))
                    .error_x(ErrorData::new(ErrorType::Data).array(obs_ra_err).symmetric(true))
                    .error_y(ErrorData::new(ErrorType::Data).array(obs_dec_err).symmetric(true)),
            );

            plot.add_trace(
                Scatter::new(pred_ra, pred_dec)
                    .name("Kalman prediction (pre-update)")
                    .mode(Mode::Markers)
                    .marker(Marker::new().color("#d2422d").size(6))
                    .error_x(ErrorData::new(ErrorType::Data).array(pred_ra_err).symmetric(true))
                    .error_y(ErrorData::new(ErrorType::Data).array(pred_dec_err).symmetric(true)),
            );

            let layout = Layout::new()
                .height(420)
                .margin(Margin::new().top(20).right(20))
                .x_axis(Axis::new().title(Title::from("RA (deg)")))
                .y_axis(Axis::new().title(Title::from("Dec (deg)")));
            plot.set_layout(layout);

            spawn(async move {
                if *drawn.peek() {
                    plotly::bindings::react("lineage-trajectory-plot", &plot).await;
                } else {
                    plotly::bindings::new_plot("lineage-trajectory-plot", &plot).await;
                    drawn.set(true);
                }
            });
        }
    }));

    rsx! {
        div { class: "card bg-base-100 shadow-sm flex-1",
            div { class: "card-body",
                h2 { class: "card-title", "Trajectory & Kalman predictions" }
                p { class: "text-xs opacity-60",
                    "Error bars: astrometric 1σ for observations, propagated sky covariance (pre-update) for predictions."
                }
                div {
                    id: "lineage-trajectory-plot",
                    style: "width: 100%;",
                    onmounted: move |_| is_mounted.set(true),
                }
            }
        }
    }
}
