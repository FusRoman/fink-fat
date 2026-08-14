#[cfg(target_arch = "wasm32")]
use super::kf_replay::{HypothesisSnapshot, KfStep};

/// Which quantity the x-axis of the replay plots represents. Shared by every
/// mini-plot in `metrics_plot.rs`/`rho_evolution_plot.rs`/`hypotheses_plot.rs`
/// via a single selector in `plot_tabs.rs`, so switching it redraws every
/// plot consistently rather than per-tab.
#[derive(Clone, Copy, PartialEq)]
pub enum XAxisUnit {
    ObservationIndex,
    DaysSinceFirst,
    IsoUtcDate,
}

impl XAxisUnit {
    pub const ALL: [XAxisUnit; 3] = [
        XAxisUnit::ObservationIndex,
        XAxisUnit::DaysSinceFirst,
        XAxisUnit::IsoUtcDate,
    ];

    pub fn label(self) -> &'static str {
        match self {
            XAxisUnit::ObservationIndex => "Steps",
            XAxisUnit::DaysSinceFirst => "Days",
            XAxisUnit::IsoUtcDate => "Date",
        }
    }

    pub fn axis_title(self) -> &'static str {
        match self {
            XAxisUnit::ObservationIndex => "Real observation #",
            XAxisUnit::DaysSinceFirst => "Days since first observation",
            XAxisUnit::IsoUtcDate => "Date (UTC)",
        }
    }
}

/// The x-values a mini-plot should feed to `plotly::Scatter`: either a plain
/// numeric axis (observation index or Δt in days), or ISO-8601 date strings
/// for a native Plotly date axis (`AxisType::Date`) — Plotly spaces date
/// axes proportionally to real elapsed time on its own, no need to convert
/// to a numeric timestamp ourselves.
#[cfg(target_arch = "wasm32")]
pub enum XAxisValues {
    Numeric(Vec<f64>),
    Date(Vec<String>),
}

/// Convert a replayed MJD(TT) epoch to an ISO-8601 UTC string. Pure
/// computation, no network access (unlike `fink-fat-engine`'s UT1 provider
/// use of `hifitime`) — safe to run in the browser.
#[cfg(target_arch = "wasm32")]
fn epoch_to_iso_utc(mjd_tt: f64) -> String {
    use hifitime::{Epoch, TimeScale};

    Epoch::from_mjd_in_time_scale(mjd_tt, TimeScale::TT)
        .to_time_scale(TimeScale::UTC)
        .to_isoformat()
}

#[cfg(target_arch = "wasm32")]
pub fn x_values_for_steps(unit: XAxisUnit, replay: &[KfStep]) -> XAxisValues {
    match unit {
        XAxisUnit::ObservationIndex => {
            XAxisValues::Numeric(replay.iter().map(|s| s.step as f64).collect())
        }
        XAxisUnit::DaysSinceFirst => {
            let first_epoch = replay.first().map(|s| s.epoch).unwrap_or(0.0);
            XAxisValues::Numeric(replay.iter().map(|s| s.epoch - first_epoch).collect())
        }
        XAxisUnit::IsoUtcDate => {
            XAxisValues::Date(replay.iter().map(|s| epoch_to_iso_utc(s.epoch)).collect())
        }
    }
}

/// Same as [`x_values_for_steps`], but for [`HypothesisSnapshot`]s, which
/// only carry a `step` index — the corresponding epoch is looked up in
/// `replay` by matching `step`.
#[cfg(target_arch = "wasm32")]
pub fn x_values_for_hypotheses(
    unit: XAxisUnit,
    hypotheses: &[HypothesisSnapshot],
    replay: &[KfStep],
) -> XAxisValues {
    if let XAxisUnit::ObservationIndex = unit {
        return XAxisValues::Numeric(hypotheses.iter().map(|h| h.step as f64).collect());
    }

    let epoch_by_step: std::collections::HashMap<i32, f64> =
        replay.iter().map(|s| (s.step, s.epoch)).collect();
    let first_epoch = replay.first().map(|s| s.epoch).unwrap_or(0.0);

    match unit {
        XAxisUnit::ObservationIndex => unreachable!(),
        XAxisUnit::DaysSinceFirst => XAxisValues::Numeric(
            hypotheses
                .iter()
                .map(|h| epoch_by_step.get(&h.step).copied().unwrap_or(first_epoch) - first_epoch)
                .collect(),
        ),
        XAxisUnit::IsoUtcDate => XAxisValues::Date(
            hypotheses
                .iter()
                .map(|h| {
                    epoch_to_iso_utc(epoch_by_step.get(&h.step).copied().unwrap_or(first_epoch))
                })
                .collect(),
        ),
    }
}
