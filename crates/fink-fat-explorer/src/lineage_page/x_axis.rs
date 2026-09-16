#[cfg(target_arch = "wasm32")]
use super::kf_replay::{HypothesisSnapshot, KfStep};
#[cfg(target_arch = "wasm32")]
use super::observations_table::ObservationRow;

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

#[cfg(target_arch = "wasm32")]
use crate::format_epoch::iso_utc as epoch_to_iso_utc;

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

/// Same as [`x_values_for_steps`], but for [`ObservationRow`]s (`position`
/// for the step index, `mjd_tt` for the epoch).
#[cfg(target_arch = "wasm32")]
pub fn x_values_for_observations(unit: XAxisUnit, observations: &[ObservationRow]) -> XAxisValues {
    match unit {
        XAxisUnit::ObservationIndex => {
            XAxisValues::Numeric(observations.iter().map(|o| o.position as f64).collect())
        }
        XAxisUnit::DaysSinceFirst => {
            let first_epoch = observations.first().map(|o| o.mjd_tt).unwrap_or(0.0);
            XAxisValues::Numeric(
                observations
                    .iter()
                    .map(|o| o.mjd_tt - first_epoch)
                    .collect(),
            )
        }
        XAxisUnit::IsoUtcDate => XAxisValues::Date(
            observations
                .iter()
                .map(|o| epoch_to_iso_utc(o.mjd_tt))
                .collect(),
        ),
    }
}

/// Human-readable label per point, matching the wording of the light
/// curve's axis-native hover template (`x_hover_format` in
/// `light_curve_plot.rs`), but as plain text — for plots like the
/// trajectory view where time isn't the x/y axis and must be folded into a
/// `customdata` string instead of a `%{x}` placeholder.
///
/// `iso_utc` truncates to `YYYY-MM-DDTHH:MM:SS.ffffff` with no trailing
/// `Z`/offset, so `..19` trims to whole seconds before the `" UTC"` suffix
/// makes the timezone explicit.
#[cfg(target_arch = "wasm32")]
pub fn format_time_labels(unit: XAxisUnit, values: &XAxisValues) -> Vec<String> {
    match values {
        XAxisValues::Numeric(v) => v
            .iter()
            .map(|x| match unit {
                XAxisUnit::ObservationIndex => format!("Obs #{x:.0}"),
                XAxisUnit::DaysSinceFirst => format!("{x:.2} d since first obs"),
                XAxisUnit::IsoUtcDate => unreachable!(),
            })
            .collect(),
        XAxisValues::Date(v) => v
            .iter()
            .map(|d| format!("{} UTC", d.get(..19).unwrap_or(d).replace('T', " ")))
            .collect(),
    }
}
