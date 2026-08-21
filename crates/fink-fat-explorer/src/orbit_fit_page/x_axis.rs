#[cfg(target_arch = "wasm32")]
use crate::orbit_fit::ObsResidual;

/// Which quantity the residuals plots' shared x-axis represents. Mirrors
/// `lineage_page::x_axis::XAxisUnit` for the Kalman-replay plots, but keyed
/// off `ObsResidual` (position in the fit's residual list, not a `KfStep`).
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
            XAxisUnit::ObservationIndex => "Observation #",
            XAxisUnit::DaysSinceFirst => "Days since first observation",
            XAxisUnit::IsoUtcDate => "Date (UTC)",
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub enum XAxisValues {
    Numeric(Vec<f64>),
    Date(Vec<String>),
}

#[cfg(target_arch = "wasm32")]
use crate::format_epoch::iso_utc as epoch_to_iso_utc;

#[cfg(target_arch = "wasm32")]
pub fn x_values_for_residuals(unit: XAxisUnit, residuals: &[ObsResidual]) -> XAxisValues {
    match unit {
        XAxisUnit::ObservationIndex => {
            XAxisValues::Numeric((0..residuals.len()).map(|i| i as f64).collect())
        }
        XAxisUnit::DaysSinceFirst => {
            let first_mjd = residuals.first().map(|r| r.mjd_tt).unwrap_or(0.0);
            XAxisValues::Numeric(residuals.iter().map(|r| r.mjd_tt - first_mjd).collect())
        }
        XAxisUnit::IsoUtcDate => XAxisValues::Date(
            residuals
                .iter()
                .map(|r| epoch_to_iso_utc(r.mjd_tt))
                .collect(),
        ),
    }
}
