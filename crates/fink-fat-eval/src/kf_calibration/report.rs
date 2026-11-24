//! [`CalibrationReport`]: the JSON-serializable record of a whole
//! [`crate::kf_calibration::search::calibrate`] run, plus its stdout
//! printing and the YAML config snippet handed back to the user.

use serde::{Deserialize, Serialize};

use crate::kf_calibration::params::CalibrationParams;

/// One round of [`crate::kf_calibration::search::calibrate`]: the sample it
/// ran coordinate descent against, the resulting params, and where they
/// landed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundResult {
    pub round_index: usize,
    pub sample_size: usize,
    pub n_failing: usize,
    #[serde(with = "crate::trajectory_processing::finite_f64")]
    pub recall: f64,
    #[serde(with = "crate::trajectory_processing::finite_f64")]
    pub cost_mean_search_radius_arcsec: f64,
    pub params: CalibrationParams,
    /// Per-parameter score delta from this round's last coordinate-descent
    /// sweep (recall gain while below `target_recall`, cost reduction once
    /// at/above it — see
    /// [`crate::kf_calibration::search::coordinate_descent`]), in the fixed
    /// sweep order (`max_arcsec` first). The single largest entry is this
    /// round's "which parameter to look at next" signal.
    pub param_deltas: Vec<(String, f64)>,
}

/// Full record of a `calibrate` run: every round in order, plus the final
/// recommended [`CalibrationParams`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationReport {
    pub rounds: Vec<RoundResult>,
    pub final_params: CalibrationParams,
}

impl CalibrationReport {
    /// The parameter whose last round's sweep produced the largest score
    /// delta — the "what to look at next" hint the user asked for. `None`
    /// if the report has no rounds (shouldn't happen for a real run, but
    /// `calibrate` can't panic its way out of an empty `all_traj_ids`).
    pub fn top_param_by_last_delta(&self) -> Option<(&str, f64)> {
        self.rounds.last().and_then(|round| {
            round
                .param_deltas
                .iter()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(name, delta)| (name.as_str(), *delta))
        })
    }
}

/// Print a one-line-per-round table, then the final recommended params and
/// the "next parameter to look at" hint.
pub fn print_report(report: &CalibrationReport) {
    let sep = "=".repeat(78);
    println!("\n{sep}");
    println!(
        "[kf_calibrate] Calibration report — {} round(s)",
        report.rounds.len()
    );
    println!("{sep}");
    println!(
        "{:>6}  {:>10}  {:>8}  {:>9}  {:>10}",
        "round", "n_sample", "n_fail", "recall%", "cost(\")"
    );
    for r in &report.rounds {
        println!(
            "{:>6}  {:>10}  {:>8}  {:>9.2}  {:>10.2}",
            r.round_index,
            r.sample_size,
            r.n_failing,
            r.recall * 100.0,
            r.cost_mean_search_radius_arcsec,
        );
    }

    if let Some((name, delta)) = report.top_param_by_last_delta() {
        println!("\n  Next parameter worth looking at: {name} (last-sweep score delta {delta:.4})");
    }

    println!("\n-- Final recommended parameters --");
    println!("{}", to_yaml_snippet(&report.final_params));
}

/// A YAML fragment for `advance_params`/`kfbank_config` and the shared
/// Kalman `q0`/`dt_ref`, using the exact field names the engine config
/// expects — ready to copy into `kalman_shared_context.config`,
/// `kfbank_config` and `advance_params` in the user's config file.
///
/// Deliberately hand-built rather than `serde_yaml::to_string`'d off
/// [`CalibrationParams`] directly: the flat calibration vector doesn't match
/// the engine's nested config shape (e.g. `max_arcsec` lives inside
/// `radius_strategy: Clamped { .. }`, `obs_noise` is a squared `[rad^2;
/// rad^2]` pair) — see [`CalibrationParams::apply`].
pub fn to_yaml_snippet(params: &CalibrationParams) -> String {
    let sigma_rad = params.obs_noise_sigma_arcsec / (3600.0 * 180.0 / std::f64::consts::PI);
    let obs_noise_rad2 = sigma_rad * sigma_rad;
    format!(
        "kalman_shared_context:\n  config:\n    q0: {q0:e}\n    dt_ref: {dt_ref}\n\
         kfbank_config:\n  gate_chi2: {gate_chi2}\n  search_region_chi2: {search_region_chi2}\n  weight_floor: {weight_floor:e}\n\
         advance_params:\n  obs_noise: [{obs_noise_rad2:e}, {obs_noise_rad2:e}]  # {sigma_arcsec:.4}\" 1-sigma\n  top_k:\n    WeightThreshold: {weight_threshold}\n  radius_strategy:\n    Clamped:\n      inner: MixtureCovariance\n      max_arcsec: {max_arcsec}\n",
        q0 = params.q0,
        dt_ref = params.dt_ref,
        gate_chi2 = params.gate_chi2,
        search_region_chi2 = params.search_region_chi2,
        weight_floor = params.weight_floor,
        obs_noise_rad2 = obs_noise_rad2,
        sigma_arcsec = params.obs_noise_sigma_arcsec,
        weight_threshold = params.weight_threshold,
        max_arcsec = params.max_arcsec,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_params() -> CalibrationParams {
        CalibrationParams {
            max_arcsec: 900.0,
            obs_noise_sigma_arcsec: 0.1,
            weight_threshold: 0.99,
            gate_chi2: 23.0,
            search_region_chi2: 400.0,
            weight_floor: 1e-4,
            q0: 1e-16,
            dt_ref: 1.0,
        }
    }

    #[test]
    fn top_param_by_last_delta_picks_the_max() {
        let report = CalibrationReport {
            rounds: vec![RoundResult {
                round_index: 0,
                sample_size: 10,
                n_failing: 1,
                recall: 0.9,
                cost_mean_search_radius_arcsec: 12.0,
                params: sample_params(),
                param_deltas: vec![
                    ("max_arcsec".to_string(), 0.05),
                    ("gate_chi2".to_string(), 0.4),
                    ("dt_ref".to_string(), 0.1),
                ],
            }],
            final_params: sample_params(),
        };

        assert_eq!(report.top_param_by_last_delta(), Some(("gate_chi2", 0.4)));
    }

    #[test]
    fn top_param_by_last_delta_none_when_no_rounds() {
        let report = CalibrationReport {
            rounds: vec![],
            final_params: sample_params(),
        };
        assert_eq!(report.top_param_by_last_delta(), None);
    }

    #[test]
    fn yaml_snippet_contains_every_field() {
        let snippet = to_yaml_snippet(&sample_params());
        assert!(snippet.contains("max_arcsec: 900"));
        assert!(snippet.contains("gate_chi2: 23"));
        assert!(snippet.contains("WeightThreshold: 0.99"));
    }
}
