//! Running a fit, and the shape of what comes out of it.
//!
//! Both fit paths — the single-lineage page and the bulk job — go through
//! [`fit_seedless`] / [`fit_from_seed`] and receive a [`FitProduct`]. That is
//! the point of this module: the two paths have twice drifted apart on
//! differences invisible from the outside (a missing trajectory index that
//! silently disabled the batch-RMS correction, then a stray RNG draw that
//! changed which Gauss solution seeded the correction), each time making the
//! same branch converge on one path and diverge on the other.

use serde::{Deserialize, Serialize};

#[cfg(feature = "server")]
use outfit::constants::FitOrbitResult;
#[cfg(feature = "server")]
use outfit::differential_orbit_correction::{
    differential_correction, run_differential_correction, DifferentialCorrectionConfig,
    DifferentialCorrectionOutput, ObsFitData, ObsSelection,
};
#[cfg(feature = "server")]
use photom::observation_dataset::observation::Observation;
#[cfg(feature = "server")]
use rand::{rngs::SmallRng, SeedableRng};

#[cfg(feature = "server")]
use super::params::FIT_RNG_SEED;

/// How a stored fit was actually obtained — the `orbit_fits.fit_method` column.
///
/// `outfit`'s `differential_correction` never reports a diverged correction as
/// an error: it falls back to the preliminary Gauss orbit and returns it as a
/// success. That fallback is what [`Self::IodOnly`] records, and it is the
/// difference between "this orbit was least-squares fitted" and "this orbit is
/// a preliminary estimate the correction could not improve on".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FitMethod {
    DifferentialCorrection,
    IodOnly,
}

impl FitMethod {
    /// Parses the stored `orbit_fits.fit_method` value. Anything unrecognised
    /// reads as [`Self::IodOnly`]: the column is free-form `TEXT` written by
    /// this crate, so a defensive fallback beats failing a whole query over a
    /// value that can only be one of two strings.
    pub fn from_column(s: &str) -> Self {
        match s {
            "differential_correction" => Self::DifferentialCorrection,
            _ => Self::IodOnly,
        }
    }

    /// The value written to `orbit_fits.fit_method` — the single definition of
    /// those two strings, which used to be spelled out at each insert site.
    pub fn as_column(self) -> &'static str {
        match self {
            Self::DifferentialCorrection => "differential_correction",
            Self::IodOnly => "iod_only",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::DifferentialCorrection => "Differential correction",
            Self::IodOnly => "Gauss IOD only",
        }
    }
}

/// Above this normalised RMS a numerically-finished fit is not considered
/// converged — a least-squares solution whose residuals are ten times their
/// own uncertainties has not really fitted anything.
pub const MAX_CONVERGED_RMS: f64 = 10.0;

/// Whether a fit is worth calling converged, for `orbit_fits.converged`.
///
/// # Arguments
///
/// - `fit_method` — [`FitMethod::IodOnly`] never counts: the correction
///   diverged and the orbit is the preliminary Gauss estimate.
/// - `normalised_rms` — the fit's own RMS, which must be finite and below
///   [`MAX_CONVERGED_RMS`].
pub fn is_converged(fit_method: FitMethod, normalised_rms: f64) -> bool {
    fit_method == FitMethod::DifferentialCorrection
        && normalised_rms.is_finite()
        && normalised_rms < MAX_CONVERGED_RMS
}

/// One Keplerian orbital element with its 1-sigma uncertainty (`None` when the
/// covariance wasn't propagated, e.g. an IOD-only orbit).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KeplerianView {
    pub semi_major_axis_au: f64,
    pub eccentricity: f64,
    pub inclination_deg: f64,
    pub ascending_node_longitude_deg: f64,
    pub periapsis_argument_deg: f64,
    pub mean_anomaly_deg: f64,

    pub sigma_semi_major_axis_au: Option<f64>,
    pub sigma_eccentricity: Option<f64>,
    pub sigma_inclination_deg: Option<f64>,
    pub sigma_ascending_node_longitude_deg: Option<f64>,
    pub sigma_periapsis_argument_deg: Option<f64>,
    pub sigma_mean_anomaly_deg: Option<f64>,
}

/// Whether an observation still took part in the final iteration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObsSelectionView {
    Kept,
    Rejected,
}

/// One observation's contribution to the fit, as stored in
/// `orbit_fits.residuals` and plotted by the fit page.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObsResidual {
    pub obs_id: i64,
    pub mjd_tt: f64,
    pub residual_ra_arcsec: f64,
    pub residual_dec_arcsec: f64,
    pub chi: f64,
    pub selection: ObsSelectionView,
}

/// Everything a finished fit contributes to storage and display, whichever
/// path produced it and whichever way it ended.
#[cfg(feature = "server")]
pub struct FitProduct {
    pub fit_method: FitMethod,
    pub elements: outfit::EquinoctialElements,
    pub keplerian: KeplerianView,
    /// Flattened covariance matrix; empty for an orbit that carries none.
    pub covariance: Vec<f64>,
    pub normalised_rms: f64,
    pub total_newton_iterations: usize,
    pub num_measurements: usize,
    pub n_observations_used: usize,
    pub n_observations_rejected: usize,
    /// Empty unless the fit ran with [`Diagnostics::Full`].
    pub residuals: Vec<ObsResidual>,
}

#[cfg(feature = "server")]
impl FitProduct {
    /// See [`is_converged`].
    pub fn converged(&self) -> bool {
        is_converged(self.fit_method, self.normalised_rms)
    }

    /// Builds a product from `outfit`'s own fit result, without
    /// per-observation diagnostics.
    ///
    /// Covers both outcomes of [`differential_correction`]: a converged
    /// correction, and the Gauss fallback it returns when the correction
    /// diverged.
    ///
    /// # Arguments
    ///
    /// - `fit` — what `outfit` returned.
    /// - `n_observations` — how many observations were submitted; the
    ///   post-rejection count is unknown here, so all of them are reported as
    ///   used.
    ///
    /// # Errors
    ///
    /// An orbit that converts to neither equinoctial nor Keplerian elements —
    /// a degenerate solution the caller should count as a failed fit rather
    /// than store with empty elements.
    pub fn from_fit_result(fit: &FitOrbitResult, n_observations: usize) -> Result<Self, String> {
        let fit_method = match fit {
            FitOrbitResult::DifferentialCorrection(_) => FitMethod::DifferentialCorrection,
            FitOrbitResult::IODGauss(_) => FitMethod::IodOnly,
        };
        let orbit = fit.orbital_elements();
        let (elements, covariance) = equinoctial_with_covariance(orbit)?;

        Ok(Self {
            fit_method,
            elements,
            keplerian: keplerian_view(orbit)?,
            covariance,
            normalised_rms: fit.orbit_quality(),
            total_newton_iterations: 0,
            num_measurements: n_observations * 2,
            n_observations_used: n_observations,
            n_observations_rejected: 0,
            residuals: Vec::new(),
        })
    }

    /// Builds a product from a differential correction's full output, keeping
    /// the per-observation residuals and the outlier selection.
    ///
    /// # Arguments
    ///
    /// - `dc_output` — the correction's output.
    /// - `observations` — the observations it was handed, in the same order,
    ///   so each residual can be attributed to its observation.
    /// - `normalised_rms` — the RMS to report. Passed in rather than read off
    ///   `dc_output` because [`fit_seedless`]'s diagnostics pass re-derives the
    ///   correction from its own solution, and the headline number must stay
    ///   the one the fit itself produced — that is what makes a single-lineage
    ///   fit comparable to the bulk row for the same branch.
    pub fn from_dc_output(
        dc_output: DifferentialCorrectionOutput,
        observations: &[Observation],
        normalised_rms: f64,
    ) -> Result<Self, String> {
        let residuals: Vec<ObsResidual> = observations
            .iter()
            .zip(dc_output.final_obs_fit_data.iter())
            .map(|(obs, fit_data)| ObsResidual {
                obs_id: *obs.id() as i64,
                mjd_tt: obs.mjd_tt(),
                residual_ra_arcsec: fit_data.residual_ra.to_degrees() * 3600.0,
                residual_dec_arcsec: fit_data.residual_dec.to_degrees() * 3600.0,
                chi: fit_data.chi,
                selection: if fit_data.selection == ObsSelection::Active {
                    ObsSelectionView::Kept
                } else {
                    ObsSelectionView::Rejected
                },
            })
            .collect();

        let n_observations_used = dc_output
            .final_obs_fit_data
            .iter()
            .filter(|o| o.selection == ObsSelection::Active)
            .count();
        let n_observations_rejected = dc_output.final_obs_fit_data.len() - n_observations_used;
        let total_newton_iterations = dc_output.total_newton_iterations;
        let num_measurements = dc_output.num_measurements;
        let elements = dc_output.elements.clone();
        let covariance: Vec<f64> = dc_output.uncertainty.covariance.iter().copied().collect();
        // `outfit`'s own conversion, so the Keplerian 1-sigmas are derived from
        // the correction's covariance the way the crate intends.
        let orbit: outfit::OrbitalElements = dc_output.into();

        Ok(Self {
            fit_method: FitMethod::DifferentialCorrection,
            elements,
            keplerian: keplerian_view(&orbit)?,
            covariance,
            normalised_rms,
            total_newton_iterations,
            num_measurements,
            n_observations_used,
            n_observations_rejected,
            residuals,
        })
    }
}

/// Whether a fit should also produce per-observation diagnostics.
#[cfg(feature = "server")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Diagnostics {
    /// Re-derive the correction from its own solution to recover residuals,
    /// outlier selection and iteration counts. Converging from the optimum
    /// takes one or two iterations, so this roughly doubles a single fit —
    /// affordable for one branch, not for a whole bulk run.
    Full,
    /// Orbit and RMS only.
    Skip,
}

/// The deterministic RNG a branch's Gauss IOD draws its Monte-Carlo noise
/// realizations from.
///
/// Fitting the same branch twice only gives the same orbit because both runs
/// start from this exact stream — and a single-lineage fit only reproduces the
/// bulk row for that branch because both call this same function. See
/// [`FIT_RNG_SEED`].
#[cfg(feature = "server")]
pub fn branch_rng(branch_id: i64) -> SmallRng {
    SmallRng::seed_from_u64(FIT_RNG_SEED ^ super::dataset::traj_id(branch_id).stable_hash())
}

/// Fits a branch from scratch: Gauss IOD for the starting orbit, then the
/// differential correction.
///
/// **Always go through `outfit`'s `differential_correction` wrapper for this**,
/// never through `FitIOD::fit_iod`: `fit_iod` calls `prepare_iod`, which draws
/// a `u64` from the RNG and discards it before Gauss runs. That offsets the
/// whole noise stream by one draw, so a different triplet wins, a different
/// Gauss orbit (and reference epoch) seeds the correction, and the fit can
/// diverge where the bulk fit converged on the very same branch with the very
/// same parameters. `run_iod_on_observations`, the function that would let us
/// seed a correction ourselves without that draw, is `pub(crate)` in `outfit`.
///
/// # Arguments
///
/// - `observations` — the branch's observations, already error-corrected
///   (`super::dataset::apply_error_model`).
/// - `cache` — the geometry cache built from the same corrected dataset.
/// - `branch_id` — picks the RNG stream, see [`branch_rng`].
/// - `diagnostics` — whether to run the extra pass that recovers residuals.
///
/// # Returns
///
/// A [`FitProduct`], tagged [`FitMethod::IodOnly`] when the correction diverged
/// and `outfit` fell back to the Gauss orbit.
///
/// # Errors
///
/// The Gauss IOD itself failing (no viable orbit, no feasible triplet), or the
/// resulting orbit not converting to equinoctial/Keplerian elements.
#[cfg(feature = "server")]
pub fn fit_seedless(
    observations: &[Observation],
    cache: &outfit::cache::OutfitCache,
    jpl: &outfit::JPLEphem,
    iod_params: &outfit::IODParams,
    dc_config: &DifferentialCorrectionConfig,
    branch_id: i64,
    diagnostics: Diagnostics,
) -> Result<FitProduct, String> {
    let mut rng = branch_rng(branch_id);
    let fit = differential_correction(
        observations,
        cache,
        jpl,
        iod_params,
        dc_config,
        None,
        &mut rng,
    )
    .map_err(|e| e.to_string())?;

    let (elements, normalised_rms) = match (diagnostics, &fit) {
        (Diagnostics::Full, FitOrbitResult::DifferentialCorrection((elements, rms))) => {
            (elements.clone(), *rms)
        }
        // Either diagnostics weren't asked for, or the correction diverged and
        // `fit` holds the Gauss fallback — nothing to re-derive in that case.
        _ => return FitProduct::from_fit_result(&fit, observations.len()),
    };

    // Re-run the correction from its own solution purely to recover what the
    // wrapper drops (per-observation residuals, outlier selection, iteration
    // count). It starts at the optimum, so it settles immediately; should it
    // fail anyway, the orbit itself is still good and we report it without the
    // extra detail rather than losing the fit.
    let seed = match to_equinoctial(elements) {
        Ok(seed) => seed,
        Err(_) => return FitProduct::from_fit_result(&fit, observations.len()),
    };
    match run_differential_correction(
        observations,
        &obs_fit_data(observations),
        &seed,
        cache,
        jpl,
        dc_config,
    ) {
        Ok(dc_output) => FitProduct::from_dc_output(dc_output, observations, normalised_rms),
        Err(_) => FitProduct::from_fit_result(&fit, observations.len()),
    }
}

/// Fits a branch starting from an orbit the caller already has — the
/// single-lineage page's "refine the current Kalman estimate" mode.
///
/// Unlike [`fit_seedless`] there is no Gauss fallback to land on: a diverging
/// correction is an error, since the only other orbit available is the one the
/// caller passed in.
///
/// # Arguments
///
/// - `observations` / `cache` — as in [`fit_seedless`].
/// - `seed` — the starting orbit.
///
/// # Errors
///
/// The correction diverging, stagnating or failing to invert its normal matrix.
#[cfg(feature = "server")]
pub fn fit_from_seed(
    observations: &[Observation],
    cache: &outfit::cache::OutfitCache,
    jpl: &outfit::JPLEphem,
    dc_config: &DifferentialCorrectionConfig,
    seed: &outfit::EquinoctialElements,
) -> Result<FitProduct, String> {
    let dc_output = run_differential_correction(
        observations,
        &obs_fit_data(observations),
        seed,
        cache,
        jpl,
        dc_config,
    )
    .map_err(|e| e.to_string())?;

    let normalised_rms = dc_output.normalised_rms;
    FitProduct::from_dc_output(dc_output, observations, normalised_rms)
}

/// Per-observation weights for the correction, straight from the astrometric
/// uncertainties the error model left on each observation.
#[cfg(feature = "server")]
fn obs_fit_data(observations: &[Observation]) -> Vec<ObsFitData> {
    observations
        .iter()
        .map(|obs| ObsFitData::new(obs.equ_coord().ra_error, obs.equ_coord().dec_error))
        .collect()
}

/// Converts orbital elements to the equinoctial representation the correction
/// seeds from.
///
/// # Errors
///
/// A conversion `outfit` refuses (degenerate orbit), as a display string.
#[cfg(feature = "server")]
pub fn to_equinoctial(
    orbit: outfit::OrbitalElements,
) -> Result<outfit::EquinoctialElements, String> {
    orbit
        .to_equinoctial()
        .map_err(|e| e.to_string())?
        .as_equinoctial()
        .ok_or_else(|| "failed to convert orbital elements to equinoctial elements".to_string())
}

/// Converts orbital elements to their Keplerian form.
///
/// # Errors
///
/// A conversion `outfit` refuses (degenerate orbit), as a display string.
#[cfg(feature = "server")]
pub fn to_keplerian(orbit: outfit::OrbitalElements) -> Result<outfit::KeplerianElements, String> {
    orbit
        .to_keplerian()
        .map_err(|e| e.to_string())?
        .as_keplerian()
        .ok_or_else(|| "failed to convert orbital elements to Keplerian elements".to_string())
}

/// Splits orbital elements into their equinoctial form and a flattened
/// covariance matrix (empty when the orbit carries none, as an IOD orbit does).
#[cfg(feature = "server")]
fn equinoctial_with_covariance(
    orbit: &outfit::OrbitalElements,
) -> Result<(outfit::EquinoctialElements, Vec<f64>), String> {
    match orbit.to_equinoctial().map_err(|e| e.to_string())? {
        outfit::OrbitalElements::Equinoctial {
            elements,
            covariance,
            ..
        } => Ok((
            elements,
            covariance
                .map(|c| c.matrix.iter().copied().collect())
                .unwrap_or_default(),
        )),
        _ => Err("expected equinoctial orbital elements after conversion".to_string()),
    }
}

/// The display/storage view of an orbit's Keplerian elements, with 1-sigma
/// uncertainties when the orbit carries a covariance.
#[cfg(feature = "server")]
fn keplerian_view(orbit: &outfit::OrbitalElements) -> Result<KeplerianView, String> {
    let (elements, uncertainty) = match orbit.to_keplerian().map_err(|e| e.to_string())? {
        outfit::OrbitalElements::Keplerian {
            elements,
            uncertainty,
            ..
        } => (elements, uncertainty),
        _ => return Err("expected Keplerian orbital elements after conversion".to_string()),
    };
    let uncertainty = uncertainty.as_ref();

    Ok(KeplerianView {
        semi_major_axis_au: elements.semi_major_axis,
        eccentricity: elements.eccentricity,
        inclination_deg: elements.inclination.to_degrees(),
        ascending_node_longitude_deg: elements.ascending_node_longitude.to_degrees(),
        periapsis_argument_deg: elements.periapsis_argument.to_degrees(),
        mean_anomaly_deg: elements.mean_anomaly.to_degrees(),
        sigma_semi_major_axis_au: uncertainty.map(|u| u.semi_major_axis),
        sigma_eccentricity: uncertainty.map(|u| u.eccentricity),
        sigma_inclination_deg: uncertainty.map(|u| u.inclination.to_degrees()),
        sigma_ascending_node_longitude_deg: uncertainty
            .map(|u| u.ascending_node_longitude.to_degrees()),
        sigma_periapsis_argument_deg: uncertainty.map(|u| u.periapsis_argument.to_degrees()),
        sigma_mean_anomaly_deg: uncertainty.map(|u| u.mean_anomaly.to_degrees()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_method_column_round_trips() {
        for method in [FitMethod::DifferentialCorrection, FitMethod::IodOnly] {
            assert_eq!(FitMethod::from_column(method.as_column()), method);
        }
    }

    #[test]
    fn unknown_fit_method_reads_as_iod_only() {
        assert_eq!(FitMethod::from_column("something_else"), FitMethod::IodOnly);
    }

    #[test]
    fn iod_only_is_never_converged() {
        assert!(!is_converged(FitMethod::IodOnly, 0.5));
    }

    #[test]
    fn converged_requires_a_finite_rms_under_the_cap() {
        assert!(is_converged(FitMethod::DifferentialCorrection, 0.5));
        assert!(!is_converged(
            FitMethod::DifferentialCorrection,
            MAX_CONVERGED_RMS
        ));
        assert!(!is_converged(FitMethod::DifferentialCorrection, f64::NAN));
        assert!(!is_converged(
            FitMethod::DifferentialCorrection,
            f64::INFINITY
        ));
    }

    #[cfg(feature = "server")]
    #[test]
    fn branch_rng_is_deterministic_per_branch() {
        use rand::Rng;

        let draw = |branch_id| branch_rng(branch_id).random::<u64>();

        assert_eq!(draw(1661353), draw(1661353));
        assert_ne!(draw(1661353), draw(1661354));
    }
}
