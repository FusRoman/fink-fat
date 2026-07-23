//! The tunable parameter vector for KF-bank calibration, and the generic
//! [`ParamSpec`] machinery [`crate::kf_calibration::search`]'s coordinate
//! descent uses to perturb one field at a time without per-field
//! boilerplate in the search loop itself.

use fink_fat_engine::engine_config::{
    kalman_context::KalmanContext, kf_bank_config::KFBankConfig, main_config::EngineConfig,
    night_advance_params::NightAdvanceParams, single_kalman_config::KalmanConfig,
};
use fink_fat_engine::topocentric_kf::kalman_bank::ellipse_region_finder::{
    radius_strategy::{MixOrMax, RadiusStrategy},
    top_k::TopK,
};

const RAD_TO_ARCSEC: f64 = 3600.0 * 180.0 / std::f64::consts::PI;
const ARCSEC_TO_RAD: f64 = 1.0 / RAD_TO_ARCSEC;

/// The eight knobs [`crate::kf_calibration`] calibrates, spanning three
/// underlying config types (see each field's doc for its source):
/// search-region sizing (`max_arcsec`, `obs_noise_sigma_arcsec`,
/// `weight_threshold`), hypothesis-bank gating
/// (`gate_chi2`, `search_region_chi2`, `weight_floor`), and the filter's own
/// process-noise model (`q0`, `dt_ref`).
///
/// Kept as a flat, independent `f64` vector (rather than nesting the real
/// config types) so [`ParamSpec`]'s getter/setter closures and the
/// coordinate-descent loop in [`crate::kf_calibration::search`] can treat
/// every field uniformly.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CalibrationParams {
    /// `NightAdvanceParams.radius_strategy`'s `Clamped.max_arcsec` — the hard
    /// cap on the predicted search-region radius, in arcseconds. The primary
    /// lever: shrinking it directly shrinks how many candidate observations
    /// (and thus branches) a search can ever pull in.
    pub max_arcsec: f64,
    /// `NightAdvanceParams.obs_noise`'s 1-sigma value (symmetric RA/Dec), in
    /// arcseconds. Stored/applied at 1-sigma for readability; squared into
    /// the `[rad^2, rad^2]` pair the engine expects.
    pub obs_noise_sigma_arcsec: f64,
    /// `NightAdvanceParams.top_k`'s `WeightThreshold(theta)` credible-mass
    /// cutoff, in `(0, 1]`.
    pub weight_threshold: f64,
    /// `KFBankConfig.gate_chi2` — the update-time chi-square gate.
    pub gate_chi2: f64,
    /// `KFBankConfig.search_region_chi2` — the search-region sizing
    /// chi-square factor (decoupled from `gate_chi2`).
    pub search_region_chi2: f64,
    /// `KFBankConfig.weight_floor` — relative weight floor for hypothesis
    /// pruning.
    pub weight_floor: f64,
    /// `KalmanConfig.q0` (via `KalmanContext.config`) — baseline
    /// acceleration process-noise PSD, AU² day⁻³.
    pub q0: f64,
    /// `KalmanConfig.dt_ref` (via `KalmanContext.config`) — perturbation
    /// scaling reference interval, days.
    pub dt_ref: f64,
}

impl CalibrationParams {
    /// Extract the starting point for calibration from a loaded
    /// [`EngineConfig`]/[`KalmanContext`] — i.e. whatever the user already
    /// has in their config file.
    ///
    /// Falls back to `max_arcsec = 30 * 60.0` (30 arcminutes, the engine's
    /// own default — see [`NightAdvanceParams::default`]) if
    /// `radius_strategy` isn't `Clamped`, since the other strategies have no
    /// `max_arcsec` to read: calibration always operates in `Clamped` form
    /// (see [`Self::apply`]).
    pub fn from_engine_config(config: &EngineConfig, context: &KalmanContext) -> Self {
        Self::from_parts(
            &config.advance_params,
            &config.kfbank_config,
            &context.config,
        )
    }

    /// Pure variant of [`Self::from_engine_config`], operating on the three
    /// underlying config values directly instead of the types that own them
    /// ([`EngineConfig`], [`KalmanContext`]). Doesn't touch `KalmanContext`
    /// at all, so — unlike `from_engine_config` — it never needs a live
    /// ephemeris and is safe to call from a fast, offline unit test (see
    /// this crate's [`kalman_traj`](crate::kalman_traj) module doc on why a
    /// real `KalmanContext` can't be built in one).
    ///
    /// Falls back to `max_arcsec = 30 * 60.0` (30 arcminutes, the engine's
    /// own default — see [`NightAdvanceParams::default`]) if
    /// `radius_strategy` isn't `Clamped`, since the other strategies have no
    /// `max_arcsec` to read: calibration always operates in `Clamped` form
    /// (see [`Self::apply`]).
    pub fn from_parts(
        advance_params: &NightAdvanceParams,
        bank_config: &KFBankConfig,
        kalman_config: &KalmanConfig,
    ) -> Self {
        let max_arcsec = match advance_params.radius_strategy {
            RadiusStrategy::Clamped { max_arcsec, .. } => max_arcsec,
            RadiusStrategy::MaxEllipse | RadiusStrategy::MixtureCovariance => 30.0 * 60.0,
        };
        let obs_noise_sigma_arcsec = (advance_params.obs_noise[0].max(0.0)).sqrt() * RAD_TO_ARCSEC;
        let weight_threshold = match advance_params.top_k {
            TopK::WeightThreshold(theta) => theta,
            TopK::All | TopK::Map | TopK::Best(_) => 0.99,
        };

        Self {
            max_arcsec,
            obs_noise_sigma_arcsec,
            weight_threshold,
            gate_chi2: bank_config.gate_chi2,
            search_region_chi2: bank_config.search_region_chi2,
            weight_floor: bank_config.weight_floor,
            q0: kalman_config.q0,
            dt_ref: kalman_config.dt_ref,
        }
    }

    /// Write this parameter set into a clone of `base_config`'s
    /// `advance_params`/`kfbank_config` (every other field left untouched).
    ///
    /// `radius_strategy` is always written back as `Clamped { inner:
    /// MixtureCovariance, max_arcsec }`, regardless of what `base_config` had
    /// — calibration only ever tunes the clamp, and `MixtureCovariance` is
    /// the engine's own default inner strategy.
    pub fn apply(&self, base_config: &EngineConfig) -> EngineConfig {
        let mut config = base_config.clone();
        config.advance_params.radius_strategy = RadiusStrategy::Clamped {
            inner: MixOrMax::MixtureCovariance,
            max_arcsec: self.max_arcsec,
        };
        let sigma_rad = self.obs_noise_sigma_arcsec * ARCSEC_TO_RAD;
        config.advance_params.obs_noise = [sigma_rad * sigma_rad, sigma_rad * sigma_rad];
        config.advance_params.top_k = TopK::WeightThreshold(self.weight_threshold);
        config.kfbank_config.gate_chi2 = self.gate_chi2;
        config.kfbank_config.search_region_chi2 = self.search_region_chi2;
        config.kfbank_config.weight_floor = self.weight_floor;
        config
    }

    /// Clone `base_ctx` with `q0`/`dt_ref` overridden.
    ///
    /// Cheap: `KalmanContext`'s ephemeris is `Arc`-shared, so this never
    /// reloads it — see the module-level warning in
    /// [`crate::kf_calibration`] about never calling
    /// `EngineConfig::build_context`/`KalmanContextConfig::build` per
    /// candidate.
    pub fn build_context(&self, base_ctx: &KalmanContext) -> KalmanContext {
        let mut ctx = base_ctx.clone();
        ctx.config = self.apply_kalman_config(&ctx.config);
        ctx
    }

    /// Pure part of [`Self::build_context`]: `base`, with `q0`/`dt_ref`
    /// overridden. Split out so it's testable without a live
    /// `KalmanContext` (see [`Self::from_parts`]'s doc for why that matters).
    pub fn apply_kalman_config(&self, base: &KalmanConfig) -> KalmanConfig {
        let mut cfg = base.clone();
        cfg.q0 = self.q0;
        cfg.dt_ref = self.dt_ref;
        cfg
    }
}

/// Reads one [`CalibrationParams`] field.
type ParamGetter = Box<dyn Fn(&CalibrationParams) -> f64 + Sync>;
/// Writes one [`CalibrationParams`] field.
type ParamSetter = Box<dyn Fn(&mut CalibrationParams, f64) + Sync>;

/// One tunable field of [`CalibrationParams`], described generically enough
/// that [`crate::kf_calibration::search`]'s coordinate descent can iterate
/// over [`default_param_specs`] without a per-field match arm.
pub struct ParamSpec {
    /// Human-readable name, used in [`crate::kf_calibration::report`] and
    /// `--params` CLI filtering.
    pub name: &'static str,
    pub get: ParamGetter,
    pub set: ParamSetter,
    pub min: f64,
    pub max: f64,
    /// Whether this field is read anywhere by `KFBank::step_with_geometry`
    /// (gating/pruning/propagation) — `false` — or only by
    /// `predict_search_region`/`search_region` (a diagnostic derived from
    /// the trajectory, never fed back into it) — `true`.
    ///
    /// `crate::kf_calibration::search::coordinate_descent` uses this to
    /// decide whether sweeping this field's candidates requires re-running
    /// the whole Kalman filter loop (`false`) or can reuse a cached
    /// [`crate::kf_calibration::objective::ReferenceRun`] and only
    /// recompute the search-region diagnostic (`true`) — see that module's
    /// doc for why this split is safe.
    pub is_diagnostic_only: bool,
}

impl ParamSpec {
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: &'static str,
        min: f64,
        max: f64,
        is_diagnostic_only: bool,
        get: impl Fn(&CalibrationParams) -> f64 + Sync + 'static,
        set: impl Fn(&mut CalibrationParams, f64) + Sync + 'static,
    ) -> Self {
        Self {
            name,
            get: Box::new(get),
            set: Box::new(set),
            min,
            max,
            is_diagnostic_only,
        }
    }

    /// Log-spaced candidate values around `current` (×0.7, ×0.85, current,
    /// ×1.15, ×1.3), clamped to `[min, max]` and deduplicated (candidates
    /// that clamp to the same value, e.g. near a bound, are only tried once).
    pub fn candidates(&self, current: f64) -> Vec<f64> {
        const FACTORS: [f64; 5] = [0.7, 0.85, 1.0, 1.15, 1.3];
        let mut values: Vec<f64> = FACTORS
            .iter()
            .map(|f| (current * f).clamp(self.min, self.max))
            .collect();
        values.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);
        values
    }
}

/// Every calibrated field, in the order coordinate descent sweeps them.
///
/// Dynamics-affecting fields first (`gate_chi2`, `weight_floor`, `q0`,
/// `dt_ref` — read by `KFBank::step_with_geometry`, changing them requires a
/// full KF re-run), then diagnostic-only fields (`max_arcsec`,
/// `obs_noise_sigma_arcsec`, `weight_threshold`, `search_region_chi2` — read
/// only by the search-region diagnostic, cheaply re-swept against a cached
/// [`crate::kf_calibration::objective::ReferenceRun`] once the
/// dynamics-affecting fields have settled for this sweep — see
/// [`ParamSpec::is_diagnostic_only`] and
/// `crate::kf_calibration::search::coordinate_descent`). This grouping —
/// not the within-group order — is what `coordinate_descent` relies on to
/// build the reference-run cache once per sweep, right before the first
/// diagnostic-only spec.
///
/// Bounds are deliberately generous (an order of magnitude either side of
/// sane defaults): the search never has to hug a bound in practice, it's
/// just a numerical safety net against a runaway candidate.
pub fn default_param_specs() -> Vec<ParamSpec> {
    vec![
        // ── Dynamics-affecting (require a full KF re-run) ──────────────
        ParamSpec::new(
            "gate_chi2",
            5.0,
            200.0,
            false,
            |p| p.gate_chi2,
            |p, v| p.gate_chi2 = v,
        ),
        ParamSpec::new(
            "weight_floor",
            1e-8,
            1e-2,
            false,
            |p| p.weight_floor,
            |p, v| p.weight_floor = v,
        ),
        ParamSpec::new("q0", 1e-20, 1e-10, false, |p| p.q0, |p, v| p.q0 = v),
        ParamSpec::new(
            "dt_ref",
            0.01,
            30.0,
            false,
            |p| p.dt_ref,
            |p, v| p.dt_ref = v,
        ),
        // ── Diagnostic-only (cheap re-sweep against a cached reference run) ──
        ParamSpec::new(
            "max_arcsec",
            1.0,
            3600.0 * 6.0,
            true,
            |p| p.max_arcsec,
            |p, v| p.max_arcsec = v,
        ),
        ParamSpec::new(
            "obs_noise_sigma_arcsec",
            0.01,
            60.0,
            true,
            |p| p.obs_noise_sigma_arcsec,
            |p, v| p.obs_noise_sigma_arcsec = v,
        ),
        ParamSpec::new(
            "weight_threshold",
            0.5,
            0.999999,
            true,
            |p| p.weight_threshold,
            |p, v| p.weight_threshold = v,
        ),
        ParamSpec::new(
            "search_region_chi2",
            5.0,
            2000.0,
            true,
            |p| p.search_region_chi2,
            |p, v| p.search_region_chi2 = v,
        ),
    ]
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

    // `KalmanContext::build`/`EngineConfig::build_context` load a real
    // ephemeris (network I/O, can panic offline — see this crate's
    // `kalman_traj` module doc). Round-trip tests below therefore go through
    // `apply`/`apply_kalman_config`/`from_parts`, which only ever touch
    // plain, ephemeris-free config values.

    #[test]
    fn apply_then_from_parts_round_trips() {
        let base = EngineConfig::default();
        let base_kalman_config = KalmanConfig::default();
        let params = sample_params();

        let derived_config = params.apply(&base);
        let derived_kalman_config = params.apply_kalman_config(&base_kalman_config);

        let round_tripped = CalibrationParams::from_parts(
            &derived_config.advance_params,
            &derived_config.kfbank_config,
            &derived_kalman_config,
        );

        assert!((round_tripped.max_arcsec - params.max_arcsec).abs() < 1e-9);
        assert!(
            (round_tripped.obs_noise_sigma_arcsec - params.obs_noise_sigma_arcsec).abs() < 1e-9
        );
        assert!((round_tripped.weight_threshold - params.weight_threshold).abs() < 1e-9);
        assert!((round_tripped.gate_chi2 - params.gate_chi2).abs() < 1e-9);
        assert!((round_tripped.search_region_chi2 - params.search_region_chi2).abs() < 1e-9);
        assert!((round_tripped.weight_floor - params.weight_floor).abs() < 1e-9);
        assert!((round_tripped.q0 - params.q0).abs() < 1e-30);
        assert!((round_tripped.dt_ref - params.dt_ref).abs() < 1e-9);
    }

    #[test]
    fn apply_kalman_config_only_touches_q0_and_dt_ref() {
        let base = KalmanConfig::default();
        let params = sample_params();

        let derived = params.apply_kalman_config(&base);

        assert_eq!(derived.q0, params.q0);
        assert_eq!(derived.dt_ref, params.dt_ref);
    }

    #[test]
    fn candidates_are_clamped_and_deduplicated() {
        let spec = ParamSpec::new(
            "x",
            0.0,
            10.0,
            true,
            |p| p.max_arcsec,
            |p, v| p.max_arcsec = v,
        );
        let values = spec.candidates(9.5);
        assert!(values.iter().all(|v| (0.0..=10.0).contains(v)));
        // 9.5 * 1.15 and 9.5 * 1.3 both clamp to 10.0 — deduplicated.
        assert_eq!(values.last(), Some(&10.0));
        assert!(values.windows(2).all(|w| w[0] <= w[1]));
    }
}
