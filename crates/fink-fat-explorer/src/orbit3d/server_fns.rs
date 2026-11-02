//! Server functions backing the 3D orbit views: the homepage's population
//! scatter and the lineage page's single-object orbit tab.
//!
//! The homepage view is positions-only by design — one marker per lineage,
//! no orbit curve — because tracing thousands of full ellipses at once would
//! be unreadable and expensive to ship; the lineage view, showing exactly
//! one object, draws its full ellipse instead. See the plan this feature was
//! built from for the rationale.

use dioxus::prelude::*;
use hifitime::Epoch;

use crate::orbit3d::geometry;
use crate::orbit3d::types::{
    Body3D, HomepageObjects3D, LineageOrbit3D, ObjectPoint3D, ObservationPoint3D,
};

/// The current epoch, as Modified Julian Date in Terrestrial Time — the
/// convention every stored epoch in this crate uses (see
/// `crate::format_epoch::iso_utc`). Used as "now" for every 3D endpoint
/// below, so a homepage/lineage page reload always reflects the current
/// positions rather than a fixed reference epoch.
///
/// # Panics
///
/// Panics if the system clock cannot be read (`hifitime::Epoch::now`
/// failing means the host itself is unusable well beyond this feature).
fn now_mjd_tt() -> f64 {
    Epoch::now()
        .expect("failed to read the system clock")
        .to_mjd_tt_days()
}

/// Starts loading the JPL ephemeris and filling the planets cache in the
/// background, without waiting for it.
///
/// The first [`get_planets_3d`]/[`get_lineage_orbit3d`] call otherwise pays
/// the whole lazy `crate::get_kalman_context` load (JPL/SPK kernels, UT1) at
/// the moment the user first opens a 3D view, so the plot first draws without
/// its planets and then redraws. Called from the first homepage request
/// instead, so that cost is paid while the user is still on another view.
/// Both `get_kalman_context`'s and `get_planets`'s caches deduplicate, so a
/// request racing this task simply waits on the same initialisation.
///
/// Explicitly `#[cfg(feature = "server")]`, like [`keplerian_from_branch`]:
/// it is a plain free function outside any `#[server]` macro.
#[cfg(feature = "server")]
pub(crate) fn warm_up_planets() {
    tokio::spawn(async {
        crate::orbit3d::ephem_provider::get_planets(now_mjd_tt()).await;
    });
}

/// The planets and tracked perturbers, positioned at the current epoch — see
/// `crate::orbit3d::ephem_provider`.
#[server]
pub async fn get_planets_3d() -> Result<Vec<Body3D>, ServerFnError> {
    use crate::orbit3d::ephem_provider;

    let bodies = ephem_provider::get_planets(now_mjd_tt()).await;
    Ok((*bodies).clone())
}

/// One point per lineage — its best branch's current heliocentric position —
/// for the homepage's 3D scatter, plus how many lineages could not be placed.
/// `None` while the homepage snapshot is still (re)building, mirroring
/// `homepage::dynamic_pop_plot::query_orbital_elements`'s own "warming up"
/// contract so both plots poll the same way.
///
/// A lineage whose best state is not a closed ellipse has no orbit to
/// position it on; it is left out and counted in
/// [`HomepageObjects3D::n_excluded`], so the plot can say why its total is
/// lower than the (a, e) plot's.
#[server]
pub async fn get_homepage_orbit3d() -> Result<Option<HomepageObjects3D>, ServerFnError> {
    let Some(snapshot) = crate::homepage::snapshot::snapshot().await else {
        return Ok(None);
    };

    let now = now_mjd_tt();
    let points: Vec<ObjectPoint3D> = snapshot
        .lineages
        .iter()
        .filter_map(|entry| {
            let branch = &snapshot.branches[entry.best as usize];
            let elems = keplerian_from_branch(branch)?;
            Some(ObjectPoint3D {
                lineage_id: entry.lineage_id,
                family: branch.family,
                tier: branch.quality_tier,
                position: geometry::position_at_epoch(&elems, now),
            })
        })
        .collect();

    Ok(Some(HomepageObjects3D {
        n_excluded: snapshot.lineages.len() - points.len(),
        points,
    }))
}

/// Converts a branch's raw Kalman attributable state (topocentric ra/dec/
/// rho and their rates, plus the observer state it was measured against —
/// the same inputs `homepage::family::classify_from_attributable_state`
/// already converts for the dynamic-family classification) into heliocentric
/// osculating [`geometry::Keplerian`] elements.
///
/// # Returns
///
/// `None` if the branch's attributable state does not resolve to a closed
/// ellipse — not expected for a real tracked object, but not assumed away.
///
/// Explicitly `#[cfg(feature = "server")]`: unlike the `#[server]`-decorated
/// functions above (whose bodies the macro itself compiles only under this
/// feature), this is a plain free function outside any such macro, and its
/// signature names `homepage::snapshot::BranchRow` — a type that only exists
/// under the `server` feature — so it must be gated explicitly to keep a
/// wasm-only client build compiling.
#[cfg(feature = "server")]
fn keplerian_from_branch(
    branch: &crate::homepage::snapshot::BranchRow,
) -> Option<geometry::Keplerian> {
    use nalgebra::{Vector3, Vector6};

    let state = Vector6::new(
        branch.ra,
        branch.dec,
        branch.ra_dot,
        branch.dec_dot,
        branch.rho,
        branch.rho_dot,
    );
    let r_obs = Vector3::new(branch.r_obs_x, branch.r_obs_y, branch.r_obs_z);
    let v_obs = Vector3::new(branch.v_obs_x, branch.v_obs_y, branch.v_obs_z);

    crate::orbit3d::ephem_provider::keplerian_from_attributable_state(
        &state,
        &r_obs,
        &v_obs,
        branch.epoch,
    )
}

/// A single lineage's orbit and current position, plus the planets/
/// perturbers for context.
///
/// # Arguments
///
/// * `lineage_designation` — the lineage's designation string, exactly as
///   `orbit_fit::latest::get_latest_orbit_fit_result` already keys on.
///
/// # Returns
///
/// `None` if the lineage has no `orbit_fits` row yet (it has never been
/// fitted, individually or in bulk) — the caller should invite the user to
/// run a fit rather than showing an empty plot.
#[server]
pub async fn get_lineage_orbit3d(
    lineage_designation: String,
) -> Result<Option<LineageOrbit3D>, ServerFnError> {
    use crate::fit_pipeline::fit::{FitMethod, KeplerianView};
    use crate::orbit3d::ephem_provider;
    use crate::orbit3d::uncertainty::FitCovariance;

    #[derive(sqlx::FromRow)]
    struct Row {
        reference_epoch: f64,
        keplerian: sqlx::types::Json<KeplerianView>,
        semi_major_axis: f64,
        eccentricity_sin_lon: f64,
        eccentricity_cos_lon: f64,
        tan_half_incl_sin_node: f64,
        tan_half_incl_cos_node: f64,
        mean_longitude: f64,
        covariance: Vec<f64>,
        normalised_rms: f64,
        converged: bool,
        fit_method: String,
    }

    let pool = crate::get_pool().await;
    let row: Option<Row> = sqlx::query_as(
        "SELECT reference_epoch, keplerian, semi_major_axis, eccentricity_sin_lon, \
                eccentricity_cos_lon, tan_half_incl_sin_node, tan_half_incl_cos_node, \
                mean_longitude, covariance, normalised_rms, converged, fit_method \
         FROM orbit_fits \
         WHERE lineage_designation = $1 \
         ORDER BY fitted_at DESC \
         LIMIT 1",
    )
    .bind(&lineage_designation)
    .fetch_optional(pool)
    .await
    .map_err(|e| ServerFnError::new(e.to_string()))?;

    let Some(row) = row else {
        return Ok(None);
    };

    let elems = keplerian_from_view(&row.keplerian.0, row.reference_epoch);
    let now = now_mjd_tt();

    let planets = (*ephem_provider::get_planets(now).await).clone();

    // The crosses and the uncertainty cloud are bonuses on top of the
    // orbit: failing to load the observations must not blank the whole view.
    let observations = match crate::lineage_page::observations_table::get_lineage_observations(
        lineage_designation.clone(),
    )
    .await
    {
        Ok(Some(data)) => data.observations,
        Ok(None) => Vec::new(),
        Err(e) => {
            tracing::warn!("orbit3d: failed to load the lineage's observations: {e}");
            Vec::new()
        }
    };

    let last_observation_mjd = observations
        .iter()
        .map(|o| o.mjd_tt)
        .fold(row.reference_epoch, f64::max);
    let fit_covariance = FitCovariance {
        elements: nalgebra::Vector6::new(
            row.semi_major_axis,
            row.eccentricity_sin_lon,
            row.eccentricity_cos_lon,
            row.tan_half_incl_sin_node,
            row.tan_half_incl_cos_node,
            row.mean_longitude,
        ),
        covariance: row.covariance,
        reference_epoch: row.reference_epoch,
        converged: row.converged,
        differential_correction: FitMethod::from_column(&row.fit_method)
            == FitMethod::DifferentialCorrection,
        normalised_rms: row.normalised_rms,
    };

    let object_position = geometry::position_at_epoch(&elems, now);

    let (uncertainty, uncertainty_unavailable_reason) = match lineage_uncertainty(
        &lineage_designation,
        &fit_covariance,
        last_observation_mjd,
        now,
    ) {
        Ok(cloud) => (Some(cloud), None),
        Err(why) => (None, Some(why)),
    };

    Ok(Some(LineageOrbit3D {
        summary: crate::orbit3d::summary::build_orbit_summary(&elems, object_position, &planets),
        object_position,
        object_orbit: geometry::ellipse_points(&elems, LINEAGE_ORBIT_CURVE_SAMPLES),
        observation_points: observation_points(&elems, &observations).await,
        uncertainty,
        uncertainty_unavailable_reason,
        planets,
    }))
}

/// The lineage's orbit uncertainty cloud, drawn from the N-body fit's
/// covariance.
///
/// The cloud is a bonus on top of the orbit: when it cannot be built the
/// reason is returned for the UI rather than failing the view.
///
/// # Arguments
///
/// * `lineage_designation` — seeds the clone sampling, so a lineage's cloud
///   is stable across page loads.
/// * `fit` — the latest stored fit.
/// * `last_observation_mjd` — epoch of the last observation, MJD-TT.
/// * `now_mjd` — the view's epoch, MJD-TT.
///
/// # Returns
///
/// The cloud.
///
/// # Errors
///
/// A sentence saying why there is no cloud (the fit did not converge, is an
/// IOD-only solution, or has no usable covariance).
///
/// Explicitly `#[cfg(feature = "server")]`, like [`keplerian_from_branch`].
#[cfg(feature = "server")]
fn lineage_uncertainty(
    lineage_designation: &str,
    fit: &crate::orbit3d::uncertainty::FitCovariance,
    last_observation_mjd: f64,
    now_mjd: f64,
) -> Result<crate::orbit3d::types::UncertaintyCloud3D, String> {
    use crate::orbit3d::uncertainty;

    let selected = uncertainty::select_covariance(fit)?;
    uncertainty::build_uncertainty_cloud(
        &selected,
        uncertainty::seed_from_designation(lineage_designation),
        last_observation_mjd,
        now_mjd,
    )
}

/// Each observation's heliocentric ecliptic position, for the lineage 3D
/// view's crosses.
///
/// For every observation: the observer's heliocentric position at the
/// observation epoch (real ephemeris, via
/// `EphemState::helio_observer_state`, the same call the Kalman replay
/// makes), plus the measured line of sight — the ra/dec unit vector rotated
/// from equatorial to ecliptic J2000 — at the distance where it passes
/// nearest to `elems`'s predicted position at that epoch
/// ([`geometry::point_on_line_of_sight_nearest`]). Light-time and
/// aberration are ignored: negligible at plot scale.
///
/// An observation whose MPC code is malformed or unknown, or whose observer
/// state cannot be computed, is skipped with a warning so one bad row does
/// not drop every cross.
///
/// # Arguments
///
/// * `elems` — the lineage's fitted orbit, used only to choose each
///   observation's distance along its line of sight.
/// * `observations` — the lineage's observations (ra/dec in radians,
///   MJD-TT).
///
/// # Returns
///
/// One [`ObservationPoint3D`] per resolved observation, in input order: the
/// position plus the observation epoch, the phase angle and the
/// heliocentric/topocentric distances its hover shows.
///
/// Explicitly `#[cfg(feature = "server")]`, like [`keplerian_from_branch`].
#[cfg(feature = "server")]
async fn observation_points(
    elems: &geometry::Keplerian,
    observations: &[crate::lineage_page::observations_table::ObservationRow],
) -> Vec<ObservationPoint3D> {
    use nalgebra::Vector3;
    use outfit::constants::ROT_EQUMJ2000_TO_ECLMJ2000;

    if observations.is_empty() {
        return Vec::new();
    }

    let observatories = crate::get_observatories().await;
    let ephem = crate::get_kalman_context().await.get_ephem();

    observations
        .iter()
        .filter_map(|obs| {
            let code: [u8; 3] = obs.mpc_code_obs.as_bytes().try_into().ok().or_else(|| {
                tracing::warn!("orbit3d: invalid MPC code {:?}, skipping", obs.mpc_code_obs);
                None
            })?;
            let observer = observatories.get(&code).or_else(|| {
                tracing::warn!("orbit3d: unknown MPC code {:?}, skipping", obs.mpc_code_obs);
                None
            })?;
            let state = ephem
                .helio_observer_state(observer, obs.mjd_tt)
                .map_err(|e| {
                    tracing::warn!("orbit3d: observer state failed for obs {}: {e}", obs.id)
                })
                .ok()?;

            let [ex, ey, ez] = geometry::equatorial_unit_vector(obs.ra, obs.dec);
            let direction = ROT_EQUMJ2000_TO_ECLMJ2000 * Vector3::new(ex, ey, ez);
            let observer_pos = state.helio_cart_pos;

            let observer_pos = [observer_pos.x, observer_pos.y, observer_pos.z];
            let position = geometry::point_on_line_of_sight_nearest(
                observer_pos,
                [direction.x, direction.y, direction.z],
                geometry::position_at_epoch(elems, obs.mjd_tt),
            );

            let phase_rad = geometry::phase_angle_rad(position, observer_pos);
            let heliocentric_distance_au = geometry::distance(position, [0.0; 3]);
            let topocentric_distance_au = geometry::distance(position, observer_pos);

            Some(ObservationPoint3D {
                position,
                observer_position: observer_pos,
                mjd_tt: obs.mjd_tt,
                magnitude: obs.magnitude,
                mag_err: obs.mag_err,
                filter: obs.filter,
                mpc_code: obs.mpc_code_obs.clone(),
                elongation_deg: geometry::solar_elongation_rad(position, observer_pos).to_degrees(),
                absolute_magnitude: geometry::absolute_magnitude_hg(
                    obs.magnitude,
                    heliocentric_distance_au,
                    topocentric_distance_au,
                    phase_rad,
                    ABSOLUTE_MAGNITUDE_SLOPE_G,
                ),
                phase_angle_deg: phase_rad.to_degrees(),
                heliocentric_distance_au,
                topocentric_distance_au,
            })
        })
        .collect()
}

/// The H,G slope parameter used for each observation's absolute-magnitude
/// estimate — the customary default for an asteroid of unknown taxonomy.
const ABSOLUTE_MAGNITUDE_SLOPE_G: f64 = 0.15;

/// Number of points sampled for a single lineage's own orbit — denser than
/// [`crate::orbit3d::ephem_provider::TRACKED_BODIES`]'s curves since it is
/// the one shape the lineage page's 3D tab is actually about.
const LINEAGE_ORBIT_CURVE_SAMPLES: usize = 360;

/// [`KeplerianView`]'s fields are already in the same units as
/// [`geometry::Keplerian`] (AU, degrees) — this is a plain field-for-field
/// copy, not a unit conversion, unlike
/// `ephem_provider::keplerian_from_outfit_elements`.
fn keplerian_from_view(
    view: &crate::fit_pipeline::fit::KeplerianView,
    epoch_mjd_tt: f64,
) -> geometry::Keplerian {
    geometry::Keplerian {
        epoch_mjd_tt,
        semi_major_axis_au: view.semi_major_axis_au,
        eccentricity: view.eccentricity,
        inclination_deg: view.inclination_deg,
        ascending_node_longitude_deg: view.ascending_node_longitude_deg,
        periapsis_argument_deg: view.periapsis_argument_deg,
        mean_anomaly_deg: view.mean_anomaly_deg,
    }
}

#[cfg(all(test, feature = "server"))]
mod tests {
    use super::*;
    use crate::fit_pipeline::fit::KeplerianView;
    use crate::homepage::family::DynamicalFamily;
    use crate::homepage::quality_tier::QualityTier;
    use crate::homepage::snapshot::BranchRow;

    /// `keplerian_from_view` must be a plain field-for-field copy — no unit
    /// conversion, since `KeplerianView` is already in degrees/AU.
    #[test]
    fn keplerian_from_view_copies_fields_unchanged() {
        let view = KeplerianView {
            semi_major_axis_au: 2.741,
            eccentricity: 0.187,
            inclination_deg: 9.32,
            ascending_node_longitude_deg: 202.5,
            periapsis_argument_deg: 88.1,
            mean_anomaly_deg: 317.9,
            sigma_semi_major_axis_au: None,
            sigma_eccentricity: None,
            sigma_inclination_deg: None,
            sigma_ascending_node_longitude_deg: None,
            sigma_periapsis_argument_deg: None,
            sigma_mean_anomaly_deg: None,
        };

        let elems = keplerian_from_view(&view, 60_123.5);

        assert_eq!(elems.epoch_mjd_tt, 60_123.5);
        assert_eq!(elems.semi_major_axis_au, view.semi_major_axis_au);
        assert_eq!(elems.eccentricity, view.eccentricity);
        assert_eq!(elems.inclination_deg, view.inclination_deg);
        assert_eq!(
            elems.ascending_node_longitude_deg,
            view.ascending_node_longitude_deg
        );
        assert_eq!(elems.periapsis_argument_deg, view.periapsis_argument_deg);
        assert_eq!(elems.mean_anomaly_deg, view.mean_anomaly_deg);
    }

    fn branch_with_attributable_state(
        ra: f64,
        dec: f64,
        ra_dot: f64,
        dec_dot: f64,
        rho: f64,
        rho_dot: f64,
    ) -> BranchRow {
        BranchRow {
            branch_id: 1,
            lineage_id: 1,
            designation: "test".into(),
            lineage_designation: "test".into(),
            cumulative_llr: 0.0,
            n_real_updates: 0,
            arc_length_days: 0.0,
            n_nights: 0,
            median_inter_night_dt_days: None,
            family: DynamicalFamily::MbOuter,
            semi_major_axis: 0.0,
            eccentricity: 0.0,
            quality_tier: QualityTier::NotFitted,
            ra,
            dec,
            ra_dot,
            dec_dot,
            rho,
            rho_dot,
            epoch: 60_000.0,
            // A roughly Earth-like observer state: 1 AU out along x, moving
            // at the ~0.0172 AU/day of Earth's mean orbital speed along y.
            r_obs_x: 1.0,
            r_obs_y: 0.0,
            r_obs_z: 0.0,
            v_obs_x: 0.0,
            v_obs_y: 0.017_2,
            v_obs_z: 0.0,
        }
    }

    /// A line of sight along the observer-Sun axis (`ra = dec = 0`), with a
    /// slowly changing range, resolves to a bound (`e < 1`) outer main-belt
    /// -like orbit: `keplerian_from_branch` must return `Some` with sane
    /// values, not silently drop the branch.
    ///
    /// `rho = 0.5` AU (heliocentric distance ≈ 1.5 AU) is load-bearing, not
    /// arbitrary: with the observer's Earth-like ~0.0172 AU/day added
    /// straight onto the line-of-sight velocity, anything past
    /// `rho` ≈ 1.0 AU pushes the total speed above the local escape
    /// velocity and the orbit becomes hyperbolic (`keplerian_from_branch`
    /// then correctly returns `None` — that isn't a bug to work around).
    #[test]
    fn keplerian_from_branch_resolves_a_bound_orbit() {
        let branch = branch_with_attributable_state(0.0, 0.0, 0.0, 0.0, 0.5, 0.001);

        let elems = keplerian_from_branch(&branch).expect("expected a closed-ellipse orbit");

        assert!(elems.semi_major_axis_au > 0.0);
        assert!((0.0..1.0).contains(&elems.eccentricity));
        assert_eq!(elems.epoch_mjd_tt, branch.epoch);
    }
}
