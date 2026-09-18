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
use crate::orbit3d::types::{Body3D, LineageOrbit3D, ObjectPoint3D};

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

/// The planets and tracked perturbers, positioned at the current epoch — see
/// `crate::orbit3d::ephem_provider`.
#[server]
pub async fn get_planets_3d() -> Result<Vec<Body3D>, ServerFnError> {
    use crate::orbit3d::ephem_provider;

    let bodies = ephem_provider::get_planets(now_mjd_tt()).await;
    Ok((*bodies).clone())
}

/// One point per lineage — its best branch's current heliocentric position —
/// for the homepage's 3D scatter. `None` while the homepage snapshot is
/// still (re)building, mirroring `homepage::dynamic_pop_plot
/// ::query_orbital_elements`'s own "warming up" contract so both plots poll
/// the same way.
#[server]
pub async fn get_homepage_orbit3d() -> Result<Option<Vec<ObjectPoint3D>>, ServerFnError> {
    let Some(snapshot) = crate::homepage::snapshot::snapshot().await else {
        return Ok(None);
    };

    let now = now_mjd_tt();
    let points = snapshot
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

    Ok(Some(points))
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
    use fink_fat_engine::topocentric_kf::conversion::attributable_to_cartesian;
    use nalgebra::{Vector3, Vector6};
    use outfit::OrbitalElements;

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
    let cartesian = attributable_to_cartesian(&state, &r_obs, &v_obs);

    let elems = OrbitalElements::from_orbital_state(&cartesian.pos, &cartesian.vel, branch.epoch)
        .as_keplerian()?;
    Some(crate::orbit3d::ephem_provider::keplerian_from_outfit_elements(&elems, branch.epoch))
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
    use crate::fit_pipeline::fit::KeplerianView;
    use crate::orbit3d::ephem_provider;

    #[derive(sqlx::FromRow)]
    struct Row {
        reference_epoch: f64,
        keplerian: sqlx::types::Json<KeplerianView>,
    }

    let pool = crate::get_pool().await;
    let row: Option<Row> = sqlx::query_as(
        "SELECT reference_epoch, keplerian FROM orbit_fits \
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

    Ok(Some(LineageOrbit3D {
        object_position: geometry::position_at_epoch(&elems, now),
        object_orbit: geometry::ellipse_points(&elems, LINEAGE_ORBIT_CURVE_SAMPLES),
        planets,
    }))
}

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
    #[test]
    fn keplerian_from_branch_resolves_a_bound_orbit() {
        let branch = branch_with_attributable_state(0.0, 0.0, 0.0, 0.0, 1.5, 0.001);

        let elems = keplerian_from_branch(&branch).expect("expected a closed-ellipse orbit");

        assert!(elems.semi_major_axis_au > 0.0);
        assert!((0.0..1.0).contains(&elems.eccentricity));
        assert_eq!(elems.epoch_mjd_tt, branch.epoch);
    }
}
