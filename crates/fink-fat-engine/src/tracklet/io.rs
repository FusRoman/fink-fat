use camino::Utf8Path;
use outfit::{GaussResult, OrbitalElements, constants::FitOrbitResult};
use polars::prelude::*;

use crate::tracklet::Tracklet;

/// Write the track storage to a Parquet file.
///
/// Each tracklet is serialized as a single row. The `obs_ids` field is
/// stored as a pipe-separated string (`"id1|id2|id3"`) to avoid the
/// `List<String>` column type, which has significant per-row overhead.
/// Orbital elements are converted to Keplerian form. `Seed` and `Filter`
/// tracklets have `null` orbital columns.
///
/// Arguments
/// ---------
/// * `tracklets` – Iterator over [`Tracklet`] references to serialize.
/// * `path` – Destination path for the Parquet file.
///
/// Return
/// ------
/// * `Ok(())` – File written successfully.
/// * `Err(PolarsError)` – If DataFrame construction or file I/O fails.
pub(crate) fn write_tracklets_parquet<'a>(
    tracklets: impl Iterator<Item = &'a Tracklet>,
    path: impl AsRef<Utf8Path>,
) -> PolarsResult<()> {
    let mut track_ids: Vec<u32> = Vec::new();
    let mut tracklet_types: Vec<&'static str> = Vec::new();
    let mut obs_ids: Vec<String> = Vec::new();
    let mut ref_mags: Vec<f64> = Vec::new();
    let mut ref_mag_errs: Vec<f64> = Vec::new();
    let mut orbit_types: Vec<Option<&'static str>> = Vec::new();
    let mut orbit_epochs: Vec<Option<f64>> = Vec::new();
    let mut semi_major_axes: Vec<Option<f64>> = Vec::new();
    let mut eccentricities: Vec<Option<f64>> = Vec::new();
    let mut inclinations: Vec<Option<f64>> = Vec::new();
    let mut ascending_nodes: Vec<Option<f64>> = Vec::new();
    let mut periapsis_args: Vec<Option<f64>> = Vec::new();
    let mut mean_anomalies: Vec<Option<f64>> = Vec::new();
    let mut sigma_as: Vec<Option<f64>> = Vec::new();
    let mut sigma_es: Vec<Option<f64>> = Vec::new();
    let mut sigma_is: Vec<Option<f64>> = Vec::new();
    let mut sigma_nodes: Vec<Option<f64>> = Vec::new();
    let mut sigma_peris: Vec<Option<f64>> = Vec::new();
    let mut sigma_ms: Vec<Option<f64>> = Vec::new();
    let mut fit_types: Vec<Option<&'static str>> = Vec::new();
    let mut chi2s: Vec<Option<f64>> = Vec::new();
    let mut iod_rmss: Vec<Option<f64>> = Vec::new();

    for tracklet in tracklets {
        match tracklet {
            Tracklet::Seed(data) => {
                track_ids.push(data.key.0);
                tracklet_types.push("Seed");
                obs_ids.push(
                    data.obs_keys
                        .iter()
                        .map(|id| id.to_string())
                        .collect::<Vec<_>>()
                        .join("|"),
                );
                ref_mags.push(data.ref_mag);
                ref_mag_errs.push(data.ref_mag_err);
                push_nulls(
                    &mut orbit_types,
                    &mut orbit_epochs,
                    &mut semi_major_axes,
                    &mut eccentricities,
                    &mut inclinations,
                    &mut ascending_nodes,
                    &mut periapsis_args,
                    &mut mean_anomalies,
                    &mut sigma_as,
                    &mut sigma_es,
                    &mut sigma_is,
                    &mut sigma_nodes,
                    &mut sigma_peris,
                    &mut sigma_ms,
                    &mut fit_types,
                    &mut chi2s,
                    &mut iod_rmss,
                );
            }
            Tracklet::Filter(data) => {
                track_ids.push(data.key.0);
                tracklet_types.push("Filter");
                obs_ids.push(
                    data.obs_keys
                        .iter()
                        .map(|id| id.to_string())
                        .collect::<Vec<_>>()
                        .join("|"),
                );
                ref_mags.push(data.ref_mag);
                ref_mag_errs.push(data.ref_mag_err);
                push_nulls(
                    &mut orbit_types,
                    &mut orbit_epochs,
                    &mut semi_major_axes,
                    &mut eccentricities,
                    &mut inclinations,
                    &mut ascending_nodes,
                    &mut periapsis_args,
                    &mut mean_anomalies,
                    &mut sigma_as,
                    &mut sigma_es,
                    &mut sigma_is,
                    &mut sigma_nodes,
                    &mut sigma_peris,
                    &mut sigma_ms,
                    &mut fit_types,
                    &mut chi2s,
                    &mut iod_rmss,
                );
            }
            Tracklet::Orbit(data) => {
                track_ids.push(data.key.0);
                tracklet_types.push("Orbit");
                obs_ids.push(
                    data.obs_keys
                        .iter()
                        .map(|id| id.to_string())
                        .collect::<Vec<_>>()
                        .join("|"),
                );
                ref_mags.push(data.ref_mag);
                ref_mag_errs.push(data.ref_mag_err);
                push_orbit(
                    &data.state,
                    &mut orbit_types,
                    &mut orbit_epochs,
                    &mut semi_major_axes,
                    &mut eccentricities,
                    &mut inclinations,
                    &mut ascending_nodes,
                    &mut periapsis_args,
                    &mut mean_anomalies,
                    &mut sigma_as,
                    &mut sigma_es,
                    &mut sigma_is,
                    &mut sigma_nodes,
                    &mut sigma_peris,
                    &mut sigma_ms,
                    &mut fit_types,
                    &mut chi2s,
                    &mut iod_rmss,
                );
            }
        }
    }

    let height = track_ids.len();
    let mut df = DataFrame::new(
        height,
        vec![
            Column::new("track_id".into(), track_ids),
            Column::new("tracklet_type".into(), tracklet_types),
            Column::new("obs_ids".into(), obs_ids),
            Column::new("ref_mag".into(), ref_mags),
            Column::new("ref_mag_err".into(), ref_mag_errs),
            Column::new("orbit_type".into(), orbit_types),
            Column::new("orbit_epoch".into(), orbit_epochs),
            Column::new("semi_major_axis".into(), semi_major_axes),
            Column::new("eccentricity".into(), eccentricities),
            Column::new("inclination".into(), inclinations),
            Column::new("ascending_node_longitude".into(), ascending_nodes),
            Column::new("periapsis_argument".into(), periapsis_args),
            Column::new("mean_anomaly".into(), mean_anomalies),
            Column::new("sigma_a".into(), sigma_as),
            Column::new("sigma_e".into(), sigma_es),
            Column::new("sigma_i".into(), sigma_is),
            Column::new("sigma_node".into(), sigma_nodes),
            Column::new("sigma_peri".into(), sigma_peris),
            Column::new("sigma_M".into(), sigma_ms),
            Column::new("fit_type".into(), fit_types),
            Column::new("chi2".into(), chi2s),
            Column::new("iod_rms".into(), iod_rmss),
        ],
    )?;

    let file = std::fs::File::create(path.as_ref())?;
    ParquetWriter::new(file)
        .with_compression(ParquetCompression::Snappy)
        .finish(&mut df)?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn push_nulls(
    orbit_types: &mut Vec<Option<&'static str>>,
    orbit_epochs: &mut Vec<Option<f64>>,
    semi_major_axes: &mut Vec<Option<f64>>,
    eccentricities: &mut Vec<Option<f64>>,
    inclinations: &mut Vec<Option<f64>>,
    ascending_nodes: &mut Vec<Option<f64>>,
    periapsis_args: &mut Vec<Option<f64>>,
    mean_anomalies: &mut Vec<Option<f64>>,
    sigma_as: &mut Vec<Option<f64>>,
    sigma_es: &mut Vec<Option<f64>>,
    sigma_is: &mut Vec<Option<f64>>,
    sigma_nodes: &mut Vec<Option<f64>>,
    sigma_peris: &mut Vec<Option<f64>>,
    sigma_ms: &mut Vec<Option<f64>>,
    fit_types: &mut Vec<Option<&'static str>>,
    chi2s: &mut Vec<Option<f64>>,
    iod_rmss: &mut Vec<Option<f64>>,
) {
    orbit_types.push(None);
    orbit_epochs.push(None);
    semi_major_axes.push(None);
    eccentricities.push(None);
    inclinations.push(None);
    ascending_nodes.push(None);
    periapsis_args.push(None);
    mean_anomalies.push(None);
    sigma_as.push(None);
    sigma_es.push(None);
    sigma_is.push(None);
    sigma_nodes.push(None);
    sigma_peris.push(None);
    sigma_ms.push(None);
    fit_types.push(None);
    chi2s.push(None);
    iod_rmss.push(None);
}

#[allow(clippy::too_many_arguments)]
fn push_orbit(
    state: &FitOrbitResult,
    orbit_types: &mut Vec<Option<&'static str>>,
    orbit_epochs: &mut Vec<Option<f64>>,
    semi_major_axes: &mut Vec<Option<f64>>,
    eccentricities: &mut Vec<Option<f64>>,
    inclinations: &mut Vec<Option<f64>>,
    ascending_nodes: &mut Vec<Option<f64>>,
    periapsis_args: &mut Vec<Option<f64>>,
    mean_anomalies: &mut Vec<Option<f64>>,
    sigma_as: &mut Vec<Option<f64>>,
    sigma_es: &mut Vec<Option<f64>>,
    sigma_is: &mut Vec<Option<f64>>,
    sigma_nodes: &mut Vec<Option<f64>>,
    sigma_peris: &mut Vec<Option<f64>>,
    sigma_ms: &mut Vec<Option<f64>>,
    fit_types: &mut Vec<Option<&'static str>>,
    chi2s: &mut Vec<Option<f64>>,
    iod_rmss: &mut Vec<Option<f64>>,
) {
    let (elements, fit_type, chi2, iod_rms) = match state {
        FitOrbitResult::IODGauss((gauss_result, rms)) => {
            let (elements, fit_type) = match gauss_result {
                GaussResult::PrelimOrbit(e) => (e, "IODGauss_Prelim"),
                GaussResult::CorrectedOrbit(e) => (e, "IODGauss_Corrected"),
            };
            (elements, fit_type, None, Some(*rms))
        }
        FitOrbitResult::DifferentialCorrection((elements, chi2)) => {
            (elements, "DifferentialCorrection", Some(*chi2), None)
        }
    };

    orbit_types.push(Some(match elements {
        OrbitalElements::Keplerian { .. } => "Keplerian",
        OrbitalElements::Equinoctial { .. } => "Equinoctial",
        OrbitalElements::Cometary { .. } => "Cometary",
    }));
    fit_types.push(Some(fit_type));
    chi2s.push(chi2);
    iod_rmss.push(iod_rms);

    match elements.to_keplerian() {
        Ok(OrbitalElements::Keplerian {
            elements: e,
            uncertainty,
            ..
        }) => {
            orbit_epochs.push(Some(e.reference_epoch));
            semi_major_axes.push(Some(e.semi_major_axis));
            eccentricities.push(Some(e.eccentricity));
            inclinations.push(Some(e.inclination));
            ascending_nodes.push(Some(e.ascending_node_longitude));
            periapsis_args.push(Some(e.periapsis_argument));
            mean_anomalies.push(Some(e.mean_anomaly));
            if let Some(unc) = uncertainty {
                sigma_as.push(Some(unc.semi_major_axis));
                sigma_es.push(Some(unc.eccentricity));
                sigma_is.push(Some(unc.inclination));
                sigma_nodes.push(Some(unc.ascending_node_longitude));
                sigma_peris.push(Some(unc.periapsis_argument));
                sigma_ms.push(Some(unc.mean_anomaly));
            } else {
                sigma_as.push(None);
                sigma_es.push(None);
                sigma_is.push(None);
                sigma_nodes.push(None);
                sigma_peris.push(None);
                sigma_ms.push(None);
            }
        }
        _ => {
            orbit_epochs.push(None);
            semi_major_axes.push(None);
            eccentricities.push(None);
            inclinations.push(None);
            ascending_nodes.push(None);
            periapsis_args.push(None);
            mean_anomalies.push(None);
            sigma_as.push(None);
            sigma_es.push(None);
            sigma_is.push(None);
            sigma_nodes.push(None);
            sigma_peris.push(None);
            sigma_ms.push(None);
        }
    }
}
