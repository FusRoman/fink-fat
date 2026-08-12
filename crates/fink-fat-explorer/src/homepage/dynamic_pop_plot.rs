use dioxus::prelude::*;

#[cfg(feature = "server")]
use rayon::iter::ParallelIterator;

#[cfg(feature = "server")]
use rayon::prelude::*;

#[cfg(target_arch = "wasm32")]
use plotly::{
    common::{Mode, TickMode, Title},
    layout::{Axis, AxisType, Layout},
    Plot, Scatter,
};

use serde::{Deserialize, Serialize};

#[cfg(feature = "server")]
use nalgebra::{Vector3, Vector6};

// The engine's attributable-to-Cartesian converter.
// Adapt the import path to match your crate structure.
#[cfg(feature = "server")]
use fink_fat_engine::topocentric_kf::conversion::attributable_to_cartesian;

/// Minimal row fetched from the DB — no covariance, no kalman gain, no nis_ema.
#[cfg(feature = "server")]
#[derive(sqlx::FromRow)]
struct MinimalStateRow {
    hypothesis_id: i64,
    branch_id: i64,
    // Attributable state vector components
    ra: f64,
    dec: f64,
    ra_dot: f64,
    dec_dot: f64,
    rho: f64,
    rho_dot: f64,
    // reference epoch
    epoch: f64,
    // Observer heliocentric state
    r_obs_x: f64,
    r_obs_y: f64,
    r_obs_z: f64,
    v_obs_x: f64,
    v_obs_y: f64,
    v_obs_z: f64,
}

/// One point in the (a, e) distribution plot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OrbitalPoint {
    pub hypothesis_id: i64,
    pub branch_id: i64,
    pub semi_major_axis: f64,
    pub eccentricity: f64,
    pub family: String,
}

/// Classify an asteroid into its dynamical family based on the IMCCE SSP
/// population table: https://ssp.imcce.fr/webservices/skybot/
/// a = semi-major axis (AU), e = eccentricity.
#[cfg(feature = "server")]
fn classify_asteroid(a: f64, e: f64) -> &'static str {
    let perihelion = a * (1.0 - e);
    let aphelion = a * (1.0 + e);

    // Threshold for KBO>SDO: a(1−e) ≤ 30.1 * 2^(2/3) * (1−0.24) ≈ 36.3 AU
    const KBO_SDO_THRESHOLD: f64 = 30.1 * 1.587_401_05 * (1.0 - 0.24);

    match a {
        a if a < 0.08 => "Unknown",
        a if a < 0.21 => "Vulcanoid",
        a if a < 1.0 => {
            if aphelion < 0.983 {
                "NEA>Atira"
            } else {
                "NEA>Aten"
            }
        }
        a if a < 2.0 => {
            if perihelion < 1.017 {
                "NEA>Apollo"
            } else if perihelion < 1.3 {
                "NEA>Amor"
            } else if perihelion <= 1.58 {
                "Mars-Crosser>Deep"
            } else if perihelion <= 1.666 {
                "Mars-Crosser>Shallow"
            } else {
                "Hungaria"
            }
        }
        a if a < 2.5 => "MB>Inner",
        a if a < 2.82 => "MB>Middle",
        a if a < 3.27 => "MB>Outer",
        a if a < 3.7 => "MB>Cybele",
        a if a < 4.6 => "MB>Hilda",
        a if a < 5.5 => "Trojan",
        a if a < 30.1 => "Centaur",
        a if a < 2000.0 => {
            if perihelion <= KBO_SDO_THRESHOLD {
                "KBO>SDO"
            } else if e >= 0.24 {
                "KBO>Detached"
            } else if a < 39.4 {
                "KBO>Classical>Inner"
            } else if a < 47.8 {
                "KBO>Classical>Main"
            } else {
                "KBO>Classical>Outer"
            }
        }
        _ => "IOC",
    }
}

#[server]
pub async fn query_orbital_elements() -> Result<Vec<OrbitalPoint>, ServerFnError> {
    use crate::get_pool;

    let pool = get_pool().await;

    let rows: Vec<MinimalStateRow> = sqlx::query_as(
        "SELECT bh.hypothesis_id, bh.branch_id,
            ks.ra, ks.dec, ks.ra_dot, ks.dec_dot, ks.rho, ks.rho_dot,
            ks.epoch,
            ks.r_obs_x, ks.r_obs_y, ks.r_obs_z,
            ks.v_obs_x, ks.v_obs_y, ks.v_obs_z
     FROM branches b
     CROSS JOIN LATERAL (
         SELECT hypothesis_id, branch_id
         FROM hypotheses h
         WHERE h.branch_id = b.branch_id
         ORDER BY h.log_weight DESC
         LIMIT 1
     ) bh
     JOIN kf_state ks ON ks.hypothesis_id = bh.hypothesis_id",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| ServerFnError::new(e.to_string()))?;

    let points = rows
        .into_par_iter()
        .filter_map(|row| {
            use outfit::OrbitalElements;

            let state = Vector6::new(
                row.ra,
                row.dec,
                row.ra_dot,
                row.dec_dot,
                row.rho,
                row.rho_dot,
            );
            let r_obs = Vector3::new(row.r_obs_x, row.r_obs_y, row.r_obs_z);
            let v_obs = Vector3::new(row.v_obs_x, row.v_obs_y, row.v_obs_z);

            // Convert attributable + observer → heliocentric Cartesian.
            // attributable_to_cartesian returns CartesianState { r, v } — adapt
            // the field names if your CartesianState uses different ones.
            let cartesian = attributable_to_cartesian(&state, &r_obs, &v_obs);

            let orbit =
                OrbitalElements::from_orbital_state(&cartesian.pos, &cartesian.vel, row.epoch)
                    .as_keplerian()?;

            Some(OrbitalPoint {
                hypothesis_id: row.hypothesis_id,
                branch_id: row.branch_id,
                semi_major_axis: orbit.semi_major_axis,
                eccentricity: orbit.eccentricity,
                family: classify_asteroid(orbit.semi_major_axis, orbit.eccentricity).to_string(),
            })
        })
        .collect();

    Ok(points)
}

#[component]
pub fn DynamicPopPlot() -> Element {
    let orbital_data = use_resource(|| query_orbital_elements());
    let mut is_mounted = use_signal(|| false);

    let points = use_memo(move || match &*orbital_data.read() {
        Some(Ok(pts)) => Some(pts.clone()),
        _ => None,
    });

    let status_text = match &*orbital_data.read() {
        Some(Ok(pts)) => format!("{} objects plotted", pts.len()),
        Some(Err(e)) => format!("Error: {e}"),
        None => "Loading...".to_string(),
    };

    use_effect(move || {
        #[cfg(target_arch = "wasm32")]
        if is_mounted() {
            if let Some(pts) = points() {
                spawn(async move {
                    if pts.is_empty() {
                        return;
                    }

                    let height = web_sys::window()
                        .and_then(|w| w.inner_height().ok())
                        .and_then(|h| h.as_f64())
                        .map(|h| (h * 0.68) as usize)
                        .unwrap_or(700);

                    let a_vals: Vec<f32> = pts.iter().map(|p| p.semi_major_axis as f32).collect();
                    let e_vals: Vec<f32> = pts.iter().map(|p| p.eccentricity as f32).collect();

                    let mut families: std::collections::BTreeMap<String, (Vec<f32>, Vec<f32>)> =
                        std::collections::BTreeMap::new();
                    for pt in &pts {
                        let entry = families.entry(pt.family.clone()).or_default();
                        entry.0.push(pt.semi_major_axis as f32);
                        entry.1.push(pt.eccentricity as f32);
                    }

                    let mut plot = Plot::new();
                    for (family, (a_vals, e_vals)) in families {
                        let trace = Scatter::new(a_vals, e_vals)
                            .name(&family)
                            .mode(Mode::Markers)
                            .web_gl_mode(true);
                        plot.add_trace(trace);
                    }

                    let layout = Layout::new()
                        .height(height)
                        .x_axis(
                            Axis::new()
                                .type_(AxisType::Log)
                                .title(Title::from("Semi-major axis (AU)"))
                                .tick_mode(TickMode::Array)
                                // Moins de ticks, espacés de façon lisible sur log
                                .tick_values(vec![
                                    0.5, 1.0, 2.0, 3.0, 4.6, 5.5, 10.0, 30.0, 100.0, 1000.0,
                                ])
                                .tick_text(vec![
                                    "0.5",
                                    "1",
                                    "2 (MB)",
                                    "3",
                                    "4.6 (Trojan)",
                                    "5.5 (Centaur)",
                                    "10",
                                    "30 (KBO)",
                                    "100",
                                    "1000",
                                ])
                                .tick_angle(-35.0), // ← incline pour éviter le chevauchement
                        )
                        .y_axis(Axis::new().title(Title::from("Eccentricity")));

                    plot.set_layout(layout);
                    plotly::bindings::new_plot("ae-plot-div", &plot).await;
                });
            }
        }
    });

    rsx! {
        div { class: "card bg-base-100 shadow-sm",
            div { class: "card-body",
                div { class: "flex items-center justify-between mb-2",
                    h2 { class: "card-title", "Dynamical families — (a, e)" }
                    span { class: "text-sm opacity-60", "{status_text}" }
                }
                div {
                    id: "ae-plot-div",
                    style: "width: 100%;",
                    onmounted: move |_| {
                        is_mounted.set(true);
                    },
                }
            }
        }
    }
}
