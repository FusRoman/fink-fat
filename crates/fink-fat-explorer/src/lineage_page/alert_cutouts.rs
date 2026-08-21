use std::collections::HashMap;

use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

use crate::format_epoch::iso_utc;
use crate::survey::Survey;

use super::observations_table::ObservationRow;

/// Which of the three cutout stamps attached to every ZTF/LSST alert is
/// being requested. `label()` doubles as the display heading and, verbatim,
/// as the Fink `/api/v1/cutouts` `kind` request value.
#[derive(Clone, Copy, PartialEq)]
enum CutoutKind {
    Science,
    Template,
    Difference,
}

impl CutoutKind {
    const ALL: [CutoutKind; 3] = [
        CutoutKind::Science,
        CutoutKind::Template,
        CutoutKind::Difference,
    ];

    fn label(self) -> &'static str {
        match self {
            CutoutKind::Science => "Science",
            CutoutKind::Template => "Template",
            CutoutKind::Difference => "Difference",
        }
    }
}

/// The three cutout stamps of one alert, each a `data:image/png;base64,...`
/// URI ready to drop straight into an `<img src>`.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
struct AlertCutouts {
    science: String,
    template: String,
    difference: String,
}

impl AlertCutouts {
    fn get(&self, kind: CutoutKind) -> &str {
        match kind {
            CutoutKind::Science => &self.science,
            CutoutKind::Template => &self.template,
            CutoutKind::Difference => &self.difference,
        }
    }
}

/// Fetch all three cutouts of one alert. `object_id`/`alert_id` are
/// `ObservationRow::object_id`/`::id` — see `Survey::cutout_request_body` for
/// why both are needed (ZTF's `object_id` alone only identifies the object,
/// not a specific alert).
#[server]
async fn get_alert_cutouts(
    mpc_code_obs: String,
    object_id: String,
    alert_id: i64,
) -> Result<AlertCutouts, ServerFnError> {
    use crate::get_http_client;

    let survey = Survey::from_code_obs(&mpc_code_obs)
        .ok_or_else(|| ServerFnError::new(format!("unknown observatory code: {mpc_code_obs}")))?;
    let client = get_http_client().await;

    let (science, template, difference) = tokio::try_join!(
        fetch_cutout_data_uri(client, survey, &object_id, alert_id, CutoutKind::Science),
        fetch_cutout_data_uri(client, survey, &object_id, alert_id, CutoutKind::Template),
        fetch_cutout_data_uri(client, survey, &object_id, alert_id, CutoutKind::Difference),
    )?;

    Ok(AlertCutouts {
        science,
        template,
        difference,
    })
}

#[cfg(feature = "server")]
async fn fetch_cutout_data_uri(
    client: &reqwest::Client,
    survey: Survey,
    object_id: &str,
    alert_id: i64,
    kind: CutoutKind,
) -> Result<String, ServerFnError> {
    use base64::{engine::general_purpose::STANDARD, Engine as _};

    let response = client
        .post(survey.cutout_api_url())
        .json(&survey.cutout_request_body(object_id, alert_id, kind.label()))
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| ServerFnError::new(format!("{} cutout request failed: {e}", kind.label())))?;

    let bytes = response
        .bytes()
        .await
        .map_err(|e| ServerFnError::new(format!("failed to read {} cutout: {e}", kind.label())))?;

    Ok(format!("data:image/png;base64,{}", STANDARD.encode(bytes)))
}

/// One row of `/api/v1/sources`' `r:diaSourceId,r:snr` column selection.
#[cfg(feature = "server")]
#[derive(Deserialize)]
struct RawLsstSource {
    #[serde(rename = "r:diaSourceId")]
    dia_source_id: i64,
    #[serde(rename = "r:snr")]
    snr: f64,
}

/// Best-effort real SNR for every alert of one LSST object, keyed by
/// `diaSourceId` (== `ObservationRow::id` for LSST rows). `dia_object_id_candidate`
/// is a *guess*: LSST's `diaObjectId` equals the `diaSourceId` of the object's
/// very first detection, and fink-fat only ever stores each alert's own
/// `diaSourceId` — so this only resolves real data when the guessed id
/// actually is the object's founding alert. An empty map (guess didn't
/// resolve) is a normal, expected outcome, not an error — callers fall back
/// to `pogson_snr` per alert when a `diaSourceId` isn't in the map.
#[server]
async fn get_lsst_snr_map(
    dia_object_id_candidate: String,
) -> Result<HashMap<i64, f64>, ServerFnError> {
    use crate::get_http_client;

    let client = get_http_client().await;
    let body = serde_json::json!({
        "diaObjectId": dia_object_id_candidate,
        "columns": "r:diaSourceId,r:snr",
        "output-format": "json",
    });

    let response = client
        .post("https://api.lsst.fink-portal.org/api/v1/sources")
        .json(&body)
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| ServerFnError::new(format!("LSST sources request failed: {e}")))?;

    let rows: Vec<RawLsstSource> = response
        .json()
        .await
        .map_err(|e| ServerFnError::new(format!("failed to parse LSST sources response: {e}")))?;

    Ok(rows.into_iter().map(|r| (r.dia_source_id, r.snr)).collect())
}

/// Standard photometric SNR ≈ 1.0857 / mag_err (Pogson relation between a
/// magnitude error and the underlying flux SNR) — the only SNR source for
/// ZTF (no `snr` field exists in its alert schema) and the fallback for LSST
/// whenever `get_lsst_snr_map`'s guess doesn't cover a given alert.
fn pogson_snr(mag_err: f64) -> f64 {
    1.0857362 / mag_err
}

#[derive(Clone, Copy, PartialEq)]
enum SnrSource {
    Fetched,
    Pogson,
}

impl SnrSource {
    fn label(self) -> &'static str {
        match self {
            SnrSource::Fetched => "fetched",
            SnrSource::Pogson => "Pogson",
        }
    }
}

struct SnrValue {
    value: f64,
    source: SnrSource,
}

fn snr_for(obs: &ObservationRow, lsst_snr_map: &HashMap<i64, f64>) -> SnrValue {
    if let Some(&snr) = lsst_snr_map.get(&obs.id) {
        debug_log(&format!(
            "snr_for: obs.id={} found in lsst_snr_map ({} entries) -> {snr:.2} (fetched)",
            obs.id,
            lsst_snr_map.len()
        ));
        return SnrValue {
            value: snr,
            source: SnrSource::Fetched,
        };
    }
    let value = pogson_snr(obs.mag_err);
    debug_log(&format!(
        "snr_for: obs.id={} mpc_code_obs={} NOT in lsst_snr_map ({} entries) -> pogson {value:.2} from mag_err={}",
        obs.id,
        obs.mpc_code_obs,
        lsst_snr_map.len(),
        obs.mag_err
    ));
    SnrValue {
        value,
        source: SnrSource::Pogson,
    }
}

/// Browser-console debug logging for the LSST SNR pipeline (candidate
/// `diaObjectId` guess → `get_lsst_snr_map` outcome → per-alert lookup) — a
/// no-op on the native/server build, since this only runs client-side where
/// `web_sys::console` is actually available.
#[cfg(target_arch = "wasm32")]
fn debug_log(message: &str) {
    web_sys::console::log_1(&format!("[alert-carousel] {message}").into());
}

#[cfg(not(target_arch = "wasm32"))]
fn debug_log(_message: &str) {}

/// Both overlay stamps are laid out identically — pinned to the frame's
/// top-left at its full size, so the Science layer's clipping wrapper reveals
/// it against the Template underneath without either image shifting or
/// rescaling as the divider moves.
const DIFF_IMAGE_STYLE: &str = "position: absolute; top: 0; left: 0; width: 384px; height: 384px; \
                                max-width: none; object-fit: cover; object-position: center; \
                                image-rendering: pixelated;";

/// LSST's DIA association re-clusters into a *new* `diaObjectId` every night
/// for a fast-moving object (its spatial cross-match can't link positions
/// across nights — bridging exactly that gap is the point of fink-fat's own
/// tracker). So a lineage's LSST observations don't share one `diaObjectId`;
/// they're split into one cluster per night. This groups them by a gap in
/// `mjd_tt` and returns each cluster's earliest alert's `object_id` — per
/// `get_lsst_snr_map`'s doc, that's the best per-cluster `diaObjectId` guess
/// available from what fink-fat stores.
fn cluster_lsst_dia_object_id_candidates(observations: &[ObservationRow]) -> Vec<String> {
    /// LSST visits within one night span at most a few hours; consecutive
    /// nights for a tracked lineage are at least ~1 day apart — comfortably
    /// splits nights without ever splitting one.
    const NIGHT_GAP_DAYS: f64 = 0.5;

    let mut lsst: Vec<&ObservationRow> = observations
        .iter()
        .filter(|o| o.mpc_code_obs == "X05")
        .collect();
    lsst.sort_by(|a, b| a.mjd_tt.total_cmp(&b.mjd_tt));

    let mut candidates = Vec::new();
    let mut prev_mjd: Option<f64> = None;
    for obs in lsst {
        let starts_new_cluster = match prev_mjd {
            Some(prev) => obs.mjd_tt - prev > NIGHT_GAP_DAYS,
            None => true,
        };
        if starts_new_cluster {
            candidates.push(obs.object_id.clone());
        }
        prev_mjd = Some(obs.mjd_tt);
    }
    candidates
}

/// Browse the Science/Template/Difference cutout stamps of a lineage's real
/// observations one alert at a time. The three images always move together
/// (Previous/Next step through alerts, not through the three kinds) — reuses
/// the same MJD/UTC toggle as `ObservationsTable`.
#[component]
pub fn AlertCarousel(observations: Vec<ObservationRow>) -> Element {
    let mut index = use_signal(|| 0usize);
    let mut show_utc = use_signal(|| true);
    let mut diff_open = use_signal(|| false);
    // Divider position in the Science/Template overlay, as a percentage from
    // the left edge.
    let mut diff_pos = use_signal(|| 50.0_f64);
    let mut cutouts_cache: Signal<HashMap<i64, Result<AlertCutouts, String>>> =
        use_signal(HashMap::new);

    // One diaObjectId candidate per night (see `cluster_lsst_dia_object_id_candidates`
    // doc — LSST assigns a fresh diaObjectId per night for a fast mover).
    let lsst_dia_object_id_candidates = cluster_lsst_dia_object_id_candidates(&observations);

    debug_log(&format!(
        "lsst diaObjectId candidates = {lsst_dia_object_id_candidates:?} (clustered from {} LSST observation(s))",
        observations.iter().filter(|o| o.mpc_code_obs == "X05").count()
    ));

    let lsst_snr_resource = use_resource(use_reactive!(
        |(lsst_dia_object_id_candidates,)| async move {
            let mut merged = HashMap::new();
            for candidate in &lsst_dia_object_id_candidates {
                debug_log(&format!(
                    "get_lsst_snr_map: calling with candidate = {candidate}"
                ));
                match get_lsst_snr_map(candidate.clone()).await {
                    Ok(map) => {
                        debug_log(&format!(
                            "get_lsst_snr_map: candidate {candidate} resolved {} source(s)",
                            map.len()
                        ));
                        merged.extend(map);
                    }
                    Err(e) => debug_log(&format!(
                        "get_lsst_snr_map: candidate {candidate} failed: {e}"
                    )),
                }
            }
            debug_log(&format!(
                "get_lsst_snr_map: merged total = {} source(s) across {} night(s)",
                merged.len(),
                lsst_dia_object_id_candidates.len()
            ));
            Ok::<HashMap<i64, f64>, ServerFnError>(merged)
        }
    ));

    {
        let observations = observations.clone();
        use_effect(move || {
            let Some(obs) = observations.get(index()).cloned() else {
                return;
            };
            if cutouts_cache.read().contains_key(&obs.id) {
                return;
            }
            spawn(async move {
                let result =
                    get_alert_cutouts(obs.mpc_code_obs.clone(), obs.object_id.clone(), obs.id)
                        .await
                        .map_err(|e| e.to_string());
                cutouts_cache.write().insert(obs.id, result);
            });
        });
    }

    let lsst_snr_map: HashMap<i64, f64> = match &*lsst_snr_resource.read() {
        Some(Ok(map)) => map.clone(),
        _ => HashMap::new(),
    };

    let n = observations.len();
    let current = observations.get(index()).cloned();

    rsx! {
        div { class: "card bg-base-100 shadow-sm",
            div { class: "card-body gap-3",
                div { class: "flex flex-wrap items-center justify-between gap-2",
                    h2 { class: "card-title", "Alert images ({(index() + 1).min(n)}/{n})" }
                    label { class: "flex items-center gap-2 cursor-pointer text-sm",
                        span { class: if !show_utc() { "font-bold" } else { "opacity-50" }, "MJD (TT)" }
                        input {
                            r#type: "checkbox",
                            class: "toggle toggle-sm",
                            checked: show_utc(),
                            onchange: move |evt| show_utc.set(evt.checked()),
                        }
                        span { class: if show_utc() { "font-bold" } else { "opacity-50" }, "ISO (UTC)" }
                    }
                }

                match current {
                    Some(obs) => {
                        let epoch = if show_utc() {
                            iso_utc(obs.mjd_tt)
                        } else {
                            format!("{:.5}", obs.mjd_tt)
                        };
                        let snr = snr_for(&obs, &lsst_snr_map);
                        let cutout = cutouts_cache.read().get(&obs.id).cloned();
                        rsx! {
                            div { class: "flex items-center justify-center gap-3",
                                button {
                                    class: "btn btn-circle btn-sm",
                                    disabled: index() == 0,
                                    onclick: move |_| {
                                        index.set(index().saturating_sub(1));
                                        diff_open.set(false);
                                    },
                                    "‹"
                                }
                                for kind in CutoutKind::ALL {
                                    CutoutCard {
                                        key: "{kind.label()}",
                                        kind_label: kind.label(),
                                        cutout: match &cutout {
                                            Some(Ok(c)) => Some(Ok(c.get(kind).to_string())),
                                            Some(Err(e)) => Some(Err(e.clone())),
                                            None => None,
                                        },
                                        band: obs.filter,
                                        epoch: epoch.clone(),
                                        magnitude: obs.magnitude,
                                        mag_err: obs.mag_err,
                                        snr: snr.value,
                                        snr_source_label: snr.source.label(),
                                        clickable: kind != CutoutKind::Difference && matches!(cutout, Some(Ok(_))),
                                        on_click: move |_| {
                                            diff_pos.set(50.0);
                                            diff_open.set(true);
                                        },
                                    }
                                }
                                button {
                                    class: "btn btn-circle btn-sm",
                                    disabled: index() + 1 >= n,
                                    onclick: move |_| {
                                        index.set(index() + 1);
                                        diff_open.set(false);
                                    },
                                    "›"
                                }
                            }
                            if diff_open() {
                                if let Some(Ok(c)) = &cutout {
                                    div {
                                        class: "fixed inset-0 z-50 flex items-center justify-center bg-black/70",
                                        onclick: move |_| diff_open.set(false),
                                        // Hand-rolled rather than daisyUI's `diff`: that
                                        // component drives its divider from the browser's
                                        // native `resize: horizontal` corner handle, sized in
                                        // container-query units that collapse here — leaving
                                        // the divider pinned left with no usable drag target.
                                        // A range input over the stack is fully deterministic.
                                        div {
                                            class: "relative overflow-hidden select-none rounded",
                                            style: "width: 384px; height: 384px;",
                                            onclick: move |evt| evt.stop_propagation(),

                                            // Template fills the frame; Science is clipped over
                                            // it from the left, so the divider reveals one
                                            // against the other.
                                            img {
                                                src: "{c.template}",
                                                style: "{DIFF_IMAGE_STYLE}",
                                            }
                                            div {
                                                style: "position: absolute; top: 0; left: 0; height: 384px; overflow: hidden; width: {diff_pos()}%;",
                                                img {
                                                    src: "{c.science}",
                                                    style: "{DIFF_IMAGE_STYLE}",
                                                }
                                            }

                                            div {
                                                style: "position: absolute; top: 0; bottom: 0; left: {diff_pos()}%; width: 2px; margin-left: -1px; background: white; box-shadow: 0 0 3px rgba(0,0,0,0.8); pointer-events: none;",
                                            }
                                            span {
                                                class: "absolute top-1 left-2 text-xs text-white",
                                                style: "text-shadow: 0 0 3px black; pointer-events: none;",
                                                "Science"
                                            }
                                            span {
                                                class: "absolute top-1 right-2 text-xs text-white",
                                                style: "text-shadow: 0 0 3px black; pointer-events: none;",
                                                "Template"
                                            }

                                            input {
                                                r#type: "range",
                                                min: "0",
                                                max: "100",
                                                step: "0.1",
                                                value: "{diff_pos()}",
                                                style: "position: absolute; top: 0; left: 0; width: 384px; height: 384px; margin: 0; opacity: 0; cursor: ew-resize;",
                                                oninput: move |evt| {
                                                    if let Ok(v) = evt.value().parse::<f64>() {
                                                        diff_pos.set(v);
                                                    }
                                                },
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    None => rsx! {
                        div { class: "text-sm opacity-60", "No observations to display." }
                    },
                }
            }
        }
    }
}

#[component]
fn CutoutCard(
    kind_label: &'static str,
    cutout: Option<Result<String, String>>,
    band: i16,
    epoch: String,
    magnitude: f64,
    mag_err: f64,
    snr: f64,
    snr_source_label: &'static str,
    clickable: bool,
    on_click: EventHandler<()>,
) -> Element {
    rsx! {
        div { class: "flex flex-col items-center gap-1 p-2 bg-base-200 rounded-box",
            span { class: "text-xs font-semibold", "{kind_label}" }
            match cutout {
                Some(Ok(src)) => rsx! {
                    img {
                        src: "{src}",
                        class: if clickable { "w-40 h-40 object-contain bg-black rounded cursor-pointer" } else { "w-40 h-40 object-contain bg-black rounded" },
                        style: "image-rendering: pixelated;",
                        onclick: move |_| {
                            if clickable {
                                on_click.call(());
                            }
                        },
                    }
                },
                Some(Err(message)) => rsx! {
                    div {
                        class: "w-40 h-40 flex items-center justify-center bg-black rounded p-2",
                        title: "{message}",
                        span { class: "text-error text-xs text-center", "Failed to load" }
                    }
                },
                None => rsx! {
                    div { class: "w-40 h-40 flex items-center justify-center bg-black rounded",
                        span { class: "loading loading-spinner loading-sm" }
                    }
                },
            }
            div { class: "text-xs opacity-70 text-center",
                p { "band {band}" }
                p { "{epoch}" }
                p { "mag {magnitude:.2} ± {mag_err:.2}" }
                p { "SNR {snr:.1} ({snr_source_label})" }
            }
        }
    }
}
