//! Modal collecting the ADES header fields not tracked by the pipeline, then
//! driving local `submit.xsd` validation (the gate on the download button)
//! and, once valid, an informational submission to MPC's `submit_xml_test`
//! endpoint. See `crate::ades` for the full pipeline this drives.

use dioxus::prelude::*;

use crate::ades::model::AdesHeaderInput;
use crate::ades::server_fns::{export_and_validate_ades, AdesExportResult};
use crate::survey::Survey;

/// Where the export flow currently stands, driving which controls are shown.
#[derive(Clone, Copy, PartialEq)]
enum ExportState {
    Idle,
    Checking,
    Valid,
    Invalid,
}

/// Telescope defaults for the two surveys fink-fat ingests today, keyed off
/// the station codes present in the lineage's observations
/// ([`crate::survey::Survey::from_code_obs`]). Left blank (forcing the user
/// to fill them in) when the observations come from an unrecognized station.
fn default_telescope(mpc_codes: &[String]) -> (String, String, String) {
    let survey = mpc_codes
        .iter()
        .find_map(|code| Survey::from_code_obs(code));
    match survey {
        Some(Survey::ZTF) => (
            "Schmidt".to_string(),
            "1.2".to_string(),
            "CCD Mosaic".to_string(),
        ),
        Some(Survey::LSST) => (
            "Reflector".to_string(),
            "8.4".to_string(),
            "CCD Mosaic".to_string(),
        ),
        None => (String::new(), String::new(), String::new()),
    }
}

/// A one-click fill for `measurers`/`observers`, one per survey fink-fat
/// ingests. `measurers` credits whoever measured the astrometry — for both
/// surveys that's the survey's own alert-production pipeline upstream of
/// Fink, not Fink itself (Fink only brokers and enriches already-measured
/// alerts) — and `observers` names that same pipeline explicitly.
#[derive(Clone, Copy)]
struct SurveyPreset {
    button_label: &'static str,
    measurers: &'static str,
    observers: &'static str,
}

const RUBIN_PRESET: SurveyPreset = SurveyPreset {
    button_label: "Rubin (X05)",
    measurers: "Vera C. Rubin Observatory (LSST)",
    observers: "Rubin Observatory Alert Production Pipeline",
};

const ZTF_PRESET: SurveyPreset = SurveyPreset {
    button_label: "ZTF (I41)",
    measurers: "Zwicky Transient Facility",
    observers: "ZTF Alert Production Pipeline",
};

/// Default `ack` message: the lineage designation and today's date, so the
/// eventual MPC acknowledgement/email report can be matched back to which
/// lineage and submission day it came from without the user typing anything.
fn default_ack_message(lineage_designation: &str) -> String {
    let date = crate::format_epoch::today_utc_date().unwrap_or_default();
    format!("fink-fat export - lineage {lineage_designation} - {date}")
}

fn default_header(mpc_codes: &[String], lineage_designation: &str) -> AdesHeaderInput {
    let (telescope_design, telescope_aperture, telescope_detector) = default_telescope(mpc_codes);
    AdesHeaderInput {
        submitter_name: String::new(),
        submitter_institution: None,
        observers: vec![],
        measurers: vec![],
        telescope_design,
        telescope_aperture,
        telescope_detector,
        ast_cat: "Gaia2".to_string(),
        mode: "CCD".to_string(),
        funding_source: None,
        ack_message: default_ack_message(lineage_designation),
        ac2_email: String::new(),
    }
}

/// Modal offered from the lineage page toolbar to export the lineage's
/// observations as an ADES pre-submission XML file.
#[component]
pub fn AdesExportModal(
    lineage_designation: String,
    mpc_codes: Vec<String>,
    open: bool,
    on_close: EventHandler<()>,
) -> Element {
    if !open {
        return rsx! {};
    }

    let mut header = use_signal(|| default_header(&mpc_codes, &lineage_designation));
    let mut state = use_signal(|| ExportState::Idle);
    let mut result = use_signal(|| None::<AdesExportResult>);
    let mut request_error = use_signal(|| None::<String>);

    let mut reset_to_idle = move || {
        state.set(ExportState::Idle);
    };

    let check = {
        let lineage_designation = lineage_designation.clone();
        move |_| {
            let lineage_designation = lineage_designation.clone();
            let header_value = header();
            state.set(ExportState::Checking);
            request_error.set(None);
            spawn(async move {
                match export_and_validate_ades(lineage_designation, header_value).await {
                    Ok(export_result) => {
                        state.set(if export_result.schema_valid {
                            ExportState::Valid
                        } else {
                            ExportState::Invalid
                        });
                        result.set(Some(export_result));
                    }
                    Err(err) => {
                        state.set(ExportState::Invalid);
                        result.set(None);
                        request_error.set(Some(err.to_string()));
                    }
                }
            });
        }
    };

    let download = move |_| {
        if let Some(export_result) = result() {
            let payload = serde_json::json!({
                "xml": export_result.xml,
                "fileName": export_result.file_name,
            });
            let eval = document::eval(
                "const data = await dioxus.recv();
                 const blob = new Blob([data.xml], { type: 'application/xml' });
                 const url = URL.createObjectURL(blob);
                 const a = document.createElement('a');
                 a.href = url;
                 a.download = data.fileName;
                 document.body.appendChild(a);
                 a.click();
                 a.remove();
                 URL.revokeObjectURL(url);",
            );
            let _ = eval.send(payload);
        }
    };

    rsx! {
        div {
            class: "fixed inset-0 z-50 bg-black/40",
            onclick: move |_| on_close.call(()),
        }
        div { class: "fixed inset-0 z-50 flex items-center justify-center p-4",
            div {
                class: "card bg-base-100 shadow-xl w-full max-w-lg max-h-[90vh] overflow-y-auto",
                onclick: move |evt| evt.stop_propagation(),
                div { class: "card-body gap-3",
                    div { class: "flex items-center justify-between",
                        h3 { class: "font-semibold", "Export ADES pre-submission file" }
                        button {
                            class: "btn btn-sm btn-circle btn-ghost",
                            r#type: "button",
                            onclick: move |_| on_close.call(()),
                            "✕"
                        }
                    }

                    label { class: "form-control",
                        span { class: "label-text", "Submitter name" }
                        input {
                            class: "input input-sm input-bordered",
                            value: "{header().submitter_name}",
                            oninput: move |evt| {
                                header.write().submitter_name = evt.value();
                                reset_to_idle();
                            },
                        }
                    }
                    div { class: "flex items-center gap-2",
                        span { class: "text-xs opacity-70", "Prefill measurers/observers:" }
                        for preset in [RUBIN_PRESET, ZTF_PRESET] {
                            button {
                                key: "{preset.button_label}",
                                class: "btn btn-xs btn-outline",
                                r#type: "button",
                                onclick: move |_| {
                                    header.write().measurers = vec![preset.measurers.to_string()];
                                    header.write().observers = vec![preset.observers.to_string()];
                                    reset_to_idle();
                                },
                                "{preset.button_label}"
                            }
                        }
                    }
                    label { class: "form-control",
                        span { class: "label-text", "Measurers (comma-separated, required)" }
                        input {
                            class: "input input-sm input-bordered",
                            value: "{header().measurers.join(\", \")}",
                            oninput: move |evt| {
                                header.write().measurers = evt
                                    .value()
                                    .split(',')
                                    .map(|s| s.trim().to_string())
                                    .filter(|s| !s.is_empty())
                                    .collect();
                                reset_to_idle();
                            },
                        }
                    }
                    label { class: "form-control",
                        span { class: "label-text", "Observers (comma-separated, optional)" }
                        input {
                            class: "input input-sm input-bordered",
                            value: "{header().observers.join(\", \")}",
                            oninput: move |evt| {
                                header.write().observers = evt
                                    .value()
                                    .split(',')
                                    .map(|s| s.trim().to_string())
                                    .filter(|s| !s.is_empty())
                                    .collect();
                                reset_to_idle();
                            },
                        }
                    }
                    div { class: "grid grid-cols-3 gap-2",
                        label { class: "form-control",
                            span { class: "label-text", "Telescope design" }
                            input {
                                class: "input input-sm input-bordered",
                                value: "{header().telescope_design}",
                                oninput: move |evt| {
                                    header.write().telescope_design = evt.value();
                                    reset_to_idle();
                                },
                            }
                        }
                        label { class: "form-control",
                            span { class: "label-text", "Aperture (m)" }
                            input {
                                class: "input input-sm input-bordered",
                                value: "{header().telescope_aperture}",
                                oninput: move |evt| {
                                    header.write().telescope_aperture = evt.value();
                                    reset_to_idle();
                                },
                            }
                        }
                        label { class: "form-control",
                            span { class: "label-text", "Detector" }
                            input {
                                class: "input input-sm input-bordered",
                                value: "{header().telescope_detector}",
                                oninput: move |evt| {
                                    header.write().telescope_detector = evt.value();
                                    reset_to_idle();
                                },
                            }
                        }
                    }
                    div { class: "grid grid-cols-2 gap-2",
                        label { class: "form-control",
                            span { class: "label-text", "astCat" }
                            input {
                                class: "input input-sm input-bordered",
                                value: "{header().ast_cat}",
                                oninput: move |evt| {
                                    header.write().ast_cat = evt.value();
                                    reset_to_idle();
                                },
                            }
                        }
                        label { class: "form-control",
                            span { class: "label-text", "mode" }
                            input {
                                class: "input input-sm input-bordered",
                                value: "{header().mode}",
                                oninput: move |evt| {
                                    header.write().mode = evt.value();
                                    reset_to_idle();
                                },
                            }
                        }
                    }
                    label { class: "form-control",
                        span { class: "label-text", "Acknowledgment message (required by MPC)" }
                        input {
                            class: "input input-sm input-bordered",
                            value: "{header().ack_message}",
                            oninput: move |evt| {
                                header.write().ack_message = evt.value();
                                reset_to_idle();
                            },
                        }
                    }
                    label { class: "form-control",
                        span { class: "label-text", "Acknowledgment email (required by MPC)" }
                        input {
                            class: "input input-sm input-bordered",
                            r#type: "email",
                            value: "{header().ac2_email}",
                            oninput: move |evt| {
                                header.write().ac2_email = evt.value();
                                reset_to_idle();
                            },
                        }
                    }

                    if let Some(export_result) = result() {
                        if export_result.singleton_summary.removed_observation_count > 0 {
                            div { class: "alert alert-warning text-xs py-2",
                                "{export_result.singleton_summary.removed_observation_count} observation(s) from {export_result.singleton_summary.singleton_night_count} singleton night(s) were removed from the ADES file — the MPC rejects any batch containing a night with a single position."
                            }
                        }
                        if !export_result.advisory.warnings.is_empty() {
                            div { class: "alert alert-warning text-xs py-2 flex-col items-start",
                                for warning in &export_result.advisory.warnings {
                                    div { "{warning}" }
                                }
                            }
                        }
                    }

                    match state() {
                        ExportState::Idle => rsx! {
                            button {
                                class: "btn btn-primary btn-sm",
                                r#type: "button",
                                onclick: check,
                                "Generate and check"
                            }
                        },
                        ExportState::Checking => rsx! {
                            div { class: "flex items-center gap-2",
                                span { class: "loading loading-spinner loading-sm" }
                                span { "Checking..." }
                            }
                        },
                        ExportState::Valid => rsx! {
                            div { class: "flex flex-col gap-2",
                                div { class: "flex items-center gap-2",
                                    span { class: "badge badge-success", "Valid" }
                                    button {
                                        class: "btn btn-primary btn-sm",
                                        r#type: "button",
                                        onclick: download,
                                        "Download"
                                    }
                                }
                                if let Some(export_result) = result() {
                                    match &export_result.mpc_submission {
                                        Some(Ok(submission_id)) => rsx! {
                                            div { class: "text-xs opacity-70",
                                                "Submitted to the MPC for full validation (ID {submission_id}) — the detailed report will be emailed to {header().ac2_email}."
                                            }
                                        },
                                        Some(Err(message)) => rsx! {
                                            div { class: "text-xs text-warning",
                                                "The file is locally valid, but submitting it to the MPC failed: {message}"
                                            }
                                        },
                                        None => rsx! {},
                                    }
                                }
                            }
                        },
                        ExportState::Invalid => rsx! {
                            div { class: "flex flex-col gap-2",
                                span { class: "badge badge-error", "Invalid" }
                                if let Some(message) = request_error() {
                                    div { class: "text-xs text-error", "{message}" }
                                }
                                if let Some(export_result) = result() {
                                    ul { class: "text-xs text-error list-disc pl-4",
                                        for violation in &export_result.schema_violations {
                                            li { "{violation}" }
                                        }
                                    }
                                }
                                button {
                                    class: "btn btn-sm",
                                    r#type: "button",
                                    onclick: check,
                                    "Re-check"
                                }
                            }
                        },
                    }
                }
            }
        }
    }
}
