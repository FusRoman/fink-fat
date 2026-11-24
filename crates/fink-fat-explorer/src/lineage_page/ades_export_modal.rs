//! Modal collecting the ADES header fields not tracked by the pipeline, then
//! driving local `submit.xsd` validation (a fast pre-check) and MPC's real
//! validation verdict (the actual gate on the download button — see
//! `crate::ades::server_fns` for why that takes a submission plus a status
//! poll rather than a single request).

use dioxus::prelude::*;
use fink_fat_ades::submitter_config::{default_ack_message, SubmitterConfig};

use crate::ades::mpc_submission::MPC_SUBMISSION_STATUS_URL;
use crate::ades::server_fns::{
    check_mpc_submission_status, export_and_validate_ades, AdesExportResult, McpVerdict,
};
use crate::submitter_config_form::{default_telescope, SubmitterConfigFields};

/// Link to MPC's test-submission status page for a given submission, so the
/// user can double-check the verdict themselves.
fn mpc_status_url(submission_id: &str) -> String {
    format!("{MPC_SUBMISSION_STATUS_URL}?id={submission_id}")
}

/// Where the export flow currently stands, driving which controls are shown.
#[derive(Clone, Copy, PartialEq)]
enum ExportState {
    Idle,
    Checking,
    Valid,
    Invalid,
}

/// The modal's initial [`SubmitterConfig`]: blank submitter/contact fields,
/// telescope auto-filled from the lineage's own observation station codes
/// ([`default_telescope`]), and a prefilled `ack_message` so the field shows
/// useful default text immediately (still editable, and still auto-derived
/// the same way if the user clears it — see
/// [`SubmitterConfig::into_header`]).
fn default_submitter_config(mpc_codes: &[String], lineage_designation: &str) -> SubmitterConfig {
    let (telescope_design, telescope_aperture, telescope_detector) = default_telescope(mpc_codes);
    let today = fink_fat_ades::format_epoch::today_utc_date().unwrap_or_default();
    SubmitterConfig {
        telescope_design,
        telescope_aperture,
        telescope_detector,
        ack_message: Some(default_ack_message(lineage_designation, &today)),
        ..crate::submitter_config_form::default_submitter_config()
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

    let config = use_signal(|| default_submitter_config(&mpc_codes, &lineage_designation));
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
            let today = fink_fat_ades::format_epoch::today_utc_date().unwrap_or_default();
            let header_value = config().into_header(&lineage_designation, &today);
            state.set(ExportState::Checking);
            request_error.set(None);
            spawn(async move {
                match export_and_validate_ades(lineage_designation, header_value).await {
                    Ok(export_result) => {
                        state.set(if export_result.download_allowed() {
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

    // Re-polls MPC's status for an already-submitted lineage, without
    // resubmitting the XML — used after a `PollTimedOut` verdict, since
    // resubmitting identical content risks MPC flagging it as a duplicate.
    let mut check_status_again = move |submission_id: String| {
        state.set(ExportState::Checking);
        spawn(async move {
            match check_mpc_submission_status(submission_id).await {
                Ok(mpc_verdict) => {
                    if let Some(mut export_result) = result() {
                        export_result.mpc_verdict = mpc_verdict;
                        state.set(if export_result.download_allowed() {
                            ExportState::Valid
                        } else {
                            ExportState::Invalid
                        });
                        result.set(Some(export_result));
                    }
                }
                Err(err) => {
                    state.set(ExportState::Invalid);
                    request_error.set(Some(err.to_string()));
                }
            }
        });
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

                    SubmitterConfigFields { config, on_change: move |_| reset_to_idle() }

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
                                span { "Checking with the MPC (this can take up to a minute)..." }
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
                                    if let McpVerdict::Valid { submission_id } = &export_result.mpc_verdict {
                                        div { class: "text-xs opacity-70",
                                            "Confirmed valid by the MPC (submission "
                                            a {
                                                class: "link",
                                                href: "{mpc_status_url(submission_id)}",
                                                target: "_blank",
                                                rel: "noopener noreferrer",
                                                "{submission_id}"
                                            }
                                            ")."
                                        }
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
                                    if !export_result.schema_valid {
                                        ul { class: "text-xs text-error list-disc pl-4",
                                            for violation in &export_result.schema_violations {
                                                li { "{violation}" }
                                            }
                                        }
                                    }
                                    match &export_result.mpc_verdict {
                                        McpVerdict::Invalid { submission_id, comments } => rsx! {
                                            div { class: "text-xs text-error",
                                                "Rejected by the MPC (submission "
                                                a {
                                                    class: "link",
                                                    href: "{mpc_status_url(submission_id)}",
                                                    target: "_blank",
                                                    rel: "noopener noreferrer",
                                                    "{submission_id}"
                                                }
                                                "):"
                                            }
                                            for comment in comments {
                                                pre { class: "text-xs text-error whitespace-pre-wrap", "{comment}" }
                                            }
                                        },
                                        McpVerdict::PollTimedOut { submission_id } => rsx! {
                                            div { class: "text-xs text-warning",
                                                "The MPC hasn't finished processing submission "
                                                a {
                                                    class: "link",
                                                    href: "{mpc_status_url(submission_id)}",
                                                    target: "_blank",
                                                    rel: "noopener noreferrer",
                                                    "{submission_id}"
                                                }
                                                " yet."
                                            }
                                            button {
                                                class: "btn btn-sm",
                                                r#type: "button",
                                                onclick: {
                                                    let submission_id = submission_id.clone();
                                                    move |_| check_status_again(submission_id.clone())
                                                },
                                                "Check status again"
                                            }
                                        },
                                        McpVerdict::SubmissionFailed(message) => rsx! {
                                            div { class: "text-xs text-error", "MPC submission failed: {message}" }
                                        },
                                        McpVerdict::NotSubmitted => rsx! {},
                                        McpVerdict::Valid { .. } => rsx! {},
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
