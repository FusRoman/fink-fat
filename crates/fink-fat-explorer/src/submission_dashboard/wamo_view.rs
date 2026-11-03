//! How the Submission page shows a production submission's WAMO state: the
//! compact badge of the table's "MPC processing" column
//! ([`WamoSummaryBadge`]) and the per-state breakdown of the expanded row
//! ([`WamoDetailSection`]).
//!
//! Every observation is classified from its `status_decoded` sentence by
//! [`fink_fat_ades::wamo::classify_status_decoded`], including the ones MPC
//! still hides (`"hidden"` designation): those still name the processing
//! queue they sit in, which is exactly what tells a user that MPC is
//! working on the submission. The pure helpers below turn the classified
//! observations into labels, colors and explanations; the components only
//! assemble them.

use dioxus::prelude::*;
use fink_fat_ades::wamo::{
    queue_description, queue_needs_mpc_attention, wamo_field_is_hidden, WamoObservation, WamoStage,
    WamoStatus, MPC_WAMO_HELP_URL, MPC_WAMO_MANUAL_URL,
};

/// Parses a row's stored `wamo_detail` JSON back into the observations it
/// holds. Defensive rather than fallible: this is fink-fat's own
/// previously-written JSON, so a parse failure would mean a schema drift
/// bug, not bad external input — surfacing it as an empty list (a state the
/// UI already renders sensibly, as "not linked yet") is better than an
/// error banner for what is, at worst, a display glitch.
///
/// # Arguments
/// * `wamo_detail` — the row's `wamo_detail` field.
///
/// # Return
/// The observations, empty if `wamo_detail` is `None` or fails to parse.
fn parse_wamo_observations(wamo_detail: &Option<serde_json::Value>) -> Vec<WamoObservation> {
    wamo_detail
        .as_ref()
        .and_then(|value| serde_json::from_value(value.clone()).ok())
        .unwrap_or_default()
}

/// The one-line summary badge text for a WAMO observation's IAU designation
/// — the actual payoff of a real submission once MPC assigns one. Pure.
///
/// # Arguments
/// * `observation` — one WAMO match.
///
/// # Return
/// `Some("Designated (NNNNN)")` if MPC assigned a designation, `None`
/// otherwise — either no designation exists (e.g. an identification against
/// a known object, without a fresh designation), or MPC hasn't processed
/// the observation yet ([`WamoObservation::is_pending`] — its `iau_desig`
/// would otherwise be the literal string `"hidden"`, not a real value).
fn wamo_designation_label(observation: &WamoObservation) -> Option<String> {
    observation
        .iau_desig
        .as_deref()
        .filter(|desig| !wamo_field_is_hidden(desig))
        .map(|desig| format!("Designated ({desig})"))
}

/// daisyUI badge color class for a [`WamoStage`].
///
/// # Arguments
/// * `stage` — the stage to color.
///
/// # Return
/// Green when done, blue in progress, orange when waiting on MPC staff, red
/// for a discarded observation.
fn stage_badge_class(stage: WamoStage) -> &'static str {
    match stage {
        WamoStage::Done => "badge-success",
        WamoStage::InProgress => "badge-info",
        WamoStage::NeedsMpcAttention => "badge-warning",
        WamoStage::Problem => "badge-error",
    }
}

/// Short human-readable label of a [`WamoStatus`], for badges.
///
/// # Arguments
/// * `status` — the classified status.
///
/// # Return
/// The label, e.g. `"In MPC queue: whatcode"`.
fn status_label(status: &WamoStatus) -> String {
    match status {
        WamoStatus::InQueue { queue } => format!("In MPC queue: {queue}"),
        WamoStatus::NotProcessed => "Not processed yet".to_string(),
        WamoStatus::IdentifiedPendingPublication { designation } => {
            format!("Identified as {designation}, publication pending")
        }
        WamoStatus::Published {
            designation,
            reference,
        } => format!("Published: {designation} ({reference})"),
        WamoStatus::OnConfirmationPage => "On the NEOCP/PCCP".to_string(),
        WamoStatus::Itf => "Isolated Tracklet File".to_string(),
        WamoStatus::UnidentifiedIsolated => "Unidentified isolated observation".to_string(),
        WamoStatus::Rejected => "Rejected".to_string(),
        WamoStatus::Deleted => "Deleted".to_string(),
        WamoStatus::NotMinorPlanet => "Not a minor planet".to_string(),
        WamoStatus::SuspectedArtificial => "Suspected artificial".to_string(),
        WamoStatus::NearDuplicate => "Near-duplicate".to_string(),
        WamoStatus::UnknownStatus => "Unknown status".to_string(),
        WamoStatus::Unrecognized => "Unrecognized status".to_string(),
    }
}

/// One-sentence explanation of a [`WamoStatus`], from the WAMO help page.
///
/// # Arguments
/// * `status` — the classified status.
///
/// # Return
/// The explanation, `None` when the label says it all.
fn status_explanation(status: &WamoStatus) -> Option<String> {
    match status {
        WamoStatus::InQueue { queue } => {
            let holds = queue_description(queue).map_or_else(
                || "This queue is not described on the WAMO help page.".to_string(),
                |description| format!("This MPC queue holds {description}."),
            );
            Some(if queue_needs_mpc_attention(queue) {
                format!(
                    "{holds} It needs manual action at MPC and can stay unchanged \
                     for a few to several days."
                )
            } else {
                holds
            })
        }
        WamoStatus::NotProcessed => Some("Received by MPC, not processed yet.".to_string()),
        WamoStatus::OnConfirmationPage => {
            Some("Posted on MPC's confirmation page, awaiting follow-up.".to_string())
        }
        WamoStatus::Itf => Some(
            "Not linked to a known object: kept in the Isolated Tracklet File for a \
             future linkage."
                .to_string(),
        ),
        WamoStatus::UnidentifiedIsolated => Some("Not linked to any known object.".to_string()),
        WamoStatus::Rejected => Some("Usually due to poor residuals.".to_string()),
        WamoStatus::UnknownStatus => {
            Some("MPC itself reports this observation's status as unknown.".to_string())
        }
        WamoStatus::Unrecognized => Some(
            "This sentence is not on the WAMO help page: see MPC's raw text below.".to_string(),
        ),
        WamoStatus::IdentifiedPendingPublication { .. }
        | WamoStatus::Published { .. }
        | WamoStatus::Deleted
        | WamoStatus::NotMinorPlanet
        | WamoStatus::SuspectedArtificial
        | WamoStatus::NearDuplicate => None,
    }
}

/// A submission's observations that share one [`WamoStatus`].
#[derive(Debug, PartialEq)]
struct StatusGroup<'a> {
    status: WamoStatus,
    observations: Vec<&'a WamoObservation>,
}

impl StatusGroup<'_> {
    /// MPC's raw `status_decoded` sentences for this group, without
    /// repeats (every observation of a submission usually shares one).
    ///
    /// # Return
    /// The distinct sentences, in observation order.
    fn distinct_sentences(&self) -> Vec<&str> {
        self.observations
            .iter()
            .map(|observation| observation.status_decoded.as_str())
            .fold(Vec::new(), |mut sentences, sentence| {
                if !sentences.contains(&sentence) {
                    sentences.push(sentence);
                }
                sentences
            })
    }
}

/// Group observations by their classified status, so a submission whose
/// observations all sit in the same state reads as one block.
///
/// # Arguments
/// * `observations` — a submission's WAMO observations.
///
/// # Return
/// One group per distinct status, in order of first appearance.
fn group_by_status(observations: &[WamoObservation]) -> Vec<StatusGroup<'_>> {
    observations
        .iter()
        .fold(Vec::<StatusGroup<'_>>::new(), |mut groups, observation| {
            let status = observation.classify();
            match groups.iter_mut().find(|group| group.status == status) {
                Some(group) => group.observations.push(observation),
                None => groups.push(StatusGroup {
                    status,
                    observations: vec![observation],
                }),
            }
            groups
        })
}

/// The table badge summarising a whole submission.
#[derive(Debug, PartialEq)]
struct WamoSummary {
    stage: WamoStage,
    label: String,
    /// MPC's raw sentences behind the label, one per line (hover text).
    tooltip: String,
}

/// Summarise a submission's WAMO state as its worst status group: one
/// observation still in a queue means the submission is not done, one
/// rejected observation deserves a look.
///
/// # Arguments
/// * `observations` — a submission's WAMO observations.
///
/// # Return
/// The summary, `None` when WAMO returned no observation.
fn wamo_summary(observations: &[WamoObservation]) -> Option<WamoSummary> {
    let groups = group_by_status(observations);
    let worst = groups.iter().max_by_key(|group| group.status.stage())?;
    let count = if groups.len() == 1 {
        format!("{} obs", observations.len())
    } else {
        format!("{}/{} obs", worst.observations.len(), observations.len())
    };
    Some(WamoSummary {
        stage: worst.status.stage(),
        label: format!("{} · {count}", status_label(&worst.status)),
        tooltip: worst.distinct_sentences().join("\n"),
    })
}

/// The "MPC processing" table cell of a production submission.
///
/// # Arguments
/// * `wamo_detail` — the row's stored WAMO observations, `None` if WAMO was
///   never queried.
#[component]
pub(super) fn WamoSummaryBadge(wamo_detail: Option<serde_json::Value>) -> Element {
    if wamo_detail.is_none() {
        return rsx! { span { class: "text-xs opacity-50", "—" } };
    }
    match wamo_summary(&parse_wamo_observations(&wamo_detail)) {
        Some(summary) => rsx! {
            span {
                class: "badge badge-sm whitespace-nowrap {stage_badge_class(summary.stage)}",
                title: "{summary.tooltip}",
                "{summary.label}"
            }
        },
        None => rsx! { span { class: "text-xs opacity-60", "Not in WAMO yet" } },
    }
}

/// The "WAMO detail" block of an expanded production row: one block per
/// status group, then links to check the submission by hand on MPC's site.
///
/// # Arguments
/// * `submission_id` — the MPC submission ID, copied for the manual WAMO
///   page (`None` if MPC never returned one).
/// * `wamo_detail` — the row's stored WAMO observations.
/// * `wamo_checked_at` — when WAMO was last queried.
#[component]
pub(super) fn WamoDetailSection(
    submission_id: Option<String>,
    wamo_detail: Option<serde_json::Value>,
    wamo_checked_at: Option<String>,
) -> Element {
    let mut copied = use_signal(|| false);
    let observations = parse_wamo_observations(&wamo_detail);
    let groups = group_by_status(&observations);

    rsx! {
        div { class: "flex flex-col gap-2",
            div { class: "font-semibold text-xs opacity-70 flex items-center gap-2",
                "WAMO detail"
                if let Some(checked_at) = &wamo_checked_at {
                    span { class: "font-normal opacity-60", "(checked {checked_at})" }
                }
            }
            if wamo_detail.is_none() {
                p { class: "text-xs opacity-60", "Not checked yet — click Refresh to query WAMO." }
            } else if groups.is_empty() {
                p { class: "text-xs opacity-60",
                    "MPC hasn't linked this submission in WAMO yet — normal right after \
                     submitting, check back later."
                }
            }
            for group in &groups {
                div {
                    key: "{status_label(&group.status)}",
                    class: "border border-base-300 rounded p-2 flex flex-col gap-1",
                    div { class: "flex items-center gap-2",
                        span { class: "badge badge-sm {stage_badge_class(group.status.stage())}",
                            "{status_label(&group.status)}"
                        }
                        span { class: "text-xs opacity-70",
                            "{group.observations.len()} observation(s)"
                        }
                    }
                    if let Some(explanation) = status_explanation(&group.status) {
                        span { class: "text-xs", "{explanation}" }
                    }
                    for sentence in group.distinct_sentences() {
                        span { key: "{sentence}", class: "text-xs opacity-60 font-mono", "MPC: {sentence}" }
                    }
                    for observation in group.observations.iter().filter(|o| !o.is_pending()) {
                        div { key: "{observation.obsid}", class: "flex flex-col gap-1 pl-2 border-l border-base-300",
                            if let Some(label) = wamo_designation_label(observation) {
                                span { class: "badge badge-success badge-sm self-start", "{label}" }
                            }
                            if let Some(obs80) = observation.obs80.as_deref().filter(|v| !wamo_field_is_hidden(v)) {
                                code { class: "text-xs block", "{obs80}" }
                            }
                            if let Some(reference) = observation.reference.as_deref().filter(|v| !wamo_field_is_hidden(v)) {
                                span { class: "text-xs opacity-70", "Reference: {reference}" }
                            }
                        }
                    }
                }
            }
            div { class: "flex flex-wrap items-center gap-3 text-xs",
                if let Some(submission_id) = submission_id {
                    button {
                        class: "btn btn-xs btn-outline",
                        r#type: "button",
                        title: "Copy the submission ID, to paste on the WAMO page",
                        onclick: move |_| {
                            let eval = document::eval("navigator.clipboard.writeText(await dioxus.recv());");
                            let _ = eval.send(submission_id.clone());
                            copied.set(true);
                        },
                        if copied() { "✓ Copied" } else { "📋 Copy ID" }
                    }
                }
                a {
                    class: "link",
                    href: MPC_WAMO_MANUAL_URL,
                    target: "_blank",
                    rel: "noopener noreferrer",
                    "Check on WAMO ↗"
                }
                a {
                    class: "link opacity-70",
                    href: MPC_WAMO_HELP_URL,
                    target: "_blank",
                    rel: "noopener noreferrer",
                    "What do these statuses mean? ↗"
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A WAMO observation as stored in `wamo_detail`.
    fn observation(obsid: &str, iau_desig: &str, status_decoded: &str) -> WamoObservation {
        let mut observations = parse_wamo_observations(&Some(serde_json::json!([{
            "iau_desig": iau_desig,
            "input_type": "submission_id",
            "obs80": "hidden",
            "obsid": obsid,
            "obssubid": null,
            "ref": "hidden",
            "status": "W",
            "status_decoded": status_decoded,
            "submission_block_id": null,
            "submission_id": null
        }])));
        observations.remove(0)
    }

    const WHATCODE: &str = "The submission_id 'S' is in the 'whatcode' processing queue.";
    const MBA_UNN: &str = "The submission_id 'S' is in the 'mba/unn' processing queue.";

    #[test]
    fn parse_wamo_observations_empty_for_none() {
        assert!(parse_wamo_observations(&None).is_empty());
    }

    #[test]
    fn parse_wamo_observations_empty_for_malformed_json() {
        assert!(parse_wamo_observations(&Some(serde_json::json!({"not": "a list"}))).is_empty());
    }

    #[test]
    fn wamo_designation_label_present_when_designated() {
        assert_eq!(
            wamo_designation_label(&observation("obs-1", "380635", "matched")),
            Some("Designated (380635)".to_string())
        );
    }

    #[test]
    fn wamo_designation_label_absent_when_mpc_hides_it_pending_processing() {
        // MPC's real, live behavior on an unprocessed production submission
        // (see `fink_fat_ades::wamo`'s module docs): `iau_desig` is the
        // literal string "hidden", not `null`.
        let observation = observation("obs-1", "hidden", WHATCODE);
        assert_eq!(wamo_designation_label(&observation), None);
        assert!(observation.is_pending());
    }

    #[test]
    fn group_by_status_merges_observations_in_the_same_queue() {
        let observations = [
            observation("obs-1", "hidden", WHATCODE),
            observation("obs-2", "hidden", WHATCODE),
        ];
        let groups = group_by_status(&observations);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].observations.len(), 2);
        assert_eq!(groups[0].distinct_sentences(), vec![WHATCODE]);
    }

    #[test]
    fn group_by_status_keeps_first_appearance_order() {
        let observations = [
            observation("obs-1", "hidden", MBA_UNN),
            observation("obs-2", "hidden", WHATCODE),
            observation("obs-3", "hidden", MBA_UNN),
        ];
        let queues: Vec<String> = group_by_status(&observations)
            .iter()
            .map(|group| status_label(&group.status))
            .collect();
        assert_eq!(queues, ["In MPC queue: mba/unn", "In MPC queue: whatcode"]);
    }

    #[test]
    fn wamo_summary_none_without_observations() {
        assert_eq!(wamo_summary(&[]), None);
    }

    #[test]
    fn wamo_summary_of_a_single_state_counts_every_observation() {
        let observations = [
            observation("obs-1", "hidden", WHATCODE),
            observation("obs-2", "hidden", WHATCODE),
        ];
        assert_eq!(
            wamo_summary(&observations),
            Some(WamoSummary {
                stage: WamoStage::NeedsMpcAttention,
                label: "In MPC queue: whatcode · 2 obs".to_string(),
                tooltip: WHATCODE.to_string(),
            })
        );
    }

    #[test]
    fn wamo_summary_of_mixed_states_shows_the_worst_one() {
        let observations = [
            observation("obs-1", "hidden", MBA_UNN),
            observation("obs-2", "hidden", WHATCODE),
            observation("obs-3", "hidden", MBA_UNN),
        ];
        let summary = wamo_summary(&observations).unwrap();
        assert_eq!(summary.stage, WamoStage::NeedsMpcAttention);
        assert_eq!(summary.label, "In MPC queue: whatcode · 1/3 obs");
    }

    #[test]
    fn stage_badge_class_covers_every_stage() {
        assert_eq!(stage_badge_class(WamoStage::Done), "badge-success");
        assert_eq!(stage_badge_class(WamoStage::InProgress), "badge-info");
        assert_eq!(
            stage_badge_class(WamoStage::NeedsMpcAttention),
            "badge-warning"
        );
        assert_eq!(stage_badge_class(WamoStage::Problem), "badge-error");
    }

    #[test]
    fn status_explanation_flags_queues_needing_mpc_staff() {
        let whatcode = status_explanation(&WamoStatus::InQueue {
            queue: "whatcode".to_string(),
        })
        .unwrap();
        assert!(whatcode.contains("awaiting a program code"));
        assert!(whatcode.contains("manual action at MPC"));

        let mba = status_explanation(&WamoStatus::InQueue {
            queue: "mba/unn".to_string(),
        })
        .unwrap();
        assert!(!mba.contains("manual action"));
    }

    #[test]
    fn status_explanation_handles_an_undocumented_queue() {
        let text = status_explanation(&WamoStatus::InQueue {
            queue: "rare".to_string(),
        })
        .unwrap();
        assert!(text.contains("not described"));
    }
}
