//! Shared submitter/telescope identity form: the field bindings and survey
//! presets used both by the lineage page's `AdesExportModal` (private to
//! `lineage_page`, so not directly linkable from here — a single-lineage
//! MPC test-tier check) and by the Submission page's
//! generator for `fink-fat submit`'s `--submitter-config` YAML file. Both
//! ultimately edit a
//! [`fink_fat_ades::submitter_config::SubmitterConfig`] — the exact type
//! `fink-fat submit` deserializes — so a value built here and downloaded as
//! YAML is guaranteed to load back without surprises.

use dioxus::prelude::*;
use fink_fat_ades::submitter_config::SubmitterConfig;

use crate::survey::Survey;

/// One-click prefill for measurers/observers/telescope — one per survey
/// fink-fat ingests. `measurers` credits whoever measured the astrometry —
/// for both surveys that's the survey's own alert-production pipeline
/// upstream of Fink, not Fink itself (Fink only brokers and enriches
/// already-measured alerts) — and `observers` names that same pipeline
/// explicitly.
#[derive(Clone, Copy)]
struct SurveyPreset {
    button_label: &'static str,
    measurers: &'static str,
    observers: &'static str,
    telescope_design: &'static str,
    telescope_aperture: &'static str,
    telescope_detector: &'static str,
}

const RUBIN_PRESET: SurveyPreset = SurveyPreset {
    button_label: "Rubin (X05)",
    measurers: "Vera C. Rubin Observatory (LSST)",
    observers: "Rubin Observatory Alert Production Pipeline",
    telescope_design: "Reflector",
    telescope_aperture: "8.4",
    telescope_detector: "CCD Mosaic",
};

const ZTF_PRESET: SurveyPreset = SurveyPreset {
    button_label: "ZTF (I41)",
    measurers: "Zwicky Transient Facility",
    observers: "ZTF Alert Production Pipeline",
    telescope_design: "Schmidt",
    telescope_aperture: "1.2",
    telescope_detector: "CCD Mosaic",
};

impl SurveyPreset {
    /// Overwrites `config`'s measurer/observer/telescope fields with this
    /// preset's values. Pure.
    fn apply(self, config: &mut SubmitterConfig) {
        config.measurers = vec![self.measurers.to_string()];
        config.observers = vec![self.observers.to_string()];
        config.telescope_design = self.telescope_design.to_string();
        config.telescope_aperture = self.telescope_aperture.to_string();
        config.telescope_detector = self.telescope_detector.to_string();
    }
}

/// Baseline defaults shared by every consumer of this form — the fields
/// that don't depend on a specific lineage or survey: `ast_cat` defaults to
/// `"Gaia2"` (the reference catalog fink-fat's surveys reduce astrometry
/// against) and `mode` to `"CCD"` (the only detector mode either survey
/// uses). Everything else starts blank; callers layer their own additions
/// on top (lineage-specific `ack_message`, survey-detected telescope
/// fields, ...) via struct-update syntax.
///
/// # Return
/// A [`SubmitterConfig`] with just `ast_cat`/`mode` prefilled.
pub fn default_submitter_config() -> SubmitterConfig {
    SubmitterConfig {
        ast_cat: "Gaia2".to_string(),
        mode: "CCD".to_string(),
        ..Default::default()
    }
}

/// Telescope defaults for the two surveys fink-fat ingests, keyed off the
/// station codes present in a lineage's observations
/// ([`Survey::from_code_obs`]) — used to auto-fill the form when it opens
/// for a specific lineage. The Submission page's generic form has no single
/// lineage to detect from, so it starts blank and relies on the preset
/// buttons ([`SurveyPreset::apply`]) instead.
///
/// # Arguments
/// * `mpc_codes` — the observation station codes to detect a survey from.
///
/// # Return
/// `(design, aperture, detector)`, blank if no known survey matched.
pub fn default_telescope(mpc_codes: &[String]) -> (String, String, String) {
    let survey = mpc_codes
        .iter()
        .find_map(|code| Survey::from_code_obs(code));
    match survey {
        Some(Survey::ZTF) => (
            ZTF_PRESET.telescope_design.to_string(),
            ZTF_PRESET.telescope_aperture.to_string(),
            ZTF_PRESET.telescope_detector.to_string(),
        ),
        Some(Survey::LSST) => (
            RUBIN_PRESET.telescope_design.to_string(),
            RUBIN_PRESET.telescope_aperture.to_string(),
            RUBIN_PRESET.telescope_detector.to_string(),
        ),
        None => (String::new(), String::new(), String::new()),
    }
}

/// The submitter/telescope identity form shared by the lineage page's
/// `AdesExportModal` and the Submission page's config generator. Mutates
/// `config` in place; the
/// caller supplies `on_change` for whatever it needs to happen on every
/// edit (the modal resets its check state, the Submission page does
/// nothing). Also shows [`SubmitterConfig::validation_errors`] inline, so
/// every consumer surfaces exactly the same gate `fink-fat submit` itself
/// enforces.
#[component]
pub fn SubmitterConfigFields(
    config: Signal<SubmitterConfig>,
    on_change: EventHandler<()>,
) -> Element {
    rsx! {
        label { class: "form-control",
            span { class: "label-text", "Submitter name" }
            input {
                class: "input input-sm input-bordered",
                value: "{config().submitter_name}",
                oninput: move |evt| {
                    config.write().submitter_name = evt.value();
                    on_change.call(());
                },
            }
        }
        div { class: "flex items-center gap-2",
            span { class: "text-xs opacity-70", "Prefill from a survey:" }
            for preset in [RUBIN_PRESET, ZTF_PRESET] {
                button {
                    key: "{preset.button_label}",
                    class: "btn btn-xs btn-outline",
                    r#type: "button",
                    onclick: move |_| {
                        preset.apply(&mut config.write());
                        on_change.call(());
                    },
                    "{preset.button_label}"
                }
            }
        }
        label { class: "form-control",
            span { class: "label-text", "Measurers (comma-separated, required)" }
            input {
                class: "input input-sm input-bordered",
                value: "{config().measurers.join(\", \")}",
                oninput: move |evt| {
                    config.write().measurers = evt
                        .value()
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                    on_change.call(());
                },
            }
        }
        label { class: "form-control",
            span { class: "label-text", "Observers (comma-separated, optional)" }
            input {
                class: "input input-sm input-bordered",
                value: "{config().observers.join(\", \")}",
                oninput: move |evt| {
                    config.write().observers = evt
                        .value()
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                    on_change.call(());
                },
            }
        }
        div { class: "grid grid-cols-3 gap-2",
            label { class: "form-control",
                span { class: "label-text", "Telescope design" }
                input {
                    class: "input input-sm input-bordered",
                    value: "{config().telescope_design}",
                    oninput: move |evt| {
                        config.write().telescope_design = evt.value();
                        on_change.call(());
                    },
                }
            }
            label { class: "form-control",
                span { class: "label-text", "Aperture (m)" }
                input {
                    class: "input input-sm input-bordered",
                    value: "{config().telescope_aperture}",
                    oninput: move |evt| {
                        config.write().telescope_aperture = evt.value();
                        on_change.call(());
                    },
                }
            }
            label { class: "form-control",
                span { class: "label-text", "Detector" }
                input {
                    class: "input input-sm input-bordered",
                    value: "{config().telescope_detector}",
                    oninput: move |evt| {
                        config.write().telescope_detector = evt.value();
                        on_change.call(());
                    },
                }
            }
        }
        div { class: "grid grid-cols-2 gap-2",
            label { class: "form-control",
                span { class: "label-text", "astCat" }
                input {
                    class: "input input-sm input-bordered",
                    value: "{config().ast_cat}",
                    oninput: move |evt| {
                        config.write().ast_cat = evt.value();
                        on_change.call(());
                    },
                }
            }
            label { class: "form-control",
                span { class: "label-text", "mode" }
                input {
                    class: "input input-sm input-bordered",
                    value: "{config().mode}",
                    oninput: move |evt| {
                        config.write().mode = evt.value();
                        on_change.call(());
                    },
                }
            }
        }
        label { class: "form-control",
            span { class: "label-text",
                "Acknowledgment message (optional — auto-generated per lineage if left blank)"
            }
            input {
                class: "input input-sm input-bordered",
                value: "{config().ack_message.clone().unwrap_or_default()}",
                oninput: move |evt| {
                    let value = evt.value();
                    config.write().ack_message = if value.trim().is_empty() { None } else { Some(value) };
                    on_change.call(());
                },
            }
        }
        label { class: "form-control",
            span { class: "label-text", "Acknowledgment email (required by MPC)" }
            input {
                class: "input input-sm input-bordered",
                r#type: "email",
                value: "{config().ac2_email}",
                oninput: move |evt| {
                    config.write().ac2_email = evt.value();
                    on_change.call(());
                },
            }
        }
        if !config().validation_errors().is_empty() {
            div { class: "alert alert-warning text-xs py-2 flex-col items-start",
                for reason in config().validation_errors() {
                    div { key: "{reason}", "⚠ {reason}" }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ztf_preset_fills_measurers_observers_and_telescope() {
        let mut config = SubmitterConfig::default();
        ZTF_PRESET.apply(&mut config);
        assert_eq!(
            config.measurers,
            vec!["Zwicky Transient Facility".to_string()]
        );
        assert_eq!(config.telescope_design, "Schmidt");
        assert_eq!(config.telescope_aperture, "1.2");
    }

    #[test]
    fn rubin_preset_fills_measurers_observers_and_telescope() {
        let mut config = SubmitterConfig::default();
        RUBIN_PRESET.apply(&mut config);
        assert_eq!(
            config.measurers,
            vec!["Vera C. Rubin Observatory (LSST)".to_string()]
        );
        assert_eq!(config.telescope_design, "Reflector");
    }

    #[test]
    fn default_telescope_detects_ztf_from_station_code() {
        let (design, aperture, detector) = default_telescope(&["I41".to_string()]);
        assert_eq!(design, "Schmidt");
        assert_eq!(aperture, "1.2");
        assert_eq!(detector, "CCD Mosaic");
    }

    #[test]
    fn default_telescope_detects_lsst_from_station_code() {
        let (design, ..) = default_telescope(&["X05".to_string()]);
        assert_eq!(design, "Reflector");
    }

    #[test]
    fn default_telescope_blank_for_unknown_codes() {
        let (design, aperture, detector) = default_telescope(&["ZZZ".to_string()]);
        assert!(design.is_empty());
        assert!(aperture.is_empty());
        assert!(detector.is_empty());
    }
}
