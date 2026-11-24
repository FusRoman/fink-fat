use serde_json::{json, Value};

/// Which real-time alert survey an observation comes from, keyed off its
/// `mpc_code_obs` (ZTF's Palomar code `I41`, LSST's Rubin code `X05`).
#[derive(Clone, Copy, PartialEq)]
pub enum Survey {
    ZTF,
    LSST,
}

impl Survey {
    pub fn get_link(&self) -> &'static str {
        match self {
            Survey::ZTF => "https://ztf.fink-portal.org/",
            Survey::LSST => "https://lsst.fink-portal.org/",
        }
    }

    pub fn from_code_obs(code_obs: &str) -> Option<Self> {
        match code_obs {
            "X05" => Some(Survey::LSST),
            "I41" => Some(Survey::ZTF),
            _ => None,
        }
    }

    pub fn cutout_api_url(&self) -> &'static str {
        match self {
            Survey::ZTF => "https://api.ztf.fink-portal.org/api/v1/cutouts",
            Survey::LSST => "https://api.lsst.fink-portal.org/api/v1/cutouts",
        }
    }

    /// Body for a single-kind `/api/v1/cutouts` request. `object_id` and
    /// `alert_id` are `ObservationRow::object_id`/`::id` — for ZTF, `id` is
    /// the alert's `candid` (its own `object_id` is only the persistent,
    /// object-wide `objectId`, not enough to pick one specific alert); for
    /// LSST, `id` is the alert's own `diaSourceId`, which the cutout API
    /// wants directly (`object_id` happens to carry the same value there,
    /// but `id` is the authoritative one). `kind_label` is
    /// `CutoutKind::label()`, passed as a plain string so this module has no
    /// dependency on the cutout-feature module.
    pub fn cutout_request_body(&self, object_id: &str, alert_id: i64, kind_label: &str) -> Value {
        match self {
            Survey::ZTF => json!({
                "objectId": object_id,
                "candid": alert_id,
                "kind": kind_label,
            }),
            Survey::LSST => json!({
                "diaSourceId": alert_id.to_string(),
                "kind": kind_label,
            }),
        }
    }
}

pub enum ObsLink {
    Valid(String),
    Unknown(String),
}

/// Fink-portal link for an observation, or `Unknown` (carrying the raw code)
/// if `mpc_code_obs` isn't a recognised survey.
pub fn observation_link(object_id: &str, mpc_code_obs: &str) -> ObsLink {
    match Survey::from_code_obs(mpc_code_obs) {
        Some(s) => ObsLink::Valid(format!("{}{}", s.get_link(), object_id)),
        None => ObsLink::Unknown(mpc_code_obs.to_string()),
    }
}
