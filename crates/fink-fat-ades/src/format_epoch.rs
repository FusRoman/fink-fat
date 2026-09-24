//! MJD(TT) epoch formatting, shared by ADES `obsTime` construction
//! ([`crate::model::mjd_tt_to_ades_obs_time`]) and `fink-fat-explorer`'s
//! human-facing displays.

use hifitime::{Epoch, TimeScale};

/// Convert an MJD(TT) epoch (the unit stored throughout the
/// database/engine) to an ISO-8601 UTC string. Pure computation, no network
/// access — safe to run server-side, in the browser (wasm), or in a native
/// CLI process.
pub fn iso_utc(mjd_tt: f64) -> String {
    Epoch::from_mjd_in_time_scale(mjd_tt, TimeScale::TT)
        .to_time_scale(TimeScale::UTC)
        .to_isoformat()
}

/// Format a reference epoch for display: ISO-8601 UTC first — what users
/// actually read — with the raw MJD(TT) value in parentheses for
/// cross-referencing against the database/engine internals.
pub fn format_epoch(mjd_tt: f64) -> String {
    format!("{} (MJD-TT {mjd_tt:.5})", iso_utc(mjd_tt))
}

/// Today's UTC calendar date as `YYYY-MM-DD`, for prefilling
/// submission-time timestamps (e.g. the ADES export's default acknowledgment
/// message). `None` only if the system/browser clock is unavailable, which
/// doesn't happen in this app's actual environments (native server, browser
/// wasm — `hifitime::Epoch::now` uses JS interop under `wasm32-unknown-unknown`).
pub fn today_utc_date() -> Option<String> {
    let (year, month, day, ..) = Epoch::now().ok()?.to_gregorian_utc();
    Some(format!("{year:04}-{month:02}-{day:02}"))
}
