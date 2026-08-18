use hifitime::{Epoch, TimeScale};

/// Convert an MJD(TT) epoch (the unit stored throughout the database/engine)
/// to an ISO-8601 UTC string. Pure computation, no network access (unlike
/// `fink-fat-engine`'s UT1 provider use of `hifitime`) — safe to run
/// server-side or in the browser.
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
