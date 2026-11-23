//! [`NightClock`]: when the open night closes.
//!
//! Closing a night runs N-scan pruning, the stale purge and the discovery of
//! new lineages from the night's leftover alerts (see
//! [`NightSession::close`](fink_fat_engine::topocentric_kf::branching::NightSession::close)),
//! so it must happen once, after the night's last alert, and never on a mere
//! pause. Three triggers close the night:
//!
//! 1. **A newer night starts**: an alert of a later `dayObs` arrives (the
//!    driver checks it through [`NightClock::classify`]). Rubin's `dayObs`
//!    rolls over at 12:00 UTC, so a night is never split.
//! 2. **Morning twilight**: the clock passed the Sun's crossing of
//!    [`ClockParams::twilight_altitude`] (−12° by default) plus a grace
//!    period, the source is caught up and no visit is still being
//!    assembled.
//! 3. **Safety net**: the clock passed the `dayObs` rollover (12:00 UTC the
//!    next day) plus a grace period, whatever the source's state.
//!
//! There is deliberately **no inactivity trigger**: a weather pause leaves
//! the night open, and alerts that resume before twilight join it.
//!
//! The clock is the driver's *event time* (see [`crate::driver`]): while it
//! catches up on a backlog, the night closes on the backlog's own times, not
//! on the wall clock.
//!
//! # Timeline of a night (Cerro Pachón, late October)
//!
//! ```text
//!  dayObs D, 12:00 UTC          ~23:30 UTC        D+1 ~08:50 UTC      D+1 12:00 UTC
//!  ─────┬─────────────────────────┬──────── … ──────────┬──────┬───────────┬──────┬──►
//!       │ alerts of D are "D"     │ first visit         │ −12° │ +grace    │      │ +grace
//!       │                         │                     twilight  trigger 2  rollover trigger 3
//!       │ (an alert of D+1 can only arrive after D+1 12:00 UTC → trigger 1)
//! ```
//!
//! In normal operation trigger 2 closes the night shortly after dawn; trigger
//! 1 covers a source that never reports itself caught up (and every replay);
//! trigger 3 covers a night whose last alerts never come.
//!
//! # Purity
//!
//! [`NightClock`] only holds two deadlines computed at [`NightClock::open`];
//! every decision takes the current time as an argument. The caller (the
//! driver) owns the clock and the source's state.

pub mod twilight;

use fink_fat_engine::engine_config::units::{de_angle_rad, de_time_days};
use photom::NightId;
use serde::Deserialize;

pub use twilight::{Site, morning_twilight, sun_altitude};

/// Tuning of the [`NightClock`] (`clock:` section of the stream
/// configuration; durations and angles accept units, e.g. `"30 min"`,
/// `"-12 deg"`).
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ClockParams {
    /// Sun altitude whose morning crossing ends the night, radians
    /// (default −12°, nautical twilight).
    #[serde(deserialize_with = "de_angle_rad")]
    pub twilight_altitude: f64,
    /// Delay after the twilight crossing before trigger 2 may fire, days
    /// (default 30 min) — room for the last alerts in flight.
    #[serde(deserialize_with = "de_time_days")]
    pub twilight_grace: f64,
    /// Delay after the 12:00 UTC `dayObs` rollover before the safety net
    /// fires, days (default 30 min).
    #[serde(deserialize_with = "de_time_days")]
    pub rollover_grace: f64,
    /// Observing site (default Cerro Pachón).
    pub site: Site,
}

impl Default for ClockParams {
    fn default() -> Self {
        Self {
            twilight_altitude: (-12.0_f64).to_radians(),
            twilight_grace: 30.0 / 1_440.0,
            rollover_grace: 30.0 / 1_440.0,
            site: Site::default(),
        }
    }
}

/// Where an alert's night stands relative to the open one (see
/// [`NightClock::classify`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertNight {
    /// The open night.
    Current,
    /// A later night: the open one must close (trigger 1).
    Newer,
    /// An earlier night, already closed: the alert is late and not tracked.
    Older,
}

/// Why [`NightClock::close_trigger`] closes the night.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloseTrigger {
    /// An alert of a later night arrived.
    NewerNight,
    /// Morning twilight plus grace passed, source caught up, nothing pending.
    MorningTwilight,
    /// The `dayObs` rollover plus grace passed.
    Rollover,
    /// The source ended (a replay reached its last alert). Set by the
    /// driver, never returned by [`NightClock::close_trigger`].
    EndOfStream,
}

/// Close deadlines of one open night (see the [module doc](self)).
///
/// Built once per night by [`Self::open`]; immutable afterwards.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NightClock {
    night_id: NightId,
    /// Twilight + grace, MJD UTC; `None` when the site has no such twilight.
    twilight_close: Option<f64>,
    /// Rollover + grace, MJD UTC.
    rollover_close: f64,
}

impl NightClock {
    /// Compute the deadlines of `night_id` under `params`.
    ///
    /// * twilight close = the morning crossing of
    ///   [`ClockParams::twilight_altitude`] at [`ClockParams::site`] (see
    ///   [`morning_twilight`]) + [`ClockParams::twilight_grace`]; `None`
    ///   when the Sun does not cross that altitude that morning (polar
    ///   sites), which disables trigger 2;
    /// * rollover close = [`rollover_mjd`] + [`ClockParams::rollover_grace`].
    ///
    /// # Arguments
    /// * `night_id` – The open night (MJD of its `dayObs`).
    /// * `params` – Altitude, graces and site.
    pub fn open(night_id: NightId, params: &ClockParams) -> Self {
        Self {
            night_id,
            twilight_close: morning_twilight(night_id, params.twilight_altitude, &params.site)
                .map(|twilight| twilight + params.twilight_grace),
            rollover_close: rollover_mjd(night_id) + params.rollover_grace,
        }
    }

    /// The open night.
    pub fn night_id(&self) -> NightId {
        self.night_id
    }

    /// When trigger 2 may fire (twilight + grace), MJD UTC.
    pub fn twilight_close(&self) -> Option<f64> {
        self.twilight_close
    }

    /// When trigger 3 fires (rollover + grace), MJD UTC.
    pub fn rollover_close(&self) -> f64 {
        self.rollover_close
    }

    /// Place an alert of `alert_night` relative to the open night — the
    /// input of trigger 1: [`AlertNight::Newer`] means the open night must
    /// close before the alert is processed.
    pub fn classify(&self, alert_night: NightId) -> AlertNight {
        match alert_night.cmp(&self.night_id) {
            std::cmp::Ordering::Equal => AlertNight::Current,
            std::cmp::Ordering::Greater => AlertNight::Newer,
            std::cmp::Ordering::Less => AlertNight::Older,
        }
    }

    /// Whether the clock closes the night now (triggers 2 and 3; trigger 1
    /// comes from [`Self::classify`]).
    ///
    /// Rollover is checked first: past it, the night closes whatever the
    /// source or the assembler say (the caller drains pending visits).
    /// Twilight only closes a night that is *finished*: nothing left in the
    /// source, nothing waiting in the assembler. Both comparisons are
    /// strict (`now > deadline`).
    ///
    /// # Arguments
    /// * `now` – Driver clock, MJD UTC.
    /// * `caught_up` – The source has delivered everything available.
    /// * `visits_pending` – Some visit is still being assembled.
    ///
    /// # Returns
    /// The trigger that fires, or `None` to keep the night open.
    pub fn close_trigger(
        &self,
        now: f64,
        caught_up: bool,
        visits_pending: bool,
    ) -> Option<CloseTrigger> {
        if now > self.rollover_close {
            return Some(CloseTrigger::Rollover);
        }
        let past_twilight = self.twilight_close.is_some_and(|close| now > close);
        (past_twilight && caught_up && !visits_pending).then_some(CloseTrigger::MorningTwilight)
    }
}

/// The `dayObs` rollover ending `night_id`: 12:00 UTC on the date after
/// `dayObs`, as an MJD (`night_id + 1.5`). Every alert of the night has an
/// exposure mid-point before it, by Rubin's definition of `dayObs`
/// (UTC−12 h).
pub fn rollover_mjd(night_id: NightId) -> f64 {
    f64::from(night_id.0) + 1.5
}

#[cfg(test)]
mod tests {
    use super::*;

    const NIGHT: NightId = NightId(61_344); // 2026-10-31
    const MINUTE: f64 = 1.0 / 1_440.0;

    fn clock() -> NightClock {
        NightClock::open(NIGHT, &ClockParams::default())
    }

    #[test]
    fn deadlines_follow_twilight_and_rollover() {
        let clock = clock();
        let twilight_close = clock.twilight_close().unwrap();
        // Twilight 08:52 UTC + 30 min grace, on 2026-11-01.
        assert!((twilight_close - (61_345.369_79 + 30.0 * MINUTE)).abs() < MINUTE);
        assert_eq!(clock.rollover_close(), 61_345.5 + 30.0 * MINUTE);
        assert!(twilight_close < clock.rollover_close());
    }

    #[test]
    fn alerts_are_classified_against_the_open_night() {
        let clock = clock();
        assert_eq!(clock.classify(NIGHT), AlertNight::Current);
        assert_eq!(clock.classify(NightId(61_345)), AlertNight::Newer);
        assert_eq!(clock.classify(NightId(61_343)), AlertNight::Older);
    }

    #[test]
    fn twilight_closes_only_when_caught_up_with_nothing_pending() {
        let clock = clock();
        let after = clock.twilight_close().unwrap() + MINUTE;
        assert_eq!(
            clock.close_trigger(after, true, false),
            Some(CloseTrigger::MorningTwilight)
        );
        assert_eq!(clock.close_trigger(after, false, false), None, "backlog");
        assert_eq!(
            clock.close_trigger(after, true, true),
            None,
            "visit pending"
        );
    }

    #[test]
    fn a_weather_pause_never_closes_the_night() {
        let clock = clock();
        // Caught up, nothing pending, hours without alerts — but before twilight.
        let midnight = f64::from(NIGHT.0) + 1.2;
        assert_eq!(clock.close_trigger(midnight, true, false), None);
    }

    #[test]
    fn a_backlog_catch_up_closes_on_event_time_only() {
        let clock = clock();
        // Replaying the night's backlog: event time is mid-night, so nothing
        // closes even though the source is not caught up for hours.
        let mid_night = f64::from(NIGHT.0) + 1.1;
        assert_eq!(clock.close_trigger(mid_night, false, true), None);
        // Once the backlog reaches past twilight and drains, it closes.
        let after = clock.twilight_close().unwrap() + MINUTE;
        assert_eq!(
            clock.close_trigger(after, true, false),
            Some(CloseTrigger::MorningTwilight)
        );
    }

    #[test]
    fn the_rollover_safety_net_fires_whatever_the_state() {
        let clock = clock();
        let after = clock.rollover_close() + MINUTE;
        for (caught_up, pending) in [(false, true), (true, true), (false, false)] {
            assert_eq!(
                clock.close_trigger(after, caught_up, pending),
                Some(CloseTrigger::Rollover)
            );
        }
        assert_eq!(
            clock.close_trigger(clock.rollover_close() - MINUTE, false, true),
            None
        );
    }

    #[test]
    fn clock_params_parse_units_from_yaml() {
        let params: ClockParams = serde_yaml::from_str(
            "twilight_altitude: \"-18 deg\"\ntwilight_grace: \"10 min\"\nsite:\n  lon: \"10 deg\"\n  lat: \"-20 deg\"\n",
        )
        .unwrap();
        assert!((params.twilight_altitude - (-18.0_f64).to_radians()).abs() < 1e-12);
        assert!((params.twilight_grace - 10.0 * MINUTE).abs() < 1e-12);
        assert_eq!(params.rollover_grace, ClockParams::default().rollover_grace);
        assert!((params.site.lat - (-20.0_f64).to_radians()).abs() < 1e-12);
    }
}
