//! [`Backoff`]: when to try Postgres again after a failure.
//!
//! Pure: the current instant is always passed in, never read from a clock,
//! so the policy is unit-testable without waiting.

use std::time::{Duration, Instant};

/// Exponential backoff between retries of a failing operation: after each
/// failure the wait doubles, from `min` up to `max`; a success resets it.
///
/// ```text
/// failure → wait min → failure → wait 2·min → … → wait max → … → success → no wait
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Backoff {
    /// First wait after a success.
    min: Duration,
    /// Cap on the wait.
    max: Duration,
    /// Wait to apply after the next failure.
    next_wait: Duration,
    /// No attempt before this instant; `None` after a success.
    not_before: Option<Instant>,
}

impl Backoff {
    /// A backoff ready to try right away.
    ///
    /// # Arguments
    /// * `min` – Wait after the first failure.
    /// * `max` – Cap on the wait (raised to `min` if smaller).
    pub fn new(min: Duration, max: Duration) -> Self {
        Self {
            min,
            max: max.max(min),
            next_wait: min,
            not_before: None,
        }
    }

    /// Whether an attempt is allowed at `now`.
    pub fn ready(&self, now: Instant) -> bool {
        self.not_before.is_none_or(|not_before| now >= not_before)
    }

    /// Record a success: the next attempt may happen at once, and the next
    /// failure waits `min` again.
    pub fn succeeded(&mut self) {
        self.next_wait = self.min;
        self.not_before = None;
    }

    /// Record a failure at `now`: no attempt before `now + wait`, and the
    /// following failure waits twice as long (up to `max`).
    ///
    /// # Returns
    /// The wait imposed by this failure.
    pub fn failed(&mut self, now: Instant) -> Duration {
        let wait = self.next_wait;
        self.not_before = Some(now + wait);
        self.next_wait = (wait * 2).min(self.max);
        wait
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIN: Duration = Duration::from_secs(10);
    const MAX: Duration = Duration::from_secs(60);

    #[test]
    fn a_fresh_backoff_is_ready() {
        assert!(Backoff::new(MIN, MAX).ready(Instant::now()));
    }

    #[test]
    fn waits_double_up_to_the_cap() {
        let mut backoff = Backoff::new(MIN, MAX);
        let now = Instant::now();
        let waits: Vec<u64> = (0..5).map(|_| backoff.failed(now).as_secs()).collect();
        assert_eq!(waits, vec![10, 20, 40, 60, 60]);
    }

    #[test]
    fn no_attempt_before_the_wait_is_over() {
        let mut backoff = Backoff::new(MIN, MAX);
        let now = Instant::now();
        backoff.failed(now);
        assert!(!backoff.ready(now + Duration::from_secs(9)));
        assert!(backoff.ready(now + MIN));
    }

    #[test]
    fn a_success_resets_the_wait() {
        let mut backoff = Backoff::new(MIN, MAX);
        let now = Instant::now();
        backoff.failed(now);
        backoff.failed(now);
        backoff.succeeded();
        assert!(backoff.ready(now));
        assert_eq!(backoff.failed(now), MIN);
    }

    #[test]
    fn a_cap_below_the_minimum_is_raised_to_it() {
        let mut backoff = Backoff::new(MIN, Duration::from_secs(1));
        let now = Instant::now();
        assert_eq!(backoff.failed(now), MIN);
        assert_eq!(backoff.failed(now), MIN);
    }
}
