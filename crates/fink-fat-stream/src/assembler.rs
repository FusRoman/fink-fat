//! [`VisitAssembler`]: regroups a night's alerts into complete visits.
//!
//! # Why
//!
//! The tracker folds a night in **one visit at a time**, and each visit must
//! be complete: clutter density, top-K candidate selection and the single
//! null branch per lineage are per-visit quantities (see
//! [`NightSession::advance_visit`](fink_fat_engine::topocentric_kf::branching::NightSession::advance_visit)).
//! Alerts, however, arrive one by one: a visit's alerts spread over a few
//! tens of seconds, possibly interleaved with the next visit's, and nothing
//! in the stream says "this visit is over".
//!
//! # How
//!
//! The assembler buffers alerts per `visit` and declares a visit **complete
//! once it received no alert for `idle_timeout`** (90 s by default). A visit
//! goes through three states:
//!
//! ```text
//!   unknown ──push──► open ──poll (idle) / take / drain──► emitted
//!                      ▲ │                                    │
//!                      └─┘ push (buffered,                    └─ push → Leftover
//!                           refreshes last_arrival)
//! ```
//!
//! Visits are emitted in `visit` order — `dayObs × 100000 + seqNum`, i.e.
//! exposure order within a night — and **never ahead of a lower visit still
//! open**, so the tracker always sees the night's visits chronologically
//! even if a later exposure's alerts go quiet first.
//!
//! Example (timeout 90 s; times in seconds):
//!
//! ```text
//!   t=0   push(v1)      t=30  push(v2)      t=40  push(v1)
//!   t=100 poll → []          v1 idle 60 s only, so v2 (idle 70 s) waits too
//!   t=130 poll → [v1, v2]    v1 idle 90 s, then v2 idle 100 s
//!   t=140 push(v1) → Leftover
//! ```
//!
//! # Late alerts and the timeout trade-off
//!
//! An alert of a visit already emitted is handed back as a
//! [`Placement::Leftover`]: too late to take part in that visit's
//! association, it still reaches the night's discovery step (see
//! [`LiveNight::add_leftovers`](crate::night::LiveNight::add_leftovers)).
//! A longer timeout makes leftovers rarer but delays every visit — and
//! delays the next one in line, since visits never overtake.
//!
//! # Purity
//!
//! Time is injected (`arrival`, `now`, MJD — any scale, as long as both are
//! on the same), never read from a clock, and the buffered item type `T` is
//! generic: the assembler knows nothing of alerts beyond their `visit` and
//! arrival time.

use std::collections::{BTreeMap, HashSet};

/// A visit the assembler declared complete.
#[derive(Debug, Clone, PartialEq)]
pub struct EmittedVisit<T> {
    /// Rubin `visit` identifier.
    pub visit: i64,
    /// The visit's items, in the order they were pushed (empty only for a
    /// [`VisitAssembler::take`] of a visit with nothing buffered).
    pub items: Vec<T>,
}

/// Where [`VisitAssembler::push`] put an item.
#[derive(Debug, Clone, PartialEq)]
pub enum Placement<T> {
    /// Buffered with its (still open) visit.
    Buffered,
    /// Its visit was already emitted tonight: the item is handed back to
    /// the caller, unbuffered.
    Leftover(T),
}

/// An open visit: its items so far, and when the latest one arrived.
#[derive(Debug, Clone)]
struct OpenVisit<T> {
    /// Latest arrival time among `items` (the idle clock starts here).
    last_arrival: f64,
    /// Items pushed so far, in push order.
    items: Vec<T>,
}

/// Per-visit buffer of **one night's** alerts (see the [module doc](self)).
///
/// One assembler per night: the set of emitted visits is never cleared, so a
/// new night needs a new assembler (the driver builds one at each night
/// opening).
#[derive(Debug, Clone)]
pub struct VisitAssembler<T> {
    /// A visit idle for at least this long (days) is complete.
    idle_timeout: f64,
    /// Open visits, keyed and therefore ordered by `visit`.
    open: BTreeMap<i64, OpenVisit<T>>,
    /// Visits already emitted tonight: their later alerts are leftovers.
    emitted: HashSet<i64>,
}

impl<T> VisitAssembler<T> {
    /// Empty assembler.
    ///
    /// # Arguments
    /// * `idle_timeout` – Days without a new alert after which a visit is
    ///   complete (see the [module doc](self#late-alerts-and-the-timeout-trade-off)).
    pub fn new(idle_timeout: f64) -> Self {
        Self {
            idle_timeout,
            open: BTreeMap::new(),
            emitted: HashSet::new(),
        }
    }

    /// Add `item`, an alert of `visit` that arrived at `arrival`.
    ///
    /// * Visit already emitted → [`Placement::Leftover`], the item is handed
    ///   back.
    /// * Otherwise the item joins its visit (opened if new) and the visit's
    ///   idle clock restarts at `max(last_arrival, arrival)` — an alert
    ///   delivered out of order never makes a visit look older than it is.
    ///
    /// # Arguments
    /// * `visit` – Rubin `visit` of the alert.
    /// * `arrival` – When the alert arrived (MJD, same scale as `now` in
    ///   [`Self::poll`]).
    /// * `item` – What to buffer.
    pub fn push(&mut self, visit: i64, arrival: f64, item: T) -> Placement<T> {
        if self.emitted.contains(&visit) {
            return Placement::Leftover(item);
        }
        let open = self.open.entry(visit).or_insert_with(|| OpenVisit {
            last_arrival: arrival,
            items: Vec::new(),
        });
        open.last_arrival = open.last_arrival.max(arrival);
        open.items.push(item);
        Placement::Buffered
    }

    /// Emit the visits that are complete at `now`.
    ///
    /// Walks the open visits from the lowest `visit` up and emits each one
    /// idle for at least the timeout (`now − last_arrival ≥ idle_timeout`),
    /// **stopping at the first that is not**: a visit never overtakes a
    /// lower one still receiving alerts, even if it is itself idle.
    ///
    /// # Arguments
    /// * `now` – Current time (MJD, same scale as the `arrival`s).
    ///
    /// # Returns
    /// The emitted visits, in `visit` order (possibly none). They are
    /// marked emitted: their later alerts become leftovers.
    pub fn poll(&mut self, now: f64) -> Vec<EmittedVisit<T>> {
        let mut out = Vec::new();
        while let Some(entry) = self.open.first_entry() {
            if now - entry.get().last_arrival < self.idle_timeout {
                break;
            }
            let (visit, open) = entry.remove_entry();
            out.push(self.mark_emitted(visit, open.items));
        }
        out
    }

    /// Emit `visit` now, whether idle or not, and whatever lower visits are
    /// still open.
    ///
    /// Used by the night journal's replay, which must re-emit each visit at
    /// the exact point of the alert sequence where the interrupted run
    /// emitted it, regardless of what the clock says after the restart (see
    /// [`crate::journal`]).
    ///
    /// # Returns
    /// The visit with its buffered items — none if nothing was buffered for
    /// it, in which case it is still marked emitted.
    pub fn take(&mut self, visit: i64) -> EmittedVisit<T> {
        let items = self.open.remove(&visit).map_or_else(Vec::new, |o| o.items);
        self.mark_emitted(visit, items)
    }

    /// Emit every open visit, idle or not, in `visit` order — the night is
    /// closing and no more alerts will join them.
    ///
    /// # Returns
    /// Every visit that was open; the assembler is empty afterwards.
    pub fn drain(&mut self) -> Vec<EmittedVisit<T>> {
        std::mem::take(&mut self.open)
            .into_iter()
            .map(|(visit, open)| self.mark_emitted(visit, open.items))
            .collect()
    }

    /// Number of visits still open (buffered, not emitted).
    pub fn n_open_visits(&self) -> usize {
        self.open.len()
    }

    /// Whether no visit is open — nothing is waiting to be advanced.
    pub fn is_empty(&self) -> bool {
        self.open.is_empty()
    }

    /// Record `visit` as emitted and package it.
    fn mark_emitted(&mut self, visit: i64, items: Vec<T>) -> EmittedVisit<T> {
        self.emitted.insert(visit);
        EmittedVisit { visit, items }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECOND: f64 = 1.0 / 86_400.0;
    const TIMEOUT: f64 = 90.0 * SECOND;

    fn visits(emitted: &[EmittedVisit<u32>]) -> Vec<i64> {
        emitted.iter().map(|v| v.visit).collect()
    }

    #[test]
    fn a_visit_is_emitted_once_idle_for_the_timeout() {
        let mut asm = VisitAssembler::new(TIMEOUT);
        asm.push(1, 0.0, 10);
        asm.push(1, 30.0 * SECOND, 11);

        assert!(asm.poll(100.0 * SECOND).is_empty(), "idle for 70 s only");
        let out = asm.poll(121.0 * SECOND);
        assert_eq!(
            out,
            vec![EmittedVisit {
                visit: 1,
                items: vec![10, 11]
            }]
        );
        assert!(asm.is_empty());
    }

    #[test]
    fn visits_are_emitted_in_seq_num_order() {
        let mut asm = VisitAssembler::new(TIMEOUT);
        asm.push(3, 0.0, 30);
        asm.push(2, 0.0, 20);
        asm.push(1, 0.0, 10);
        assert_eq!(visits(&asm.poll(TIMEOUT)), vec![1, 2, 3]);
    }

    #[test]
    fn an_idle_visit_waits_for_a_lower_open_one() {
        let mut asm = VisitAssembler::new(TIMEOUT);
        asm.push(1, 60.0 * SECOND, 10); // still receiving
        asm.push(2, 0.0, 20); // idle first

        assert!(asm.poll(100.0 * SECOND).is_empty());
        assert_eq!(asm.n_open_visits(), 2);
        assert_eq!(visits(&asm.poll(151.0 * SECOND)), vec![1, 2]);
    }

    #[test]
    fn an_alert_of_an_emitted_visit_is_a_leftover() {
        let mut asm = VisitAssembler::new(TIMEOUT);
        asm.push(1, 0.0, 10);
        asm.poll(TIMEOUT);
        assert_eq!(asm.push(1, 200.0 * SECOND, 11), Placement::Leftover(11));
        assert!(asm.is_empty());
    }

    #[test]
    fn drain_emits_everything_in_order() {
        let mut asm = VisitAssembler::new(TIMEOUT);
        asm.push(5, 0.0, 50);
        asm.push(4, 0.0, 40);
        let out = asm.drain();
        assert_eq!(visits(&out), vec![4, 5]);
        assert!(asm.is_empty());
        assert_eq!(asm.push(4, 0.0, 41), Placement::Leftover(41));
    }

    #[test]
    fn take_forces_one_visit_out() {
        let mut asm = VisitAssembler::new(TIMEOUT);
        asm.push(1, 0.0, 10);
        asm.push(2, 0.0, 20);
        assert_eq!(asm.take(2).items, vec![20]);
        assert_eq!(asm.take(7).items, Vec::<u32>::new());
        assert_eq!(asm.push(7, 0.0, 70), Placement::Leftover(70));
        assert_eq!(visits(&asm.drain()), vec![1]);
    }

    #[test]
    fn a_late_alert_extends_the_idle_window() {
        let mut asm = VisitAssembler::new(TIMEOUT);
        asm.push(1, 0.0, 10);
        asm.push(1, 80.0 * SECOND, 11);
        assert!(asm.poll(TIMEOUT).is_empty());
        assert_eq!(asm.poll(171.0 * SECOND)[0].items, vec![10, 11]);
    }
}
