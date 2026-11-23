//! Structured log events of the streaming driver, following the engine's
//! `.emit()` pattern (see [`fink_fat_engine::logging`]): one enum variant
//! per message, all under the `stream` tracing target.
//!
//! | Variant | Level | When |
//! |---------|-------|------|
//! | [`StreamEvent::NightOpened`] | INFO | a night opens |
//! | [`StreamEvent::NightClosed`] | INFO | a night closes, with its summary |
//! | [`StreamEvent::Recovered`] | INFO | start-up resumed an open night from the journal |
//! | [`StreamEvent::JournalDiscarded`] | WARN | start-up found a journal of an already closed night |
//! | [`StreamEvent::VisitAdvanced`] | DEBUG | a visit was folded into the night |
//! | [`StreamEvent::AlertRejected`] | DEBUG | an alert was cut, unconvertible or a duplicate |
//! | [`StreamEvent::LateAlert`] | DEBUG | an alert of a closed night arrived |
//! | [`StreamEvent::SqlFlushed`] | DEBUG | the SQL outbox was inserted into Postgres |
//! | [`StreamEvent::SqlFlushFailed`] | WARN | Postgres refused or could not be reached; rows stay queued |
//! | [`StreamEvent::NightSynced`] | INFO | the trajectory tables were swapped in |
//! | [`StreamEvent::NightSyncFailed`] | WARN | the swap failed; retried at the next close or start-up |
//!
//! Per-alert events are DEBUG: at Rubin rates they would flood the INFO log;
//! their totals are in [`DriverStats`](crate::DriverStats). Enable them
//! with `--log-target stream=debug`.

use std::time::Duration;

use photom::NightId;

use crate::{clock::CloseTrigger, ingest::Rejection, sink::SinkFailure};

/// Driver-level events (see the [module doc](self)).
pub enum StreamEvent<'a> {
    /// A night was opened.
    NightOpened {
        /// The night (MJD of its `dayObs`).
        night_id: NightId,
    },
    /// A visit was folded into the open night.
    VisitAdvanced {
        /// Rubin `visit` identifier.
        visit: i64,
        /// Observations in the visit.
        n_observations: usize,
        /// Lineages that ran the full propagation + gate on this visit.
        n_lineages_gated: usize,
        /// Live branches after the visit's top-B cap.
        n_branches: usize,
        /// `false` when the engine could not resolve the visit's observer
        /// and skipped it.
        observer_resolved: bool,
    },
    /// An alert was not handed to the tracker.
    AlertRejected {
        /// The alert's `diaSourceId`.
        dia_source_id: i64,
        /// Why.
        rejection: &'a Rejection,
    },
    /// An alert of a night already closed: dropped, not tracked.
    LateAlert {
        /// The alert's `diaSourceId`.
        dia_source_id: i64,
        /// The alert's night.
        night_id: NightId,
    },
    /// The open night was closed and persisted.
    NightClosed {
        /// The night (MJD of its `dayObs`).
        night_id: NightId,
        /// What closed it.
        trigger: CloseTrigger,
        /// Visits handed to the engine.
        n_visits: usize,
        /// Of those, visits skipped for an unresolved observer.
        n_unresolved_visits: usize,
        /// Observations of the night (visits + leftovers).
        n_observations: usize,
        /// Observations that arrived after their visit was advanced.
        n_leftovers: usize,
        /// Live branches carried to the next night.
        n_branches: usize,
        /// Trajectories archived at this close.
        n_archived: usize,
    },
    /// The open night was rebuilt from the journal at start-up.
    Recovered {
        /// The resumed night.
        night_id: NightId,
        /// Journal entries replayed.
        n_entries: usize,
    },
    /// The journal described a night already closed (crash between the
    /// snapshot and the journal truncation): it was discarded.
    JournalDiscarded {
        /// The journal's night.
        night_id: NightId,
    },
    /// The SQL outbox was inserted into Postgres.
    SqlFlushed {
        /// Rows inserted into `observations`.
        n_observations: u64,
        /// Rows inserted into `archived_trajectories`.
        n_archived: u64,
    },
    /// A flush of the SQL outbox failed; the rows stay queued.
    SqlFlushFailed {
        /// Why.
        failure: &'a SinkFailure,
        /// No new attempt before this long.
        retry_in: Duration,
    },
    /// The trajectory tables now hold the collection of this night.
    NightSynced {
        /// Last closed night of the published collection.
        night_id: Option<NightId>,
        /// Rows loaded into `branches`.
        n_branches: usize,
        /// Staging build + swap.
        duration: Duration,
    },
    /// Publishing the trajectory tables failed; retried at the next close
    /// or start-up.
    NightSyncFailed {
        /// Last closed night of the collection that was not published.
        night_id: Option<NightId>,
        /// Why.
        failure: &'a SinkFailure,
    },
}

fink_fat_engine::impl_log_target!(
    StreamEvent<'_>,
    "stream",
    "Streaming driver: night open/close, recovery, alerts not tracked",
    [
        tracing::Level::WARN,
        tracing::Level::INFO,
        tracing::Level::DEBUG
    ]
);

impl StreamEvent<'_> {
    /// Log the event at its level (see the [module doc](self)).
    pub fn emit(&self) {
        use fink_fat_engine::logging::LogTarget;
        const TARGET: &str = <StreamEvent<'static> as LogTarget>::TARGET;
        match self {
            Self::NightOpened { night_id } => {
                tracing::info!(target: TARGET, %night_id, "night opened");
            }
            Self::VisitAdvanced {
                visit,
                n_observations,
                n_lineages_gated,
                n_branches,
                observer_resolved,
            } => tracing::debug!(
                target: TARGET,
                visit,
                n_observations,
                n_lineages_gated,
                n_branches,
                observer_resolved,
                "visit advanced"
            ),
            Self::AlertRejected {
                dia_source_id,
                rejection,
            } => tracing::debug!(target: TARGET, dia_source_id, ?rejection, "alert rejected"),
            Self::LateAlert {
                dia_source_id,
                night_id,
            } => tracing::debug!(
                target: TARGET,
                dia_source_id,
                %night_id,
                "late alert of a closed night, not tracked"
            ),
            Self::NightClosed {
                night_id,
                trigger,
                n_visits,
                n_unresolved_visits,
                n_observations,
                n_leftovers,
                n_branches,
                n_archived,
            } => tracing::info!(
                target: TARGET,
                %night_id,
                ?trigger,
                n_visits,
                n_unresolved_visits,
                n_observations,
                n_leftovers,
                n_branches,
                n_archived,
                "night closed"
            ),
            Self::Recovered {
                night_id,
                n_entries,
            } => tracing::info!(
                target: TARGET,
                %night_id,
                n_entries,
                "open night rebuilt from the journal"
            ),
            Self::JournalDiscarded { night_id } => tracing::warn!(
                target: TARGET,
                %night_id,
                "journal of an already closed night discarded"
            ),
            Self::SqlFlushed {
                n_observations,
                n_archived,
            } => tracing::debug!(
                target: TARGET,
                n_observations,
                n_archived,
                "SQL outbox flushed"
            ),
            Self::SqlFlushFailed { failure, retry_in } => tracing::warn!(
                target: TARGET,
                %failure,
                retry_in_s = retry_in.as_secs(),
                "SQL outbox flush failed, rows kept for a later attempt"
            ),
            Self::NightSynced {
                night_id,
                n_branches,
                duration,
            } => tracing::info!(
                target: TARGET,
                ?night_id,
                n_branches,
                duration_s = duration.as_secs_f64(),
                "trajectory tables swapped in"
            ),
            Self::NightSyncFailed { night_id, failure } => tracing::warn!(
                target: TARGET,
                ?night_id,
                %failure,
                "trajectory sync failed, retried at the next close or start-up"
            ),
        }
    }
}
