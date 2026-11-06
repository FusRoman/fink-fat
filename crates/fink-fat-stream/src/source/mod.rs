//! Where alerts come from: the [`AlertSource`] trait and its replay
//! implementation ([`ReplaySource`]). The Kafka consumer of phase 6 will be
//! a second implementation.

pub mod replay;

pub use replay::{ColumnMap, ReplayConfig, ReplaySource};

use crate::{error::Result, record::AlertRecord};

/// A stream of alerts delivered in batches, *at least once*.
///
/// # Protocol
///
/// The driver loops on:
///
/// 1. [`Self::next_batch`];
/// 2. process the batch and write it durably to its journal;
/// 3. [`Self::ack`].
///
/// # Contract for implementers
///
/// * **At least once**: a batch that was not acknowledged may be delivered
///   again after a restart — the driver deduplicates by `diaSourceId` and
///   drops alerts of closed nights. Never *lose* an unacknowledged alert.
/// * **Order**: alerts should come roughly in arrival order. Out-of-order
///   alerts are tolerated (the clock takes the maximum arrival seen), but an
///   alert arriving after its visit was declared complete only feeds the
///   night's discovery, not the visit's association.
/// * **`arrival_mjd`** of each record must be on UTC (see
///   [`crate::record`]): it drives visit completion and night closing.
/// * **`caught_up`** must only be `true` when no backlog is pending:
///   it allows the night to close at twilight.
pub trait AlertSource {
    /// The next batch of alerts, `Ok(None)` once the stream is over (a
    /// replay reached its end; a live source never ends). An empty batch
    /// means "nothing right now" and lets the driver advance its clock.
    ///
    /// # Errors
    /// The source failed; the driver stops.
    fn next_batch(&mut self) -> Result<Option<Vec<AlertRecord>>>;

    /// Acknowledge every batch returned so far: it is durably journaled
    /// and must not be redelivered.
    ///
    /// # Errors
    /// The acknowledgement failed; the driver stops.
    fn ack(&mut self) -> Result<()>;

    /// Whether the source has delivered everything available right now
    /// (no backlog). Required by the morning-twilight close (see
    /// [`crate::clock`]).
    fn caught_up(&self) -> bool;

    /// Wall-clock time, MJD UTC, for a live source; `None` for a source
    /// whose only clock is its alerts' event time (a replay).
    ///
    /// Only consulted while [`Self::caught_up`]: it lets the clock reach
    /// twilight after the night's last alert, when no event time advances
    /// it any more.
    fn wall_clock_mjd(&self) -> Option<f64> {
        None
    }
}
