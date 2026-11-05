//! MJD(TT) epoch formatting. Moved to the shared [`fink_fat_ades::format_epoch`]
//! crate (needed there for ADES `obsTime` construction) and re-exported here
//! under this module's original path, so every existing
//! `crate::format_epoch::*` call site needed no changes.

pub use fink_fat_ades::format_epoch::{format_epoch, iso_utc, today_utc_date};
