//! LSST photometric band convention. Moved to the shared
//! [`fink_fat_ades::lsst_band`] crate (needed there for ADES `band`
//! construction) and re-exported here under this module's original path, so
//! every existing `crate::lsst_band::*` call site needed no changes.

pub use fink_fat_ades::lsst_band::band_index_to_letter;
