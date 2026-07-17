//! Structured, per-module tracing.
//!
//! Business code never calls `tracing::trace!`/`debug!`/`info!` directly.
//! Instead, each pipeline module defines its own log-event enum — one variant
//! per distinct message — implementing [`LogTarget`] (via [`impl_log_target`](crate::impl_log_target))
//! for the module's tracing `target` name, description and levels, and an
//! inherent `emit(&self)` method that performs the actual `tracing::` call.
//!
//! This keeps the `target`/description/level metadata attached to the type
//! that defines the event (no risk of a stray, undocumented `target: "..."`
//! string), and lets [`registry::all_targets`] build the `--list-log-targets`
//! listing and the `EnvFilter` (`tracing_subscriber::EnvFilter`) directive from
//! a single, statically-checked source.
//!
//! # Example
//!
//! ```ignore
//! pub enum ExampleEvent {
//!     Started { n: usize },
//! }
//!
//! fink_fat_engine::impl_log_target!(
//!     ExampleEvent, "example", "Example stage", [tracing::Level::INFO]
//! );
//!
//! impl ExampleEvent {
//!     pub fn emit(&self) {
//!         match self {
//!             Self::Started { n } => tracing::info!(target: Self::TARGET, n, "started"),
//!         }
//!     }
//! }
//! ```

pub mod registry;

/// Implemented by each module's log-event enum. Carries the module's tracing
/// `target` name, a human-readable description (surfaced by
/// `--list-log-targets`), and the set of levels it can emit at.
pub trait LogTarget {
    const TARGET: &'static str;
    const DESCRIPTION: &'static str;
    const LEVELS: &'static [tracing::Level];
}

/// Boilerplate for `impl LogTarget for SomeEvent { ... }` — see the
/// [module docs](self) for usage.
#[macro_export]
macro_rules! impl_log_target {
    ($ty:ty, $target:literal, $description:literal, [$($level:expr),+ $(,)?]) => {
        impl $crate::logging::LogTarget for $ty {
            const TARGET: &'static str = $target;
            const DESCRIPTION: &'static str = $description;
            const LEVELS: &'static [tracing::Level] = &[$($level),+];
        }
    };
}
