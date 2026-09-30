//! Public REST API of the explorer.
//!
//! Unlike the internal `#[server]` functions (RPC endpoints whose URLs and
//! encoding are private to the Dioxus front end), the routes declared here are
//! stable, versioned (`/api/v1/...`), return plain JSON and use meaningful
//! HTTP status codes, so they can be consumed by external scripts and tools.
//!
//! Routes are declared with Dioxus' `#[get]` macro and registered
//! automatically on the same server and port as the web UI.

#[cfg(feature = "server")]
mod error;
pub mod reverse_search;
pub mod types;

pub use reverse_search::{reverse_search_alert, REVERSE_SEARCH_PATH};
pub use types::{LineageMatch, ReverseSearchResponse};
