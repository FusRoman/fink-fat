//! Heliocentric 3D orbit visualization: the homepage's population-wide
//! position scatter and the lineage page's single-object orbit plot.
//!
//! [`geometry`], [`plot3d`] and [`types`] compile on every target (native
//! and `wasm32-unknown-unknown`) — they hold no I/O, or (for [`plot3d`])
//! only the wasm-side chart code, cfg-gated internally exactly like
//! [`crate::homepage::dynamic_pop_plot`].
//!
//! [`server_fns`] holds the `#[server]` functions the wasm client calls: like
//! every other `#[server]` function in this crate (see
//! `homepage::dynamic_pop_plot::query_orbital_elements`), its *declaration*
//! must exist on both targets — the macro compiles the real body only under
//! the `server` feature and a client-calling stub otherwise — so this module
//! is *not* itself feature-gated.
//!
//! [`ephem_provider`] holds no `#[server]` functions, only server-only
//! implementation the bodies in [`server_fns`] call into (it queries the
//! JPL/ANISE ephemeris already loaded by [`crate::get_kalman_context`]), so
//! it is gated behind the `server` feature exactly like
//! [`crate::homepage::snapshot`].

pub mod geometry;
pub mod plot3d;
pub mod server_fns;
pub mod types;

#[cfg(feature = "server")]
pub mod ephem_provider;
