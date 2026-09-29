//! Captures build-time metadata — this crate's own version plus every
//! dependency's resolved version from `Cargo.lock` (the `cargo-lock`
//! feature of `built`) — into a generated `built.rs`, `include!`d by
//! `src/footer/versions.rs` to populate the site footer's "Software"
//! column. See that module's doc comment for why this is a build-time
//! lookup rather than hardcoded strings.

fn main() {
    built::write_built_file().expect("failed to collect build-time version info");
}
