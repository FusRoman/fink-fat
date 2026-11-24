//! Resolved software versions shown in the footer's "Software" column.
//!
//! `fink-fat-explorer` links against `fink-fat-engine` (which produces the
//! tracked trajectories), `outfit` (orbit fitting) and `photom` (photometric
//! data handling) — but only as `server`-feature-gated, optional
//! dependencies, and neither they nor `fink-fat-explorer` itself expose
//! their version as an importable Rust constant. Hardcoding version strings
//! here would silently go stale on the next `cargo update`; instead,
//! `build.rs` uses the `built` crate to read every dependency's resolved
//! version straight out of `Cargo.lock` at compile time and writes it to a
//! generated file this module `include!`s — so the numbers shown are always
//! exactly what was actually compiled, for both the server and wasm32
//! builds (both read the same, single workspace `Cargo.lock`, so there is
//! no risk of the two disagreeing after hydration).

mod built_info {
    include!(concat!(env!("OUT_DIR"), "/built.rs"));
}

/// One piece of software credited in the footer: its name, the version
/// actually compiled in, and where to read more about it.
pub struct SoftwareVersion {
    pub name: &'static str,
    pub version: &'static str,
    /// crates.io/docs.rs/GitHub page for this software.
    pub url: &'static str,
}

/// Looks up `name`'s resolved version in [`built_info::DEPENDENCIES`] (the
/// full, deduplicated package list `built` read from `Cargo.lock`).
///
/// # Return
/// The resolved version string, or `"unknown"` if `name` isn't found — a
/// missing entry (e.g. after a dependency rename) should degrade the
/// footer's display, not break every page.
fn dependency_version(name: &str) -> &'static str {
    built_info::DEPENDENCIES
        .iter()
        .find(|(dep_name, _)| *dep_name == name)
        .map(|(_, version)| *version)
        .unwrap_or("unknown")
}

/// This crate's own version, plus the 3 dependencies credited in the
/// footer, in display order.
///
/// # Return
/// One [`SoftwareVersion`] per credited crate: `fink-fat-explorer` itself
/// (from [`built_info::PKG_VERSION`], this crate's own `Cargo.toml`), then
/// `fink-fat-engine`, `outfit` and `photom` (resolved via
/// [`dependency_version`]).
pub fn software_versions() -> Vec<SoftwareVersion> {
    vec![
        SoftwareVersion {
            name: "fink-fat-explorer",
            version: built_info::PKG_VERSION,
            url: "https://github.com/FusRoman/fink-fat/tree/main/crates/fink-fat-explorer",
        },
        SoftwareVersion {
            name: "fink-fat-engine",
            version: dependency_version("fink-fat-engine"),
            url: "https://github.com/FusRoman/fink-fat/tree/main/crates/fink-fat-engine",
        },
        SoftwareVersion {
            name: "outfit",
            version: dependency_version("outfit"),
            url: "https://crates.io/crates/outfit",
        },
        SoftwareVersion {
            name: "photom",
            version: dependency_version("photom"),
            url: "https://crates.io/crates/photom",
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every credited crate must resolve to a real version — an
    /// `"unknown"` here means `Cargo.lock`'s package name no longer matches
    /// what this module looks for (e.g. after a rename), which should fail
    /// loudly in CI rather than silently ship a broken footer.
    #[test]
    fn every_credited_crate_resolves_to_a_known_version() {
        for sw in software_versions() {
            assert_ne!(
                sw.version, "unknown",
                "{} did not resolve a version",
                sw.name
            );
        }
    }
}
