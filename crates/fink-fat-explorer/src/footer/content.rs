//! Static text and links shown in the footer — plain data, no Dioxus, so it
//! stays trivially unit-testable independently of anything rendering it.

/// A logo bundled with the app and shown next to an [`Acknowledgment`].
///
/// An enum rather than a raw asset path: `mod.rs`'s `asset!` macro call
/// needs a string literal at its own call site to bundle the file, so it
/// can't be driven by a path read out of this const data at runtime —
/// `mod.rs` matches on this instead, and the compiler catches a variant
/// added here without a matching arm there.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Logo {
    Rust,
    Ferris,
    Dioxus,
}

/// One organization/service credited in the footer's "Acknowledgments"
/// column.
pub struct Acknowledgment {
    pub name: &'static str,
    /// One short sentence: what this app actually uses it for.
    pub blurb: &'static str,
    pub url: &'static str,
    /// Logos shown next to this entry — empty when none is credited with
    /// one; more than one for an entry crediting more than one mark (Rust
    /// gets both the gear and Ferris).
    pub logos: &'static [Logo],
}

/// Every organization/service this app's data pipeline actually depends on.
pub const ACKNOWLEDGMENTS: &[Acknowledgment] = &[
    Acknowledgment {
        name: "Vera C. Rubin Observatory",
        url: "https://rubinobservatory.org/",
        blurb: "Source of the LSST alert data this app tracks.",
        logos: &[],
    },
    Acknowledgment {
        name: "Fink broker",
        url: "https://fink-broker.org/",
        blurb: "Distributes the real-time alert stream and enriches it with the cross-match \
                and classification data this app builds on.",
        logos: &[],
    },
    Acknowledgment {
        name: "Minor Planet Center",
        url: "https://minorplanetcenter.net/",
        blurb: "Maintains the astrometric catalog this app cross-matches against and submits \
                discoveries to.",
        logos: &[],
    },
    Acknowledgment {
        name: "IMCCE",
        url: "https://ssp.imcce.fr/webservices/skybot/",
        blurb: "Provides the SkyBoT service this app's cross-match and dynamical-family \
                classification are built on.",
        logos: &[],
    },
    Acknowledgment {
        name: "Rust",
        url: "https://www.rust-lang.org/",
        blurb: "The language this entire pipeline, engine and web app are written in.",
        logos: &[Logo::Rust, Logo::Ferris],
    },
    Acknowledgment {
        name: "Dioxus",
        url: "https://dioxuslabs.com/",
        blurb: "The Rust UI framework this web app (fink-fat-explorer) is built with.",
        logos: &[Logo::Dioxus],
    },
];

/// One external resource listed in the footer's "Useful links" column.
pub struct FooterLink {
    pub label: &'static str,
    pub url: &'static str,
}

/// External resources relevant to reading and cross-checking this app's
/// asteroid data, beyond the acknowledgments above.
pub const USEFUL_LINKS: &[FooterLink] = &[
    FooterLink {
        label: "Fink broker docs (ZTF)",
        url: "https://doc.ztf.fink-broker.org/",
    },
    FooterLink {
        label: "Fink broker docs (LSST)",
        url: "https://doc.lsst.fink-broker.org/",
    },
    FooterLink {
        label: "IMCCE SkyBoT",
        url: "https://ssp.imcce.fr/webservices/skybot/",
    },
    FooterLink {
        label: "Minor Planet Center",
        url: "https://minorplanetcenter.net/",
    },
];

/// This project's GitHub repository.
pub const GITHUB_URL: &str = "https://github.com/FusRoman/fink-fat";

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn no_acknowledgment_field_is_empty() {
        for ack in ACKNOWLEDGMENTS {
            assert!(!ack.name.trim().is_empty());
            assert!(!ack.blurb.trim().is_empty());
            assert!(!ack.url.trim().is_empty());
        }
    }

    #[test]
    fn no_useful_link_field_is_empty() {
        for link in USEFUL_LINKS {
            assert!(!link.label.trim().is_empty());
            assert!(!link.url.trim().is_empty());
        }
    }

    /// Every URL in the footer (acknowledgments, useful links, and the
    /// GitHub link) must be a well-formed absolute HTTPS URL — a typo here
    /// would otherwise only surface as a dead link in production.
    #[test]
    fn every_url_is_a_well_formed_https_url() {
        let all_urls = ACKNOWLEDGMENTS
            .iter()
            .map(|a| a.url)
            .chain(USEFUL_LINKS.iter().map(|l| l.url))
            .chain(std::iter::once(GITHUB_URL));
        for url in all_urls {
            assert!(url.starts_with("https://"), "{url}");
        }
    }

    #[test]
    fn no_duplicate_acknowledgment_urls() {
        let urls: HashSet<&str> = ACKNOWLEDGMENTS.iter().map(|a| a.url).collect();
        assert_eq!(
            urls.len(),
            ACKNOWLEDGMENTS.len(),
            "duplicate acknowledgment URL"
        );
    }

    #[test]
    fn no_duplicate_useful_link_urls() {
        let urls: HashSet<&str> = USEFUL_LINKS.iter().map(|l| l.url).collect();
        assert_eq!(urls.len(), USEFUL_LINKS.len(), "duplicate useful-link URL");
    }
}
