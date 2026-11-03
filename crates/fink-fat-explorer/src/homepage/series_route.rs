//! Plain HTTP GET route serving the (a, e) plot's series payload.
//!
//! Server functions are `POST` requests, which browsers never cache: the
//! former `query_orbital_elements` re-downloaded the whole population on every
//! visit. This route serves the snapshot's pre-encoded
//! [`Snapshot::series_payload`](crate::homepage::snapshot::Snapshot::series_payload)
//! with a content-derived `ETag` and `Cache-Control: no-cache`, so the browser
//! keeps the payload and only revalidates it: an unchanged snapshot answers
//! `304 Not Modified` with an empty body.
//!
//! Server-only, like [`crate::homepage::snapshot`]. The client side lives in
//! `homepage::dynamic_pop_plot::use_population_series`.

use std::hash::{DefaultHasher, Hash, Hasher};

use dioxus::server::axum::{
    http::{
        header::{CACHE_CONTROL, CONTENT_TYPE, ETAG, IF_NONE_MATCH},
        HeaderMap, HeaderValue, StatusCode,
    },
    response::{IntoResponse, Response},
    routing::get,
    Router,
};

use crate::homepage::series_wire::POPULATION_SERIES_PATH;

/// Adds the population series route to the app's router.
///
/// # Arguments
/// * `router` — the Dioxus fullstack router.
///
/// # Return
/// `router` with `GET` [`POPULATION_SERIES_PATH`] registered.
pub fn register(router: Router) -> Router {
    router.route(POPULATION_SERIES_PATH, get(population_series))
}

/// Serves the current snapshot's series payload.
///
/// # Return
/// * `503 Service Unavailable` while the first snapshot is still building —
///   the client polls until it is ready;
/// * `304 Not Modified` if `If-None-Match` already names the current payload;
/// * `200 OK` with the binary payload otherwise.
async fn population_series(headers: HeaderMap) -> Response {
    let Some(snap) = crate::homepage::snapshot::snapshot() else {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    };

    // `no-cache` means "store, but revalidate before every use": the payload
    // changes whenever a rebuild lands, under the same URL.
    let mut cache_headers = HeaderMap::new();
    cache_headers.insert(CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    // `etag_for` only ever yields ASCII, so this always succeeds; without a
    // tag the payload is simply re-downloaded on every visit.
    if let Ok(etag) = HeaderValue::from_str(&snap.series_etag) {
        cache_headers.insert(ETAG, etag);
    }

    let revalidated = headers
        .get(IF_NONE_MATCH)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| if_none_match_hits(value, &snap.series_etag));
    if revalidated {
        return (StatusCode::NOT_MODIFIED, cache_headers).into_response();
    }

    (
        cache_headers,
        [(
            CONTENT_TYPE,
            HeaderValue::from_static("application/octet-stream"),
        )],
        snap.series_payload.clone(),
    )
        .into_response()
}

/// Entity tag of a series payload, derived from its content.
///
/// Weak (`W/`): the compression layer re-encodes the body per request, so
/// the tag vouches for the content, not for the exact bytes on the wire.
///
/// # Arguments
/// * `payload` — the encoded series payload.
///
/// # Return
/// A quoted weak entity tag, e.g. `W/"0123456789abcdef"`.
///
/// # Notes
/// [`DefaultHasher::new`] uses fixed keys, so the same payload gets the same
/// tag across rebuilds and server restarts of the same binary — which is
/// what lets a browser keep its copy after a restart. A different binary may
/// hash differently; that only costs one re-download.
pub fn etag_for(payload: &[u8]) -> String {
    let mut hasher = DefaultHasher::new();
    payload.hash(&mut hasher);
    format!("W/\"{:016x}\"", hasher.finish())
}

/// Whether an `If-None-Match` header value matches `etag`, using the weak
/// comparison RFC 9110 prescribes for this header.
///
/// # Arguments
/// * `if_none_match` — the raw header value: `*` or a comma-separated list
///   of entity tags.
/// * `etag` — the current entity tag, as produced by [`etag_for`].
///
/// # Return
/// `true` if the client's cached copy is current.
pub fn if_none_match_hits(if_none_match: &str, etag: &str) -> bool {
    let opaque = |tag: &str| tag.trim().trim_start_matches("W/").to_owned();
    let current = opaque(etag);
    if_none_match
        .split(',')
        .any(|candidate| candidate.trim() == "*" || opaque(candidate) == current)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn etag_depends_only_on_content() {
        assert_eq!(etag_for(b"abc"), etag_for(b"abc"));
        assert_ne!(etag_for(b"abc"), etag_for(b"abd"));
        let tag = etag_for(b"abc");
        assert!(tag.starts_with("W/\"") && tag.ends_with('"'));
    }

    #[test]
    fn if_none_match_uses_weak_comparison() {
        let tag = etag_for(b"payload");
        let strong = tag.trim_start_matches("W/");
        assert!(if_none_match_hits(&tag, &tag));
        assert!(if_none_match_hits(strong, &tag));
    }

    #[test]
    fn if_none_match_accepts_lists_and_wildcard() {
        let tag = etag_for(b"payload");
        assert!(if_none_match_hits(&format!("W/\"other\", {tag}"), &tag));
        assert!(if_none_match_hits("*", &tag));
    }

    #[test]
    fn if_none_match_misses_on_another_payload() {
        let tag = etag_for(b"payload");
        assert!(!if_none_match_hits(&etag_for(b"changed"), &tag));
        assert!(!if_none_match_hits("", &tag));
    }
}
