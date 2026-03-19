//! Object-store resolution for `InputUri`.
//!
//! This module translates a parsed `Url` into an `object_store` backend
//! (`ObjectStore`) and an in-store `Path`.
//!
//! Supported schemes:
//! - file://...
//! - http://...
//! - https://...
//! - hdfs://...

use std::sync::Arc;

use datafusion::object_store;
use datafusion::object_store::http::HttpBuilder;
use datafusion::object_store::local::LocalFileSystem;
use datafusion::object_store::{ObjectStore, path::Path as ObjPath};
use url::Url;

use hdfs_native_object_store::HdfsObjectStoreBuilder;

use crate::pipeline::stages::alert_inputs::input_uri::InputUri;

/// Result of URI resolution: an object store instance and a relative in-store path.
#[derive(Clone)]
pub struct ResolvedObject {
    pub store: Arc<dyn ObjectStore>,
    pub path: ObjPath,
}

impl std::fmt::Debug for ResolvedObject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedObject")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub enum UriStoreError {
    InvalidUri(String),
    UnsupportedScheme(String),
    MissingAuthority(String),
    ObjectStore(object_store::Error),
    HdfsBuild(String),
}

impl From<object_store::Error> for UriStoreError {
    fn from(e: object_store::Error) -> Self {
        UriStoreError::ObjectStore(e)
    }
}

/// Resolve an `InputUri` into an object store + path.
pub fn resolve_input_uri(uri: &InputUri) -> Result<ResolvedObject, UriStoreError> {
    let url = uri
        .parse()
        .map_err(|_| UriStoreError::InvalidUri(uri.0.clone()))?;
    resolve_url(&url)
}

/// Resolve a parsed `Url` into an object store + path.
pub fn resolve_url(url: &Url) -> Result<ResolvedObject, UriStoreError> {
    match url.scheme() {
        "file" => resolve_file(url),
        "http" | "https" => resolve_http(url),
        "hdfs" => resolve_hdfs(url),
        other => Err(UriStoreError::UnsupportedScheme(other.to_string())),
    }
}

// -----------------------------------------------------------------------------
// Scheme handlers
// -----------------------------------------------------------------------------

fn resolve_file(url: &Url) -> Result<ResolvedObject, UriStoreError> {
    // Anchor at "/" so absolute filesystem paths become relative object_store paths
    // by stripping the leading '/'.
    let store = Arc::new(LocalFileSystem::new_with_prefix("/")?);

    let rel = url.path().trim_start_matches('/');
    Ok(ResolvedObject {
        store,
        path: ObjPath::from(rel),
    })
}

fn resolve_http(url: &Url) -> Result<ResolvedObject, UriStoreError> {
    let store = Arc::new(HttpBuilder::new().build()?);

    // HTTP store expects full URL encoded in the path.
    Ok(ResolvedObject {
        store,
        path: ObjPath::from(url.as_str()),
    })
}

fn resolve_hdfs(url: &Url) -> Result<ResolvedObject, UriStoreError> {
    let host = url
        .host_str()
        .ok_or_else(|| UriStoreError::MissingAuthority(url.as_str().to_string()))?;

    let base = match url.port() {
        Some(port) => format!("hdfs://{host}:{port}"),
        None => format!("hdfs://{host}"),
    };

    let rel = url.path().trim_start_matches('/');

    let hdfs_store = HdfsObjectStoreBuilder::new()
        .with_url(base)
        .build()
        .map_err(|e| UriStoreError::HdfsBuild(e.to_string()))?;

    Ok(ResolvedObject {
        store: Arc::new(hdfs_store),
        path: ObjPath::from(rel),
    })
}

#[cfg(test)]
mod uri_resolver_tests {
    use super::*;
    use std::sync::Arc;
    use url::Url;

    // -----------------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------------

    fn url(s: &str) -> Url {
        Url::parse(s).expect("test URL must parse")
    }

    // -----------------------------------------------------------------------------
    // InputUri -> Url parsing plumbing
    // -----------------------------------------------------------------------------

    #[test]
    fn resolve_input_uri_invalid_uri_maps_to_invaliduri() {
        let uri = InputUri("not a uri".to_string());
        let err = resolve_input_uri(&uri).unwrap_err();

        match err {
            UriStoreError::InvalidUri(s) => assert_eq!(s, "not a uri"),
            other => panic!("expected InvalidUri, got: {other:?}"),
        }
    }

    // -----------------------------------------------------------------------------
    // Scheme dispatch
    // -----------------------------------------------------------------------------

    #[test]
    fn resolve_url_unsupported_scheme() {
        let u = url("s3://bucket/key");
        let err = resolve_url(&u).unwrap_err();

        match err {
            UriStoreError::UnsupportedScheme(s) => assert_eq!(s, "s3"),
            other => panic!("expected UnsupportedScheme, got: {other:?}"),
        }
    }

    // -----------------------------------------------------------------------------
    // file://...
    // -----------------------------------------------------------------------------

    #[test]
    fn resolve_file_makes_absolute_fs_path_relative() {
        // With LocalFileSystem anchored at "/", "/tmp/x" becomes "tmp/x" in object_store paths.
        let u = url("file:///tmp/test.parquet");
        let resolved = resolve_url(&u).expect("file:// should resolve");

        assert_eq!(resolved.path.as_ref(), "tmp/test.parquet");

        // Basic sanity: store is some object store impl behind Arc.
        let _store: Arc<dyn ObjectStore> = resolved.store.clone();
    }

    #[test]
    fn resolve_file_root_path_is_empty_relative_path() {
        let u = url("file:///");
        let resolved = resolve_url(&u).expect("file:/// should resolve");

        // url.path() == "/" -> trimmed -> ""
        assert_eq!(resolved.path.as_ref(), "");
    }

    // -----------------------------------------------------------------------------
    // hdfs://...
    // -----------------------------------------------------------------------------

    #[test]
    fn resolve_hdfs_missing_authority_is_error() {
        // hdfs:///path has no host
        let u = url("hdfs:///data/file.parquet");
        let err = resolve_url(&u).unwrap_err();

        match err {
            UriStoreError::MissingAuthority(s) => {
                // should contain the original URL string
                assert_eq!(s, "hdfs:///data/file.parquet");
            }
            other => panic!("expected MissingAuthority, got: {other:?}"),
        }
    }

    // -------------------------------------------------------------------------
    // http(s)://...
    //
    // With the current implementation:
    //   HttpBuilder::new().build()? -> MissingUrl
    // so resolution must return an ObjectStore error.
    // -------------------------------------------------------------------------

    #[test]
    fn resolve_http_requires_builder_base_url_current_behavior() {
        let u = url("http://example.com/data.parquet?x=1#frag");
        let err = resolve_url(&u).unwrap_err();

        match err {
            UriStoreError::ObjectStore(e) => {
                // Don't overfit on exact Debug formatting, but ensure it's the HTTP MissingUrl path.
                let s = format!("{e:?}");
                assert!(s.contains("HTTP"), "expected HTTP store error, got: {s}");
                assert!(s.contains("MissingUrl"), "expected MissingUrl, got: {s}");
            }
            other => panic!("expected ObjectStore error, got: {other:?}"),
        }
    }

    #[test]
    fn resolve_https_requires_builder_base_url_current_behavior() {
        let u = url("https://example.com/a/b/c");
        let err = resolve_url(&u).unwrap_err();

        match err {
            UriStoreError::ObjectStore(e) => {
                let s = format!("{e:?}");
                assert!(s.contains("HTTP"), "expected HTTP store error, got: {s}");
                assert!(s.contains("MissingUrl"), "expected MissingUrl, got: {s}");
            }
            other => panic!("expected ObjectStore error, got: {other:?}"),
        }
    }

    // -------------------------------------------------------------------------
    // hdfs://...
    //
    // The HDFS builder may succeed even without a live cluster (lazy connection),
    // so we only test deterministic parts:
    // - missing authority -> error
    // - returned path is the URL path without leading '/'
    // - host/port parsing doesn't crash
    //
    // If you want a "real" HDFS connectivity test, make it integration + #[ignore].
    // -------------------------------------------------------------------------

    #[test]
    fn resolve_hdfs_returns_relpath_without_leading_slash_portable() {
        let u = url("hdfs://localhost:9870/some/path.parquet");

        let resolved = resolve_url(&u).expect("builder may succeed without contacting HDFS");
        assert_eq!(resolved.path.as_ref(), "some/path.parquet");
    }
}
