use std::{fmt::Display, str::FromStr};

use serde::{Deserialize, Serialize};
use url::Url;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct InputUri(pub String);

impl InputUri {
    pub fn parse(&self) -> Result<Url, url::ParseError> {
        Url::parse(&self.0)
    }
}

impl FromStr for InputUri {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(InputUri(s.to_string()))
    }
}

impl Display for InputUri {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod input_uri_tests {
    use super::*;
    use serde_json;

    // -------------------------------------------------------------------------
    // Basic parsing tests
    // -------------------------------------------------------------------------

    #[test]
    fn parse_valid_http_uri() {
        let uri = InputUri("https://example.com/data.parquet".to_string());

        let parsed = uri.parse().expect("URI should parse");

        assert_eq!(parsed.scheme(), "https");
        assert_eq!(parsed.host_str(), Some("example.com"));
        assert_eq!(parsed.path(), "/data.parquet");
    }

    #[test]
    fn parse_valid_file_uri() {
        let uri = InputUri("file:///tmp/test.parquet".to_string());

        let parsed = uri.parse().expect("file URI should parse");

        assert_eq!(parsed.scheme(), "file");
        assert_eq!(parsed.path(), "/tmp/test.parquet");
    }

    #[test]
    fn parse_invalid_uri() {
        let uri = InputUri("not a valid uri".to_string());

        let parsed = uri.parse();

        assert!(parsed.is_err(), "Invalid URI should return error");
    }

    // -------------------------------------------------------------------------
    // Serde behavior (transparent wrapper)
    // -------------------------------------------------------------------------

    #[test]
    fn serde_roundtrip() {
        let uri = InputUri("https://example.com/foo".to_string());

        let json = serde_json::to_string(&uri).expect("serialization should work");
        let de: InputUri = serde_json::from_str(&json).expect("deserialization should work");

        assert_eq!(uri.0, de.0);
    }

    #[test]
    fn serde_transparent_representation() {
        let uri = InputUri("https://example.com/bar".to_string());

        let json = serde_json::to_string(&uri).unwrap();

        // Transparent means it's serialized as a plain JSON string
        assert_eq!(json, "\"https://example.com/bar\"");
    }
}
