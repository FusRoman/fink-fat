//! Typed error of the REST API, mapped to HTTP status codes.

use dioxus::fullstack::StatusCode;
use dioxus::prelude::ServerFnError;

/// Errors returned by the REST API handlers.
#[derive(Debug, thiserror::Error)]
pub(crate) enum ApiError {
    /// The requested resource does not exist.
    #[error("no alert with object_id `{0}`")]
    NotFound(String),
    /// A database query failed.
    #[error(transparent)]
    Db(#[from] sqlx::Error),
}

impl ApiError {
    /// HTTP status code matching the error.
    ///
    /// # Return
    ///
    /// `404` for [`ApiError::NotFound`], `500` for [`ApiError::Db`].
    pub(crate) fn status(&self) -> StatusCode {
        match self {
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::Db(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl From<ApiError> for ServerFnError {
    /// Converts to a server error carrying the HTTP status. Database details
    /// are logged and replaced by a generic message for the client.
    fn from(error: ApiError) -> Self {
        let status = error.status();
        let message = match &error {
            ApiError::Db(e) => {
                tracing::error!("REST API database error: {e}");
                "internal server error".to_string()
            }
            other => other.to_string(),
        };
        ServerFnError::ServerError {
            message,
            code: status.as_u16(),
            details: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn not_found_maps_to_404() {
        let err = ApiError::NotFound("x".into());
        assert_eq!(err.status(), StatusCode::NOT_FOUND);
        assert!(matches!(
            ServerFnError::from(err),
            ServerFnError::ServerError { code: 404, .. }
        ));
    }

    #[test]
    fn db_error_maps_to_500_with_generic_message() {
        let err = ApiError::Db(sqlx::Error::PoolTimedOut);
        match ServerFnError::from(err) {
            ServerFnError::ServerError { code, message, .. } => {
                assert_eq!(code, 500);
                assert_eq!(message, "internal server error");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
