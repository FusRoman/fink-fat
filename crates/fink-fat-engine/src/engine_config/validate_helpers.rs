//! # Shared numeric-range checks for `Validate` implementations
//!
//! Small, reusable range checks used by every `impl Validate for ...` across
//! `engine_config`, so each struct's `validate()` stays a flat list of checks
//! instead of repeating the same `is_finite()` / comparison boilerplate.
//!
//! Every helper returns `Option<FieldError>` (`None` when the value is valid)
//! so call sites read as:
//!
//! ```rust, ignore
//! let mut errors = Vec::new();
//! if let Some(e) = check_finite_nonneg("max_dt", self.max_dt, "set pairs.max_dt to a non-negative duration, e.g. \"86.4 min\"") {
//!     errors.push(e);
//! }
//! ```

use crate::engine_config::error::FieldError;

/// Require `value` to be finite and `>= 0`.
pub(crate) fn check_finite_nonneg(field: &str, value: f64, hint: &str) -> Option<FieldError> {
    if !value.is_finite() || value < 0.0 {
        Some(
            FieldError::new(
                field,
                format!("must be finite and non-negative, got {value}"),
            )
            .with_hint(hint),
        )
    } else {
        None
    }
}

/// Require `value` to be finite and strictly `> 0`.
pub(crate) fn check_finite_positive(field: &str, value: f64, hint: &str) -> Option<FieldError> {
    if !value.is_finite() || value <= 0.0 {
        Some(
            FieldError::new(
                field,
                format!("must be finite and strictly positive, got {value}"),
            )
            .with_hint(hint),
        )
    } else {
        None
    }
}

/// Require `value` to simply be finite (any sign).
pub(crate) fn check_finite(field: &str, value: f64, hint: &str) -> Option<FieldError> {
    if !value.is_finite() {
        Some(FieldError::new(field, format!("must be finite, got {value}")).with_hint(hint))
    } else {
        None
    }
}

/// Require `value` to be finite and within `[low, high]` (inclusive).
pub(crate) fn check_finite_in_range(
    field: &str,
    value: f64,
    low: f64,
    high: f64,
    hint: &str,
) -> Option<FieldError> {
    if !value.is_finite() || value < low || value > high {
        Some(
            FieldError::new(
                field,
                format!("must be finite and in [{low}, {high}], got {value}"),
            )
            .with_hint(hint),
        )
    } else {
        None
    }
}

/// Require `value >= min`.
pub(crate) fn check_min_usize(
    field: &str,
    value: usize,
    min: usize,
    hint: &str,
) -> Option<FieldError> {
    if value < min {
        Some(FieldError::new(field, format!("must be >= {min}, got {value}")).with_hint(hint))
    } else {
        None
    }
}

/// Require `a < b` (strict ordering between two related f64 fields).
pub(crate) fn check_lt(
    field_a: &str,
    a: f64,
    field_b: &str,
    b: f64,
    hint: &str,
) -> Option<FieldError> {
    if !matches!(a.partial_cmp(&b), Some(std::cmp::Ordering::Less)) {
        Some(
            FieldError::new(
                field_a,
                format!("must be strictly less than `{field_b}` ({b}), got {a}"),
            )
            .with_hint(hint),
        )
    } else {
        None
    }
}

/// Require `a <= b` (non-strict ordering between two related f64 fields).
pub(crate) fn check_le(
    field_a: &str,
    a: f64,
    field_b: &str,
    b: f64,
    hint: &str,
) -> Option<FieldError> {
    if a > b {
        Some(
            FieldError::new(field_a, format!("must be <= `{field_b}` ({b}), got {a}"))
                .with_hint(hint),
        )
    } else {
        None
    }
}

/// Require a non-empty string field.
pub(crate) fn check_non_empty(field: &str, value: &str, hint: &str) -> Option<FieldError> {
    if value.is_empty() {
        Some(FieldError::new(field, "must not be empty").with_hint(hint))
    } else {
        None
    }
}
