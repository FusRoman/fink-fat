pub mod angle;
pub mod angular_speed;
pub mod bin_utils;
pub mod cli;
pub mod dataset;
pub mod grid;
pub mod io;
pub mod night_seeds;
pub mod scoring;
pub mod seeding;

/// Replace NaN/Inf by a fallback.
pub trait FiniteOr {
    /// Return `self` if it is finite, otherwise return `fallback`.
    fn if_finite_or(self, fallback: f64) -> f64;
}

impl FiniteOr for f64 {
    #[inline]
    fn if_finite_or(self, fallback: f64) -> f64 {
        if self.is_finite() { self } else { fallback }
    }
}
