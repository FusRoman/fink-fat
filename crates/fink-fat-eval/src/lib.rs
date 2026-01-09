pub mod dataset;
pub mod seeding;
pub mod angle;
pub mod grid;
pub mod angular_speed;
pub mod cli;
pub mod scoring;



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