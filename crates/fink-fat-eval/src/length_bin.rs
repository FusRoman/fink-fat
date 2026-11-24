//! Ground-truth trajectory length bins.
//!
//! [`GoldTrajectoryTracker::n_obs_so_far`](crate::tracking_report::gold_trajectory::GoldTrajectoryTracker::n_obs_so_far)
//! gives, for a trackable trajectory, how many of its observations have been
//! seen within the nights processed so far — the "number of points of the
//! reconstructible trajectory" the tracker is being judged against. Binning
//! trajectories on that count lets efficacy tables show whether performance
//! is a function of how much evidence the trajectory actually offers,
//! independent of orbital class (see [`crate::population::Population`], the
//! other efficacy breakdown axis).

/// Trajectory-length bins, from the shortest trackable trajectories (2
/// same-night observations is the minimum for
/// [`GoldTrajectoryTracker::is_trackable`](crate::tracking_report::gold_trajectory::GoldTrajectoryTracker::is_trackable))
/// up to long, well-observed ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LengthBin {
    Two,
    Three,
    Four,
    Five,
    SixToTen,
    ElevenPlus,
}

impl LengthBin {
    /// Classify from the trajectory's observation count seen so far.
    pub fn classify(n_obs: usize) -> Self {
        match n_obs {
            0..=2 => LengthBin::Two,
            3 => LengthBin::Three,
            4 => LengthBin::Four,
            5 => LengthBin::Five,
            6..=10 => LengthBin::SixToTen,
            _ => LengthBin::ElevenPlus,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            LengthBin::Two => "2",
            LengthBin::Three => "3",
            LengthBin::Four => "4",
            LengthBin::Five => "5",
            LengthBin::SixToTen => "6-10",
            LengthBin::ElevenPlus => "11+",
        }
    }

    /// All bins, in increasing order.
    pub fn all() -> [LengthBin; 6] {
        [
            LengthBin::Two,
            LengthBin::Three,
            LengthBin::Four,
            LengthBin::Five,
            LengthBin::SixToTen,
            LengthBin::ElevenPlus,
        ]
    }
}
