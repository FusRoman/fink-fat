//! Directed edge model (hypothesis linking two seeds across nights).

use crate::graph::{EdgeId, NodeId};

/// Directed link from an older seed to a newer seed (forward in time).
///
/// Cost
/// ----
/// `cost` is a **dimensionless** score already aggregated upstream:
/// lower is better. Enforce strictly **positive** costs to avoid
/// degeneracies in later solvers.
#[derive(Clone, Debug)]
pub struct Edge {
    pub id: EdgeId,
    pub from: NodeId,
    pub to: NodeId,
    pub cost: f32,
    /// Time gap in days (TT) between the two seeds (positive).
    pub dt_days: f32,
}

impl Edge {
    pub fn new(id: EdgeId, from: NodeId, to: NodeId, cost: f32, dt_days: f32) -> Self {
        assert!(
            cost.is_finite() && cost > 0.0,
            "Edge cost must be finite and > 0."
        );
        assert!(
            dt_days.is_finite() && dt_days > 0.0,
            "dt_days must be finite and > 0."
        );
        Self {
            id,
            from,
            to,
            cost,
            dt_days,
        }
    }
}
