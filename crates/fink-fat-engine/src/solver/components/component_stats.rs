use crate::night_id::NightId;

/// Fast per-component stats used for routing.
#[derive(Copy, Clone, Debug, Default)]
pub struct ComponentStats {
    pub n_nodes: u32,
    pub m_active_edges: u32,

    /// Inclusive bounds of nights spanned by the component.
    /// `None` means the component has no nodes (should not happen in normal use).
    pub night_bounds: Option<(NightId, NightId)>,
}

impl ComponentStats {
    /// Convenience: (max_night - min_night) in "night units".
    ///
    /// If `NightId` is not a simple `u32` newtype, adapt the conversion here.
    #[inline]
    pub fn night_span(&self) -> u32 {
        match self.night_bounds {
            Some((min_n, max_n)) => {
                let min: u32 = min_n.into();
                let max: u32 = max_n.into();
                max.saturating_sub(min)
            }
            None => 0,
        }
    }

    #[inline]
    pub fn min_night(&self) -> Option<NightId> {
        self.night_bounds.map(|(a, _)| a)
    }

    #[inline]
    pub fn max_night(&self) -> Option<NightId> {
        self.night_bounds.map(|(_, b)| b)
    }
}
