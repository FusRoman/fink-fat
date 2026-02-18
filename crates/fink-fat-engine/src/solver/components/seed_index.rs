use ahash::AHashMap;

use crate::{
    night_id::NightId,
    seeding::{SeedKey, store::SeedStore},
    solver::components::error::ComponentError,
};

/// Dense identifier used for seeds in global index space.
pub type DenseSeedId = u32;

/// Map `SeedKey` -> dense global index `[0..N_total)`.
///
/// This implementation leverages the internal reverse index of
/// `SeedStore` (`id_to_location`) to retrieve the intra-night
/// position of each seed.
///
/// The dense index is computed as:
///
/// ```text
/// global_idx = base[night_id] + index_in_vec
/// ```
///
/// Unlike the previous design, `SeedKey.unique_id` is no longer
/// assumed to be dense nor structured per-night.
#[derive(Debug, Clone)]
pub struct SeedGlobalIndex {
    /// Base offset per night.
    base: AHashMap<NightId, DenseSeedId>,

    /// Total number of seeds.
    n_total: u32,
}

impl SeedGlobalIndex {
    /// Build global dense indexing using per-night contiguous layout.
    ///
    /// Nights are processed in sorted deterministic order.
    /// Within each night, the natural `Vec` order is used.
    ///
    /// Complexity
    /// ----------
    /// O(K) where K is the number of nights.
    pub fn build(seed_store: &SeedStore) -> Result<Self, ComponentError> {
        let mut base = AHashMap::default();
        let mut cursor: u32 = 0;

        for night_id in seed_store.nights() {
            base.insert(*night_id, cursor);

            let len = seed_store
                .len_night(night_id)
                .ok_or(ComponentError::NightNotFound(*night_id))?;
            cursor += len as u32;
        }

        Ok(Self {
            base,
            n_total: cursor,
        })
    }

    /// Total number of indexed seeds.
    #[inline]
    pub fn n_total(&self) -> usize {
        self.n_total as usize
    }

    /// Return dense index for a seed key.
    ///
    /// Arguments
    /// ---------
    /// * `seed_store` – Seed storage to query for reverse indexing.
    /// * `key` – Seed key to index.
    ///
    /// Return
    /// ------
    /// * `Ok(usize)` – Dense index of the seed key.
    /// * `Err(ComponentError::SeedKeyInIndexNotFound)` – If the seed key is not found in the seed store's reverse index.
    #[inline]
    pub fn idx_of_key(
        &self,
        seed_store: &SeedStore,
        key: SeedKey,
    ) -> Result<usize, ComponentError> {
        let (night_id, index_in_vec) = seed_store
            .get_reverse_index(key)
            .ok_or(ComponentError::SeedKeyInIndexNotFound(key))?;

        Ok((self.base[&night_id] + (index_in_vec as u32)) as usize)
    }
}
