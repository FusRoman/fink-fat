# 🤖 Rust Coding Agent — Conventions for this Workspace

> This file defines how **AI coding agents in VS Code** should behave when
> reading, editing or generating **Rust** code in this repository.

If you propose code that does *not* follow these conventions, **fix it before
showing it** to the user.

---

## 1. 🔧 Formatting, Tooling & Lints

- Always assume **`rustfmt`** and **`clippy`** are part of the pipeline.
- Style:
  - Let `rustfmt` own all layout decisions.
  - Do **not** manually align columns or try to fight `rustfmt`.
  - Prefer line lengths ≲ 100 chars when possible, but do not break idioms just for that.
- Lints:
  - Code must be **Clippy-clean** at default settings (no warnings).
  - If you *must* allow a lint, use a **narrow scope**:
    - Prefer `#[allow(clippy::some_lint)]` on the **smallest scope** (fn, impl) instead of modules.
    - Add a short comment *why* the allow is needed.

---

## 2. 🧩 Naming & Style

- **Identifiers are in English.**
- Conventions:
  - `snake_case` for functions, variables, modules: `compute_offset`, `seed_index`.
  - `UpperCamelCase` for types and traits: `SeedNode`, `TangentPlaneModel`.
  - `SCREAMING_SNAKE_CASE` for `const`s and `static`s: `INV_COSC_MIN`, `NORM_MIN`.
- Acronyms:
  - Prefer `CamelCase` style even for acronyms: `MjdTt`, not `MJDTT`.
- Functions should have **clear verbs**:
  - Good: `compute_cone_radius`, `build_bucket_index`, `resolve_seed_members`.
  - Avoid ambiguous names: `run`, `handle`, `do_stuff`.
- Tests:
  - Use descriptive names:
    - `fn members_are_sorted_by_time_then_id()`
    - `fn prop_gnomonic_roundtrip_small_offsets()`

---

## 3. 🧱 Modules & File Organization

- One **main concept per module** when possible:
  - `astro_math.rs` – low-level spherical / tangent-plane math.
  - `seed_node.rs` – definition + behavior of `SeedNode`.
  - `spacetime_bucket.rs` – generic buckets & indexing.
- Keep **public API small & clean**:
  - Re-export types from a `mod` if needed (e.g. `pub use seed_node::SeedNode;`).
  - Prefer `pub(crate)` when the type/function is internal to the crate.
- Inside modules:
  - Order items roughly as:
    1. `use` imports
    2. Public types (structs/enums/traits)
    3. Public functions
    4. Internal helpers (`pub(crate)` / `fn` not exported)
    5. `#[cfg(test)] mod tests`

---

## 4. ⚠️ Error Handling & Panics

- **Never use `unwrap` / `expect` in library code** except in:
  - Tests
  - `main` of small binaries / examples
- Prefer explicit error types:
  - Use meaningful error enums where appropriate: `SeedBuildError`, `ConfigError`.
  - Where a generic error type is acceptable, `anyhow::Result` is fine for binaries and tools, **not** for core library APIs.
- In fallible functions:
  - Return `Result<T, E>` where `E` is a meaningful error type.
  - Do not hide logic errors with silent `Option::None` unless semantically “value not found”/“not applicable”.
- Panics:
  - Only for *truly impossible / programmer bugs* (e.g. invariants broken).
  - If you introduce a `debug_assert!`, ensure behavior still makes sense in release.

---

## 5. 📚 Documentation & Comments

### Doc comments (`///`)

- All **public types, functions and methods** must have a doc comment.
- Style:
  - First line: **short summary** on one sentence.
  - Then an optional longer description / sections:
    - `Overview`, `Arguments`, `Return`, `Panics`, `Examples`, `Notes`, `Units`.
- Keep docs **precise, technical and minimal**, not marketing.

Example:

```rust
/// Core **triplet** generation from a list of previously built pairs.
///
/// Builds `(a, b, c)` triplets by:
///
/// 1. Scanning candidate `c` in the spatio-temporal neighbors of `b`.
/// 2. Enforcing pairwise consistency between `(b, c)`:
///    - `0 < t_c − t_b ≤ max_dt_between`
///    - `ang_sep(b, c) ≤ max_pair_sep`
/// 3. Applying a **linear prediction test** on the tangent plane around `a`:
///    - estimate motion from `(a, b)`,
///    - extrapolate to the time of `c`,
///    - accept only if residual distance ≤ `max_predicted_residual`.
///
/// Arguments
/// ---------
/// * `index` – Spatio-temporal bucket index (same as used for the pair stage).
/// * `alerts` – Contiguous array of alerts (must satisfy `alert.id.idx() == index`).
/// * `sb` – Spatial binner (e.g. HEALPix) used for bucket construction.
/// * `tb` – Time binner used for bucket construction.
/// * `triplet_config` – Triplet constraints and thresholds:
///   - `max_dt_between` – maximum allowed time between b and c,
///   - `max_pair_sep` – maximum separation for (b, c),
///   - `max_predicted_residual` – maximum tangent-plane residual,
///   - `enforce_time_order` – enforce `t_a < t_b` when consuming pairs,
///   - `max_flux_difference` – flux similarity constraint.
/// * `pairs` – List of `(a, b)` pairs previously produced by [`generate_pairs`].
///
/// Return
/// ------
/// `Triplets` – vector of `(AlertId, AlertId, AlertId)` triplets with:
/// - `t_a < t_b < t_c` (if `enforce_time_order == true`),
/// - pairwise time/angle cuts satisfied for `(b, c)`,
/// - linear prediction residual below `max_predicted_residual`.
///
/// Notes
/// -----
/// The implementation is optimized for large-N usage:
/// - reuses id-indexed tables for all scalar/vector quantities,
/// - caches spatial and temporal neighbors per `(space_key, time_bin)` combo,
/// - uses binary search and early exit in each bucket,
/// - deduplicates the output at the end.
pub fn generate_triplets_from_pairs<Bs: SpatialBinner, Bt: TimeBinner>(
    index: &BucketIndex<AlertId>,
    alerts: &[Alert],
    sb: &Bs,
    tb: &Bt,
    triplet_config: &TripletConfig,
    pairs: &[Pair],
) -> Triplets {...}
```