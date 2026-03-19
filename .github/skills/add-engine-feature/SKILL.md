---
name: add-engine-feature
description: 'Step-by-step guide for adding a new feature to the fink-fat-engine Rust crate. Use when implementing a new module, solver, seeding strategy, cost function, edge feature, config option, or pipeline stage. Covers: module placement, naming conventions, error types, EngineConfig wiring, re-exports, unit/integration tests, rustdoc, and verification commands.'
argument-hint: 'Describe the feature to add (e.g. "new min-cost-flow solver variant", "new edge feature for photometry residual")'
---

# Add a Feature to fink-fat-engine

## Overview

The engine lives in `crates/fink-fat-engine/src/`. All features follow the same layered structure:
domain module → config struct → error type → pipeline wiring → tests → rustdoc.

---

## Step 1 — Decide the module location

Use the table below to find where the new code belongs.

| What you're adding | Location |
|---|---|
| New seeding strategy (pairs / triplets) | `src/seeding/` |
| New solver variant | `src/solver/` |
| New cost function or edge feature | `src/graph/edge/` |
| New spatial/temporal bucketing | `src/spacetime_bucket/` |
| New pipeline stage | `src/pipeline/stages/` |
| Reusable math primitives | `src/astro_math.rs` |
| New top-level domain | `src/<domain>/` (new folder) |

If adding a **new top-level domain**, create:
```
src/<domain>/
    mod.rs          ← public API + re-exports
    error.rs        ← domain-specific error enum
    <impl>.rs       ← implementation file(s)
```

---

## Step 2 — Create the implementation file(s)

> **Delegate to the `code-generation` agent** for this step and all subsequent implementation steps (Steps 2–5 and Step 7).

### Naming conventions
- **Newtypes** for semantic types: `MyDomainId(u32)`, `MyKey`
- **`*Store`** for indexed collections
- **`*Builder`** / `build_*` for construction functions
- **`*Config`** for configuration structs
- **`*Error`** for error enums
- **`*Pool`** for reusable / thread-local resources
- Use `AHashMap` (from `ahash`) instead of `std::HashMap` in hot paths
- Use `smallvec::SmallVec` for collections that are usually small

### Error type (src/<domain>/error.rs)

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum <Domain>Error {
    #[error("...")]
    Variant(/* cause */),
}
```

Then add a variant to `src/error.rs` wrapping it:
```rust
#[error(transparent)]
<Domain>(#[from] crate::<domain>::error::<Domain>Error),
```

---

## Step 3 — Wire up the module

### 3a. Declare the submodule in the parent `mod.rs`

```rust
// In src/<parent>/mod.rs  (or src/lib.rs for top-level domains)
pub mod <new_module>;
pub use <new_module>::{MyType, MyKey};   // re-export what belongs to the public API
```

### 3b. For top-level domains — register in `src/lib.rs`

```rust
pub mod <domain>;
// optional public re-export:
pub use <domain>::MyType;
```

### 3c. For new pipeline stages — update `src/pipeline/stages/`

1. Add a variant to the `PipelineStage` enum.
2. Implement the stage handler (input/output types must match surrounding stages).
3. Register it in `src/pipeline/mod.rs` (`PipelineRunner::run`).

---

## Step 4 — Add configuration

If the feature is configurable, add a sub-config struct in `src/engine_config/`:

```rust
// src/engine_config/<feature>_config.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct <Feature>Config {
    pub param: f64,
}

impl Default for <Feature>Config {
    fn default() -> Self { Self { param: 1.0 } }
}
```

Then embed it in `EngineConfig` in `src/engine_config/mod.rs`:
```rust
pub <feature>: <Feature>Config,
```

Config is loaded from YAML + env vars automatically (prefix `FINK_FAT__`, separator `__`).

---

## Step 5 — Write tests

### Unit tests (inline in the implementation file)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_behavior() {
        // ...
    }
}
```

### Integration tests (in `crates/fink-fat-engine/tests/`)

For features that touch the full pipeline, add a file under `tests/<domain>_test.rs` and declare it in `tests/mod.rs`:
```rust
mod <domain>_test;
```

Use `SyntheticDatasetBuilder` (from `tests/synthetic_alerts.rs`) to generate realistic multi-night datasets without needing real Parquet files:
```rust
use crate::synthetic_alerts::SyntheticDatasetBuilder;

let dataset = SyntheticDatasetBuilder::default()
    .with_nea_tracks(10)
    .build();
```

### Property-based tests (for algorithmic invariants)

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn prop_invariant_holds(input in 0.0f64..1.0) {
        // ...
    }
}
```

---

## Step 6 — Add rustdoc to the public API

> **Delegate to the `fink_fat_doc` agent** for this step. Provide it with the list of `pub` items added in Steps 2–4.

Every `pub` type, function, and module that is part of the public API needs a `///` doc comment:

```rust
/// Short one-line summary.
///
/// Longer explanation if needed. Reference related types with [`OtherType`].
///
/// # Errors
///
/// Returns [`MyDomainError::Variant`] when ...
pub fn my_function(...) -> Result<..., MyDomainError> { ... }
```

---

## Step 7 — Verify the change

Run these commands in order from the workspace root:

```bash
# 1. Compile-check the engine crate only (fast)
cargo check -p fink-fat-engine

# 2. Clippy (catches common patterns and style issues)
cargo clippy -p fink-fat-engine -- -D warnings

# 3. Run all tests
cargo test -p fink-fat-engine

# 4. Run a specific test by name
cargo test -p fink-fat-engine <test_name>

# 5. Run benchmarks (optional – only when changing hot paths)
cargo bench -p fink-fat-engine
# Or via the helper script:
# crates/fink-fat-engine/run_bench.sh
```

If adding a **new pipeline stage**, also run the eval suite to check for regressions:
```bash
cargo run -p fink-fat-eval -- --config crates/fink-fat-eval/eval_config_best.yml
```

---

## Checklist

Before marking the feature complete, verify each item:

- [ ] Module file(s) created in the right domain folder
- [ ] Domain error type created in `error.rs` using `thiserror`
- [ ] New error variant added to top-level `src/error.rs`
- [ ] `pub mod` + `pub use` added in parent `mod.rs`
- [ ] Registered in `src/lib.rs` if it's a new top-level domain
- [ ] `*Config` struct added to `src/engine_config/` (if configurable)
- [ ] Embedded in `EngineConfig` with `Default` impl
- [ ] Inline `#[cfg(test)]` unit tests written
- [ ] Integration test file added under `tests/` (if touching pipeline)
- [ ] `///` rustdoc on all `pub` items
- [ ] `cargo check`, `cargo clippy`, `cargo test` all pass
