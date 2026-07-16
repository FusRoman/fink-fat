//! Central registry of every tracing target defined in the engine.
//!
//! This is the *only* place that needs to be updated when a new module's
//! log-event enum is added — it doesn't duplicate any name/description
//! string (those are read from each type's [`LogTarget`] impl), so the
//! registry can never drift out of sync with the module that owns them.

use std::collections::BTreeMap;

use crate::engine_config::log_level::LogLevel;
use crate::logging::LogTarget;

/// Everything [`all_targets`] reports about one tracing target: its name,
/// human-readable description, and the levels it can emit at.
pub struct TargetInfo {
    pub name: &'static str,
    pub description: &'static str,
    pub levels: &'static [tracing::Level],
}

impl TargetInfo {
    pub fn of<T: LogTarget>() -> Self {
        Self {
            name: T::TARGET,
            description: T::DESCRIPTION,
            levels: T::LEVELS,
        }
    }
}

/// Every tracing target defined across the engine's pipeline, sorted by
/// name. Used by the CLI's `--list-log-targets` and by tests asserting name
/// uniqueness.
pub fn all_targets() -> Vec<TargetInfo> {
    use crate::{
        seeding::pairs::SeedingEvent,
        spacetime_bucket::bucket::SpacetimeBucketEvent,
        topocentric_kf::{
            branching::{
                candidate_search::CandidateSearchEvent, collection::CollectionEvent,
                detection_probability::DetectionProbabilityEvent, discovery::DiscoveryEvent,
                llr_score::LlrScoreEvent, orchestrate::OrchestrateEvent, pruning::PruningEvent,
                visit::VisitEvent,
            },
            kalman_bank::{
                BankEvent, ellipse_region_finder::EllipseRegionEvent, from_seeds::BankBuildEvent,
                seed_grid::SeedGridEvent,
            },
            single_kalman::{init::InitEvent, propagate::PropagationEvent, update::UpdateEvent},
        },
    };

    let mut targets = vec![
        TargetInfo::of::<CollectionEvent>(),
        TargetInfo::of::<OrchestrateEvent>(),
        TargetInfo::of::<DiscoveryEvent>(),
        TargetInfo::of::<PruningEvent>(),
        TargetInfo::of::<VisitEvent>(),
        TargetInfo::of::<CandidateSearchEvent>(),
        TargetInfo::of::<LlrScoreEvent>(),
        TargetInfo::of::<DetectionProbabilityEvent>(),
        TargetInfo::of::<BankEvent>(),
        TargetInfo::of::<EllipseRegionEvent>(),
        TargetInfo::of::<SeedGridEvent>(),
        TargetInfo::of::<BankBuildEvent>(),
        TargetInfo::of::<InitEvent>(),
        TargetInfo::of::<PropagationEvent>(),
        TargetInfo::of::<UpdateEvent>(),
        TargetInfo::of::<SeedingEvent>(),
        TargetInfo::of::<SpacetimeBucketEvent>(),
    ];
    targets.sort_by_key(|t| t.name);
    targets
}

/// Build a `tracing_subscriber::EnvFilter`-compatible directive string from a
/// default level plus per-target overrides, e.g.
/// `"info,propagation=trace,update=debug"`.
pub fn build_env_filter_directive(
    default_level: LogLevel,
    targets: &BTreeMap<String, LogLevel>,
) -> String {
    let mut directives = vec![default_level.to_string()];
    for (target, level) in targets {
        directives.push(format!("{target}={level}"));
    }
    directives.join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directive_with_no_overrides_is_just_the_default_level() {
        assert_eq!(
            build_env_filter_directive(LogLevel::Info, &BTreeMap::new()),
            "info"
        );
    }

    #[test]
    fn directive_combines_default_and_overrides_in_sorted_order() {
        let mut targets = BTreeMap::new();
        targets.insert("update".to_string(), LogLevel::Debug);
        targets.insert("propagation".to_string(), LogLevel::Trace);

        assert_eq!(
            build_env_filter_directive(LogLevel::Warn, &targets),
            "warn,propagation=trace,update=debug"
        );
    }

    #[test]
    fn directive_is_a_valid_env_filter() {
        let mut targets = BTreeMap::new();
        targets.insert("bank".to_string(), LogLevel::Trace);
        let directive = build_env_filter_directive(LogLevel::Info, &targets);

        tracing_subscriber::EnvFilter::try_new(&directive)
            .expect("directive built by build_env_filter_directive must be a valid EnvFilter");
    }

    #[test]
    fn every_target_name_is_unique() {
        let targets = all_targets();
        let mut names: Vec<&str> = targets.iter().map(|t| t.name).collect();
        let n_total = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(
            names.len(),
            n_total,
            "duplicate tracing target name found in the registry"
        );
    }

    #[test]
    fn registry_is_not_empty() {
        assert!(!all_targets().is_empty());
    }

    /// End-to-end check that a directive built by [`build_env_filter_directive`]
    /// actually gates events by *both* target and level when installed as a
    /// real `tracing_subscriber` filter — not just that the string looks
    /// right. Exercises the real `.emit()` API (not raw `tracing::` macros)
    /// on two different modules' event enums.
    #[test]
    fn end_to_end_filtering_respects_target_and_level_overrides() {
        use std::sync::{Arc, Mutex};
        use tracing_subscriber::layer::SubscriberExt;

        use crate::topocentric_kf::{
            kalman_bank::BankEvent, single_kalman::propagate::PropagationEvent,
        };

        struct CapturingLayer {
            seen_targets: Arc<Mutex<Vec<String>>>,
        }

        impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for CapturingLayer {
            fn on_event(
                &self,
                event: &tracing::Event<'_>,
                _ctx: tracing_subscriber::layer::Context<'_, S>,
            ) {
                self.seen_targets
                    .lock()
                    .unwrap()
                    .push(event.metadata().target().to_string());
            }
        }

        let seen_targets = Arc::new(Mutex::new(Vec::new()));
        let layer = CapturingLayer {
            seen_targets: seen_targets.clone(),
        };

        // Default level "warn" (too coarse for either event below), but
        // "bank" is explicitly overridden to "trace".
        let mut targets = BTreeMap::new();
        targets.insert("bank".to_string(), LogLevel::Trace);
        let directive = build_env_filter_directive(LogLevel::Warn, &targets);
        let filter = tracing_subscriber::EnvFilter::try_new(&directive).unwrap();

        let subscriber = tracing_subscriber::registry().with(filter).with(layer);

        tracing::subscriber::with_default(subscriber, || {
            // target "bank" @ trace: passes, since "bank" is overridden to trace.
            BankEvent::GateOk { hyp_id: 1, d2: 2.0 }.emit();
            // target "propagation" @ trace: filtered out, default level is "warn".
            PropagationEvent::Complete.emit();
        });

        let seen_targets = seen_targets.lock().unwrap();
        assert!(seen_targets.iter().any(|t| t == "bank"));
        assert!(!seen_targets.iter().any(|t| t == "propagation"));
    }
}
