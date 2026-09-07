use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

/// The navbar's four counters. They used to be four independent server
/// functions, each a full table scan issued on every homepage mount —
/// `COUNT(DISTINCT lineage_id)` on an unindexed column among them. All four
/// now come out of the in-RAM homepage snapshot in a single round-trip.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq)]
pub struct HomeStats {
    pub n_branches: i64,
    pub n_hypotheses: i64,
    pub n_lineages: i64,
    pub n_archived: i64,
}

/// `None` while the snapshot is still building. See
/// [`crate::homepage::snapshot`].
#[server]
pub async fn get_stats() -> Result<Option<HomeStats>, ServerFnError> {
    Ok(crate::homepage::snapshot::snapshot()
        .await
        .map(|snap| HomeStats {
            n_branches: snap.n_branches,
            n_hypotheses: snap.n_hypotheses,
            n_lineages: snap.n_lineages,
            n_archived: snap.n_archived,
        }))
}

/// The snapshot's build counter, or `None` while the first build is running.
/// The refresh button polls this to tell when a rebuild it requested has
/// actually landed — until it does, the previous snapshot is still served.
#[server]
pub async fn get_snapshot_version() -> Result<Option<u64>, ServerFnError> {
    Ok(crate::homepage::snapshot::snapshot()
        .await
        .map(|snap| snap.version))
}

/// How often to re-check whether the homepage snapshot has finished building.
const WARMUP_POLL_MS: u64 = 1000;

#[component]
fn Stat(label: &'static str, value: String, value_class: &'static str) -> Element {
    rsx! {
        span { class: "flex items-baseline gap-1",
            span { class: "opacity-60", "{label}" }
            span { class: "font-semibold {value_class}", "{value}" }
        }
    }
}

#[component]
pub fn StatsBanner(refresh_token: Signal<u64>) -> Element {
    let mut stats = use_resource(move || async move {
        let _ = refresh_token();
        get_stats().await
    });

    use_effect(move || {
        let warming = matches!(&*stats.read(), Some(Ok(None)));
        if warming {
            spawn(async move {
                crate::sleep_ms(WARMUP_POLL_MS).await;
                stats.restart();
            });
        }
    });

    // One placeholder shape for every not-yet-resolved state, so the navbar
    // does not reflow as the snapshot lands.
    let (branches, hypotheses, lineages, archived) = match &*stats.read() {
        Some(Ok(Some(s))) => (
            s.n_branches.to_string(),
            s.n_hypotheses.to_string(),
            s.n_lineages.to_string(),
            s.n_archived.to_string(),
        ),
        Some(Err(e)) => {
            let msg = format!("Error: {e}");
            return rsx! {
                span { class: "text-sm text-error", "{msg}" }
            };
        }
        _ => (
            "—".to_string(),
            "—".to_string(),
            "—".to_string(),
            "—".to_string(),
        ),
    };

    rsx! {
        div { class: "flex items-center gap-4 text-sm whitespace-nowrap overflow-x-auto",
            Stat { label: "Branches", value: branches, value_class: "text-primary" }
            span { class: "opacity-30", "·" }
            Stat { label: "Hypotheses", value: hypotheses, value_class: "" }
            span { class: "opacity-30", "·" }
            Stat { label: "Lineages", value: lineages, value_class: "text-secondary" }
            span { class: "opacity-30", "·" }
            Stat { label: "Archived", value: archived, value_class: "" }
        }
    }
}
