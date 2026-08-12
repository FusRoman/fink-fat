use std::{future::Future, pin::Pin};

use dioxus::prelude::*;

#[cfg(feature = "server")]
use crate::get_pool;

#[cfg(feature = "server")]
pub async fn count_query(sql: &'static str) -> Result<i64, ServerFnError> {
    let pool = get_pool().await;

    let row: (i64,) = sqlx::query_as(sql)
        .fetch_one(pool)
        .await
        .map_err(|e| ServerFnError::new(e.to_string()))?;

    Ok(row.0)
}

#[server]
pub async fn get_branch_count() -> Result<i64, ServerFnError> {
    count_query("SELECT COUNT(*) FROM branches").await
}

#[server]
pub async fn get_archived_count() -> Result<i64, ServerFnError> {
    count_query("SELECT COUNT(*) FROM archived_trajectories").await
}

#[server]
pub async fn get_hypothesis_count() -> Result<i64, ServerFnError> {
    count_query("SELECT COUNT(*) FROM hypotheses").await
}

#[server]
pub async fn get_unique_lineage_count() -> Result<i64, ServerFnError> {
    count_query("SELECT COUNT(DISTINCT lineage_id) FROM branches").await
}

type CountFuture = Pin<Box<dyn Future<Output = Result<i64, ServerFnError>>>>;

#[derive(Props, Clone, PartialEq)]
pub struct CountProps {
    fetcher: fn() -> CountFuture,
}

#[component]
pub fn Count(props: CountProps) -> Element {
    let fetcher = props.fetcher;
    let sql_ressource = use_resource(move || fetcher());
    let count = &*sql_ressource.read();

    match count {
        Some(Ok(n)) => rsx! {
            p { "{n}" }
        },
        Some(Err(e)) => rsx! {
            p { "Error : {e}" }
        },
        None => rsx! {
            p { "Loading..." }
        },
    }
}

#[component]
pub fn StatsBanner() -> Element {
    // Compact inline stats, meant to live in the navbar.
    rsx! {
        div { class: "flex items-center gap-4 text-sm whitespace-nowrap overflow-x-auto",
            span { class: "flex items-baseline gap-1",
                span { class: "opacity-60", "Branches" }
                span { class: "font-semibold text-primary",
                    Count { fetcher: || Box::pin(get_branch_count()) }
                }
            }
            span { class: "opacity-30", "·" }
            span { class: "flex items-baseline gap-1",
                span { class: "opacity-60", "Hypotheses" }
                span { class: "font-semibold",
                    Count { fetcher: || Box::pin(get_hypothesis_count()) }
                }
            }
            span { class: "opacity-30", "·" }
            span { class: "flex items-baseline gap-1",
                span { class: "opacity-60", "Lineages" }
                span { class: "font-semibold text-secondary",
                    Count { fetcher: || Box::pin(get_unique_lineage_count()) }
                }
            }
            span { class: "opacity-30", "·" }
            span { class: "flex items-baseline gap-1",
                span { class: "opacity-60", "Archived" }
                span { class: "font-semibold",
                    Count { fetcher: || Box::pin(get_archived_count()) }
                }
            }
        }
    }
}
