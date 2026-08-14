pub mod homepage;
pub mod lineage_page;

use dioxus::prelude::*;

#[cfg(feature = "server")]
use sqlx::postgres::PgPoolOptions;
#[cfg(feature = "server")]
use sqlx::PgPool;
#[cfg(feature = "server")]
use tokio::sync::OnceCell;

use crate::homepage::Home;
use crate::lineage_page::LineagePage;

#[cfg(feature = "server")]
static DB_POOL: OnceCell<PgPool> = OnceCell::const_new();

#[cfg(feature = "server")]
async fn get_pool() -> &'static PgPool {
    DB_POOL
        .get_or_init(|| async {
            dotenvy::dotenv().ok();

            let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");

            PgPoolOptions::new()
                .max_connections(5)
                .connect(&database_url)
                .await
                .expect("Failed to connect to Postgres")
        })
        .await
}

/// The engine's full validated configuration (same `engine_config.best.yaml`
/// shape the main pipeline loads), read once from `KALMAN_CONFIG_PATH`. The
/// lineage replay needs more than just the Kalman tuning from this file —
/// it also needs `seeding_grid_config`/`kfbank_config`/`advance_params` to
/// replicate the production multi-hypothesis bank rather than a single
/// circular-orbit guess — so the whole `EngineConfig` is cached here instead
/// of discarding everything but the derived `KalmanContext`. The path is
/// required; there is no hard-coded fallback.
#[cfg(feature = "server")]
static ENGINE_CONFIG: OnceCell<fink_fat_engine::engine_config::EngineConfig> =
    OnceCell::const_new();

#[cfg(feature = "server")]
async fn get_engine_config() -> &'static fink_fat_engine::engine_config::EngineConfig {
    ENGINE_CONFIG
        .get_or_init(|| async {
            dotenvy::dotenv().ok();

            let config_path =
                std::env::var("KALMAN_CONFIG_PATH").expect("KALMAN_CONFIG_PATH must be set");

            tokio::task::spawn_blocking(move || {
                use fink_fat_engine::engine_config::EngineConfig;

                EngineConfig::load_engine_config_validated(camino::Utf8Path::new(&config_path))
                    .expect("failed to load/validate KALMAN_CONFIG_PATH")
            })
            .await
            .expect("EngineConfig load task panicked")
        })
        .await
}

/// Shared Kalman filter runtime (JPL ephemeris + UT1 provider, expensive to
/// load) derived from [`get_engine_config`], used by every hypothesis in the
/// lineage replay's bank.
#[cfg(feature = "server")]
static KALMAN_CONTEXT: OnceCell<fink_fat_engine::engine_config::kalman_context::KalmanContext> =
    OnceCell::const_new();

#[cfg(feature = "server")]
async fn get_kalman_context() -> &'static fink_fat_engine::engine_config::kalman_context::KalmanContext
{
    KALMAN_CONTEXT
        .get_or_init(|| async {
            let engine_config = get_engine_config().await;
            tokio::task::spawn_blocking(|| engine_config.build_context())
                .await
                .expect("KalmanContext build task panicked")
        })
        .await
}

/// MPC observatory code -> `Observer` lookup table, fetched once at
/// startup. Resolved explicitly here (rather than relying on
/// `photom::ObsDataset`'s own lazy per-lookup MPC-catalogue fetch) so a
/// resolution failure surfaces as a clear panic/log at startup instead of a
/// confusing `ObserverIdIsNone` deep inside a replay.
#[cfg(feature = "server")]
static OBSERVATORIES: OnceCell<photom::observer::mpc::MpcCodeObs> = OnceCell::const_new();

#[cfg(feature = "server")]
async fn get_observatories() -> &'static photom::observer::mpc::MpcCodeObs {
    OBSERVATORIES
        .get_or_init(|| async {
            tokio::task::spawn_blocking(|| {
                photom::observer::mpc::init_observatories(&Default::default())
            })
            .await
            .expect("observatory lookup task panicked")
            .expect("failed to fetch/parse the MPC observatory list")
        })
        .await
}

#[derive(Clone, Debug, PartialEq, Routable)]
enum Route {
    #[route("/")]
    Home {},

    #[route("/lineage/:lineage_id")]
    LineagePage { lineage_id: String },
}

fn main() {
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    rsx! {
        document::Stylesheet { href: asset!("/assets/main.css") }

        Router::<Route> {}
    }
}
