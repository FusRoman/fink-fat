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
