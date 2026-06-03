// src/db/mod.rs — Postgres connection pool and schema bootstrap.

use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;
use anyhow::Result;
use std::time::Duration;

pub mod models;
pub mod notify;
pub mod queries;

pub async fn init_pool(database_url: &str) -> Result<PgPool> {
    // idle_timeout < proxy idle window to avoid RST'd sockets on re-acquire.
    let pool = PgPoolOptions::new()
        .max_connections(10)
        .min_connections(1)
        .acquire_timeout(Duration::from_secs(10))
        .idle_timeout(Duration::from_secs(120))
        .max_lifetime(Duration::from_secs(15 * 60))
        .test_before_acquire(true)
        .connect(database_url)
        .await?;

    sqlx::migrate!("./src/migrations").run(&pool).await?;

    Ok(pool)
}