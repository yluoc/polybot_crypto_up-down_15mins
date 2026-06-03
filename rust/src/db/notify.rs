// src/db/notify.rs — PgListener wrapper for the `model_updated` channel.

use anyhow::{Context, Result};
use sqlx::postgres::PgListener;
use sqlx::PgPool;

pub const CHANNEL: &str = "model_updated";

/// Listens for `NOTIFY model_updated` and returns the symbol payload. Reconnects transparently.
pub struct ModelUpdateListener {
    inner: PgListener,
}

impl ModelUpdateListener {
    pub async fn connect(pool: &PgPool) -> Result<Self> {
        let mut inner = PgListener::connect_with(pool)
            .await
            .context("PgListener::connect_with failed")?;
        inner
            .listen(CHANNEL)
            .await
            .with_context(|| format!("LISTEN {CHANNEL} failed"))?;
        Ok(Self { inner })
    }

    /// Awaits the next notification and returns its payload (symbol).
    pub async fn recv(&mut self) -> Result<String> {
        let n = self.inner.recv().await.context("PgListener::recv failed")?;
        Ok(n.payload().to_string())
    }
}
