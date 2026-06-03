// Nightly pipeline: backfill → backfill-outcomes → backfill-macro →
// backfill-coinalyze → backfill-taker → retrain, with a lifecycle row in
// `cron_runs`. Fail-fast: any hard stage error aborts the chain.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use serde_json::{json, Map, Value};
use sqlx::PgPool;
use tracing::warn;

use crate::cli::{BackfillArgs, BackfillMacroArgs, BackfillOutcomesArgs, RetrainArgs};
use crate::config::Config;
use crate::{backfill, backfill_macro, coinalyze, okx_taker_volume, outcome_backfill, retrain};

struct Recorder<'a> {
    pool: &'a PgPool,
    id: Option<i64>,
    stages: Vec<Value>,
}

impl<'a> Recorder<'a> {
    async fn start(pool: &'a PgPool, command: &str) -> Self {
        let host = std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".into());
        let id = sqlx::query_scalar::<_, i64>(
            "INSERT INTO cron_runs (command, host) VALUES ($1, $2) RETURNING id",
        )
        .bind(command)
        .bind(&host)
        .fetch_one(pool)
        .await
        .map_err(|e| warn!("cron_runs insert failed, continuing without ledger row: {e}"))
        .ok();
        Self { pool, id, stages: Vec::new() }
    }

    fn record_ok(&mut self, name: &str, elapsed_ms: u64) {
        self.stages.push(json!({
            "name": name,
            "status": "ok",
            "duration_ms": elapsed_ms,
        }));
    }

    fn record_err(&mut self, name: &str, elapsed_ms: u64, err: &anyhow::Error) {
        self.stages.push(json!({
            "name": name,
            "status": "error",
            "duration_ms": elapsed_ms,
            "error": format!("{err:#}"),
        }));
    }

    async fn finish(self, exit_code: i32) {
        let Some(id) = self.id else { return };
        let mut summary = Map::new();
        summary.insert("stages".into(), Value::Array(self.stages));
        if let Err(e) = sqlx::query(
            "UPDATE cron_runs
                SET finished_at = NOW(),
                    exit_code   = $2,
                    summary     = $3
              WHERE id = $1",
        )
        .bind(id)
        .bind(exit_code)
        .bind(Value::Object(summary))
        .execute(self.pool)
        .await
        {
            warn!("cron_runs update failed for id={id}: {e}");
        }
    }
}

pub async fn run(cfg: Arc<Config>) -> Result<()> {
    let pool = crate::db::init_pool(&cfg.database_url).await?;
    let mut rec = Recorder::start(&pool, "cron").await;

    let t = Instant::now();
    if let Err(e) = backfill::run(&pool, BackfillArgs { full: false, days: 275 }).await {
        rec.record_err("backfill", t.elapsed().as_millis() as u64, &e);
        rec.finish(1).await;
        return Err(e);
    }
    rec.record_ok("backfill", t.elapsed().as_millis() as u64);

    let t = Instant::now();
    if let Err(e) = outcome_backfill::run(
        &pool,
        &cfg.cryptos,
        &cfg.gamma_api_url,
        BackfillOutcomesArgs { lookback_days: 185 },
    )
    .await
    {
        rec.record_err("outcome_backfill", t.elapsed().as_millis() as u64, &e);
        rec.finish(1).await;
        return Err(e);
    }
    rec.record_ok("outcome_backfill", t.elapsed().as_millis() as u64);

    // backfill-macro MUST land before retrain so macro_daily has rows for
    // every bucket in the lookback window.
    let t = Instant::now();
    if let Err(e) = backfill_macro::run(&pool, &cfg.fred_api_key, BackfillMacroArgs { days: 190 })
        .await
    {
        rec.record_err("backfill_macro", t.elapsed().as_millis() as u64, &e);
        rec.finish(1).await;
        return Err(e);
    }
    rec.record_ok("backfill_macro", t.elapsed().as_millis() as u64);

    // SOFT FAIL: NaN-tolerant features; a Coinalyze outage must not block retrain.
    let t = Instant::now();
    match coinalyze::run_backfill(&pool, 20).await {
        Ok(()) => rec.record_ok("backfill_coinalyze", t.elapsed().as_millis() as u64),
        Err(e) => {
            warn!("[cron] backfill_coinalyze soft-failed (continuing): {e:#}");
            rec.record_err("backfill_coinalyze", t.elapsed().as_millis() as u64, &e);
        }
    }

    let t = Instant::now();
    match okx_taker_volume::run_backfill(&pool, 190).await {
        Ok(()) => rec.record_ok("backfill_taker", t.elapsed().as_millis() as u64),
        Err(e) => {
            warn!("[cron] backfill_taker soft-failed (continuing): {e:#}");
            rec.record_err("backfill_taker", t.elapsed().as_millis() as u64, &e);
        }
    }

    let t = Instant::now();
    if let Err(e) =
        retrain::run(
            &pool,
            &cfg.cryptos,
            RetrainArgs {
                lookback_days: 180,
                trees: 500,
                sample_weight_half_life_days: retrain::SAMPLE_WEIGHT_HALF_LIFE_DAYS,
            },
        )
        .await
    {
        rec.record_err("retrain", t.elapsed().as_millis() as u64, &e);
        rec.finish(1).await;
        return Err(e);
    }
    rec.record_ok("retrain", t.elapsed().as_millis() as u64);

    rec.finish(0).await;
    Ok(())
}
