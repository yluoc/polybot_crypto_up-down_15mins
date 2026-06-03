// Clap sub-command dispatch for the `polybot` binary.

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name    = "polybot",
    version,
    about   = "15-minute Polymarket crypto trader — single Rust binary for live inference, retraining, and candle backfill"
)]
pub struct Args {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Live trading: OKX WS → features → inference → Trader. Loads current
    /// model per symbol from Postgres and hot-reloads on NOTIFY model_updated.
    Run,

    /// Top up the `candles` table from OKX REST history. Incremental by
    /// default — only fetches candles newer than the max ts_ms already in
    /// the table per instrument. Use `--full` for a blank-slate fill.
    Backfill(BackfillArgs),

    /// Train a per-symbol LightGBM model from stored candles and promote.
    /// One call produces and promotes up to 7 models (one per configured
    /// crypto). A live `polybot run` picks up each promotion via NOTIFY.
    Retrain(RetrainArgs),

    /// Walk Gamma for Polymarket-resolved outcomes of historical 15-min
    /// windows and upsert into `window_outcomes`. Runs nightly before
    /// retrain so labels come from Polymarket's actual settled outcomes
    /// rather than OKX-derived pct-change.
    BackfillOutcomes(BackfillOutcomesArgs),

    /// Pull the configured FRED daily macro series (DXY/SPX/VIX/yields) for
    /// the last `--days` and upsert into `macro_daily`. Idempotent; runs
    /// nightly BEFORE retrain so the new global feature block has values
    /// for every bucket in the lookback window.
    BackfillMacro(BackfillMacroArgs),

    /// v14 (Phase 2 audit #5): pull aggregated OI + liquidation history
    /// from Coinalyze for the last `--days` and upsert into
    /// `open_interest_aggregated` / `liquidations_aggregated`. Requires
    /// COINALYZE_API_KEY env var. Coinalyze retention floor is ~20 days
    /// — this command pulls only what's available; the local tables
    /// accumulate forever (the table grows 1d/day from first deploy
    /// until the 180-day training window is fully covered).
    BackfillCoinalyze(BackfillCoinalyzeArgs),

    /// v14: pull OKX taker buy/sell volume for the last `--days` and
    /// upsert into `taker_volume_15m`. Aggregates the underlying 5m
    /// OKX response into 15-min buckets matching `CANDLE_INTERVAL_MS`.
    /// No env var required (public endpoint).
    BackfillTaker(BackfillTakerArgs),

    /// Nightly chain: backfill → backfill-outcomes → backfill-macro →
    /// backfill-coinalyze → backfill-taker → retrain as a single
    /// in-process pipeline, with a lifecycle ledger row in `cron_runs`.
    /// Invoked by `ops/crontab`; safe to run manually via `fly ssh
    /// console`. Fail-fast: a failing stage aborts the chain with
    /// non-zero exit code, EXCEPT backfill-coinalyze and backfill-taker
    /// which are SOFT FAILS (warned but skipped) — the v14 micro
    /// features are NaN-tolerant by design, so a Coinalyze outage or
    /// missing API key must not block retrain.
    Cron,
}

#[derive(clap::Args, Debug)]
pub struct BackfillArgs {
    /// Ignore existing candles and page back until OKX returns no more data.
    #[arg(long)]
    pub full: bool,

    /// Full-fill horizon cap in days. Ignored in incremental mode.
    #[arg(long, default_value_t = 180)]
    pub days: u32,
}

#[derive(clap::Args, Debug)]
pub struct RetrainArgs {
    /// Training window in days — only candles newer than now - N go into
    /// the feature matrix.
    #[arg(long, default_value_t = 180)]
    pub lookback_days: u32,

    /// Number of boosting iterations (cap when early stopping is active —
    /// the booster will usually stop well before this when val logloss plateaus).
    #[arg(long, default_value_t = 500)]
    pub trees: u32,

    /// Phase 3 audit #7: exponential sample-weight decay half-life in
    /// days. Weight(row) = 0.5 ^ ((union_max_ts − row_ts) / half_life).
    /// Default 45.0 is the midpoint of the audit's 30–60d band and the
    /// A/B-rollout midpoint. Clamped to [10.0, 365.0] in `run()` — see
    /// `SAMPLE_WEIGHT_HALF_LIFE_{MIN,MAX}` in retrain.rs and the
    /// weighted/unweighted matrix in `docs/phase3_7_sample_weight_decay.md`.
    #[arg(
        long,
        env = "POLYBOT_SAMPLE_WEIGHT_HALF_LIFE",
        default_value_t = crate::retrain::SAMPLE_WEIGHT_HALF_LIFE_DAYS
    )]
    pub sample_weight_half_life_days: f64,
}

#[derive(clap::Args, Debug)]
pub struct BackfillOutcomesArgs {
    /// How many days back to walk. Default 185 = 180d retrain window + 5d slack.
    #[arg(long, default_value_t = 185)]
    pub lookback_days: u32,
}

#[derive(clap::Args, Debug)]
pub struct BackfillMacroArgs {
    /// How many days back to fetch from FRED. Default 190 ≥ 180d retrain
    /// window + 10d margin so the earliest retrain bucket has a `prev`
    /// observation (see state_to_snapshot invariant in macro_poll.rs).
    #[arg(long, default_value_t = 190)]
    pub days: u32,
}

#[derive(clap::Args, Debug)]
pub struct BackfillCoinalyzeArgs {
    /// How many days back to fetch from Coinalyze. Default 20 matches
    /// the documented free-tier intraday retention; the call still
    /// succeeds for larger values but returns at most ~20 days of
    /// history. Set lower than 20 only for testing.
    #[arg(long, default_value_t = 20)]
    pub days: u32,
}

#[derive(clap::Args, Debug)]
pub struct BackfillTakerArgs {
    /// How many days back to fetch OKX taker volume. Default 190
    /// matches BackfillMacro so the v14 trailing 180-day window is
    /// fully covered for retrain replay.
    #[arg(long, default_value_t = 190)]
    pub days: u32,
}
