// `polybot` binary entry point — parses the clap sub-command and dispatches.

use std::sync::Arc;

use anyhow::Result;
use clap::Parser;

use polybot::cli::{Args, Command};
use polybot::config::Config;
use polybot::{
    backfill, backfill_macro, coinalyze, cron, db, okx_taker_volume, outcome_backfill, retrain, run,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Explicitly pick a rustls crypto provider — multiple crates in the tree
    // enable different backends, causing rustls 0.23+ to refuse auto-select.
    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .expect("failed to install rustls aws-lc-rs crypto provider");

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    // Abort the whole process on any panic so the process supervisor can restart it.
    // Default tokio behavior silently kills only the panicking task.
    std::panic::set_hook(Box::new(|info| {
        tracing::error!("PANIC: {}", info);
        std::process::exit(1);
    }));

    let args = Args::parse();
    let cfg = Arc::new(Config::from_env()?);

    match args.command {
        Command::Run => run::run(cfg).await,
        Command::Backfill(a) => {
            let pool = db::init_pool(&cfg.database_url).await?;
            backfill::run(&pool, a).await
        }
        Command::Retrain(a) => {
            let pool = db::init_pool(&cfg.database_url).await?;
            retrain::run(&pool, &cfg.cryptos, a).await
        }
        Command::BackfillOutcomes(a) => {
            let pool = db::init_pool(&cfg.database_url).await?;
            outcome_backfill::run(&pool, &cfg.cryptos, &cfg.gamma_api_url, a).await
        }
        Command::BackfillMacro(a) => {
            let pool = db::init_pool(&cfg.database_url).await?;
            backfill_macro::run(&pool, &cfg.fred_api_key, a).await
        }
        Command::BackfillCoinalyze(a) => {
            let pool = db::init_pool(&cfg.database_url).await?;
            coinalyze::run_backfill(&pool, a.days).await
        }
        Command::BackfillTaker(a) => {
            let pool = db::init_pool(&cfg.database_url).await?;
            okx_taker_volume::run_backfill(&pool, a.days).await
        }
        Command::Cron => cron::run(cfg).await,
    }
}
