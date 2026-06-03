// First-deploy self-bootstrap: if any configured symbol lacks a model row,
// run backfill + retrain in-process before entering the live loop.

use std::collections::HashSet;

use anyhow::{Context, Result};
use sqlx::PgPool;
use tracing::info;

use crate::backfill;
use crate::backfill_macro;
use crate::cli::{BackfillArgs, BackfillMacroArgs, RetrainArgs};
use crate::config::Config;
use crate::db;
use crate::inference::model_hub::CURRENT_FORMAT_VERSION;
use crate::retrain;
use crate::signal::Symbol;

/// Returns true when at least one configured symbol has no current model row.
pub fn needs_bootstrap(configured: &HashSet<Symbol>, present: &HashSet<String>) -> bool {
    configured.iter().any(|s| !present.contains(s.short()))
}

pub async fn bootstrap_if_needed(
    pool: &PgPool,
    cryptos: &HashSet<Symbol>,
    cfg: &Config,
) -> Result<()> {
    if !cfg.bootstrap_on_empty {
        info!("[bootstrap] BOOTSTRAP_ON_EMPTY=false — skipping");
        return Ok(());
    }

    let present = db::queries::load_current_model_symbols(pool, CURRENT_FORMAT_VERSION)
        .await
        .context("bootstrap: load_current_model_symbols")?;

    let total = cryptos.len();
    let have = cryptos.iter().filter(|s| present.contains(s.short())).count();

    if !needs_bootstrap(cryptos, &present) {
        info!("[bootstrap] all models present — skipping bootstrap");
        return Ok(());
    }

    let missing = total - have;
    info!(
        "[bootstrap] {}/{} models missing — running backfill + retrain",
        missing, total
    );

    backfill::run(pool, BackfillArgs { full: true, days: 120 })
        .await
        .context("bootstrap: backfill")?;
    // backfill_macro MUST land before retrain so macro_daily has rows for
    // every bucket in the lookback window.
    backfill_macro::run(pool, &cfg.fred_api_key, BackfillMacroArgs { days: 130 })
        .await
        .context("bootstrap: backfill_macro")?;
    retrain::run(
        pool,
        cryptos,
        RetrainArgs {
            lookback_days: 90,
            trees: 500,
            sample_weight_half_life_days: retrain::SAMPLE_WEIGHT_HALF_LIFE_DAYS,
        },
    )
    .await
    .context("bootstrap: retrain")?;

    info!("[bootstrap] self-bootstrap complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_four() -> HashSet<Symbol> {
        [
            Symbol::Btc,
            Symbol::Eth,
            Symbol::Sol,
            Symbol::Xrp,
        ]
        .into_iter()
        .collect()
    }

    #[test]
    fn test_needs_bootstrap_when_empty() {
        let configured = all_four();
        let present: HashSet<String> = HashSet::new();
        assert!(needs_bootstrap(&configured, &present));
    }

    #[test]
    fn test_needs_bootstrap_partial() {
        let configured = all_four();
        let present: HashSet<String> = ["BTC", "ETH", "SOL"]
            .into_iter()
            .map(String::from)
            .collect();
        assert!(needs_bootstrap(&configured, &present));
    }

    #[test]
    fn test_no_bootstrap_when_all_present() {
        let configured = all_four();
        let present: HashSet<String> = ["BTC", "ETH", "SOL", "XRP"]
            .into_iter()
            .map(String::from)
            .collect();
        assert!(!needs_bootstrap(&configured, &present));
    }

    #[test]
    fn test_skip_when_all_models_present() {
        let configured = all_four();
        let present: HashSet<String> = configured.iter().map(|s| s.short().to_string()).collect();
        assert!(!needs_bootstrap(&configured, &present));
    }
}
