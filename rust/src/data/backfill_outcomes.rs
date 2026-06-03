// Walks Gamma for resolved Polymarket outcomes and upserts into `window_outcomes`.

use std::time::Duration;

use anyhow::{Context, Result};
use serde::Deserialize;
use sqlx::PgPool;
use tokio::time::sleep;
use tracing::{debug, info, warn};

use crate::cli::BackfillOutcomesArgs;
use crate::db;
use crate::market_resolver::{slug_for, WINDOW_SECS};
use crate::resolution::parse_outcome_prices;
use crate::signal::Symbol;

/// Buffer after window end before attempting Gamma lookup (~7 min settle lag).
const SETTLE_BUFFER_SECS: u64 = 600;

const REQUEST_PACING: Duration = Duration::from_millis(200);
const RATE_LIMIT_BACKOFF: Duration = Duration::from_secs(60);

#[derive(Deserialize)]
struct GammaMarket {
    slug: String,
    #[serde(rename = "outcomePrices")]
    outcome_prices: Option<String>,
    closed: Option<bool>,
}

pub async fn run(
    pool: &PgPool,
    cfg_cryptos: &std::collections::HashSet<Symbol>,
    gamma_base: &str,
    args: BackfillOutcomesArgs,
) -> Result<()> {
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .context("build outcome-backfill HTTP client")?;

    let now_secs = chrono::Utc::now().timestamp();
    let earliest_settled = now_secs - (WINDOW_SECS as i64) - SETTLE_BUFFER_SECS as i64;
    let from_secs = now_secs - (args.lookback_days as i64) * 86_400;
    // Snap both ends to WINDOW_SECS boundaries.
    let from_aligned = (from_secs / WINDOW_SECS as i64) * WINDOW_SECS as i64;
    let to_aligned = (earliest_settled / WINDOW_SECS as i64) * WINDOW_SECS as i64;

    info!(
        "[outcome-backfill] walking {}d: from_ts={} to_ts={} for {} symbols",
        args.lookback_days,
        from_aligned,
        to_aligned,
        cfg_cryptos.len()
    );

    for &sym in cfg_cryptos.iter() {
        let short = sym.short();
        let slug_crypto = sym.as_str();
        let existing = db::queries::existing_window_ts(pool, short, from_aligned, to_aligned)
            .await
            .with_context(|| format!("existing_window_ts for {}", short))?;

        let mut fetched = 0usize;
        let mut skipped_existing = 0usize;
        let mut skipped_not_found = 0usize;
        let mut skipped_ambiguous = 0usize;

        let mut ts = from_aligned;
        while ts <= to_aligned {
            if existing.contains(&ts) {
                skipped_existing += 1;
                ts += WINDOW_SECS as i64;
                continue;
            }

            match fetch_outcome(&http, gamma_base, slug_crypto, ts as u64).await {
                Ok(FetchResult::Resolved { outcome, slug }) => {
                    db::queries::upsert_window_outcome(pool, short, ts, outcome, &slug)
                        .await
                        .with_context(|| format!("upsert {} ts={}", short, ts))?;
                    fetched += 1;
                }
                Ok(FetchResult::NotFound) => {
                    skipped_not_found += 1;
                    debug!("[outcome-backfill] {} ts={} no market", short, ts);
                }
                Ok(FetchResult::Ambiguous { slug, prices }) => {
                    skipped_ambiguous += 1;
                    warn!(
                        "[outcome-backfill] {} slug={} ambiguous prices={:?}",
                        short, slug, prices
                    );
                }
                Ok(FetchResult::NotYetClosed) => {
                    debug!("[outcome-backfill] {} ts={} not yet closed", short, ts);
                }
                Err(FetchError::RateLimited) => {
                    warn!(
                        "[outcome-backfill] 429 — backing off {}s",
                        RATE_LIMIT_BACKOFF.as_secs()
                    );
                    sleep(RATE_LIMIT_BACKOFF).await;
                    continue; // retry same ts
                }
                Err(FetchError::Other(e)) => {
                    warn!("[outcome-backfill] {} ts={} error: {:#}", short, ts, e);
                }
            }

            ts += WINDOW_SECS as i64;
            sleep(REQUEST_PACING).await;
        }

        info!(
            "[outcome-backfill:{}] fetched={} existing={} not_found={} ambiguous={}",
            short, fetched, skipped_existing, skipped_not_found, skipped_ambiguous
        );
    }

    Ok(())
}

enum FetchResult {
    Resolved { outcome: i16, slug: String },
    NotFound,
    NotYetClosed,
    Ambiguous { slug: String, prices: Vec<f64> },
}

enum FetchError {
    RateLimited,
    Other(anyhow::Error),
}

async fn fetch_outcome(
    http: &reqwest::Client,
    gamma_base: &str,
    crypto: &str,
    window_ts: u64,
) -> Result<FetchResult, FetchError> {
    let slug = slug_for(crypto, window_ts);
    let url = format!("{}/markets?slug={}&closed=true", gamma_base, slug);
    let resp = http
        .get(&url)
        .send()
        .await
        .map_err(|e| FetchError::Other(e.into()))?;

    if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
        return Err(FetchError::RateLimited);
    }
    if !resp.status().is_success() {
        return Err(FetchError::Other(anyhow::anyhow!(
            "Gamma returned {} for slug={}",
            resp.status(),
            slug
        )));
    }
    let markets: Vec<GammaMarket> = resp
        .json()
        .await
        .map_err(|e| FetchError::Other(e.into()))?;

    let Some(market) = markets.into_iter().find(|m| m.slug == slug) else {
        return Ok(FetchResult::NotFound);
    };
    if !market.closed.unwrap_or(false) {
        return Ok(FetchResult::NotYetClosed);
    }

    let prices = parse_outcome_prices(&market.outcome_prices.unwrap_or_default());
    if prices.len() < 2 {
        return Ok(FetchResult::Ambiguous { slug, prices });
    }
    if prices[0] > 0.5 && prices[1] < 0.5 {
        Ok(FetchResult::Resolved { outcome: 1, slug })
    } else if prices[1] > 0.5 && prices[0] < 0.5 {
        Ok(FetchResult::Resolved { outcome: 2, slug })
    } else {
        Ok(FetchResult::Ambiguous { slug, prices })
    }
}
