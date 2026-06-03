// Background task: polls open positions for Polymarket market resolution.

use anyhow::{bail, Result};
use chrono::Utc;
use futures::FutureExt;
use serde::Deserialize;
use sqlx::PgPool;
use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::time::{Duration, Instant};
use tracing::{info, warn};

/// Best-effort extraction of a panic message from the `Box<dyn Any>` payload
/// that `catch_unwind` hands back.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

use crate::db::queries;
use crate::market_resolver::{de_stringified_vec, slug_prefix, slug_timestamp, WINDOW_SECS};

/// Record an error in the errors table. Non-fatal — if the DB is down we just log.
async fn record_error(pool: &PgPool, context: &str, detail: &str) {
    if let Err(e) = queries::insert_error(pool, "rust", context, detail).await {
        warn!("[resolution] failed to record error in DB: {:#}", e);
    }
}

/// Seconds after window end before we start checking resolution.
const SETTLE_BUFFER_SECS: u64 = 600;

/// How often the background task runs.
const CHECK_INTERVAL_SECS: u64 = 60;

/// Seconds after window end at which we give up and treat the market as unresolvable.
const UNRESOLVABLE_AFTER_SECS: u64 = 7_200;

/// Sentinel stored in `dry_run_results.actual_outcome` when Gamma never returned a matching market.
const UNRESOLVABLE_OUTCOME: i16 = -1;

/// Backoff duration after Gamma returns 429; shared between live and dry-run resolvers.
const RATE_LIMIT_BACKOFF_SECS: u64 = 120;

/// A 429 in either task pauses both until the backoff expires.
static GAMMA_BACKOFF: std::sync::Mutex<Option<Instant>> = std::sync::Mutex::new(None);

fn in_backoff() -> bool {
    GAMMA_BACKOFF
        .lock()
        .ok()
        .and_then(|g| *g)
        .map(|t| t.elapsed() < Duration::from_secs(RATE_LIMIT_BACKOFF_SECS))
        .unwrap_or(false)
}

fn mark_rate_limited() {
    if let Ok(mut g) = GAMMA_BACKOFF.lock() {
        *g = Some(Instant::now());
    }
}

#[derive(Debug, Deserialize)]
struct GammaMarket {
    slug: String,
    #[serde(rename = "clobTokenIds", default, deserialize_with = "de_stringified_vec")]
    clob_token_ids: Option<Vec<String>>,
    closed: Option<bool>,
    #[serde(rename = "outcomePrices")]
    outcome_prices: Option<String>,
}

/// Spawn the resolution checker as a background tokio task.
pub fn spawn(pool: PgPool, http: reqwest::Client, gamma_base: String) {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_secs(CHECK_INTERVAL_SECS));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        loop {
            ticker.tick().await;
            if in_backoff() {
                continue;
            }
            let result = AssertUnwindSafe(check_open_positions(&pool, &http, &gamma_base))
                .catch_unwind()
                .await;
            match result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    warn!("[resolution] check failed: {:#}", e);
                    record_error(&pool, "resolution_check", &format!("{:#}", e)).await;
                }
                Err(panic) => {
                    let msg = panic_message(&*panic);
                    warn!("[resolution] tick panicked: {}", msg);
                    record_error(&pool, "resolution_panic", &msg).await;
                }
            }
        }
    });
}

async fn check_open_positions(
    pool: &PgPool,
    http: &reqwest::Client,
    gamma_base: &str,
) -> Result<()> {
    let positions = queries::get_all_open_positions(pool).await?;
    if positions.is_empty() {
        return Ok(());
    }

    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    for pos in &positions {
        let window_ts = slug_timestamp(&pos.market_id);
        if window_ts == 0 {
            let msg = format!("cannot extract window_ts from slug '{}'", pos.market_id);
            warn!("[resolution] {}", msg);
            record_error(pool, "resolution_slug", &msg).await;
            continue;
        }

        // Only check after window has ended + settlement buffer
        let window_end = window_ts + WINDOW_SECS;
        if now_secs < window_end + SETTLE_BUFFER_SECS {
            continue;
        }

        match resolve_market(http, gamma_base, &pos.market_id, &pos.token_id).await {
            Ok(Some(exit_price)) => {
                let now = Utc::now();
                let shares = if pos.avg_entry_price > 0.0 {
                    pos.usdc / pos.avg_entry_price
                } else {
                    0.0
                };
                let exit_value = shares * exit_price;
                let pnl = exit_value - pos.usdc;
                let pnl_pct = if pos.usdc > 0.0 { pnl / pos.usdc } else { 0.0 };

                info!(
                    "[resolution] closing position #{} slug={} entry={:.4} exit={:.4} pnl={:.4} ({:.1}%)",
                    pos.id, pos.market_id, pos.avg_entry_price, exit_price, pnl, pnl_pct * 100.0
                );

                if let Err(e) = queries::close_position(pool, pos.id, now).await {
                    let msg = format!("close_position #{} failed: {:#}", pos.id, e);
                    warn!("[resolution] {}", msg);
                    record_error(pool, "close_position", &msg).await;
                    continue;
                }
                if let Err(e) = queries::insert_trade(
                    pool,
                    None,
                    &pos.market_id,
                    &pos.side,
                    pos.avg_entry_price,
                    exit_price,
                    pos.usdc,
                    pnl,
                    pnl_pct,
                    pos.opened_at,
                    now,
                )
                .await
                {
                    let msg = format!("insert_trade for position #{} failed: {:#}", pos.id, e);
                    warn!("[resolution] {}", msg);
                    record_error(pool, "insert_trade", &msg).await;
                }
            }
            Ok(None) => {
                // Market not resolved yet — warn if overdue
                let age_secs = now_secs - window_end;
                if age_secs > 600 {
                    warn!(
                        "[resolution] position #{} slug={} ended {}s ago, not yet resolved",
                        pos.id, pos.market_id, age_secs
                    );
                }
            }
            Err(e) => {
                let msg = format!("Gamma API check for '{}' failed: {:#}", pos.market_id, e);
                warn!("[resolution] {}", msg);
                record_error(pool, "resolve_market", &msg).await;
            }
        }
    }

    Ok(())
}

/// Query Gamma API for a resolved market.
/// Returns Ok(Some(exit_price)) if closed, Ok(None) if still open, Err on API failure.
async fn resolve_market(
    http: &reqwest::Client,
    gamma_base: &str,
    slug: &str,
    our_token_id: &str,
) -> Result<Option<f64>> {
    // `closed=true` required: Gamma's default query filters out resolved markets.
    let url = format!("{}/markets?slug={}&closed=true", gamma_base, slug);
    let resp = http.get(&url).send().await?;

    if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
        mark_rate_limited();
        bail!("Gamma API returned status 429 Too Many Requests");
    }
    if !resp.status().is_success() {
        bail!("Gamma API returned status {}", resp.status());
    }

    let markets: Vec<GammaMarket> = resp.json().await?;
    let market = match markets.into_iter().find(|m| m.slug == slug) {
        Some(m) => m,
        None => bail!("no market found for slug={}", slug),
    };

    if !market.closed.unwrap_or(false) {
        return Ok(None);
    }

    let token_ids = market.clob_token_ids.unwrap_or_default();
    let prices = parse_outcome_prices(&market.outcome_prices.unwrap_or_default());

    if token_ids.len() < 2 || prices.len() < 2 {
        bail!(
            "resolved market slug={} incomplete: tokens={} prices={}",
            slug,
            token_ids.len(),
            prices.len()
        );
    }

    match token_ids.iter().position(|id| id == our_token_id) {
        Some(idx) => Ok(Some(prices[idx])),
        None => bail!(
            "token_id {} not found in market {} tokens",
            our_token_id,
            slug
        ),
    }
}

/// Parse outcome_prices string from Gamma API. Handles `[1, 0]`, `"1, 0"`, `["1","0"]`.
/// Non-finite values are rejected.
pub fn parse_outcome_prices(raw: &str) -> Vec<f64> {
    raw.trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .filter_map(|s| s.trim().trim_matches('"').parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .collect()
}

/// Result of one walk attempt.
struct WalkResult {
    /// `Some(1|2)` = closed; `None` = still open.
    outcome: Option<i16>,
    cursor: u64,
}

enum ResolveError {
    RateLimited,
    NotFoundYet { cursor: u64 },
    Other(anyhow::Error),
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RateLimited => write!(f, "Gamma API returned status 429 Too Many Requests"),
            Self::NotFoundYet { cursor } => {
                write!(f, "no market found, walked to ts={}", cursor)
            }
            Self::Other(e) => write!(f, "{:#}", e),
        }
    }
}

enum TickOutcome {
    Ok,
    RateLimited,
}

/// Spawn the dry-run resolver. Only started when `DRY_RUN=true`.
pub fn spawn_dry_run(pool: PgPool, http: reqwest::Client, gamma_base: String) {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_secs(CHECK_INTERVAL_SECS));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // signal_id -> next ts to resume the forward-walk from.
        let mut cursors: HashMap<i64, u64> = HashMap::new();

        loop {
            ticker.tick().await;

            if in_backoff() {
                continue;
            }

            let result = AssertUnwindSafe(check_dry_run_signals(
                &pool,
                &http,
                &gamma_base,
                &mut cursors,
            ))
            .catch_unwind()
            .await;

            match result {
                Ok(Ok(TickOutcome::Ok)) => {}
                Ok(Ok(TickOutcome::RateLimited)) => {
                    warn!(
                        "[resolution/dry_run] Gamma rate-limited; backing off for {}s",
                        RATE_LIMIT_BACKOFF_SECS
                    );
                    mark_rate_limited();
                    record_error(
                        &pool,
                        "dry_run_rate_limited",
                        &format!("backing off {}s", RATE_LIMIT_BACKOFF_SECS),
                    )
                    .await;
                }
                Ok(Err(e)) => {
                    warn!("[resolution/dry_run] check failed: {:#}", e);
                    record_error(&pool, "dry_run_check", &format!("{:#}", e)).await;
                }
                Err(panic) => {
                    let msg = panic_message(&*panic);
                    warn!("[resolution/dry_run] tick panicked: {}", msg);
                    record_error(&pool, "dry_run_panic", &msg).await;
                }
            }
        }
    });
}

async fn check_dry_run_signals(
    pool: &PgPool,
    http: &reqwest::Client,
    gamma_base: &str,
    cursors: &mut HashMap<i64, u64>,
) -> Result<TickOutcome> {
    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Eligible once `now > ts_ms/1000 + 2*WINDOW_SECS + buffer` (prediction covers the next window).
    let cutoff_ms = now_secs
        .saturating_sub(2 * WINDOW_SECS + SETTLE_BUFFER_SECS)
        .saturating_mul(1000) as i64;

    let rows = queries::get_unresolved_dry_run_signals(pool, cutoff_ms).await?;
    info!("[resolution/dry_run] tick: {} eligible", rows.len());
    if rows.is_empty() {
        return Ok(TickOutcome::Ok);
    }

    let live_ids: std::collections::HashSet<i64> = rows.iter().map(|r| r.signal_id).collect();
    cursors.retain(|id, _| live_ids.contains(id));

    for row in rows {
        let start_ts = cursors
            .get(&row.signal_id)
            .copied()
            .unwrap_or_else(|| slug_timestamp(&row.market_id));

        match resolve_outcome(http, gamma_base, &row.market_id, start_ts).await {
            Ok(WalkResult { outcome: Some(outcome), .. }) => {
                cursors.remove(&row.signal_id);
                let correct = row.signal == outcome;
                info!(
                    "[resolution/dry_run] signal #{} slug={} predicted={} actual={} {}",
                    row.signal_id, row.market_id, row.signal, outcome,
                    if correct { "HIT" } else { "MISS" }
                );
                if let Err(e) = queries::insert_dry_run_result(
                    pool,
                    row.signal_id,
                    &row.market_id,
                    row.signal,
                    row.confidence,
                    outcome,
                    correct,
                    row.entry_price,
                    row.shares,
                    row.fee_rate_bps,
                    row.skip_reason.as_deref(),
                )
                .await
                {
                    let msg = format!(
                        "insert_dry_run_result signal_id={} failed: {:#}",
                        row.signal_id, e
                    );
                    warn!("[resolution/dry_run] {}", msg);
                    record_error(pool, "insert_dry_run_result", &msg).await;
                }
            }
            Ok(WalkResult { outcome: None, cursor }) => {
                cursors.insert(row.signal_id, cursor);
            }
            Err(ResolveError::RateLimited) => {
                return Ok(TickOutcome::RateLimited);
            }
            Err(ResolveError::NotFoundYet { cursor: _ }) => {
                // Do not persist the cursor: with `closed=true`, an empty response is
                // ambiguous between "not yet listed" and "not yet closed". Re-walk from base_ts.
                cursors.remove(&row.signal_id);
                let window_ts = slug_timestamp(&row.market_id);
                let window_end = window_ts + WINDOW_SECS;
                let age_secs = now_secs.saturating_sub(window_end);
                if window_ts > 0 && age_secs > UNRESOLVABLE_AFTER_SECS {
                    warn!(
                        "[resolution/dry_run] signal #{} slug={} unresolvable after {}s — \
                         recording sentinel so it stops being re-polled",
                        row.signal_id, row.market_id, age_secs
                    );
                    if let Err(e) = queries::insert_dry_run_result(
                        pool,
                        row.signal_id,
                        &row.market_id,
                        row.signal,
                        row.confidence,
                        UNRESOLVABLE_OUTCOME,
                        false,
                        row.entry_price,
                        row.shares,
                        row.fee_rate_bps,
                        row.skip_reason.as_deref(),
                    )
                    .await
                    {
                        let msg = format!(
                            "insert_dry_run_result (unresolvable) signal_id={} failed: {:#}",
                            row.signal_id, e
                        );
                        warn!("[resolution/dry_run] {}", msg);
                        record_error(pool, "insert_dry_run_result", &msg).await;
                    }
                    record_error(
                        pool,
                        "unresolvable_market",
                        &format!(
                            "dry-run signal #{} slug={} not found on Gamma after {}s",
                            row.signal_id, row.market_id, age_secs
                        ),
                    )
                    .await;
                }
            }
            Err(ResolveError::Other(e)) => {
                let msg = format!(
                    "Gamma outcome lookup for '{}' failed: {:#}",
                    row.market_id, e
                );
                warn!("[resolution/dry_run] {}", msg);
                record_error(pool, "dry_run_resolve", &msg).await;
            }
        }
    }

    Ok(TickOutcome::Ok)
}

/// Walk the slug-ts forward from `start_ts` until a matching closed market is found.
async fn resolve_outcome(
    http: &reqwest::Client,
    gamma_base: &str,
    slug: &str,
    start_ts: u64,
) -> std::result::Result<WalkResult, ResolveError> {
    let base_ts = slug_timestamp(slug);
    if base_ts == 0 {
        return Err(ResolveError::Other(anyhow::anyhow!(
            "could not parse timestamp from slug={}",
            slug
        )));
    }
    let prefix = slug_prefix(slug);
    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut ts = start_ts.max(base_ts);
    loop {
        let current_slug = format!("{}-{}", prefix, ts);
        // `closed=true` required: Gamma hides resolved markets from the default slug query.
        let url = format!("{}/markets?slug={}&closed=true", gamma_base, current_slug);
        let resp = http
            .get(&url)
            .send()
            .await
            .map_err(|e| ResolveError::Other(e.into()))?;
        if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(ResolveError::RateLimited);
        }
        if !resp.status().is_success() {
            return Err(ResolveError::Other(anyhow::anyhow!(
                "Gamma API returned status {}",
                resp.status()
            )));
        }
        let markets: Vec<GammaMarket> = resp
            .json()
            .await
            .map_err(|e| ResolveError::Other(e.into()))?;

        if let Some(market) = markets.into_iter().find(|m| m.slug == current_slug) {
            if !market.closed.unwrap_or(false) {
                return Ok(WalkResult { outcome: None, cursor: ts });
            }
            let prices = parse_outcome_prices(&market.outcome_prices.unwrap_or_default());
            if prices.len() < 2 {
                return Err(ResolveError::Other(anyhow::anyhow!(
                    "resolved market slug={} has {} outcome prices (need 2)",
                    current_slug,
                    prices.len()
                )));
            }
            if prices[0] > 0.5 && prices[1] < 0.5 {
                return Ok(WalkResult { outcome: Some(1), cursor: ts });
            } else if prices[1] > 0.5 && prices[0] < 0.5 {
                return Ok(WalkResult { outcome: Some(2), cursor: ts });
            } else {
                return Err(ResolveError::Other(anyhow::anyhow!(
                    "ambiguous outcome prices for slug={}: {:?}",
                    current_slug,
                    prices
                )));
            }
        }

        ts += WINDOW_SECS;
        if ts > now_secs + WINDOW_SECS {
            return Err(ResolveError::NotFoundYet { cursor: ts });
        }
    }
}
