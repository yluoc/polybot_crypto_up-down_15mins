// Coinalyze polling + backfill for aggregated open-interest and liquidation history.

use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use reqwest::Client;
use serde::Deserialize;
use sqlx::PgPool;
use tokio::time::sleep;
use tracing::{info, warn};

use crate::db::queries;
use crate::feature_engine::INSTRUMENT_ORDER;

pub const COINALYZE_BASE: &str = "https://api.coinalyze.net/v1";

const POLL_INTERVAL: Duration = Duration::from_secs(60);
const HTTP_TIMEOUT:  Duration = Duration::from_secs(15);
const REQ_SLEEP: Duration = Duration::from_millis(1600);

const MAX_ATTEMPTS:   u32 = 2;
const RETRY_BACKOFF:  Duration = Duration::from_millis(750);

/// (polybot short_symbol, Coinalyze aggregated-perp symbol).
pub const COINALYZE_SYMBOLS: [(&str, &str); 4] = [
    ("BTC", "BTCUSDT_PERP.A"),
    ("ETH", "ETHUSDT_PERP.A"),
    ("XRP", "XRPUSDT_PERP.A"),
    ("SOL", "SOLUSDT_PERP.A"),
];

#[derive(Deserialize, Debug)]
struct OiEnvelope {
    symbol:  String,
    history: Vec<OiBucket>,
}

#[derive(Deserialize, Debug)]
struct OiBucket {
    t: i64, // bucket-open unix seconds
    c: f64, // close-of-bucket OI in USD notional
    // o/h/l also present; intentionally unused
}

#[derive(Deserialize, Debug)]
struct LiqEnvelope {
    symbol:  String,
    history: Vec<LiqBucket>,
}

#[derive(Deserialize, Debug)]
struct LiqBucket {
    t: i64, // bucket-open unix seconds
    #[serde(default)]
    l: f64, // long liquidations in USD notional
    #[serde(default)]
    s: f64, // short liquidations in USD notional
}

#[derive(Debug, Clone)]
pub struct OiRow {
    pub symbol: String,
    pub ts_ms:  i64,
    pub oi_usd: f64,
}

#[derive(Debug, Clone)]
pub struct LiqRow {
    pub symbol:        String,
    pub ts_ms:         i64,
    pub long_liq_usd:  f64,
    pub short_liq_usd: f64,
}

/// Spawn the background poller. If `COINALYZE_API_KEY` is unset the poller is disabled.
pub fn spawn_coinalyze_poller(pool: PgPool) {
    let api_key = match std::env::var("COINALYZE_API_KEY") {
        Ok(k) if !k.trim().is_empty() => k,
        _ => {
            warn!(
                "[coinalyze] COINALYZE_API_KEY unset or empty — poller DISABLED. \
                 v14 micro features (oi_change_pct_4, liq_imbalance_4) will be NaN \
                 until a key is provisioned. Set the env var and restart."
            );
            return;
        }
    };
    tokio::spawn(async move {
        let http = match Client::builder().timeout(HTTP_TIMEOUT).build() {
            Ok(c) => c,
            Err(e) => {
                warn!("[coinalyze] failed to build HTTP client: {e:#} — poller exiting");
                return;
            }
        };
        loop {
            let cycle_start = std::time::Instant::now();
            match poll_cycle(&http, &api_key).await {
                Ok((oi_rows, liq_rows, ratelimit_429s)) => {
                    let oi_persisted = match queries::insert_oi_aggregated_batch(&pool, &oi_rows).await {
                        Ok(n) => n,
                        Err(e) => {
                            warn!("[coinalyze] oi insert failed: {e:#}");
                            0
                        }
                    };
                    let liq_persisted = match queries::insert_liq_aggregated_batch(&pool, &liq_rows).await {
                        Ok(n) => n,
                        Err(e) => {
                            warn!("[coinalyze] liq insert failed: {e:#}");
                            0
                        }
                    };
                    info!(
                        "[coinalyze] cycle ok symbols={} oi_rows_persisted={} liq_rows_persisted={} \
                         429s={} elapsed_ms={}",
                        COINALYZE_SYMBOLS.len(),
                        oi_persisted,
                        liq_persisted,
                        ratelimit_429s,
                        cycle_start.elapsed().as_millis()
                    );
                }
                Err(e) => warn!("[coinalyze] cycle errored: {e:#}"),
            }
            sleep(POLL_INTERVAL).await;
        }
    });
}

/// One full poll cycle covering all 4 symbols × (OI + liquidations).
async fn poll_cycle(
    http: &Client,
    api_key: &str,
) -> Result<(Vec<OiRow>, Vec<LiqRow>, u32)> {
    let now_secs = chrono::Utc::now().timestamp();
    let lookback_secs = 10 * 15 * 60; // 10 × 15 min ≈ 2.5h
    let from = now_secs - lookback_secs;
    let to = now_secs;

    let mut oi_rows = Vec::new();
    let mut liq_rows = Vec::new();
    let mut rl_429s: u32 = 0;

    for (short, coinalyze_sym) in COINALYZE_SYMBOLS.iter() {
        match fetch_oi(http, api_key, coinalyze_sym, from, to).await {
            Ok(rows) => {
                for b in rows {
                    oi_rows.push(OiRow {
                        symbol: short.to_string(),
                        ts_ms:  b.t * 1000,
                        oi_usd: b.c,
                    });
                }
            }
            Err(e) => {
                if format!("{e:#}").contains("429") {
                    rl_429s += 1;
                }
                warn!("[coinalyze] OI fetch {short} ({coinalyze_sym}) failed: {e:#}");
            }
        }
        sleep(REQ_SLEEP).await;

        match fetch_liq(http, api_key, coinalyze_sym, from, to).await {
            Ok(rows) => {
                for b in rows {
                    liq_rows.push(LiqRow {
                        symbol:        short.to_string(),
                        ts_ms:         b.t * 1000,
                        long_liq_usd:  b.l,
                        short_liq_usd: b.s,
                    });
                }
            }
            Err(e) => {
                if format!("{e:#}").contains("429") {
                    rl_429s += 1;
                }
                warn!("[coinalyze] LIQ fetch {short} ({coinalyze_sym}) failed: {e:#}");
            }
        }
        sleep(REQ_SLEEP).await;
    }
    Ok((oi_rows, liq_rows, rl_429s))
}

/// One-shot historical backfill. Idempotent (UPSERTs use `DO NOTHING` on conflict).
pub async fn run_backfill(pool: &PgPool, days: u32) -> Result<()> {
    let api_key = std::env::var("COINALYZE_API_KEY")
        .map_err(|_| anyhow!("COINALYZE_API_KEY not set — backfill-coinalyze requires a key"))?;
    if api_key.trim().is_empty() {
        return Err(anyhow!("COINALYZE_API_KEY is empty"));
    }
    let http = Client::builder().timeout(HTTP_TIMEOUT).build()?;
    let now_secs = chrono::Utc::now().timestamp();
    let from = now_secs - (days as i64) * 86400;
    let to = now_secs;

    let mut total_oi = 0u64;
    let mut total_liq = 0u64;
    for (short, coinalyze_sym) in COINALYZE_SYMBOLS.iter() {
        let oi = fetch_oi(&http, &api_key, coinalyze_sym, from, to)
            .await
            .with_context(|| format!("backfill OI {short}"))?;
        let oi_rows: Vec<OiRow> = oi
            .into_iter()
            .map(|b| OiRow { symbol: short.to_string(), ts_ms: b.t * 1000, oi_usd: b.c })
            .collect();
        let n_oi = queries::insert_oi_aggregated_batch(pool, &oi_rows).await?;
        total_oi += n_oi;
        sleep(REQ_SLEEP).await;

        let liq = fetch_liq(&http, &api_key, coinalyze_sym, from, to)
            .await
            .with_context(|| format!("backfill liq {short}"))?;
        let liq_rows: Vec<LiqRow> = liq
            .into_iter()
            .map(|b| LiqRow {
                symbol:        short.to_string(),
                ts_ms:         b.t * 1000,
                long_liq_usd:  b.l,
                short_liq_usd: b.s,
            })
            .collect();
        let n_liq = queries::insert_liq_aggregated_batch(pool, &liq_rows).await?;
        total_liq += n_liq;
        sleep(REQ_SLEEP).await;

        info!(
            "[coinalyze:backfill] {short} ({coinalyze_sym}) oi_new={n_oi} liq_new={n_liq}"
        );
    }
    info!(
        "[coinalyze:backfill] done — days={days} oi_total={total_oi} liq_total={total_liq} \
         symbols={}",
        INSTRUMENT_ORDER.len()
    );
    Ok(())
}

async fn fetch_oi(
    http: &Client,
    api_key: &str,
    symbol: &str,
    from_secs: i64,
    to_secs: i64,
) -> Result<Vec<OiBucket>> {
    let url = format!("{COINALYZE_BASE}/open-interest-history");
    let env: Vec<OiEnvelope> = request_with_retry(http, &url, &[
        ("symbols", symbol),
        ("interval", "15min"),
        ("from", &from_secs.to_string()),
        ("to", &to_secs.to_string()),
        ("api_key", api_key),
    ])
    .await
    .with_context(|| format!("OI fetch {symbol}"))?;
    Ok(env.into_iter().find(|e| e.symbol == symbol).map(|e| e.history).unwrap_or_default())
}

async fn fetch_liq(
    http: &Client,
    api_key: &str,
    symbol: &str,
    from_secs: i64,
    to_secs: i64,
) -> Result<Vec<LiqBucket>> {
    let url = format!("{COINALYZE_BASE}/liquidation-history");
    let env: Vec<LiqEnvelope> = request_with_retry(http, &url, &[
        ("symbols", symbol),
        ("interval", "15min"),
        ("from", &from_secs.to_string()),
        ("to", &to_secs.to_string()),
        ("api_key", api_key),
    ])
    .await
    .with_context(|| format!("LIQ fetch {symbol}"))?;
    Ok(env.into_iter().find(|e| e.symbol == symbol).map(|e| e.history).unwrap_or_default())
}

/// HTTP GET with retry and 429-aware backoff.
async fn request_with_retry<T: for<'de> Deserialize<'de>>(
    http: &Client,
    url: &str,
    query: &[(&str, &str)],
) -> Result<T> {
    let mut last_err: Option<anyhow::Error> = None;
    for attempt in 1..=MAX_ATTEMPTS {
        let resp = http.get(url).query(query).send().await;
        let resp = match resp {
            Ok(r) => r,
            Err(e) => {
                last_err = Some(anyhow!("network error attempt {attempt}: {e}"));
                sleep(RETRY_BACKOFF).await;
                continue;
            }
        };
        let status = resp.status();
        if status.as_u16() == 429 {
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .map(Duration::from_secs)
                .unwrap_or(RETRY_BACKOFF);
            last_err = Some(anyhow!("rate-limited (429) attempt {attempt}"));
            sleep(retry_after).await;
            continue;
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("HTTP {} — body: {}", status, body));
        }
        return resp.json::<T>().await.map_err(|e| anyhow!("JSON decode: {e}"));
    }
    Err(last_err.unwrap_or_else(|| anyhow!("request_with_retry exhausted")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coinalyze_symbols_match_instrument_order() {
        use std::collections::HashSet;
        let our_shorts: HashSet<&str> =
            COINALYZE_SYMBOLS.iter().map(|(s, _)| *s).collect();
        let inst_shorts: HashSet<&str> = INSTRUMENT_ORDER
            .iter()
            .map(|inst| inst.split('-').next().unwrap_or(""))
            .collect();
        assert_eq!(
            our_shorts, inst_shorts,
            "COINALYZE_SYMBOLS must cover every short in INSTRUMENT_ORDER"
        );
    }
}
