// OKX taker-volume polling + backfill.
// Fetches at 5m granularity (the only supported period) and aggregates 3-into-1 for 15m.

use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use reqwest::Client;
use serde::Deserialize;
use sqlx::PgPool;
use tokio::time::sleep;
use tracing::{info, warn};

use crate::db::queries;
use crate::feature_engine::INSTRUMENT_ORDER;

pub const OKX_TAKER_VOLUME_URL: &str = "https://www.okx.com/api/v5/rubik/stat/taker-volume";

/// 5 minutes in ms — the underlying OKX bucket size.
const FIVE_MIN_MS: i64 = 5 * 60 * 1000;
/// 15 minutes in ms — polybot's bucket size; aggregation target.
const FIFTEEN_MIN_MS: i64 = 15 * 60 * 1000;

const POLL_INTERVAL: Duration = Duration::from_secs(60);
const HTTP_TIMEOUT:  Duration = Duration::from_secs(15);

const REQ_SLEEP: Duration = Duration::from_millis(600);

const MAX_ATTEMPTS:  u32 = 3;
const RETRY_BACKOFF: Duration = Duration::from_millis(500);

/// Safety cap on backfill paging (240 × 100 × 5min ≈ 83 days).
const BACKFILL_MAX_PAGES: u32 = 240;

// taker buy vs sell ordering: data rows are [ts_ms, sellVol, buyVol]
const OKX_SELL_IDX: usize = 1;
const OKX_BUY_IDX:  usize = 2;

/// (polybot inst_id, OKX `ccy` query param).
pub const OKX_TAKER_SYMBOLS: [(&str, &str); 4] = [
    ("BTC-USDT-SWAP", "BTC"),
    ("ETH-USDT-SWAP", "ETH"),
    ("XRP-USDT-SWAP", "XRP"),
    ("SOL-USDT-SWAP", "SOL"),
];

#[derive(Deserialize, Debug)]
struct OkxRubikEnvelope {
    code: String,
    #[serde(default)]
    data: Vec<Vec<String>>,
    #[serde(default)]
    msg:  String,
}

/// One 5-minute taker-volume datapoint from OKX.
#[derive(Debug, Clone, Copy)]
struct FiveMinPoint {
    ts_ms:        i64,
    sell_vol:     f64,
    buy_vol:      f64,
}

#[derive(Debug, Clone)]
pub struct TakerVolRow {
    pub inst_id:        String,
    pub ts_ms:          i64,
    pub taker_buy_vol:  f64,
    pub taker_sell_vol: f64,
}

/// Spawn the background poller.
pub fn spawn_okx_taker_poller(pool: PgPool) {
    tokio::spawn(async move {
        let http = match Client::builder().timeout(HTTP_TIMEOUT).build() {
            Ok(c) => c,
            Err(e) => {
                warn!("[okx_taker] HTTP client build failed: {e:#} — poller exiting");
                return;
            }
        };
        loop {
            let cycle_start = std::time::Instant::now();
            match poll_cycle(&http).await {
                Ok(rows) => {
                    let n = match queries::insert_taker_volume_batch(&pool, &rows).await {
                        Ok(n) => n,
                        Err(e) => {
                            warn!("[okx_taker] insert failed: {e:#}");
                            0
                        }
                    };
                    info!(
                        "[okx_taker] cycle ok symbols={} rows_persisted={n} elapsed_ms={}",
                        OKX_TAKER_SYMBOLS.len(),
                        cycle_start.elapsed().as_millis()
                    );
                }
                Err(e) => warn!("[okx_taker] cycle errored: {e:#}"),
            }
            sleep(POLL_INTERVAL).await;
        }
    });
}

/// One poll cycle covering all 4 symbols, ~2.5 hours of lookback.
async fn poll_cycle(http: &Client) -> Result<Vec<TakerVolRow>> {
    let now_ms = chrono::Utc::now().timestamp_millis();
    let begin = now_ms - 10 * FIFTEEN_MIN_MS;
    let end = now_ms;

    let mut out = Vec::new();
    for (inst_id, ccy) in OKX_TAKER_SYMBOLS.iter() {
        match fetch_5m(http, ccy, begin, end).await {
            Ok(pts) => {
                for r in aggregate_to_15m(inst_id, &pts) {
                    out.push(r);
                }
            }
            Err(e) => warn!("[okx_taker] fetch {inst_id} ({ccy}) failed: {e:#}"),
        }
        sleep(REQ_SLEEP).await;
    }
    Ok(out)
}

/// One-shot historical backfill.
pub async fn run_backfill(pool: &PgPool, days: u32) -> Result<()> {
    let http = Client::builder().timeout(HTTP_TIMEOUT).build()?;
    let end_ms = chrono::Utc::now().timestamp_millis();
    let begin_ms = end_ms - (days as i64) * 86_400_000;

    let mut total = 0u64;
    for (inst_id, ccy) in OKX_TAKER_SYMBOLS.iter() {
        let mut window_end = end_ms;
        let mut all_pts: Vec<FiveMinPoint> = Vec::new();
        let mut pages: u32 = 0;
        loop {
            if pages >= BACKFILL_MAX_PAGES {
                warn!(
                    "[okx_taker:backfill] {inst_id} hit BACKFILL_MAX_PAGES={BACKFILL_MAX_PAGES} \
                     — capping at window_end={window_end}; expand the cap if you need deeper history"
                );
                break;
            }
            let window_begin = (window_end - 100 * FIVE_MIN_MS).max(begin_ms);
            if window_begin >= window_end {
                break;
            }
            let pts = fetch_5m(&http, ccy, window_begin, window_end)
                .await
                .with_context(|| format!("backfill {inst_id} window {window_begin}-{window_end}"))?;
            pages += 1;
            if pts.is_empty() && window_begin <= begin_ms {
                break;
            }
            all_pts.extend(pts);
            if window_begin <= begin_ms {
                break;
            }
            window_end = window_begin;
            sleep(REQ_SLEEP).await;
        }
        let rows = aggregate_to_15m(inst_id, &all_pts);
        let n = queries::insert_taker_volume_batch(pool, &rows).await?;
        total += n;
        info!("[okx_taker:backfill] {inst_id} ({ccy}) rows_new={n} (5m_pts={})", all_pts.len());
    }
    info!(
        "[okx_taker:backfill] done — days={days} rows_total={total} symbols={}",
        INSTRUMENT_ORDER.len()
    );
    Ok(())
}

async fn fetch_5m(
    http: &Client,
    ccy: &str,
    begin_ms: i64,
    end_ms: i64,
) -> Result<Vec<FiveMinPoint>> {
    for attempt in 1..=MAX_ATTEMPTS {
        let resp = http
            .get(OKX_TAKER_VOLUME_URL)
            .query(&[
                ("ccy", ccy),
                ("instType", "CONTRACTS"),
                ("period", "5m"),
                ("begin", &begin_ms.to_string()),
                ("end", &end_ms.to_string()),
            ])
            .send()
            .await;
        let resp = match resp {
            Ok(r) => r,
            Err(e) => {
                if attempt == MAX_ATTEMPTS {
                    return Err(anyhow!("OKX network error: {e}"));
                }
                sleep(RETRY_BACKOFF).await;
                continue;
            }
        };
        if resp.status().as_u16() == 429 {
            sleep(RETRY_BACKOFF).await;
            continue;
        }
        if !resp.status().is_success() {
            return Err(anyhow!("OKX HTTP {}", resp.status()));
        }
        let env: OkxRubikEnvelope = resp.json().await.context("OKX JSON decode")?;
        if env.code != "0" {
            return Err(anyhow!("OKX code={} msg={}", env.code, env.msg));
        }
        return Ok(parse_5m_data(&env.data));
    }
    Err(anyhow!("fetch_5m exhausted retries"))
}

/// Parse OKX `data` rows; malformed/non-finite values are silently dropped.
fn parse_5m_data(data: &[Vec<String>]) -> Vec<FiveMinPoint> {
    let mut out = Vec::with_capacity(data.len());
    for row in data {
        if row.len() < 3 {
            continue;
        }
        let Ok(ts_ms) = row[0].parse::<i64>() else { continue };
        let Ok(v_sell) = row[OKX_SELL_IDX].parse::<f64>() else { continue };
        let Ok(v_buy)  = row[OKX_BUY_IDX].parse::<f64>()  else { continue };
        if !v_sell.is_finite() || !v_buy.is_finite() {
            continue;
        }
        out.push(FiveMinPoint { ts_ms, sell_vol: v_sell, buy_vol: v_buy });
    }
    out
}

/// Aggregate 5m points into 15m buckets; drops partial windows (< 3 sub-buckets).
fn aggregate_to_15m(inst_id: &str, pts: &[FiveMinPoint]) -> Vec<TakerVolRow> {
    use std::collections::BTreeMap;

    let mut by_5m: BTreeMap<i64, FiveMinPoint> = BTreeMap::new();
    for p in pts {
        let aligned = (p.ts_ms / FIVE_MIN_MS) * FIVE_MIN_MS; // align to 5m boundary
        by_5m.insert(aligned, FiveMinPoint { ts_ms: aligned, ..*p });
    }

    let mut groups: BTreeMap<i64, Vec<FiveMinPoint>> = BTreeMap::new();
    for (ts_5m, p) in by_5m {
        let bucket_15 = (ts_5m / FIFTEEN_MIN_MS) * FIFTEEN_MIN_MS;
        groups.entry(bucket_15).or_default().push(p);
    }

    let mut out = Vec::with_capacity(groups.len());
    for (bucket_15, pts3) in groups {
        if pts3.len() != 3 {
            continue;
        }
        let buy: f64  = pts3.iter().map(|p| p.buy_vol).sum();
        let sell: f64 = pts3.iter().map(|p| p.sell_vol).sum();
        out.push(TakerVolRow {
            inst_id:        inst_id.to_string(),
            ts_ms:          bucket_15,
            taker_buy_vol:  buy,
            taker_sell_vol: sell,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taker_symbols_match_instrument_order() {
        use std::collections::HashSet;
        let ours: HashSet<&str> = OKX_TAKER_SYMBOLS.iter().map(|(i, _)| *i).collect();
        let inst: HashSet<&str> = INSTRUMENT_ORDER.iter().copied().collect();
        assert_eq!(ours, inst);
    }

    #[test]
    fn aggregate_drops_partial_windows() {
        // Two 5m buckets at the same 15m bucket → no row emitted.
        let pts = vec![
            FiveMinPoint { ts_ms: 0,                sell_vol: 1.0, buy_vol: 2.0 },
            FiveMinPoint { ts_ms: 5 * 60 * 1000,    sell_vol: 3.0, buy_vol: 4.0 },
        ];
        let rows = aggregate_to_15m("BTC-USDT-SWAP", &pts);
        assert!(rows.is_empty(), "partial 15m window must not emit a row");
    }

    #[test]
    fn aggregate_sums_buy_and_sell_correctly() {
        let pts = vec![
            FiveMinPoint { ts_ms: 0,                sell_vol: 1.0, buy_vol: 10.0 },
            FiveMinPoint { ts_ms: 5  * 60 * 1000,   sell_vol: 2.0, buy_vol: 20.0 },
            FiveMinPoint { ts_ms: 10 * 60 * 1000,   sell_vol: 3.0, buy_vol: 30.0 },
        ];
        let rows = aggregate_to_15m("BTC-USDT-SWAP", &pts);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].ts_ms, 0);
        assert!((rows[0].taker_sell_vol - 6.0).abs() < 1e-9);
        assert!((rows[0].taker_buy_vol - 60.0).abs() < 1e-9);
    }
}
