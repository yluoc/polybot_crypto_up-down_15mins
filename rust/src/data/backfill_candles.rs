use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use reqwest::Client;
use serde::Deserialize;
use sqlx::PgPool;
use tokio::time::sleep;
use tracing::{info, warn};

use crate::cli::BackfillArgs;
use crate::db::{
    self,
    models::{CandleRow, FundingRateRow, IndexCandleRow, OpenInterestRow},
};
use crate::feature_engine::{short_symbol, CANDLE_INTERVAL_MS, INSTRUMENT_ORDER};

/// Recent mark-price candles; caps at ~1,440 rows. Used for incremental mode.
const OKX_CANDLES_URL: &str = "https://www.okx.com/api/v5/market/mark-price-candles";

/// Deep-history mark-price candles; pages back arbitrarily far. Used for --full mode.
const OKX_HISTORY_CANDLES_URL: &str =
    "https://www.okx.com/api/v5/market/history-mark-price-candles";
const OKX_BAR: &str = "15m";
const OKX_PAGE_LIMIT: u32 = 100;
const OKX_RATE_SLEEP: Duration = Duration::from_millis(200);
const OKX_HTTP_TIMEOUT: Duration = Duration::from_secs(10);

const OKX_FUNDING_HISTORY_URL: &str = "https://www.okx.com/api/v5/public/funding-rate-history";
const OKX_OI_HISTORY_URL: &str =
    "https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-volume";
/// Recent index candles; caps at ~1440 rows. Used in incremental mode.
const OKX_INDEX_CANDLES_URL: &str = "https://www.okx.com/api/v5/market/index-candles";
/// Deep-history index candles; pages back arbitrarily far. Used in --full mode.
const OKX_INDEX_HISTORY_URL: &str = "https://www.okx.com/api/v5/market/history-index-candles";

const OKX_FUNDING_PAGE_LIMIT: u32 = 100;
const OKX_OI_PERIOD: &str = "5m";
const OKX_RATE_SLEEP_PERP: Duration = Duration::from_millis(250);

#[derive(Deserialize)]
struct OkxEnvelope {
    #[serde(default)]
    code: String,
    #[serde(default)]
    msg: String,
    #[serde(default)]
    data: Vec<Vec<String>>,
}

#[derive(Deserialize)]
struct OkxFundingItem {
    #[serde(default, rename = "instId")]
    _inst_id: String,
    #[serde(rename = "fundingTime")]
    funding_time: String,
    #[serde(rename = "fundingRate")]
    funding_rate: String,
}

#[derive(Deserialize)]
struct OkxFundingEnvelope {
    #[serde(default)]
    code: String,
    #[serde(default)]
    msg: String,
    #[serde(default)]
    data: Vec<OkxFundingItem>,
}

/// Strip the `-SWAP` suffix to get the index symbol (e.g. `BTC-USDT-SWAP` → `BTC-USDT`).
fn inst_id_to_index(inst_id: &str) -> Option<String> {
    inst_id.strip_suffix("-SWAP").map(|s| s.to_string())
}

/// 15m bucket open time aligned to epoch.
fn bucket_open_ms(ts_ms: i64) -> i64 {
    ts_ms - ts_ms.rem_euclid(CANDLE_INTERVAL_MS)
}

pub async fn run(pool: &PgPool, args: BackfillArgs) -> Result<()> {
    let http = Client::builder()
        .timeout(OKX_HTTP_TIMEOUT)
        .build()
        .context("build backfill HTTP client")?;

    let now_ms = chrono::Utc::now().timestamp_millis();
    let days_cap_ms = (args.days as i64) * 86_400_000;

    let url = if args.full {
        OKX_HISTORY_CANDLES_URL
    } else {
        OKX_CANDLES_URL
    };
    info!("[backfill] using endpoint: {} (full={})", url, args.full);

    let mut total_inserted: u64 = 0;
    for inst in INSTRUMENT_ORDER.iter() {
        let stop_at = if args.full {
            // exclusive lower bound
            Some(now_ms - days_cap_ms)
        } else {
            let max_ts = db::queries::get_max_candle_ts(pool, inst).await?;
            match max_ts {
                Some(ts) => Some(ts),
                None => {
                    warn!(
                        "[backfill] {} has no candles yet — falling back to {} day cap",
                        inst, args.days
                    );
                    Some(now_ms - days_cap_ms)
                }
            }
        };

        let fetched = fetch_instrument(&http, url, inst, stop_at).await?;
        info!("[backfill] {}: fetched {} candles (stop_at={:?})", inst, fetched.len(), stop_at);
        if fetched.is_empty() {
            continue;
        }
        let inserted = db::queries::insert_candles_batch(pool, &fetched).await?;
        info!("[backfill] {}: inserted {} new rows ({} duplicates skipped)",
              inst, inserted, fetched.len() as u64 - inserted);
        total_inserted += inserted;
    }
    info!("[backfill] candles done — {} total rows inserted across {} instruments",
          total_inserted, INSTRUMENT_ORDER.len());

    backfill_funding(&http, pool, args.full, days_cap_ms, now_ms).await?;
    backfill_open_interest(&http, pool, args.full, days_cap_ms, now_ms).await?;
    backfill_index_candles(&http, pool, args.full, days_cap_ms, now_ms).await?;

    info!("[backfill] all perp feature passes complete");
    Ok(())
}

async fn backfill_funding(
    http: &Client,
    pool: &PgPool,
    full: bool,
    days_cap_ms: i64,
    now_ms: i64,
) -> Result<()> {
    let mut total: u64 = 0;
    for inst in INSTRUMENT_ORDER.iter() {
        let stop_at = if full {
            Some(now_ms - days_cap_ms)
        } else {
            db::queries::get_max_funding_ts(pool, inst).await?.or(Some(now_ms - days_cap_ms))
        };

        let fetched = fetch_funding_history(http, inst, stop_at).await?;
        info!("[backfill] funding {}: fetched {} (stop_at={:?})", inst, fetched.len(), stop_at);
        if fetched.is_empty() {
            continue;
        }
        let inserted = db::queries::insert_funding_batch(pool, &fetched).await?;
        info!("[backfill] funding {}: inserted {} new rows", inst, inserted);
        total += inserted;
    }
    info!("[backfill] funding done — {} new rows", total);
    Ok(())
}

/// Pages OKX funding-rate-history newest-first using the `after` cursor.
/// `settle_period_secs` is derived from the gap between consecutive settlements.
async fn fetch_funding_history(
    http: &Client,
    inst_id: &str,
    stop_at_ms: Option<i64>,
) -> Result<Vec<FundingRateRow>> {
    let mut out: Vec<FundingRateRow> = Vec::new();
    // `after` pages back (records earlier than this fundingTime).
    let mut after: Option<String> = None;

    loop {
        let mut query: Vec<(&str, String)> = vec![
            ("instId", inst_id.to_string()),
            ("limit", OKX_FUNDING_PAGE_LIMIT.to_string()),
        ];
        if let Some(a) = after.as_ref() {
            query.push(("after", a.clone()));
        }

        let resp: OkxFundingEnvelope = http
            .get(OKX_FUNDING_HISTORY_URL)
            .query(&query)
            .send()
            .await
            .with_context(|| format!("OKX funding GET {}", inst_id))?
            .error_for_status()
            .with_context(|| format!("OKX funding status {}", inst_id))?
            .json()
            .await
            .with_context(|| format!("OKX funding JSON {}", inst_id))?;

        if resp.code != "0" && !resp.code.is_empty() {
            return Err(anyhow!(
                "OKX funding error inst={} code={} msg={}",
                inst_id, resp.code, resp.msg
            ));
        }
        if resp.data.is_empty() {
            break;
        }

        let mut parsed: Vec<(i64, f64)> = Vec::with_capacity(resp.data.len());
        let mut oldest_in_batch: Option<i64> = None;
        for it in &resp.data {
            let ts: i64 = it.funding_time.parse()
                .with_context(|| format!("parse fundingTime '{}'", it.funding_time))?;
            let rate: f64 = it.funding_rate.parse()
                .with_context(|| format!("parse fundingRate '{}'", it.funding_rate))?;
            oldest_in_batch = Some(match oldest_in_batch {
                Some(prev) => prev.min(ts),
                None => ts,
            });
            parsed.push((ts, rate));
        }
        parsed.sort_by_key(|(ts, _)| *ts);

        for (i, (ts, rate)) in parsed.iter().enumerate() {
            if let Some(cutoff) = stop_at_ms {
                if *ts <= cutoff {
                    continue;
                }
            }
            let settle_period_secs = if i == 0 {
                None // no predecessor in this page
            } else {
                let gap_ms = ts - parsed[i - 1].0;
                if gap_ms > 0 { Some((gap_ms / 1000) as i32) } else { None }
            };
            out.push(FundingRateRow {
                inst_id: inst_id.to_string(),
                ts_ms: *ts,
                rate: *rate,
                settle_period_secs,
            });
        }

        if let (Some(cutoff), Some(oldest)) = (stop_at_ms, oldest_in_batch) {
            if oldest <= cutoff {
                break;
            }
        }

        after = oldest_in_batch.map(|t| t.to_string());

        sleep(OKX_RATE_SLEEP_PERP).await;
    }
    Ok(out)
}

async fn backfill_open_interest(
    http: &Client,
    pool: &PgPool,
    full: bool,
    days_cap_ms: i64,
    now_ms: i64,
) -> Result<()> {
    let mut total: u64 = 0;
    for (i, inst) in INSTRUMENT_ORDER.iter().enumerate() {
        if i > 0 {
            sleep(OKX_RATE_SLEEP_PERP).await;
        }
        let stop_at = if full {
            Some(now_ms - days_cap_ms)
        } else {
            db::queries::get_max_oi_ts(pool, inst).await?.or(Some(now_ms - days_cap_ms))
        };

        let fetched = fetch_open_interest_history(http, inst, stop_at).await?;
        info!("[backfill] OI {}: fetched {} (stop_at={:?})", inst, fetched.len(), stop_at);
        if fetched.is_empty() {
            continue;
        }
        let inserted = db::queries::insert_oi_batch(pool, &fetched).await?;
        info!("[backfill] OI {}: inserted {} new rows", inst, inserted);
        total += inserted;
    }
    info!("[backfill] OI done — {} new rows", total);
    Ok(())
}

/// Single-window GET to the rubik OI endpoint with retry on 429 / 5xx.
/// Tiles the range in 42h windows; treats code=50030 as a retention-cliff stop.
async fn fetch_oi_window_with_retry(
    http: &Client,
    ccy: &str,
    begin: i64,
    end: i64,
    inst_id: &str,
) -> Result<OkxEnvelope> {
    const MAX_ATTEMPTS: u32 = 4;
    let mut attempt: u32 = 0;
    loop {
        attempt += 1;
        let resp = http
            .get(OKX_OI_HISTORY_URL)
            .query(&[
                ("ccy", ccy.to_string()),
                ("period", OKX_OI_PERIOD.to_string()),
                ("begin", begin.to_string()),
                ("end", end.to_string()),
            ])
            .send()
            .await
            .with_context(|| format!("OKX OI GET {}", inst_id))?;

        let status = resp.status();
        let retryable = status.as_u16() == 429 || status.is_server_error();
        if retryable && attempt < MAX_ATTEMPTS {
            let backoff = resp
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|h| h.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .map(Duration::from_secs)
                .unwrap_or_else(|| Duration::from_millis(1000u64 << (attempt - 1)));
            warn!(
                "[backfill] OI {} status={} (attempt {}/{}); retrying in {}ms",
                inst_id,
                status.as_u16(),
                attempt,
                MAX_ATTEMPTS,
                backoff.as_millis()
            );
            sleep(backoff).await;
            continue;
        }
        return resp
            .error_for_status()
            .with_context(|| format!("OKX OI status {}", inst_id))?
            .json::<OkxEnvelope>()
            .await
            .with_context(|| format!("OKX OI JSON {}", inst_id));
    }
}

async fn fetch_open_interest_history(
    http: &Client,
    inst_id: &str,
    stop_at_ms: Option<i64>,
) -> Result<Vec<OpenInterestRow>> {
    use std::collections::BTreeMap;

    const OI_WINDOW_MS: i64 = 42 * 3600 * 1000; // largest window rubik consistently accepts
    const OI_END_BUFFER_MS: i64 = 5 * 60 * 1000; // avoid the future-edge variant of code 50030

    let ccy = short_symbol(inst_id);
    let now_ms = chrono::Utc::now().timestamp_millis();
    let stop = stop_at_ms.unwrap_or(now_ms - 30 * 86_400_000);

    let mut by_bucket: BTreeMap<i64, (f64, f64, i64)> = BTreeMap::new();
    let mut end = now_ms - OI_END_BUFFER_MS;

    while end > stop {
        let begin = (end - OI_WINDOW_MS).max(stop);
        let resp: OkxEnvelope = fetch_oi_window_with_retry(http, ccy, begin, end, inst_id).await?;

        if resp.code == "50030" {
            // rubik retention cliff — stop walking
            info!(
                "[backfill] OI {}: hit rubik retention cliff at end={} ({} buckets so far)",
                inst_id, end, by_bucket.len()
            );
            break;
        }
        if resp.code != "0" && !resp.code.is_empty() {
            return Err(anyhow!(
                "OKX OI error inst={} code={} msg={} (begin={} end={})",
                inst_id, resp.code, resp.msg, begin, end
            ));
        }

        for row in &resp.data {
            if row.len() < 3 {
                continue;
            }
            let ts: i64 = row[0].parse().with_context(|| format!("parse OI ts '{}'", row[0]))?;
            let oi_ccy: f64 = row[1].parse().context("parse oiCcy")?;
            let oi_usd: f64 = row[2].parse().context("parse oiUsd")?;
            let bucket = bucket_open_ms(ts);
            match by_bucket.get(&bucket) {
                Some((_, _, prev_ts)) if *prev_ts >= ts => {}
                _ => { by_bucket.insert(bucket, (oi_ccy, oi_usd, ts)); }
            }
        }

        end = begin;
        sleep(OKX_RATE_SLEEP_PERP).await;
    }

    let mut out: Vec<OpenInterestRow> = Vec::with_capacity(by_bucket.len());
    for (bucket, (oi_ccy, oi_usd, _)) in by_bucket {
        if bucket <= stop {
            continue;
        }
        out.push(OpenInterestRow {
            inst_id: inst_id.to_string(),
            ts_ms: bucket,
            oi_ccy,
            oi_usd,
        });
    }
    Ok(out)
}

async fn backfill_index_candles(
    http: &Client,
    pool: &PgPool,
    full: bool,
    days_cap_ms: i64,
    now_ms: i64,
) -> Result<()> {
    let url = if full {
        OKX_INDEX_HISTORY_URL
    } else {
        OKX_INDEX_CANDLES_URL
    };

    let mut total: u64 = 0;
    for inst in INSTRUMENT_ORDER.iter() {
        let index_inst = match inst_id_to_index(inst) {
            Some(v) => v,
            None => {
                warn!("[backfill] index: skipping {} — no -SWAP suffix to strip", inst);
                continue;
            }
        };

        let stop_at = if full {
            Some(now_ms - days_cap_ms)
        } else {
            db::queries::get_max_index_ts(pool, &index_inst).await?
                .or(Some(now_ms - days_cap_ms))
        };

        let fetched = fetch_index_candles(http, url, &index_inst, stop_at).await?;
        info!("[backfill] index {}: fetched {} (stop_at={:?})", index_inst, fetched.len(), stop_at);
        if fetched.is_empty() {
            continue;
        }
        let inserted = db::queries::insert_index_candles_batch(pool, &fetched).await?;
        info!("[backfill] index {}: inserted {} new rows", index_inst, inserted);
        total += inserted;
    }
    info!("[backfill] index done — {} new rows", total);
    Ok(())
}

async fn fetch_index_candles(
    http: &Client,
    url: &str,
    index_inst: &str,
    stop_at_ms: Option<i64>,
) -> Result<Vec<IndexCandleRow>> {
    let mut out: Vec<IndexCandleRow> = Vec::new();
    let mut after: Option<String> = None;

    loop {
        let mut query: Vec<(&str, String)> = vec![
            ("instId", index_inst.to_string()),
            ("bar", OKX_BAR.to_string()),
            ("limit", OKX_PAGE_LIMIT.to_string()),
        ];
        if let Some(a) = after.as_ref() {
            query.push(("after", a.clone()));
        }

        let resp: OkxEnvelope = http
            .get(url)
            .query(&query)
            .send()
            .await
            .with_context(|| format!("OKX index GET {}", index_inst))?
            .error_for_status()
            .with_context(|| format!("OKX index status {}", index_inst))?
            .json()
            .await
            .with_context(|| format!("OKX index JSON {}", index_inst))?;

        if resp.code != "0" && !resp.code.is_empty() {
            return Err(anyhow!(
                "OKX index error inst={} code={} msg={}",
                index_inst, resp.code, resp.msg
            ));
        }
        if resp.data.is_empty() {
            break;
        }

        let mut oldest_in_batch: Option<i64> = None;
        let mut last_ts_str: Option<String> = None;
        for c in &resp.data {
            if c.len() < 5 {
                continue;
            }
            let ts: i64 = c[0].parse().with_context(|| format!("parse idx ts '{}'", c[0]))?;
            oldest_in_batch = Some(match oldest_in_batch {
                Some(prev) => prev.min(ts),
                None => ts,
            });
            last_ts_str = Some(c[0].clone());
            if let Some(cutoff) = stop_at_ms {
                if ts <= cutoff {
                    continue;
                }
            }
            let open: f64 = c[1].parse().context("parse idx open")?;
            let high: f64 = c[2].parse().context("parse idx high")?;
            let low:  f64 = c[3].parse().context("parse idx low")?;
            let close: f64 = c[4].parse().context("parse idx close")?;
            out.push(IndexCandleRow {
                inst_id: index_inst.to_string(),
                ts_ms: ts,
                open,
                high,
                low,
                close,
            });
        }

        if let (Some(cutoff), Some(oldest)) = (stop_at_ms, oldest_in_batch) {
            if oldest <= cutoff {
                break;
            }
        }

        after = match last_ts_str {
            Some(ts) => Some(ts),
            None => break,
        };

        if out.len() as i64 * CANDLE_INTERVAL_MS > 5 * 365 * 86_400_000 {
            warn!("[backfill] {} hit 5y safety cap — stopping", index_inst);
            break;
        }

        sleep(OKX_RATE_SLEEP_PERP).await;
    }
    Ok(out)
}

/// Pages OKX newest-first, collecting candles with `ts_ms > stop_at_ms`.
async fn fetch_instrument(
    http: &Client,
    url: &str,
    inst_id: &str,
    stop_at_ms: Option<i64>,
) -> Result<Vec<CandleRow>> {
    let mut out: Vec<CandleRow> = Vec::new();
    let mut after: Option<String> = None;

    let sleep_duration = if url == OKX_HISTORY_CANDLES_URL {
        Duration::from_millis(250)
    } else {
        OKX_RATE_SLEEP
    };

    loop {
        let mut query: Vec<(&str, String)> = vec![
            ("instId", inst_id.to_string()),
            ("bar", OKX_BAR.to_string()),
            ("limit", OKX_PAGE_LIMIT.to_string()),
        ];
        if let Some(a) = after.as_ref() {
            query.push(("after", a.clone()));
        }

        let resp: OkxEnvelope = http
            .get(url)
            .query(&query)
            .send()
            .await
            .with_context(|| format!("OKX GET {}", inst_id))?
            .error_for_status()
            .with_context(|| format!("OKX status {}", inst_id))?
            .json()
            .await
            .with_context(|| format!("OKX JSON {}", inst_id))?;

        if resp.code != "0" && !resp.code.is_empty() {
            return Err(anyhow!("OKX error code={} msg={}", resp.code, resp.msg));
        }
        if resp.data.is_empty() {
            break;
        }

        let mut oldest_in_batch: Option<i64> = None;
        let mut last_ts_str: Option<String> = None;

        for c in &resp.data {
            if c.len() < 5 {
                continue;
            }
            let ts: i64 = c[0].parse().with_context(|| format!("parse ts '{}'", c[0]))?;
            oldest_in_batch = Some(match oldest_in_batch {
                Some(prev) => prev.min(ts),
                None => ts,
            });
            last_ts_str = Some(c[0].clone());
            if let Some(cutoff) = stop_at_ms {
                if ts <= cutoff {
                    continue;
                }
            }
            let open: f64 = c[1].parse().context("parse open")?;
            let high: f64 = c[2].parse().context("parse high")?;
            let low:  f64 = c[3].parse().context("parse low")?;
            let close: f64 = c[4].parse().context("parse close")?;
            out.push(CandleRow {
                inst_id: inst_id.to_string(),
                ts_ms: ts,
                open,
                high,
                low,
                close,
                tick_count: 1,
            });
        }

        if let (Some(cutoff), Some(oldest)) = (stop_at_ms, oldest_in_batch) {
            if oldest <= cutoff {
                break;
            }
        }

        after = match last_ts_str {
            Some(ts) => Some(ts),
            None => break,
        };

        if out.len() as i64 * CANDLE_INTERVAL_MS > 5 * 365 * 86_400_000 {
            warn!("[backfill] {} hit 5y safety cap — stopping pagination", inst_id);
            break;
        }

        sleep(sleep_duration).await;
    }
    Ok(out)
}
