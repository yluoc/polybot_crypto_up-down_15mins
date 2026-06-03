// Live polling task for perp feature inputs (funding rate, index close).
// Polls OKX public endpoints every 60s; per-instrument errors keep the
// prior snapshot rather than publishing a partial update.

use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, Result};
use reqwest::Client;
use serde::Deserialize;
use tokio::sync::watch;
use tokio::time::sleep;
use tracing::{debug, info, warn};

use crate::feature_engine::{short_symbol, PerpSample, INSTRUMENT_ORDER};

const POLL_INTERVAL: Duration = Duration::from_secs(60);
const REQ_SLEEP:     Duration = Duration::from_millis(250);
const HTTP_TIMEOUT:  Duration = Duration::from_secs(10);

const FUNDING_HISTORY_URL: &str = "https://www.okx.com/api/v5/public/funding-rate-history";
const INDEX_CANDLES_URL:   &str = "https://www.okx.com/api/v5/market/index-candles";

/// Snapshot keyed by SWAP inst_id (e.g. "BTC-USDT-SWAP" → PerpSample).
pub type PerpSnapshot = HashMap<String, PerpSample>;

#[derive(Deserialize)]
struct FundingItem {
    #[serde(rename = "fundingTime")]
    funding_time: String,
    #[serde(rename = "fundingRate")]
    funding_rate: String,
}

#[derive(Deserialize)]
struct FundingEnvelope {
    #[serde(default)]
    code: String,
    #[serde(default)]
    msg: String,
    #[serde(default)]
    data: Vec<FundingItem>,
}

/// `index-candles` payload; last column is `confirm` ("0"=in-progress, "1"=closed).
#[derive(Deserialize)]
struct IndexCandlesEnvelope {
    #[serde(default)]
    code: String,
    #[serde(default)]
    msg: String,
    #[serde(default)]
    data: Vec<Vec<String>>,
}

/// Spawn the perp polling task; returns the watch receiver (initial value: empty map).
pub fn spawn() -> watch::Receiver<PerpSnapshot> {
    let (tx, rx) = watch::channel::<PerpSnapshot>(HashMap::new());
    tokio::spawn(async move {
        let http = match Client::builder().timeout(HTTP_TIMEOUT).build() {
            Ok(c) => c,
            Err(e) => {
                warn!("[perp_poll] failed to build HTTP client: {e:#}");
                return;
            }
        };
        loop {
            match poll_once(&http).await {
                Ok(snapshot) => {
                    if snapshot.is_empty() {
                        warn!("[perp_poll] every instrument failed this cycle — keeping prior snapshot");
                    } else {
                        info!(
                            "[perp_poll] published snapshot for {} instruments",
                            snapshot.len()
                        );
                        if tx.send(snapshot).is_err() {
                            info!("[perp_poll] watch closed — exiting");
                            return;
                        }
                    }
                }
                Err(e) => {
                    warn!("[perp_poll] poll cycle errored: {e:#}");
                }
            }
            sleep(POLL_INTERVAL).await;
        }
    });
    rx
}

/// One polling cycle; instruments with any fetch failure are omitted from the result.
async fn poll_once(http: &Client) -> Result<PerpSnapshot> {
    let mut out = PerpSnapshot::with_capacity(INSTRUMENT_ORDER.len());
    for inst in INSTRUMENT_ORDER.iter() {
        match fetch_one(http, inst).await {
            Ok(sample) => {
                out.insert(inst.to_string(), sample);
            }
            Err(e) => {
                warn!("[perp_poll] {} fetch failed (skipping this cycle): {e:#}", inst);
            }
        }
    }
    Ok(out)
}

async fn fetch_one(http: &Client, inst_id: &str) -> Result<PerpSample> {
    let f: FundingEnvelope = http
        .get(FUNDING_HISTORY_URL)
        .query(&[("instId", inst_id), ("limit", "1")])
        .send()
        .await
        .with_context(|| format!("funding GET {inst_id}"))?
        .error_for_status()
        .with_context(|| format!("funding status {inst_id}"))?
        .json()
        .await
        .with_context(|| format!("funding JSON {inst_id}"))?;
    if f.code != "0" && !f.code.is_empty() {
        anyhow::bail!("funding error inst={inst_id} code={} msg={}", f.code, f.msg);
    }
    let f0 = f.data.first().context("funding empty data")?;
    let funding_settled_at_ms: i64 = f0.funding_time.parse()
        .with_context(|| format!("parse fundingTime '{}'", f0.funding_time))?;
    let funding_rate: f64 = f0.funding_rate.parse()
        .with_context(|| format!("parse fundingRate '{}'", f0.funding_rate))?;

    sleep(REQ_SLEEP).await;

    // Use the most recent CONFIRMED 15m index candle (confirm=="1").
    let index_inst = match inst_id.strip_suffix("-SWAP") {
        Some(s) => s,
        None => anyhow::bail!("inst_id {inst_id} doesn't have -SWAP suffix"),
    };
    let i: IndexCandlesEnvelope = http
        .get(INDEX_CANDLES_URL)
        .query(&[("instId", index_inst), ("bar", "15m"), ("limit", "2")])
        .send()
        .await
        .with_context(|| format!("index GET {inst_id}"))?
        .error_for_status()
        .with_context(|| format!("index status {inst_id}"))?
        .json()
        .await
        .with_context(|| format!("index JSON {inst_id}"))?;
    if i.code != "0" && !i.code.is_empty() {
        anyhow::bail!("index error inst={inst_id} code={} msg={}", i.code, i.msg);
    }
    let mut index_close: Option<f64> = None;
    for row in &i.data {
        if row.len() < 6 {
            continue;
        }
        if row[5] == "1" {
            let c: f64 = row[4].parse().with_context(|| format!("parse idx close '{}'", row[4]))?;
            index_close = Some(c);
            break;
        }
    }
    let index_close = index_close.context("no confirmed index candle in last 2 rows")?;

    debug!(
        "[perp_poll] {} ccy={} fund={} idx={}",
        inst_id, short_symbol(inst_id), funding_rate, index_close
    );

    Ok(PerpSample {
        funding_rate,
        funding_settled_at_ms,
        index_close,
    })
}
