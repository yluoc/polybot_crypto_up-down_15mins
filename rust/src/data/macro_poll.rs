// Live polling task for the global macro (FRED) feature inputs. Polls every
// 6h to catch the daily FRED refresh; per-series errors keep the prior
// snapshot value rather than publishing a partial update.

use std::collections::{BTreeMap, HashMap};
use std::time::Duration;

use anyhow::{Context, Result};
use chrono::NaiveDate;
use reqwest::Client;
use serde::Deserialize;
use sqlx::PgPool;
use tokio::sync::watch;
use tokio::time::sleep;
use tracing::{info, warn};

use crate::db::{models::MacroDailyRow, queries};
use crate::feature_engine::{MacroSeries, MacroSnapshot, MACRO_SERIES_IDS};

const POLL_INTERVAL: Duration = Duration::from_secs(6 * 3600);
const HTTP_TIMEOUT:  Duration = Duration::from_secs(15);
const REQ_SLEEP:     Duration = Duration::from_millis(250);

// Retry transient errors with backoff; 4xx is non-retryable.
const FRED_MAX_ATTEMPTS:  u32 = 3;
const FRED_RETRY_BACKOFF: Duration = Duration::from_millis(500);

const FRED_OBSERVATIONS_URL: &str = "https://api.stlouisfed.org/fred/series/observations";

#[derive(Deserialize)]
struct FredObservation {
    date:  String,
    value: String,
}

#[derive(Deserialize)]
struct FredEnvelope {
    #[serde(default)]
    observations: Vec<FredObservation>,
}

/// Spawn the FRED poller; returns the watch receiver (initial value: empty map).
pub fn spawn(pool: PgPool, fred_api_key: String) -> watch::Receiver<MacroSnapshot> {
    let (tx, rx) = watch::channel::<MacroSnapshot>(HashMap::new());
    tokio::spawn(async move {
        let http = match Client::builder().timeout(HTTP_TIMEOUT).build() {
            Ok(c) => c,
            Err(e) => {
                warn!("[macro_poll] failed to build HTTP client: {e:#}");
                return;
            }
        };
        let mut last_snap: MacroSnapshot = HashMap::with_capacity(MACRO_SERIES_IDS.len());
        loop {
            match poll_once(&http, &fred_api_key, &last_snap).await {
                Ok((snap, db_rows)) => {
                    if snap.is_empty() {
                        warn!("[macro_poll] every series failed this cycle — keeping prior snapshot");
                    } else {
                        info!(
                            "[macro_poll] published snapshot for {} series",
                            snap.len()
                        );
                        last_snap = snap.clone();
                        if tx.send(snap).is_err() {
                            info!("[macro_poll] watch closed — exiting");
                            return;
                        }
                        match queries::insert_macro_batch(&pool, &db_rows).await {
                            Ok(0) => {}
                            Ok(n) => info!("[macro_poll] persisted {} new macro_daily rows", n),
                            Err(e) => warn!("[macro_poll] macro_daily persist failed: {e:#}"),
                        }
                    }
                }
                Err(e) => {
                    warn!("[macro_poll] poll cycle errored: {e:#}");
                }
            }
            sleep(POLL_INTERVAL).await;
        }
    });
    rx
}

/// One polling cycle. Returns `(snapshot, db_rows)`; failed series inherit
/// the prior snapshot entry rather than being dropped.
async fn poll_once(
    http: &Client,
    api_key: &str,
    last_snap: &MacroSnapshot,
) -> Result<(MacroSnapshot, Vec<MacroDailyRow>)> {
    let mut snap = MacroSnapshot::with_capacity(MACRO_SERIES_IDS.len());
    let mut db_rows = Vec::with_capacity(MACRO_SERIES_IDS.len() * 2);
    for series_id in MACRO_SERIES_IDS.iter() {
        match fetch_latest_two(http, series_id, api_key).await {
            Ok(rows) if rows.len() >= 2 => {
                let (current_d, current_v) = rows[0];
                let (_prev_d, prev_v) = rows[1];
                snap.insert(series_id.to_string(), MacroSeries {
                    date_utc: current_d,
                    current:  current_v,
                    prev:     prev_v,
                });
                for (d, v) in &rows {
                    db_rows.push(MacroDailyRow {
                        series_id: series_id.to_string(),
                        date_utc:  *d,
                        value:     *v,
                    });
                }
            }
            Ok(rows) => {
                warn!(
                    "[macro_poll] {} returned {} observations (< 2) — keeping prior snapshot entry",
                    series_id, rows.len()
                );
                if let Some(prior) = last_snap.get(*series_id) {
                    snap.insert(series_id.to_string(), *prior);
                }
            }
            Err(e) => {
                warn!("[macro_poll] {} fetch failed (keeping prior): {e:#}", series_id);
                if let Some(prior) = last_snap.get(*series_id) {
                    snap.insert(series_id.to_string(), *prior);
                }
            }
        }
        sleep(REQ_SLEEP).await;
    }
    Ok((snap, db_rows))
}

/// Per-series fold state: `series_id → (current_date, current_value, prev (date, value))`.
type MacroFoldState = HashMap<String, (NaiveDate, f64, Option<(NaiveDate, f64)>)>;

/// Build a `BTreeMap<NaiveDate, MacroSnapshot>` from chronologically-sorted
/// `macro_daily` rows (sorted by `(date_utc, series_id)`). Used for
/// forward-fill lookups via `btree.range(..=date).next_back()`.
pub fn build_macro_btree(rows: &[MacroDailyRow]) -> BTreeMap<NaiveDate, MacroSnapshot> {
    // running per-series (current, prev) — populated as we walk rows
    let mut state: MacroFoldState = HashMap::new();
    let mut out: BTreeMap<NaiveDate, MacroSnapshot> = BTreeMap::new();
    let mut current_date: Option<NaiveDate> = None;

    for r in rows {
        // Snapshot state into BTreeMap when crossing a date boundary.
        if let Some(prev_date) = current_date {
            if r.date_utc != prev_date {
                out.insert(prev_date, state_to_snapshot(&state));
            }
        }
        current_date = Some(r.date_utc);

        match state.get(&r.series_id).copied() {
            Some((prev_d, prev_v, _)) if prev_d < r.date_utc => {
                state.insert(r.series_id.clone(), (r.date_utc, r.value, Some((prev_d, prev_v))));
            }
            Some((prev_d, _, prior_prior)) if prev_d == r.date_utc => {
                // Same-day duplicate: keep latest value, leave prior unchanged.
                state.insert(r.series_id.clone(), (r.date_utc, r.value, prior_prior));
            }
            Some(_) => {
                // Out-of-order row — skip; input is documented as chronological.
            }
            None => {
                state.insert(r.series_id.clone(), (r.date_utc, r.value, None));
            }
        }
    }
    if let Some(d) = current_date {
        out.insert(d, state_to_snapshot(&state));
    }
    out
}

fn state_to_snapshot(state: &MacroFoldState) -> MacroSnapshot {
    let mut snap = MacroSnapshot::with_capacity(state.len());
    for (s, (d, v, prev)) in state {
        // Both current and prev are required; series with only one observation are omitted.
        if let Some((_, pv)) = prev {
            snap.insert(s.clone(), MacroSeries {
                date_utc: *d,
                current:  *v,
                prev:     *pv,
            });
        }
    }
    snap
}

/// Fetch the two most recent non-missing observations for one series, newest-first.
async fn fetch_latest_two(
    http: &Client,
    series_id: &str,
    api_key: &str,
) -> Result<Vec<(NaiveDate, f64)>> {
    let env: FredEnvelope = fetch_with_retry(http, series_id, api_key).await?;

    let mut out: Vec<(NaiveDate, f64)> = Vec::with_capacity(2);
    for o in env.observations {
        if o.value == "." {
            continue;
        }
        let d = NaiveDate::parse_from_str(&o.date, "%Y-%m-%d")
            .with_context(|| format!("parse FRED date '{}'", o.date))?;
        let v: f64 = o.value
            .parse()
            .with_context(|| format!("parse FRED value '{}'", o.value))?;
        out.push((d, v));
        if out.len() == 2 {
            break;
        }
    }
    Ok(out)
}

/// One series GET with retry on transient failures.
async fn fetch_with_retry(
    http: &Client,
    series_id: &str,
    api_key: &str,
) -> Result<FredEnvelope> {
    let mut attempt: u32 = 0;
    loop {
        attempt += 1;
        let send_result = http
            .get(FRED_OBSERVATIONS_URL)
            .query(&[
                ("series_id",  series_id),
                ("api_key",    api_key),
                ("file_type",  "json"),
                ("sort_order", "desc"),
                ("limit",      "10"), // a few extra so the missing-value filter still leaves ≥2
            ])
            .send()
            .await;

        let resp = match send_result {
            Ok(r) => r,
            Err(e) => {
                let transient = e.is_timeout() || e.is_connect() || e.is_request();
                if transient && attempt < FRED_MAX_ATTEMPTS {
                    let backoff = FRED_RETRY_BACKOFF * 2u32.pow(attempt - 1);
                    warn!(
                        "[macro_poll] {} attempt {}/{} transient send error, retrying in {:?}: {:#}",
                        series_id, attempt, FRED_MAX_ATTEMPTS, backoff,
                        anyhow::Error::new(e),
                    );
                    sleep(backoff).await;
                    continue;
                }
                return Err(anyhow::Error::new(e))
                    .with_context(|| format!("FRED GET {series_id}"));
            }
        };

        let status = resp.status();
        if status.is_server_error() && attempt < FRED_MAX_ATTEMPTS {
            let backoff = FRED_RETRY_BACKOFF * 2u32.pow(attempt - 1);
            warn!(
                "[macro_poll] {} attempt {}/{} HTTP {}, retrying in {:?}",
                series_id, attempt, FRED_MAX_ATTEMPTS, status, backoff,
            );
            sleep(backoff).await;
            continue;
        }

        let resp = resp
            .error_for_status()
            .with_context(|| format!("FRED status {series_id}"))?;
        return resp
            .json::<FredEnvelope>()
            .await
            .with_context(|| format!("FRED JSON {series_id}"));
    }
}
