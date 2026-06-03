// Backfill FRED daily series into the `macro_daily` table. Idempotent;
// FRED's missing-value sentinel "." is filtered before insert.

use std::time::Duration;

use anyhow::{Context, Result};
use chrono::{NaiveDate, Utc};
use reqwest::Client;
use serde::Deserialize;
use sqlx::PgPool;
use tokio::time::sleep;
use tracing::{info, warn};

use crate::cli::BackfillMacroArgs;
use crate::db::{models::MacroDailyRow, queries};
use crate::feature_engine::MACRO_SERIES_IDS;

const FRED_OBSERVATIONS_URL: &str = "https://api.stlouisfed.org/fred/series/observations";
const FRED_HTTP_TIMEOUT: Duration = Duration::from_secs(15);
const FRED_REQ_SLEEP:    Duration = Duration::from_millis(250);

#[derive(Deserialize)]
struct FredObservation {
    date:  String,   // "YYYY-MM-DD"
    value: String,   // "." for missing, else stringified float
}

#[derive(Deserialize)]
struct FredEnvelope {
    #[serde(default)]
    observations: Vec<FredObservation>,
}

pub async fn run(pool: &PgPool, fred_api_key: &str, args: BackfillMacroArgs) -> Result<()> {
    let http = Client::builder().timeout(FRED_HTTP_TIMEOUT).build()?;
    let observation_start = (Utc::now().naive_utc().date()
        - chrono::Duration::days(args.days as i64))
        .format("%Y-%m-%d")
        .to_string();
    info!(
        "[backfill-macro] pulling {} FRED series since {} ({}d lookback)",
        MACRO_SERIES_IDS.len(),
        observation_start,
        args.days
    );

    let mut total_inserted: u64 = 0;
    for series_id in MACRO_SERIES_IDS.iter() {
        let rows = match fetch_series(&http, series_id, fred_api_key, &observation_start).await {
            Ok(rs) => rs,
            Err(e) => {
                warn!(
                    "[backfill-macro] {} fetch failed (skipping this series): {e:#}",
                    series_id
                );
                continue;
            }
        };
        let inserted = queries::insert_macro_batch(pool, &rows)
            .await
            .with_context(|| format!("insert_macro_batch {series_id}"))?;
        info!(
            "[backfill-macro] {} fetched {} rows, inserted {} new",
            series_id,
            rows.len(),
            inserted
        );
        total_inserted += inserted;
        sleep(FRED_REQ_SLEEP).await;
    }
    info!("[backfill-macro] done; {} new rows total", total_inserted);
    Ok(())
}

async fn fetch_series(
    http: &Client,
    series_id: &str,
    api_key: &str,
    observation_start: &str,
) -> Result<Vec<MacroDailyRow>> {
    let env: FredEnvelope = http
        .get(FRED_OBSERVATIONS_URL)
        .query(&[
            ("series_id",         series_id),
            ("api_key",           api_key),
            ("file_type",         "json"),
            ("observation_start", observation_start),
        ])
        .send()
        .await
        .with_context(|| format!("FRED GET {series_id}"))?
        .error_for_status()
        .with_context(|| format!("FRED status {series_id}"))?
        .json()
        .await
        .with_context(|| format!("FRED JSON {series_id}"))?;

    let mut out = Vec::with_capacity(env.observations.len());
    for o in env.observations {
        if o.value == "." {  // "." = missing observation; forward-fill covers the gap
            continue;
        }
        let date_utc = NaiveDate::parse_from_str(&o.date, "%Y-%m-%d")
            .with_context(|| format!("parse FRED date '{}'", o.date))?;
        let value: f64 = o.value
            .parse()
            .with_context(|| format!("parse FRED value '{}' for {}@{}", o.value, series_id, o.date))?;
        out.push(MacroDailyRow {
            series_id: series_id.to_string(),
            date_utc,
            value,
        });
    }
    Ok(out)
}
