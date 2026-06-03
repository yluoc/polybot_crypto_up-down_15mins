
use std::collections::{BTreeMap, HashMap};

use anyhow::Result;
use chrono::DateTime;
use sqlx::PgPool;
use tracing::{info, warn};

use crate::db::{models::CandleRow, queries};
use crate::feature_engine::{
    Candle, FeatureEngineer, MacroSnapshot, Normalizer, PerpSample, CANDLE_INTERVAL_MS,
    INSTRUMENT_COUNT, INSTRUMENT_ORDER, MIN_HISTORY, NORM_WINDOW,
};
use crate::macro_poll;

/// Rows loaded per instrument to prime the normalizer and feature engine.
pub const WARMUP_PER_INST: usize = NORM_WINDOW + MIN_HISTORY + 10;

/// Replay recent candles from Postgres into `engineer` and `normalizer`.
/// Returns the number of normalized rows emitted.
/// `before_ts_ms`: when `Some`, replay candles strictly before that cutoff (used by retrain).
pub async fn warm_up_pipeline(
    pool: &PgPool,
    engineer: &mut FeatureEngineer,
    normalizer: &mut Normalizer,
    before_ts_ms: Option<i64>,
) -> Result<usize> {
    let rows =
        queries::select_recent_candles_for_warmup(pool, WARMUP_PER_INST, before_ts_ms).await?;
    if rows.is_empty() {
        warn!(
            "[warmup] no candles in DB — pipeline will take ~{}h of live ticks to warm up",
            (NORM_WINDOW + MIN_HISTORY) / 4
        );
        return Ok(0);
    }

    let per_inst_min = min_rows_per_inst(&rows);
    if per_inst_min < MIN_HISTORY {
        warn!(
            "[warmup] only {} candles for the thinnest instrument (need {} for features, \
             {} for stable z-scores) — normalizer stats will be noisy until more live \
             candles accumulate",
            per_inst_min, MIN_HISTORY, NORM_WINDOW
        );
    } else if per_inst_min < NORM_WINDOW {
        warn!(
            "[warmup] thinnest instrument has {} candles (< NORM_WINDOW={}) — normalizer \
             stats will be noisy until more live candles accumulate",
            per_inst_min, NORM_WINDOW
        );
    }

    let earliest_ts = rows.iter().map(|r| r.ts_ms).min().unwrap_or(0);
    let perp = load_perp_lookups(pool, earliest_ts).await?;
    info!(
        "[warmup] perp lookups: funding_insts={}, index_insts={}",
        perp.funding.len(), perp.index.len()
    );

    let macro_cutoff_date = DateTime::from_timestamp_millis(earliest_ts)
        .map(|d| d.date_naive() - chrono::Duration::days(7))
        .unwrap_or_else(|| chrono::NaiveDate::from_ymd_opt(2000, 1, 1).unwrap());
    let macro_rows = queries::select_macro_since(pool, macro_cutoff_date).await?;
    let macro_btree = macro_poll::build_macro_btree(&macro_rows);
    info!(
        "[warmup] macro lookups: {} rows, {} distinct dates",
        macro_rows.len(),
        macro_btree.len()
    );

    let total = rows.len();
    let mut emitted_raw = 0usize;
    let mut emitted_norm = 0usize;
    for r in rows {
        let inst_id = r.inst_id.clone();
        let bucket_ts = r.ts_ms;
        if let Some(sample) = perp.build_for(&inst_id, bucket_ts) {
            engineer.push_perp_sample(&inst_id, bucket_ts, sample);
        }
        if let Some(snap) = lookup_macro_for_bucket(&macro_btree, bucket_ts) {
            engineer.push_macro_snapshot(snap);
        }
        let candle = candle_from_row(r);
        if let Some(raw) = engineer.push_candle(candle) {
            emitted_raw += 1;
            if normalizer.push(&raw).is_some() {
                emitted_norm += 1;
            }
        }
    }

    info!(
        "[warmup] replayed {} candles across {} instruments → {} raw rows, \
         {} normalized rows emitted, pipeline ready",
        total, INSTRUMENT_COUNT, emitted_raw, emitted_norm
    );
    Ok(emitted_norm)
}

struct PerpLookups {
    funding: HashMap<String, BTreeMap<i64, f64>>, // inst_id (swap) → (settlement_ts → rate)
    index:   HashMap<String, BTreeMap<i64, f64>>, // inst_id (e.g. "BTC-USDT") → (bucket_ts → close)
}

impl PerpLookups {
    fn build_for(&self, inst_id: &str, bucket_ts: i64) -> Option<PerpSample> {
        // Forward-fill: most recent funding settlement with ts ≤ bucket_ts.
        let f = self.funding.get(inst_id)?;
        let (settled_ts, rate) = f.range(..=bucket_ts).next_back()?;
        let index_inst = inst_id.strip_suffix("-SWAP")?;
        let i = self.index.get(index_inst)?;
        let idx_close = *i.get(&bucket_ts)?;
        Some(PerpSample {
            funding_rate:          *rate,
            funding_settled_at_ms: *settled_ts,
            index_close:           idx_close,
        })
    }
}

async fn load_perp_lookups(pool: &PgPool, since_ts_ms: i64) -> Result<PerpLookups> {
    let funding_rows = queries::select_funding_since(pool, since_ts_ms).await?;
    let index_rows = queries::select_index_candles_since(pool, since_ts_ms).await?;

    let mut funding: HashMap<String, BTreeMap<i64, f64>> = HashMap::with_capacity(INSTRUMENT_COUNT);
    let mut index:   HashMap<String, BTreeMap<i64, f64>> = HashMap::with_capacity(INSTRUMENT_COUNT);
    for inst in INSTRUMENT_ORDER.iter() {
        funding.insert((*inst).to_string(), BTreeMap::new());
        if let Some(idx) = inst.strip_suffix("-SWAP") {
            index.insert(idx.to_string(), BTreeMap::new());
        }
    }
    for r in funding_rows {
        funding.entry(r.inst_id).or_default().insert(r.ts_ms, r.rate);
    }
    for r in index_rows {
        index.entry(r.inst_id).or_default().insert(r.ts_ms, r.close);
    }

    // Log once at startup for instruments with no funding rows.
    for inst in INSTRUMENT_ORDER.iter() {
        let fcount = funding.get(*inst).map(|m| m.len()).unwrap_or(0);
        if fcount == 0 {
            warn!(
                "[warmup] perp shortfall for {}: funding_rows={} since ts={}; \
                 rows containing this symbol will be blocked until backfill catches up",
                inst, fcount, since_ts_ms
            );
        }
    }

    Ok(PerpLookups { funding, index })
}

fn candle_from_row(r: CandleRow) -> Candle {
    Candle {
        inst_id: r.inst_id,
        open_ts_ms: r.ts_ms,
        close_ts_ms: r.ts_ms + CANDLE_INTERVAL_MS - 1,
        open: r.open,
        high: r.high,
        low: r.low,
        close: r.close,
        tick_count: r.tick_count.max(0) as u32,
    }
}

/// Forward-fill: return the macro snapshot for the most recent date ≤ the bucket's UTC date.
fn lookup_macro_for_bucket(
    btree: &BTreeMap<chrono::NaiveDate, MacroSnapshot>,
    bucket_ts_ms: i64,
) -> Option<&MacroSnapshot> {
    let date = DateTime::from_timestamp_millis(bucket_ts_ms)?.date_naive();
    btree.range(..=date).next_back().map(|(_, s)| s)
}

fn min_rows_per_inst(rows: &[CandleRow]) -> usize {
    use std::collections::HashMap;
    let mut counts: HashMap<&str, usize> = HashMap::with_capacity(INSTRUMENT_COUNT);
    for r in rows {
        *counts.entry(r.inst_id.as_str()).or_insert(0) += 1;
    }
    if counts.len() < INSTRUMENT_COUNT {
        return 0;
    }
    counts.values().copied().min().unwrap_or(0)
}
