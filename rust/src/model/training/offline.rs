// src/training/offline.rs — offline data-loading core (Pass 1/1b/2 replay).
// Read-only: no DB writes, no model promotion.

use std::collections::{BTreeMap, HashMap};

use anyhow::{anyhow, Result};
use sqlx::PgPool;

use crate::db::{self, models::CandleRow};
use crate::feature_engine::{
    Candle, FeatureEngineer, FeatureRow, MacroSnapshot, Normalizer, PerpSample,
    CANDLE_INTERVAL_MS, FEATURE_DIM, INSTRUMENT_ORDER, TOTAL_FEATURES,
};
use crate::macro_poll;
use crate::retrain::build_training_set;
use crate::signal::Symbol;
use crate::warmup;

/// Active symbols — the four with Polymarket history.
pub const ACTIVE_SYMBOLS: [&str; 4] = ["BTC", "ETH", "SOL", "XRP"];

/// One active symbol's offline training set (Pass 1/1b/2 replay).
/// `features_flat` is row-major `[n_rows x TOTAL_FEATURES]`, NaN-bearing.
pub struct OfflineSymbolData {
    pub symbol_short: String,
    pub features_flat: Vec<f64>,
    pub labels: Vec<f32>,
    /// Bucket open timestamps (ms) per row.
    pub timestamps_ms: Vec<i64>,
}

impl OfflineSymbolData {
    /// Number of labelled rows in this training set.
    pub fn n_rows(&self) -> usize {
        self.labels.len()
    }
}

/// Build a `PerpSample` for one (instrument, bucket) pair; `None` if either leg is missing.
fn build_perp_sample(
    inst_id: &str,
    bucket_ts: i64,
    funding: &HashMap<String, BTreeMap<i64, f64>>,
    index: &HashMap<String, BTreeMap<i64, f64>>,
) -> Option<PerpSample> {
    let (settled_ts, rate) = funding.get(inst_id)?.range(..=bucket_ts).next_back()?;
    let idx_close = *index.get(inst_id.strip_suffix("-SWAP")?)?.get(&bucket_ts)?;
    Some(PerpSample {
        funding_rate: *rate,
        funding_settled_at_ms: *settled_ts,
        index_close: idx_close,
    })
}

/// Replay `lookback_days` of data and build a per-symbol training set for each `ACTIVE_SYMBOL`.
pub async fn load_offline_training_sets(
    pool: &PgPool,
    lookback_days: i64,
) -> Result<Vec<OfflineSymbolData>> {
    let now_ms = chrono::Utc::now().timestamp_millis();
    let cutoff_ms = now_ms - lookback_days * 86_400_000;

    let candle_rows = db::queries::select_candles_since(pool, cutoff_ms).await?;
    let funding_rows = db::queries::select_funding_since(pool, cutoff_ms).await?;
    let index_rows = db::queries::select_index_candles_since(pool, cutoff_ms).await?;
    let macro_cutoff_date = chrono::DateTime::from_timestamp_millis(cutoff_ms)
        .map(|d| d.date_naive() - chrono::Duration::days(7))
        .unwrap_or_else(|| chrono::NaiveDate::from_ymd_opt(2000, 1, 1).unwrap());
    let macro_rows = db::queries::select_macro_since(pool, macro_cutoff_date).await?;
    let macro_btree = macro_poll::build_macro_btree(&macro_rows);

    let mut by_ts: BTreeMap<i64, HashMap<String, CandleRow>> = BTreeMap::new();
    for r in candle_rows {
        by_ts.entry(r.ts_ms).or_default().insert(r.inst_id.clone(), r);
    }

    let mut funding: HashMap<String, BTreeMap<i64, f64>> = HashMap::new();
    let mut index: HashMap<String, BTreeMap<i64, f64>> = HashMap::new();
    for r in funding_rows {
        funding.entry(r.inst_id).or_default().insert(r.ts_ms, r.rate);
    }
    for r in index_rows {
        index.entry(r.inst_id).or_default().insert(r.ts_ms, r.close);
    }

    let mut engineer = FeatureEngineer::new();
    let mut normalizer = Normalizer::new();
    let _ =
        warmup::warm_up_pipeline(pool, &mut engineer, &mut normalizer, Some(cutoff_ms)).await?;

    let mut feature_rows: Vec<(i64, [f32; TOTAL_FEATURES])> = Vec::new();
    for (&ts, inst_map) in by_ts.iter() {
        if let Some(date) = chrono::DateTime::from_timestamp_millis(ts).map(|d| d.date_naive()) {
            if let Some((_, snap)) = macro_btree.range(..=date).next_back() {
                engineer.push_macro_snapshot(snap as &MacroSnapshot);
            }
        }
        for inst in INSTRUMENT_ORDER.iter() {
            if let Some(sample) = build_perp_sample(inst, ts, &funding, &index) {
                engineer.push_perp_sample(inst, ts, sample);
            }
        }
        for inst in INSTRUMENT_ORDER.iter() {
            let Some(rc) = inst_map.get(*inst) else {
                continue;
            };
            let candle = Candle {
                inst_id: rc.inst_id.clone(),
                open_ts_ms: rc.ts_ms,
                close_ts_ms: rc.ts_ms + CANDLE_INTERVAL_MS,
                open: rc.open,
                high: rc.high,
                low: rc.low,
                close: rc.close,
                tick_count: rc.tick_count.max(0) as u32,
            };
            if let Some(row) = engineer.push_candle(candle) {
                if row.valid {
                    match feature_rows.last_mut() {
                        Some(last) if last.0 == ts => last.1 = row.features,
                        _ => feature_rows.push((ts, row.features)),
                    }
                }
            }
        }
    }

    // Pass 1b: Normalizer.
    for (ts, feats) in feature_rows.iter_mut() {
        let row = FeatureRow {
            candle_ts_ms: *ts,
            features: *feats,
            valid: true,
        };
        if let Some(normed) = normalizer.push(&row) {
            *feats = normed.features;
        }
    }

    // Pass 2: per-instrument close map for the label lookup.
    let mut close_by_inst: HashMap<&'static str, BTreeMap<i64, f64>> = HashMap::new();
    for inst in INSTRUMENT_ORDER.iter() {
        close_by_inst.insert(*inst, BTreeMap::new());
    }
    for (&ts, inst_map) in by_ts.iter() {
        for inst in INSTRUMENT_ORDER.iter() {
            if let Some(rc) = inst_map.get(*inst) {
                if let Some(m) = close_by_inst.get_mut(*inst) {
                    m.insert(ts, rc.close);
                }
            }
        }
    }

    let mut out = Vec::with_capacity(ACTIVE_SYMBOLS.len());
    for short in ACTIVE_SYMBOLS {
        let sym = Symbol::from_str_ci(short)?;
        let inst_id = sym.inst_id();
        let Some(closes) = close_by_inst.get(inst_id) else {
            continue; // unreachable for ACTIVE_SYMBOL, but stay defensive
        };
        let symbol_offset = INSTRUMENT_ORDER
            .iter()
            .position(|i| *i == inst_id)
            .ok_or_else(|| anyhow!("{inst_id} not in INSTRUMENT_ORDER"))?
            * FEATURE_DIM;

        let outcomes =
            db::queries::load_window_outcomes(pool, sym.short(), cutoff_ms / 1000, now_ms / 1000)
                .await?;
        let stats = build_training_set(&feature_rows, closes, &outcomes, symbol_offset);

        out.push(OfflineSymbolData {
            symbol_short: short.to_string(),
            features_flat: stats.flat,
            labels: stats.labels,
            timestamps_ms: stats.timestamps_ms,
        });
    }
    Ok(out)
}
