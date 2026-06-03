// examples/validate_per_symbol_alignment.rs — validate per-symbol feature-row alignment.
// Reports, per symbol, how many training rows the per-symbol gate yields vs. the old
// global-alignment gate.
//
// Run:  DATABASE_URL=postgres://... cargo run --example \
//           validate_per_symbol_alignment -- [lookback_days]
//
// (Read-only: no writes, no model promotion. Defaults to 180 days.)

use std::collections::{BTreeMap, HashMap};

use anyhow::{anyhow, Result};
use polybot::db;
use polybot::feature_engine::{
    Candle, FeatureEngineer, MacroSnapshot, PerpSample, FEATURE_DIM, INSTRUMENT_ORDER,
    TOTAL_FEATURES,
};
use polybot::macro_poll;

/// True iff instrument `idx`'s 22-feature block is fully finite (i.e. that
/// instrument aligned at this bucket).
fn block_present(feats: &[f32; TOTAL_FEATURES], idx: usize) -> bool {
    let base = idx * FEATURE_DIM;
    feats[base..base + FEATURE_DIM].iter().all(|v| v.is_finite())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_max_level(tracing::Level::WARN).init();

    let lookback_days: i64 = std::env::args()
        .nth(1)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(180);
    let database_url = std::env::var("DATABASE_URL")
        .map_err(|_| anyhow!("DATABASE_URL must be set"))?;
    let pool = db::init_pool(&database_url).await?;

    let now_ms = chrono::Utc::now().timestamp_millis();
    let cutoff_ms = now_ms - lookback_days * 86_400_000;

    let candle_rows = db::queries::select_candles_since(&pool, cutoff_ms).await?;
    let funding_rows = db::queries::select_funding_since(&pool, cutoff_ms).await?;
    let index_rows = db::queries::select_index_candles_since(&pool, cutoff_ms).await?;
    let macro_cutoff_date = chrono::DateTime::from_timestamp_millis(cutoff_ms)
        .map(|d| d.date_naive() - chrono::Duration::days(7))
        .unwrap_or_else(|| chrono::NaiveDate::from_ymd_opt(2000, 1, 1).unwrap());
    let macro_rows = db::queries::select_macro_since(&pool, macro_cutoff_date).await?;
    let macro_btree = macro_poll::build_macro_btree(&macro_rows);

    let mut by_ts: BTreeMap<i64, HashMap<String, _>> = BTreeMap::new();
    for r in candle_rows {
        by_ts.entry(r.ts_ms).or_default().insert(r.inst_id.clone(), r);
    }

    // Forward-fill perp lookups, keyed by instrument.
    let mut funding: HashMap<String, BTreeMap<i64, f64>> = HashMap::new();
    let mut index: HashMap<String, BTreeMap<i64, f64>> = HashMap::new();
    for r in funding_rows { funding.entry(r.inst_id).or_default().insert(r.ts_ms, r.rate); }
    for r in index_rows   { index.entry(r.inst_id).or_default().insert(r.ts_ms, r.close); }

    let build_perp = |inst: &str, ts: i64| -> Option<PerpSample> {
        let (settled_ts, rate) = funding.get(inst)?.range(..=ts).next_back()?;
        let idx_close = *index.get(inst.strip_suffix("-SWAP")?)?.get(&ts)?;
        Some(PerpSample {
            funding_rate: *rate,
            funding_settled_at_ms: *settled_ts,
            index_close: idx_close,
        })
    };

    // Single replay through the per-symbol engine, keep-last-per-ts (mirrors retrain Pass 1).
    let mut engineer = FeatureEngineer::new();
    let mut feature_rows: Vec<(i64, [f32; TOTAL_FEATURES])> = Vec::new();
    for (&ts, inst_map) in by_ts.iter() {
        if let Some(date) = chrono::DateTime::from_timestamp_millis(ts).map(|d| d.date_naive()) {
            if let Some((_, snap)) = macro_btree.range(..=date).next_back() {
                engineer.push_macro_snapshot(snap as &MacroSnapshot);
            }
        }
        for inst in INSTRUMENT_ORDER.iter() {
            if let Some(sample) = build_perp(inst, ts) {
                engineer.push_perp_sample(inst, ts, sample);
            }
        }
        for inst in INSTRUMENT_ORDER.iter() {
            let Some(rc) = inst_map.get(*inst) else { continue };
            let candle = Candle {
                inst_id: rc.inst_id.clone(),
                open_ts_ms: rc.ts_ms,
                close_ts_ms: rc.ts_ms + 15 * 60 * 1000,
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

    // Dual-count.
    let mut per_symbol = [0usize; 4]; // new per-symbol gate
    let mut global = 0usize;          // old all-4-aligned gate
    let mut present_dist = [0usize; 5]; // index = # of instruments present in a row
    for (_, feats) in feature_rows.iter() {
        let mut n_present = 0;
        for (i, count) in per_symbol.iter_mut().enumerate() {
            if block_present(feats, i) {
                *count += 1;
                n_present += 1;
            }
        }
        present_dist[n_present] += 1;
        if n_present == INSTRUMENT_ORDER.len() {
            global += 1;
        }
    }

    let total = feature_rows.len().max(1);
    println!("\n=== per-symbol alignment validation ({lookback_days}d window) ===");
    println!("Symbol  | Old (global)  | New (per-symbol) | Lift");
    for (i, inst) in INSTRUMENT_ORDER.iter().enumerate() {
        let short = inst.split('-').next().unwrap_or(inst);
        let lift = if global > 0 {
            per_symbol[i] as f64 / global as f64
        } else {
            f64::NAN
        };
        println!(
            "{:<7} | {:<13} | {:<16} | {:.2}x",
            short, global, per_symbol[i], lift
        );
    }
    println!("\nDistribution of instruments present per emitted row:");
    for (n, &count) in present_dist.iter().enumerate() {
        println!(
            "  {} of 4 present: {:<7} ({:.0}%)",
            n,
            count,
            100.0 * count as f64 / total as f64
        );
    }
    println!(
        "\nold global-aligned rows : {global}\nnew rows (any subset)   : {}\n",
        feature_rows.len()
    );
    Ok(())
}
