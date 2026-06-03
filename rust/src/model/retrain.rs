// src/retrain.rs
//
// Per-symbol LightGBM retrain driver. Reads recent candles from Postgres,
// runs them through the shared FeatureEngineer, normalises each aligned row
// with a fresh Normalizer (z-score, same transform as live inference),
// labels each row via vol-scaled threshold on OKX log returns (gated on
// a settled Polymarket window_outcomes row), trains a 2-class multiclass
// Booster, and promotes the resulting model into the `models` table.
//
// Class encoding (LightGBM-internal, 0-indexed): UP=0, DOWN=1. The DB
// `signals.signal` column uses a different encoding (1=UP, 2=DOWN); see signal.rs.
//
// Sanity gates (evaluated per symbol, BEFORE promotion):
//   1. Class distribution not degenerate.
//   2. Labeled row count not collapsed vs the prior retrain (< 50%).
//   3. Training convergence — reject if training logloss is non-finite.
//   4. Purged k-fold CV (de Prado, AFML Ch. 7) — Wilson 95% one-sided
//      lower bound must clear that fold's majority baseline by
//      GATE4_BASELINE_MARGIN; ≥ CV_MIN_FOLDS_PASSING of CV_K_FOLDS must
//      pass. Skipped when total labelled set < MIN_ROWS_FOR_GATE.

use std::collections::{BTreeMap, HashMap};

use anyhow::{anyhow, Context, Result};
use lightgbm3::{Booster, Dataset};
use serde_json::json;
use sqlx::PgPool;
use tracing::{info, warn};

use crate::calibration::{self, BetaCal};
use crate::cli::RetrainArgs;
use crate::db::{self, models::CandleRow};
use crate::feature_engine::{
    short_symbol, Candle, FeatureEngineer, FeatureRow, MacroSnapshot, MicroSample, Normalizer,
    PerpSample, CANDLE_INTERVAL_MS, FEATURE_DIM, GLOBAL_FEATURE_DIM, INSTRUMENT_COUNT,
    INSTRUMENT_ORDER, TOTAL_FEATURES,
};
use crate::inference::model_hub::{CURRENT_FORMAT_VERSION, POOLED_INPUT_DIM};
use crate::macro_poll;
use crate::signal::{symbol_id, Symbol};
use crate::training::lr_l1;
use crate::warmup::{self, WARMUP_PER_INST};

/// Per-symbol training snapshot stashed for the pooled multi-task comparison.
/// Public for diagnostic tooling (`evaluate_pooled_cv`) and tests.
pub struct PerSymbolPoolInput {
    pub sym: Symbol,
    pub features_flat: Vec<f64>,
    pub labels: Vec<f32>,
    pub timestamps_ms: Vec<i64>,
    pub per_symbol_version_id: i64,
}

pub async fn run(
    pool: &PgPool,
    cfg_cryptos: &std::collections::HashSet<Symbol>,
    args: RetrainArgs,
) -> Result<()> {
    let now_ms = chrono::Utc::now().timestamp_millis();
    let cutoff_ms = now_ms - (args.lookback_days as i64) * 86_400_000;

    info!(
        "[startup] feature layout: format_version={} FEATURE_DIM={} GLOBAL_FEATURE_DIM={} TOTAL_FEATURES={}",
        CURRENT_FORMAT_VERSION, FEATURE_DIM, GLOBAL_FEATURE_DIM, TOTAL_FEATURES
    );
    info!(
        "[retrain] loading candles >= {} ({}d lookback)",
        cutoff_ms, args.lookback_days
    );

    let half_life_days = {
        let requested = args.sample_weight_half_life_days;
        if requested < SAMPLE_WEIGHT_HALF_LIFE_MIN {
            warn!(
                "[retrain] sample_weight_half_life_days={} outside [{}, {}], \
                 clamped to {}",
                requested, SAMPLE_WEIGHT_HALF_LIFE_MIN, SAMPLE_WEIGHT_HALF_LIFE_MAX,
                SAMPLE_WEIGHT_HALF_LIFE_MIN
            );
            SAMPLE_WEIGHT_HALF_LIFE_MIN
        } else if requested > SAMPLE_WEIGHT_HALF_LIFE_MAX {
            warn!(
                "[retrain] sample_weight_half_life_days={} outside [{}, {}], \
                 clamped to {}",
                requested, SAMPLE_WEIGHT_HALF_LIFE_MIN, SAMPLE_WEIGHT_HALF_LIFE_MAX,
                SAMPLE_WEIGHT_HALF_LIFE_MAX
            );
            SAMPLE_WEIGHT_HALF_LIFE_MAX
        } else {
            requested
        }
    };
    info!(
        "[retrain] sample_weight_half_life_days={:.1}",
        half_life_days
    );
    info!(
        "[retrain] expected aligned rows ≈ {} (= {}d × 96 candles/day, minus gaps)",
        args.lookback_days * 96,
        args.lookback_days,
    );
    let rows = db::queries::select_candles_since(pool, cutoff_ms).await?;
    if rows.is_empty() {
        return Err(anyhow!(
            "[retrain] no candles in DB for the last {} days — run `polybot backfill` first",
            args.lookback_days
        ));
    }
    info!("[retrain] loaded {} candle rows", rows.len());

    // Group by ts_ms → {inst_id → CandleRow}. BTreeMap keeps chronological order.
    let mut by_ts: BTreeMap<i64, HashMap<String, CandleRow>> = BTreeMap::new();
    for r in rows {
        by_ts
            .entry(r.ts_ms)
            .or_default()
            .insert(r.inst_id.clone(), r);
    }
    info!("[retrain] {} distinct buckets", by_ts.len());

    let funding_rows = db::queries::select_funding_since(pool, cutoff_ms).await?;
    let index_rows   = db::queries::select_index_candles_since(pool, cutoff_ms).await?;
    info!(
        "[retrain] perp inputs: funding={} index_candles={}",
        funding_rows.len(), index_rows.len()
    );

    let macro_cutoff_date = chrono::DateTime::from_timestamp_millis(cutoff_ms)
        .map(|d| d.date_naive() - chrono::Duration::days(7))
        .unwrap_or_else(|| chrono::NaiveDate::from_ymd_opt(2000, 1, 1).unwrap());
    let macro_rows = db::queries::select_macro_since(pool, macro_cutoff_date).await?;
    let macro_btree: std::collections::BTreeMap<chrono::NaiveDate, MacroSnapshot> =
        macro_poll::build_macro_btree(&macro_rows);
    info!(
        "[retrain] macro inputs: {} rows across {} distinct dates",
        macro_rows.len(),
        macro_btree.len()
    );
    if macro_btree.is_empty() {
        warn!(
            "[retrain] no macro_daily rows in lookback window — every feature row will be \
             blocked by the macro gate. Run `polybot backfill-macro` first."
        );
    }

    let mut funding_by_inst: HashMap<String, BTreeMap<i64, f64>> = HashMap::with_capacity(INSTRUMENT_ORDER.len());
    let mut index_by_inst:   HashMap<String, BTreeMap<i64, f64>> = HashMap::with_capacity(INSTRUMENT_ORDER.len());
    for r in funding_rows { funding_by_inst.entry(r.inst_id).or_default().insert(r.ts_ms, r.rate); }
    for r in index_rows   { index_by_inst  .entry(r.inst_id).or_default().insert(r.ts_ms, r.close); }

    // NaN-safe: missing rows produce per-slot NaN; LightGBM splits on NaN — never blocks row emission.
    let mut oi_by_short: HashMap<String, BTreeMap<i64, f64>> = HashMap::with_capacity(4);
    let mut liq_by_short: HashMap<String, BTreeMap<i64, (f64, f64)>> = HashMap::with_capacity(4);
    let mut taker_by_inst: HashMap<String, BTreeMap<i64, (f64, f64)>> = HashMap::with_capacity(4);
    for inst in INSTRUMENT_ORDER.iter() {
        let short = inst.split('-').next().unwrap_or("");
        match db::queries::select_oi_aggregated_range(pool, short, cutoff_ms, now_ms).await {
            Ok(m) => { oi_by_short.insert(short.to_string(), m); }
            Err(e) => warn!("[retrain] load aggregated OI for {short} failed: {e:#}"),
        }
        match db::queries::select_liq_aggregated_range(pool, short, cutoff_ms, now_ms).await {
            Ok(m) => { liq_by_short.insert(short.to_string(), m); }
            Err(e) => warn!("[retrain] load aggregated liq for {short} failed: {e:#}"),
        }
        match db::queries::select_taker_volume_range(pool, inst, cutoff_ms, now_ms).await {
            Ok(m) => { taker_by_inst.insert(inst.to_string(), m); }
            Err(e) => warn!("[retrain] load taker volume for {inst} failed: {e:#}"),
        }
    }
    let total_oi:    usize = oi_by_short.values().map(|m| m.len()).sum();
    let total_liq:   usize = liq_by_short.values().map(|m| m.len()).sum();
    let total_taker: usize = taker_by_inst.values().map(|m| m.len()).sum();
    info!(
        "[retrain] micro inputs: oi_rows={total_oi} liq_rows={total_liq} taker_rows={total_taker} \
         (NaN-tolerant — partial coverage expected)"
    );

    for inst in INSTRUMENT_ORDER.iter() {
        let f = funding_by_inst.get(*inst).map(|m| m.len()).unwrap_or(0);
        if f == 0 {
            warn!(
                "[retrain] perp shortfall for {}: funding_rows=0 — \
                 buckets containing this symbol will be excluded from training",
                inst
            );
        }
    }

    let mut feature_rows: Vec<(i64, [f32; TOTAL_FEATURES])> = Vec::with_capacity(by_ts.len());
    let mut engineer = FeatureEngineer::new();
    let mut normalizer = Normalizer::new();

    info!(
        "[retrain] starting warmup pre-seed (~{} candles per instrument)",
        WARMUP_PER_INST
    );
    let seeded =
        warmup::warm_up_pipeline(pool, &mut engineer, &mut normalizer, Some(cutoff_ms)).await?;
    if seeded == 0 {
        warn!(
            "[retrain] warmup pre-seed emitted 0 aligned rows — proceeding with cold \
             feature-engine state; Pass 1 still emits training rows from in-window data. \
             (cutoff_ms={cutoff_ms})"
        );
    } else {
        info!("[retrain] warmup complete ({} rows seeded); entering per-bucket emit loop", seeded);
    }

    for (&ts, inst_map) in by_ts.iter() {
        if let Some(date) =
            chrono::DateTime::from_timestamp_millis(ts).map(|d| d.date_naive())
        {
            if let Some((_, snap)) = macro_btree.range(..=date).next_back() {
                engineer.push_macro_snapshot(snap);
            }
        }
        for inst in INSTRUMENT_ORDER.iter() {
            if let Some(sample) = build_perp_sample(inst, ts, &funding_by_inst, &index_by_inst) {
                engineer.push_perp_sample(inst, ts, sample);
            }
        }
        for inst in INSTRUMENT_ORDER.iter() {
            let short = inst.split('-').next().unwrap_or("");
            let (taker_buy, taker_sell) = taker_by_inst
                .get(*inst)
                .and_then(|m| m.get(&ts).copied())
                .map(|(b, s)| (Some(b), Some(s)))
                .unwrap_or((None, None));
            let oi_usd = oi_by_short.get(short).and_then(|m| m.get(&ts).copied());
            // Absence treated as Some(0.0) so the bucket counts toward liq_imbalance_4's window.
            let (long_liq, short_liq) = match liq_by_short.get(short).and_then(|m| m.get(&ts).copied()) {
                Some((l, s)) => (Some(l), Some(s)),
                None => (Some(0.0), Some(0.0)),
            };
            let sample = MicroSample {
                taker_buy_vol:  taker_buy,
                taker_sell_vol: taker_sell,
                oi_usd,
                long_liq_usd:   long_liq,
                short_liq_usd:  short_liq,
            };
            engineer.push_micro_sample(inst, ts, &sample);
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
                tick_count: rc.tick_count as u32,
            };
            if let Some(row) = engineer.push_candle(candle) {
                if row.valid {
                    // Keep the last emission per ts — the final one carries all aligned instruments.
                    match feature_rows.last_mut() {
                        Some(last) if last.0 == ts => last.1 = row.features,
                        _ => feature_rows.push((ts, row.features)),
                    }
                }
            }
        }
    }
    info!(
        "[retrain] {} aligned feature rows emitted",
        feature_rows.len()
    );
    if feature_rows.len() < 200 {
        return Err(anyhow!(
            "[retrain] only {} aligned rows — need at least 200 for a meaningful fit. \
             Check that all 7 instruments are being backfilled.",
            feature_rows.len()
        ));
    }

    let (mut mom_align_up, mut mom_align_flat, mut mom_align_down) = (0usize, 0usize, 0usize);
    let mut mom_1h_sum: f64 = 0.0;
    let mut mom_1h_n: usize = 0;
    for (_, feats) in feature_rows.iter() {
        for i in 0..INSTRUMENT_ORDER.len() {
            let base = i * FEATURE_DIM;
            mom_1h_sum += feats[base + 16] as f64;
            mom_1h_n += 1;
            let align = feats[base + 19];
            if align > 0.0 {
                mom_align_up += 1;
            } else if align < 0.0 {
                mom_align_down += 1;
            } else {
                mom_align_flat += 1;
            }
        }
    }
    let mom_1h_mean = if mom_1h_n > 0 { mom_1h_sum / mom_1h_n as f64 } else { 0.0 };
    info!(
        "[retrain] new momentum feature dist — mom_1h mean={:.4}, mom_align dist UP/FLAT/DOWN={}/{}/{}",
        mom_1h_mean, mom_align_up, mom_align_flat, mom_align_down
    );

    for (i, inst) in INSTRUMENT_ORDER.iter().enumerate() {
        let base = i * FEATURE_DIM;
        let (mut n_neg2, mut n_zero, mut n_pos2, mut n) = (0usize, 0usize, 0usize, 0usize);
        for (_, feats) in feature_rows.iter() {
            let v = feats[base + 17];
            if v == -2.0 {
                n_neg2 += 1;
            } else if v == 2.0 {
                n_pos2 += 1;
            } else if v == 0.0 {
                n_zero += 1;
            }
            n += 1;
        }
        if n == 0 {
            continue;
        }
        let squeeze_pct = n_neg2 as f64 / n as f64 * 100.0;
        info!(
            "[retrain:{}] funding_price_div dist — squeeze_pct={:.1}%, signed_-2/0/+2={}/{}/{}",
            short_symbol(inst), squeeze_pct, n_neg2, n_zero, n_pos2
        );
    }

    // Pass 1b: normalize every row with the same z-score transform used at inference time.
    for (ts, feats) in feature_rows.iter_mut() {
        let row = FeatureRow {
            candle_ts_ms: *ts,
            features: *feats,
            valid: true,
        };
        let normed = normalizer
            .push(&row)
            .ok_or_else(|| anyhow!("Normalizer::push returned None for valid row @ ts={ts}"))?;
        *feats = normed.features;
    }

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

    let mut promoted: Vec<String> = Vec::new();
    let mut rejections: HashMap<String, String> = HashMap::new();
    let mut pool_inputs: HashMap<Symbol, PerSymbolPoolInput> = HashMap::new();
    for &sym in cfg_cryptos.iter() {
        // Diagnostic accumulators written to retrain_diagnostics on every exit path.
        let mut diag_labeled_rows: i32 = 0;
        let mut diag_train_acc: f64 = 0.0;
        let mut diag_val_acc: Option<f64> = None;
        let mut diag_majority: Option<f64> = None;
        let mut diag_agreement: Option<f64> = None;
        let mut diag_logloss: f64 = 0.0;
        let mut diag_iterations: i32 = 0;

        macro_rules! record_diag {
            ($promoted:expr, $gate:expr) => {
                if let Err(e) = db::queries::insert_retrain_diagnostic(
                    pool,
                    sym.short(),
                    diag_labeled_rows,
                    diag_train_acc,
                    diag_val_acc,
                    diag_majority,
                    diag_agreement,
                    diag_logloss,
                    diag_iterations,
                    $promoted,
                    $gate,
                )
                .await
                {
                    warn!(
                        "[retrain:{}] insert_retrain_diagnostic failed (non-fatal): {}",
                        sym.short(),
                        e
                    );
                }
            };
        }

        let inst_id = sym.inst_id();
        let closes = close_by_inst
            .get(inst_id)
            .ok_or_else(|| anyhow!("no closes collected for {}", inst_id))?;
        let symbol_offset = INSTRUMENT_ORDER
            .iter()
            .position(|i| *i == inst_id)
            .ok_or_else(|| anyhow!("{} not in INSTRUMENT_ORDER", inst_id))?
            * FEATURE_DIM;

        let from_secs = cutoff_ms / 1000;
        let to_secs = now_ms / 1000;
        let outcomes = db::queries::load_window_outcomes(pool, sym.short(), from_secs, to_secs)
            .await
            .with_context(|| format!("load_window_outcomes {}", sym.short()))?;
        info!(
            "[retrain:{}] loaded {} Polymarket-resolved outcomes",
            sym.short(),
            outcomes.len()
        );

        let LabelingStats {
            flat: features_flat,
            labels,
            disagree,
            agree,
            dropped_vol_filter,
            dropped_no_sigma,
            median_sigma_bps,
            timestamps_ms: per_sym_timestamps,
        } = build_training_set(&feature_rows, closes, &outcomes, symbol_offset);
        let kept_rows = labels.len();
        let dropped_rows = dropped_vol_filter + dropped_no_sigma;
        let drop_pct = {
            let denom = (kept_rows + dropped_rows) as f64;
            if denom > 0.0 {
                100.0 * dropped_rows as f64 / denom
            } else {
                0.0
            }
        };
        info!(
            "[retrain:{}] vol-scaled labeler — kept={} dropped={} ({:.1}%) \
             [vol_filter={} no_sigma={}] median_sigma={:.2}bps k={}",
            sym.short(),
            kept_rows,
            dropped_rows,
            drop_pct,
            dropped_vol_filter,
            dropped_no_sigma,
            median_sigma_bps,
            VOL_SCALE_K,
        );
        if features_flat.is_empty() {
            warn!(
                "[retrain:{}] 0 labelled rows — skipping. Is {} present in every bucket, \
                 and has `polybot backfill-outcomes` populated window_outcomes?",
                sym.short(),
                inst_id
            );
            rejections.insert(sym.short().to_string(), "zero_labelled_rows".to_string());
            record_diag!(false, Some("zero_labelled_rows"));
            continue;
        }

        let per_sym_full_weights =
            exponential_decay_weights(&per_sym_timestamps, half_life_days);

        {
            let last_row_start = (kept_rows - 1) * TOTAL_FEATURES;
            let g_base = last_row_start + INSTRUMENT_COUNT * FEATURE_DIM;
            let g = &features_flat[g_base..g_base + GLOBAL_FEATURE_DIM];
            info!(
                "[retrain:{}] v13 global block — release_day={:.0} gated_dxy={:.5} \
                 gated_vix={:.3} gated_y10={:.2}bps vix_level={:.2}",
                sym.short(),
                g[0],
                g[1],
                g[2],
                g[3],
                g[4],
            );
        }

        let compared = agree + disagree;
        if compared > 0 {
            info!(
                "[retrain:{}] label source: OKX vs Polymarket agreement {}/{} = {:.2}% \
                 (disagreement {}/{} = {:.2}%)",
                sym.short(),
                agree,
                compared,
                100.0 * agree as f64 / compared as f64,
                disagree,
                compared,
                100.0 * disagree as f64 / compared as f64,
            );
            diag_agreement = Some(agree as f64 / compared as f64);
        }

        let n_rows = labels.len();
        diag_labeled_rows = n_rows.try_into().unwrap_or(i32::MAX);
        let counts = class_counts(&labels);
        log_class_dist(sym.short(), counts, n_rows);

        if let Err(reason) = check_class_distribution(counts) {
            warn!(
                "[retrain:{}] Gate 1 (class distribution) REJECTED: {} — skipping",
                sym.short(),
                reason
            );
            rejections.insert(
                sym.short().to_string(),
                "gate_1_class_distribution".to_string(),
            );
            record_diag!(false, Some("gate_1_class_distribution"));
            continue;
        }

        let prior_meta = db::queries::prior_promoted_meta(pool, sym.short())
            .await
            .with_context(|| format!("prior_promoted_meta {}", sym.short()))?;
        match prior_meta {
            None => info!(
                "[retrain:{}] Gate 2 (row-count stability) SKIPPED — no prior promoted model",
                sym.short()
            ),
            Some((prior_count, prior_format_version))
                if prior_format_version != CURRENT_FORMAT_VERSION =>
            {
                info!(
                    "[retrain:{}] Gate 2 (row-count stability) BYPASSED — format_version \
                     cutover {prior_format_version}→{} (one-time per boundary); new={} prior={:?}",
                    sym.short(),
                    CURRENT_FORMAT_VERSION,
                    n_rows,
                    prior_count,
                );
            }
            Some((None, _)) => info!(
                "[retrain:{}] Gate 2 (row-count stability) SKIPPED — prior row predates \
                 labeled_row_count column",
                sym.short()
            ),
            Some((Some(prior), _)) => {
                if let Err(reason) = check_row_count_stability(n_rows, Some(prior)) {
                    warn!(
                        "[retrain:{}] Gate 2 (row-count stability) REJECTED: {} — skipping",
                        sym.short(),
                        reason
                    );
                    rejections.insert(sym.short().to_string(), "gate_2_row_count".to_string());
                    record_diag!(false, Some("gate_2_row_count"));
                    continue;
                }
                info!(
                    "[retrain:{}] Gate 2 (row-count stability) OK — new={} prior={}",
                    sym.short(),
                    n_rows,
                    prior
                );
            }
        }

        let do_holdout = n_rows >= MIN_ROWS_FOR_GATE;
        let (train_feats, train_labels, val_feats, val_labels) = if do_holdout {
            let n_train_raw = ((n_rows as f64) * (1.0 - HOLDOUT_FRAC)) as usize;
            let n_train = n_train_raw - EMBARGO_ROWS;
            let n_val = n_rows - n_train_raw;
            info!(
                "[retrain:{}] holdout split — train={} embargo={} val={}",
                sym.short(),
                n_train,
                EMBARGO_ROWS,
                n_val
            );
            (
                &features_flat[..n_train * TOTAL_FEATURES],
                &labels[..n_train],
                &features_flat[n_train_raw * TOTAL_FEATURES..n_rows * TOTAL_FEATURES],
                &labels[n_train_raw..n_rows],
            )
        } else {
            info!(
                "[retrain:{}] Gate 4 (validation accuracy) SKIPPED — only {} rows, need ≥ {}",
                sym.short(),
                n_rows,
                MIN_ROWS_FOR_GATE
            );
            (&features_flat[..], &labels[..], &[][..], &[][..])
        };

        let labeled_row_count: i32 = n_rows.try_into().with_context(|| {
            format!("labeled_row_count overflow for {}: {}", sym.short(), n_rows)
        })?;

        if !do_holdout {
            info!(
                "[retrain:per_symbol:{}] training on {} rows (ESS={:.0}, half_life={:.1}d) — small-history fallback",
                sym.short(),
                train_labels.len(),
                effective_sample_size(&per_sym_full_weights),
                half_life_days
            );
            let booster = match train_booster(
                train_feats,
                train_labels,
                val_feats,
                val_labels,
                args.trees,
                Some(&per_sym_full_weights),
            ) {
                Ok(b) => b,
                Err(e) => {
                    warn!(
                        "[retrain:{}] train_booster failed: {} — skipping",
                        sym.short(),
                        e
                    );
                    rejections
                        .insert(sym.short().to_string(), "train_booster_failed".to_string());
                    record_diag!(false, Some("train_booster_failed"));
                    continue;
                }
            };
            diag_iterations = booster.num_iterations();
            let logloss = training_logloss(&booster, train_feats, train_labels)
                .with_context(|| format!("compute training logloss for {}", sym.short()))?;
            diag_logloss = logloss;
            if let Err(reason) = check_training_convergence(logloss) {
                warn!(
                    "[retrain:{}] Gate 3 (training convergence) REJECTED: {} — skipping",
                    sym.short(),
                    reason
                );
                rejections.insert(sym.short().to_string(), "gate_3_convergence".to_string());
                record_diag!(false, Some("gate_3_convergence"));
                continue;
            }
            diag_train_acc = validation_accuracy(&booster, train_feats, train_labels)
                .with_context(|| format!("compute training accuracy for {}", sym.short()))?;
            let fit = fit_beta_calibration_from_booster(&booster, val_feats, val_labels)
                .with_context(|| format!("fit beta calibration for {}", sym.short()))?;
            let bytes = serialise_booster(&booster, fit.cal)
                .with_context(|| format!("serialise booster for {}", sym.short()))?;
            let version_id = db::queries::promote_model(
                pool,
                sym.short(),
                &bytes,
                labeled_row_count,
                CURRENT_FORMAT_VERSION,
                "lightgbm",
            )
            .await
            .with_context(|| format!("promote_model {}", sym.short()))?;
            info!(
                "[retrain:{}] promoted family=lightgbm model_version_id={} ({} bytes) — Gate 4 SKIPPED",
                sym.short(),
                version_id,
                bytes.len()
            );
            record_diag!(true, None);
            promoted.push(format!("{}:lightgbm", sym.short()));
            continue;
        }

        let cv = run_purged_cv_gate4_both(
            &features_flat,
            &labels,
            &per_sym_timestamps,
            args.trees,
            sym.short(),
            half_life_days,
        );

        if !cv.lgbm.passed && !cv.lr_l1.passed {
            warn!(
                "[retrain:{}] Gate 4 (purged CV + Wilson) REJECTED for both families — \
                 lightgbm {}/{} folds, lr_l1 {}/{} folds, need ≥{} — skipping",
                sym.short(),
                cv.lgbm.folds_passed,
                CV_K_FOLDS,
                cv.lr_l1.folds_passed,
                CV_K_FOLDS,
                CV_MIN_FOLDS_PASSING
            );
            diag_val_acc = Some(cv.lgbm.mean_val_acc);
            diag_majority = Some(cv.lgbm.mean_majority);
            rejections.insert(sym.short().to_string(), "gate_4_purged_cv".to_string());
            record_diag!(false, Some("gate_4_purged_cv"));
            continue;
        }

        struct Candidate {
            family: &'static str,
            bytes: Vec<u8>,
            median_val_acc: f64,
            mean_val_acc: f64,
            mean_majority: f64,
            train_acc: f64,
            logloss: f64,
            iterations: i32,
        }
        let mut candidates: Vec<Candidate> = Vec::new();

        let gate4_train_weights = &per_sym_full_weights[..train_labels.len()];

        if cv.lgbm.passed {
            match train_booster(
                train_feats,
                train_labels,
                val_feats,
                val_labels,
                args.trees,
                Some(gate4_train_weights),
            ) {
                Ok(booster) => {
                    let logloss = training_logloss(&booster, train_feats, train_labels)
                        .with_context(|| {
                            format!("compute training logloss for {}", sym.short())
                        })?;
                    if let Err(reason) = check_training_convergence(logloss) {
                        warn!(
                            symbol = %sym.short(),
                            family = "lightgbm",
                            reason = %reason,
                            "gate3_family_disqualified"
                        );
                    } else {
                        let train_acc = validation_accuracy(&booster, train_feats, train_labels)
                            .with_context(|| {
                                format!("compute training accuracy for {}", sym.short())
                            })?;
                        let fit = fit_beta_calibration_from_booster(&booster, val_feats, val_labels)
                            .with_context(|| {
                                format!("fit beta calibration for {}", sym.short())
                            })?;
                        let ece_pre = calibration::expected_calibration_error(&fit.p_up, &fit.y);
                        let p_up_cal: Vec<f64> = fit
                            .p_up
                            .iter()
                            .map(|&p| calibration::apply_beta(p, &fit.cal))
                            .collect();
                        let ece_post = calibration::expected_calibration_error(&p_up_cal, &fit.y);
                        // If beta fit made ECE worse, fall back to IDENTITY.
                        let calibration_final = if ece_post > ece_pre {
                            warn!(
                                "[retrain:{}] beta fit rejected (ece_post {:.3} > ece_pre {:.3}) \
                                 — using IDENTITY",
                                sym.short(),
                                ece_post,
                                ece_pre
                            );
                            calibration::IDENTITY
                        } else {
                            fit.cal
                        };
                        info!(
                            "[retrain:{}] beta_cal family=lightgbm a={:.3} b={:.3} c={:.3} \
                             ece_pre={:.3} ece_post={:.3}",
                            sym.short(),
                            fit.cal.a,
                            fit.cal.b,
                            fit.cal.c,
                            ece_pre,
                            ece_post
                        );
                        match serialise_booster(&booster, calibration_final) {
                            Ok(bytes) => candidates.push(Candidate {
                                family: "lightgbm",
                                bytes,
                                median_val_acc: cv.lgbm.median_val_acc,
                                mean_val_acc: cv.lgbm.mean_val_acc,
                                mean_majority: cv.lgbm.mean_majority,
                                train_acc,
                                logloss,
                                iterations: booster.num_iterations(),
                            }),
                            Err(e) => warn!(
                                symbol = %sym.short(),
                                family = "lightgbm",
                                error = %e,
                                "serialise_booster_failed"
                            ),
                        }
                    }
                }
                Err(e) => warn!(
                    symbol = %sym.short(),
                    family = "lightgbm",
                    error = %e,
                    "gate3_family_disqualified"
                ),
            }
        }

        if cv.lr_l1.passed {
            match lr_l1::train_lr_l1_promoted_with_weights(
                train_feats,
                train_labels,
                val_feats,
                val_labels,
                gate4_train_weights,
            ) {
                Ok(model) => {
                    let logloss = lr_l1::lr_l1_binary_logloss(&model, train_feats, train_labels);
                    if let Err(reason) = check_training_convergence(logloss) {
                        warn!(
                            symbol = %sym.short(),
                            family = "lr_l1",
                            reason = %reason,
                            "gate3_family_disqualified"
                        );
                    } else {
                        let train_acc =
                            lr_l1::lr_l1_accuracy(&model, train_feats, train_labels);
                        match model.to_bytes() {
                            Ok(bytes) => {
                                info!(
                                    symbol = %sym.short(),
                                    family = "lr_l1",
                                    best_c = model.best_c,
                                    nonzero_coef = model.nonzero_coef_count(),
                                    beta_a = format!("{:.3}", model.beta_a),
                                    beta_b = format!("{:.3}", model.beta_b),
                                    beta_c = format!("{:.3}", model.beta_c),
                                    "lr_l1_promoted"
                                );
                                candidates.push(Candidate {
                                    family: "lr_l1",
                                    bytes,
                                    median_val_acc: cv.lr_l1.median_val_acc,
                                    mean_val_acc: cv.lr_l1.mean_val_acc,
                                    mean_majority: cv.lr_l1.mean_majority,
                                    train_acc,
                                    logloss,
                                    iterations: 0,
                                });
                            }
                            Err(e) => warn!(
                                symbol = %sym.short(),
                                family = "lr_l1",
                                error = %e,
                                "lr_l1_serialise_failed"
                            ),
                        }
                    }
                }
                Err(e) => warn!(
                    symbol = %sym.short(),
                    family = "lr_l1",
                    error = %e,
                    "gate3_family_disqualified"
                ),
            }
        }

        if candidates.is_empty() {
            warn!(
                "[retrain:{}] both families disqualified at Gate 3 — skipping",
                sym.short()
            );
            diag_val_acc = Some(cv.lgbm.mean_val_acc);
            diag_majority = Some(cv.lgbm.mean_majority);
            rejections.insert(sym.short().to_string(), "gate_3_convergence".to_string());
            record_diag!(false, Some("gate_3_convergence"));
            continue;
        }

        let lgbm_median = candidates
            .iter()
            .find(|c| c.family == "lightgbm")
            .map(|c| c.median_val_acc);
        let lr_median = candidates
            .iter()
            .find(|c| c.family == "lr_l1")
            .map(|c| c.median_val_acc);
        let winner_family = select_winner_family(lgbm_median, lr_median)
            .expect("candidates non-empty ⇒ a winner exists");
        let winner = candidates
            .iter()
            .find(|c| c.family == winner_family)
            .expect("winner_family was derived from candidates");

        info!(
            symbol = %sym.short(),
            lgbm_folds_passing = cv.lgbm.folds_passed,
            lr_l1_folds_passing = cv.lr_l1.folds_passed,
            lgbm_median_val_acc = format!("{:.4}", cv.lgbm.median_val_acc),
            lr_l1_median_val_acc = format!("{:.4}", cv.lr_l1.median_val_acc),
            winner = winner.family,
            "model_family_decision"
        );

        info!(
            "[retrain:per_symbol:{}] training on {} rows (ESS={:.0}, half_life={:.1}d) — Gate-4-passer winner family={}",
            sym.short(),
            train_labels.len(),
            effective_sample_size(gate4_train_weights),
            half_life_days,
            winner.family
        );

        diag_iterations = winner.iterations;
        diag_train_acc = winner.train_acc;
        diag_logloss = winner.logloss;
        diag_val_acc = Some(winner.mean_val_acc);
        diag_majority = Some(winner.mean_majority);

        let version_id = db::queries::promote_model(
            pool,
            sym.short(),
            &winner.bytes,
            labeled_row_count,
            CURRENT_FORMAT_VERSION,
            winner.family,
        )
        .await
        .with_context(|| format!("promote_model {}", sym.short()))?;
        info!(
            "[retrain:{}] promoted family={} model_version_id={} ({} bytes)",
            sym.short(),
            winner.family,
            version_id,
            winner.bytes.len()
        );
        record_diag!(true, None);
        promoted.push(format!("{}:{}", sym.short(), winner.family));
        pool_inputs.insert(
            sym,
            PerSymbolPoolInput {
                sym,
                features_flat: features_flat.clone(),
                labels: labels.clone(),
                timestamps_ms: per_sym_timestamps.clone(),
                per_symbol_version_id: version_id,
            },
        );
    }

    if pool_inputs.len() == cfg_cryptos.len() && !cfg_cryptos.is_empty() {
        info!(
            "[retrain:pooled] all {} symbols cleared per-symbol Gate 4 — running pooled comparison",
            cfg_cryptos.len()
        );
        let pooled_set = build_pooled_training_set(&pool_inputs);
        info!(
            "[retrain:pooled] union built: rows={} symbols={} timestamp_sorted=yes",
            pooled_set.labels.len(),
            pooled_set.per_symbol_row_counts.len()
        );

        match run_purged_cv_pooled_gate4(&pooled_set, args.trees, half_life_days) {
            Err(e) => warn!(
                "[retrain:pooled] CV failed — pooled skipped, per-symbol promotions \
                 retained as current. Error: {:#}",
                e
            ),
            Ok(cv_result) => {
                let mut any_fail = false;
                for s in 0..INSTRUMENT_COUNT {
                    let entry = &cv_result.per_sym[s];
                    let wilson_pass = entry.pooled_folds_wilson_pass >= CV_MIN_FOLDS_PASSING;
                    let median_pass =
                        entry.pooled_median_val_acc >= entry.baseline_median_val_acc;
                    if wilson_pass && median_pass {
                        info!(
                            "[retrain:pooled] symbol_id={} OK — pooled_median={:.4} \
                             baseline_median={:.4} wilson_lb_pass_folds={}/{}",
                            s,
                            entry.pooled_median_val_acc,
                            entry.baseline_median_val_acc,
                            entry.pooled_folds_wilson_pass,
                            CV_K_FOLDS
                        );
                    } else {
                        any_fail = true;
                        let reason = match (wilson_pass, median_pass) {
                            (false, false) => "both",
                            (false, true) => "wilson_floor",
                            (true, false) => "median_baseline",
                            _ => "unknown",
                        };
                        warn!(
                            "[retrain:pooled] REJECT — symbol_id={} reason={} \
                             pooled_median={:.4} baseline_median={:.4} \
                             wilson_lb_pass_folds={}/{}",
                            s, reason,
                            entry.pooled_median_val_acc,
                            entry.baseline_median_val_acc,
                            entry.pooled_folds_wilson_pass,
                            CV_K_FOLDS
                        );
                    }
                }

                if any_fail {
                    warn!(
                        "[retrain:pooled] REJECT — falling back to per-symbol promotion \
                         (per-symbol candidates above remain current)"
                    );
                } else {
                    info!(
                        "[retrain:pooled] all {} symbols passed — training final pooled booster \
                         on full union (no 80/20 split; Platts fit on OOF below)",
                        cfg_cryptos.len()
                    );
                    let final_weights = exponential_decay_weights(
                        &pooled_set.timestamps_ms,
                        half_life_days,
                    );
                    info!(
                        "[retrain:pooled_final] training on {} rows (ESS={:.0}, half_life={:.1}d) — final pooled booster",
                        pooled_set.labels.len(),
                        effective_sample_size(&final_weights),
                        half_life_days
                    );
                    let train_result = train_booster_pooled(
                        &pooled_set.features_flat_90,
                        &pooled_set.labels,
                        &[],
                        &[],
                        args.trees,
                        Some(&final_weights),
                    );
                    match train_result {
                        Err(e) => warn!(
                            "[retrain:pooled] final pooled booster train failed — fallback: {:#}",
                            e
                        ),
                        Ok(final_booster) => {
                            let betas = fit_per_symbol_beta_from_oof(
                                &cv_result.oof_p_up,
                                &cv_result.oof_labels,
                                &cv_result.oof_symbol_ids,
                            );
                            for (s, cal) in betas.iter().enumerate() {
                                info!(
                                    "[retrain:pooled] per_symbol_beta symbol_id={} a={:.3} b={:.3} c={:.3}",
                                    s, cal.a, cal.b, cal.c
                                );
                            }
                            match serialise_booster_pooled(&final_booster, betas) {
                                Err(e) => warn!(
                                    "[retrain:pooled] serialise failed — fallback: {:#}",
                                    e
                                ),
                                Ok(bytes) => {
                                    let row_counts: Vec<(&str, i32)> = pooled_set
                                        .per_symbol_row_counts
                                        .iter()
                                        .map(|(sym, n)| (sym.short(), *n))
                                        .collect();
                                    match db::queries::promote_pooled_model(
                                        pool,
                                        &bytes,
                                        &row_counts,
                                        CURRENT_FORMAT_VERSION,
                                    )
                                    .await
                                    {
                                        Ok(new_ids) => {
                                            info!(
                                                "[retrain:pooled] promoted family=lightgbm_pooled \
                                                 model_version_ids={:?} ({} bytes) — \
                                                 overrides per-symbol promotions",
                                                new_ids,
                                                bytes.len()
                                            );
                                            promoted.push("POOLED:lightgbm_pooled".to_string());
                                        }
                                        Err(e) => warn!(
                                            "[retrain:pooled] promote_pooled_model failed — \
                                             per-symbol promotions retained: {:#}",
                                            e
                                        ),
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else if !pool_inputs.is_empty() {
        info!(
            "[retrain:pooled] SKIPPED — only {}/{} symbols cleared per-symbol Gate 4; \
             pooled is all-or-nothing per design",
            pool_inputs.len(),
            cfg_cryptos.len()
        );
    }

    if promoted.is_empty() {
        let rejection_summary: Vec<String> = rejections
            .iter()
            .map(|(sym, reason)| format!("{}={}", sym, reason))
            .collect();
        warn!(
            "[retrain] no models promoted — {}/{} symbols rejected ({}). \
            This is expected during data accumulation; gates are protecting \
            against promoting underdetermined models.",
            rejections.len(),
            cfg_cryptos.len(),
            rejection_summary.join(", ")
        );
    } else {
        info!(
            "[retrain] {}/{} models promoted ({}); {} rejected",
            promoted.len(),
            cfg_cryptos.len(),
            promoted.join(", "),
            rejections.len(),
        );
    }
    Ok(())
}

/// Output of `build_training_set`: labelled rows plus drop/agreement diagnostics.
pub struct LabelingStats {
    /// Row-major features for every kept row (length = n_rows × TOTAL_FEATURES).
    pub flat: Vec<f64>,
    /// LightGBM-internal labels (0.0=UP, 1.0=DOWN).
    pub labels: Vec<f32>,
    /// OKX-vs-Polymarket disagreement count (diagnostic).
    pub disagree: usize,
    /// OKX-vs-Polymarket agreement count (diagnostic).
    pub agree: usize,
    /// Rows dropped because `|r| <= VOL_SCALE_K * σ_t`.
    pub dropped_vol_filter: usize,
    /// Rows dropped because σ_t was unavailable.
    pub dropped_no_sigma: usize,
    /// Median σ_t in basis points across kept rows (NaN when empty).
    pub median_sigma_bps: f64,
    /// Bucket open timestamp (ms, UTC) for each kept row.
    pub timestamps_ms: Vec<i64>,
}

/// Precompute σ_t for every ts in `closes` (stdev of `VOL_LOOKBACK_BUCKETS`
/// log returns strictly before bucket t — look-ahead-free). Absent entries
/// mean fewer than `VOL_LOOKBACK_BUCKETS` prior consecutive returns; callers
/// drop those rows.
pub fn compute_trailing_sigmas(closes: &BTreeMap<i64, f64>) -> BTreeMap<i64, f64> {
    use std::collections::VecDeque;
    let mut buf: VecDeque<f64> = VecDeque::with_capacity(VOL_LOOKBACK_BUCKETS);
    let mut sigmas: BTreeMap<i64, f64> = BTreeMap::new();
    let mut prev: Option<(i64, f64)> = None;
    for (&ts, &close) in closes.iter() {
        // σ_t is read before this bucket's return joins the buffer — strictly past.
        if buf.len() == VOL_LOOKBACK_BUCKETS {
            let n = VOL_LOOKBACK_BUCKETS as f64;
            let mean = buf.iter().sum::<f64>() / n;
            let var = buf.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
            sigmas.insert(ts, var.sqrt());
        }
        if let Some((pts, pclose)) = prev {
            if pts + CANDLE_INTERVAL_MS == ts && pclose > 0.0 && close > 0.0 {
                let lr = (close / pclose).ln();
                if buf.len() == VOL_LOOKBACK_BUCKETS {
                    buf.pop_front();
                }
                buf.push_back(lr);
            }
            // else: gap or invalid close — skip without resetting buf.
        }
        prev = Some((ts, close));
    }
    sigmas
}

/// Vol-scaled label: `Some(0.0)` UP, `Some(1.0)` DOWN, `None` to drop (indeterminate move).
pub fn vol_scaled_label(log_return: f64, sigma: f64) -> Option<f32> {
    let threshold = VOL_SCALE_K * sigma;
    if log_return > threshold {
        Some(0.0) // UP
    } else if log_return < -threshold {
        Some(1.0) // DOWN
    } else {
        None
    }
}

pub fn build_training_set(
    feature_rows: &[(i64, [f32; TOTAL_FEATURES])],
    closes: &BTreeMap<i64, f64>,
    outcomes: &BTreeMap<i64, i16>,
    symbol_offset: usize,
) -> LabelingStats {
    let sigmas = compute_trailing_sigmas(closes);
    let mut flat: Vec<f64> = Vec::with_capacity(feature_rows.len() * TOTAL_FEATURES);
    let mut labels: Vec<f32> = Vec::with_capacity(feature_rows.len());
    let mut timestamps_ms: Vec<i64> = Vec::with_capacity(feature_rows.len());
    let mut agreement = 0usize;
    let mut disagreement = 0usize;
    let mut dropped_vol_filter = 0usize;
    let mut dropped_no_sigma = 0usize;
    let mut kept_sigmas: Vec<f64> = Vec::with_capacity(feature_rows.len());

    for &(ts, ref feats) in feature_rows.iter() {
        // NaN first slot means this symbol had no aligned legs at this bucket.
        if feats[symbol_offset].is_nan() {
            continue;
        }
        // Gate: only train on windows that have a settled Polymarket outcome row.
        let predicted_window_ts_secs = (ts + CANDLE_INTERVAL_MS) / 1000;
        let Some(&poly_outcome) = outcomes.get(&predicted_window_ts_secs) else {
            continue;
        };
        let next_ts = ts + CANDLE_INTERVAL_MS;
        let (Some(&cur), Some(&next)) = (closes.get(&ts), closes.get(&next_ts)) else {
            continue;
        };
        if !(cur > 0.0 && next > 0.0) {
            continue;
        }
        let Some(&sigma) = sigmas.get(&ts) else {
            dropped_no_sigma += 1;
            continue;
        };
        let log_return = (next / cur).ln();
        let Some(label) = vol_scaled_label(log_return, sigma) else {
            dropped_vol_filter += 1;
            continue;
        };
        // Diagnostic: OKX-derived label vs Polymarket outcome agreement rate.
        let poly_label: Option<f32> = match poly_outcome {
            1 => Some(0.0),
            2 => Some(1.0),
            other => {
                warn!(
                    "unexpected outcome value {} in window_outcomes @ ts={}",
                    other, ts
                );
                None
            }
        };
        if let Some(p) = poly_label {
            if (p - label).abs() < f32::EPSILON {
                agreement += 1;
            } else {
                disagreement += 1;
            }
        }

        kept_sigmas.push(sigma);
        for v in feats.iter() {
            flat.push(*v as f64);
        }
        labels.push(label);
        timestamps_ms.push(ts);
    }

    let median_sigma_bps = if kept_sigmas.is_empty() {
        f64::NAN
    } else {
        kept_sigmas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        kept_sigmas[kept_sigmas.len() / 2] * 10_000.0 // σ is unit-less log-return stdev; ×10000 → bps
    };

    LabelingStats {
        flat,
        labels,
        disagree: disagreement,
        agree: agreement,
        dropped_vol_filter,
        dropped_no_sigma,
        median_sigma_bps,
        timestamps_ms,
    }
}

pub fn class_counts(labels: &[f32]) -> [usize; 2] {
    let mut counts = [0usize; 2];
    for &l in labels {
        let c = l as usize;
        if c < 2 {
            counts[c] += 1;
        }
    }
    counts
}

fn log_class_dist(sym_short: &str, counts: [usize; 2], total: usize) {
    let n = total.max(1) as f64;
    info!(
        "[retrain:{}] class dist — UP:{} ({:.1}%) DOWN:{} ({:.1}%)",
        sym_short,
        counts[0],
        100.0 * counts[0] as f64 / n,
        counts[1],
        100.0 * counts[1] as f64 / n,
    );
}

/// Fraction of labelled rows held out for validation (80/20 split).
pub const HOLDOUT_FRAC: f64 = 0.20;

/// Rows discarded between train and val slices; matches the longest rolling
/// feature window (1d SMA = 96 candles) to prevent label/feature leakage.
pub const EMBARGO_ROWS: usize = 96;

/// Minimum total row count at which Gate 4 is evaluated. Below this,
/// training runs on the full set and Gate 4 is reported as SKIPPED.
pub const MIN_ROWS_FOR_GATE: usize = 50;

/// Vol-scaled label threshold: row labelled iff `|r| > VOL_SCALE_K * σ_trailing`.
/// Rows below the threshold are dropped (not relabelled HOLD — booster is 2-class).
pub const VOL_SCALE_K: f64 = 0.3;

/// Trailing window for σ_t in 15-min buckets. 96 = 24 hours.
pub const VOL_LOOKBACK_BUCKETS: usize = 96;

/// Format-version boundary where labeling semantics changed (Gate 2 bypassed once per symbol at this boundary).
pub const MIGRATION_FORMAT_BOUNDARY: i16 = 12;

/// Gate 4 epsilon: minimum margin val_accuracy must beat the majority baseline (absolute fraction).
pub const GATE4_EPSILON: f64 = 0.05;

/// Number of contiguous chronological folds for purged k-fold CV.
pub const CV_K_FOLDS: usize = 5;

/// Symmetric per-symbol embargo rows dropped from training on both sides of each fold boundary.
pub const CV_EMBARGO_ROWS: usize = 96;

/// Gate 4 passes if at least this many folds clear their Wilson lower-bound test.
pub const CV_MIN_FOLDS_PASSING: usize = 3;

/// Significance level for the per-fold Wilson one-sided lower bound (α=0.05 → z≈1.645).
pub const WILSON_ALPHA: f64 = 0.05;

/// Margin the Wilson lower bound must clear above the fold's majority baseline.
pub const GATE4_BASELINE_MARGIN: f64 = 0.005;

/// Default half-life (days) for exponential sample-weight decay on training rows.
pub const SAMPLE_WEIGHT_HALF_LIFE_DAYS: f64 = 45.0;

/// Lower clamp for `--sample-weight-half-life-days`.
pub const SAMPLE_WEIGHT_HALF_LIFE_MIN: f64 = 10.0;

/// Upper clamp for `--sample-weight-half-life-days`.
pub const SAMPLE_WEIGHT_HALF_LIFE_MAX: f64 = 365.0;

/// Milliseconds in a day.
pub const MS_PER_DAY: f64 = 86_400_000.0;

/// Exponential sample-weight decay. Weight = `0.5 ^ (age_days / half_life_days)`;
/// freshest row gets weight 1.0. Returns empty Vec on empty input.
pub fn exponential_decay_weights(timestamps: &[i64], half_life_days: f64) -> Vec<f32> {
    if timestamps.is_empty() {
        return Vec::new();
    }
    let now_ms = *timestamps.iter().max().expect("non-empty checked above");
    let denom = half_life_days * MS_PER_DAY;
    timestamps
        .iter()
        .map(|&ts| {
            let age_ms = (now_ms - ts) as f64;
            let age_norm = age_ms / denom;
            // Floor at f32::EPSILON to avoid zero weights causing divide-by-zero in LightGBM.
            ((-std::f64::consts::LN_2 * age_norm).exp() as f32).max(f32::EPSILON)
        })
        .collect()
}

/// Effective Sample Size (Kish 1965): `ESS = (Σwᵢ)² / Σwᵢ²`.
pub fn effective_sample_size(weights: &[f32]) -> f64 {
    if weights.is_empty() {
        return 0.0;
    }
    let sum = weights.iter().map(|&w| w as f64).sum::<f64>();
    let sum_sq = weights.iter().map(|&w| (w as f64) * (w as f64)).sum::<f64>();
    if sum_sq <= 0.0 {
        return 0.0;
    }
    sum * sum / sum_sq
}

/// Gate 1: reject degenerate label distributions (missing class or >98% majority).
pub fn check_class_distribution(counts: [usize; 2]) -> Result<(), String> {
    let total: usize = counts.iter().sum();
    if total == 0 {
        return Err("0 labelled rows".to_string());
    }
    if counts[0] == 0 {
        return Err(format!("UP class has 0 samples (UP=0, DOWN={})", counts[1]));
    }
    if counts[1] == 0 {
        return Err(format!("DOWN class has 0 samples (UP={}, DOWN=0)", counts[0]));
    }
    let max_frac =
        (counts[0].max(counts[1])) as f64 / total as f64;
    if max_frac > 0.98 {
        return Err(format!(
            "majority class fraction {:.3} > 0.98 (UP={}, DOWN={})",
            max_frac, counts[0], counts[1]
        ));
    }
    Ok(())
}

/// Gate 2: reject if the new labelled-row count fell below 50% of the prior retrain's count.
/// `None` prior skips the gate (first retrain or pre-migration NULL).
pub fn check_row_count_stability(
    new_count: usize,
    prior_count: Option<i32>,
) -> Result<(), String> {
    let Some(prior) = prior_count else {
        return Ok(());
    };
    if prior <= 0 {
        return Ok(());
    }
    let threshold = 0.5 * prior as f64;
    if (new_count as f64) < threshold {
        return Err(format!(
            "labeled row count collapsed: new={new_count}, prior={prior} (< 50% of prior)"
        ));
    }
    Ok(())
}

/// Gate 3: reject only on non-finite training logloss (NaN/Inf guard).
pub fn check_training_convergence(logloss: f64) -> Result<(), String> {
    if !logloss.is_finite() {
        return Err(format!("training logloss is non-finite ({logloss})"));
    }
    Ok(())
}

/// z-score of `val_accuracy` against the majority-class baseline SE.
fn baseline_z(majority_frac: f64, n: usize, val_accuracy: f64) -> f64 {
    let se = (majority_frac * (1.0 - majority_frac) / n.max(1) as f64)
        .sqrt()
        .max(1e-9);
    (val_accuracy - majority_frac) / se
}

/// Gate 4 single-split check: val_accuracy must exceed majority baseline by `GATE4_EPSILON`.
pub fn check_validation_accuracy(
    val_counts: [usize; 2],
    val_accuracy: f64,
) -> Result<(), String> {
    let total: usize = val_counts.iter().sum();
    if total == 0 {
        return Err("0 validation rows".to_string());
    }
    if !val_accuracy.is_finite() || !(0.0..=1.0).contains(&val_accuracy) {
        return Err(format!("val_accuracy out of range: {val_accuracy}"));
    }
    let majority_frac =
        (val_counts[0].max(val_counts[1])) as f64 / total as f64;
    let threshold = majority_frac + GATE4_EPSILON;
    if val_accuracy <= threshold {
        let z = baseline_z(majority_frac, total, val_accuracy);
        return Err(format!(
            "val_accuracy {:.4} ≤ majority baseline {:.4} + {:.2} \
             (z={:+.2} SE; UP={}, DOWN={}, n={})",
            val_accuracy, majority_frac, GATE4_EPSILON, z,
            val_counts[0], val_counts[1], total
        ));
    }
    Ok(())
}

/// Purged k-fold CV index generator (de Prado, AFML Ch. 7). Returns `(train_idx, val_idx)` pairs
/// for `k` contiguous chronological folds with a symmetric `embargo`-row purge on each fold boundary.
pub fn purged_kfold_indices(
    n_rows: usize,
    k: usize,
    embargo: usize,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    if k == 0 || n_rows == 0 {
        return Vec::new();
    }
    let base = n_rows / k;
    let mut folds = Vec::with_capacity(k);
    let mut start = 0usize;
    for fold in 0..k {
        let end = if fold == k - 1 { n_rows } else { start + base }; // final fold absorbs remainder
        let val_idx: Vec<usize> = (start..end).collect();
        let purge_lo = start.saturating_sub(embargo);
        let purge_hi = (end + embargo).min(n_rows);
        let train_idx: Vec<usize> = (0..n_rows)
            .filter(|&i| i < purge_lo || i >= purge_hi)
            .collect();
        folds.push((train_idx, val_idx));
        start = end;
    }
    folds
}

/// Wilson score one-sided lower bound (Wilson 1927) for a binomial proportion.
/// Self-calibrating to `n`; only α=0.05 is calibrated (debug-asserted).
pub fn wilson_lower_bound_one_sided(successes: u32, n: u32, alpha: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    debug_assert!(
        (alpha - 0.05).abs() < 1e-9,
        "wilson_lower_bound_one_sided: only α=0.05 (z=1.645) is calibrated"
    );
    let z = 1.645_f64; // one-sided 95% z
    let n_f = n as f64;
    let p_hat = successes as f64 / n_f;
    let z2 = z * z;
    let denom = 1.0 + z2 / n_f;
    let centre = p_hat + z2 / (2.0 * n_f);
    let margin = z * (p_hat * (1.0 - p_hat) / n_f + z2 / (4.0 * n_f * n_f)).sqrt();
    (centre - margin) / denom
}

struct Gate4CvResult {
    passed: bool,
    folds_passed: usize,
    mean_val_acc: f64,
    mean_majority: f64,
    median_val_acc: f64,
}

struct Gate4BothResult {
    lgbm: Gate4CvResult,
    lr_l1: Gate4CvResult,
}

/// Gate 4 — purged k-fold CV for both model families on identical fold splits.
fn run_purged_cv_gate4_both(
    features_flat: &[f64],
    labels: &[f32],
    timestamps_ms: &[i64],
    num_trees: u32,
    sym_short: &str,
    half_life_days: f64,
) -> Gate4BothResult {
    let folds = purged_kfold_indices(labels.len(), CV_K_FOLDS, CV_EMBARGO_ROWS);

    let full_weights = exponential_decay_weights(timestamps_ms, half_life_days);
    let full_ess = effective_sample_size(&full_weights);
    info!(
        symbol = %sym_short,
        k_folds = CV_K_FOLDS,
        half_life_days,
        n_rows = labels.len(),
        full_set_ess = format!("{:.0}", full_ess),
        "retrain:cv_per_symbol weights enabled, full-set ESS reference for fold-context"
    );

    let mut lgbm_passed = 0usize;
    let mut lr_passed = 0usize;
    let mut lgbm_accs: Vec<f64> = Vec::new();
    let mut lr_accs: Vec<f64> = Vec::new();
    let mut lgbm_maj_sum = 0.0_f64;
    let mut lr_maj_sum = 0.0_f64;

    for (fold_idx, (train_idx, val_idx)) in folds.iter().enumerate() {
        let mut train_feats = Vec::with_capacity(train_idx.len() * TOTAL_FEATURES);
        let mut train_labels = Vec::with_capacity(train_idx.len());
        let mut train_weights = Vec::with_capacity(train_idx.len());
        for &i in train_idx {
            train_feats
                .extend_from_slice(&features_flat[i * TOTAL_FEATURES..(i + 1) * TOTAL_FEATURES]);
            train_labels.push(labels[i]);
            train_weights.push(full_weights[i]);
        }
        let mut val_feats = Vec::with_capacity(val_idx.len() * TOTAL_FEATURES);
        let mut val_labels = Vec::with_capacity(val_idx.len());
        for &i in val_idx {
            val_feats
                .extend_from_slice(&features_flat[i * TOTAL_FEATURES..(i + 1) * TOTAL_FEATURES]);
            val_labels.push(labels[i]);
        }

        let n_val = val_labels.len() as u32;
        let val_counts = class_counts(&val_labels);
        let fold_baseline = (val_counts[0].max(val_counts[1])) as f64 / n_val.max(1) as f64;

        // Wilson LB is computed unweighted — weights affect training only.
        let wilson_pass = |acc: f64| -> (f64, bool) {
            let successes = (acc * n_val as f64).round() as u32;
            let wlb = wilson_lower_bound_one_sided(successes, n_val, WILSON_ALPHA);
            (wlb, wlb > fold_baseline + GATE4_BASELINE_MARGIN)
        };

        let (lgbm_acc, lgbm_wlb, lgbm_fold_pass) =
            match train_booster(
                &train_feats,
                &train_labels,
                &val_feats,
                &val_labels,
                num_trees,
                Some(&train_weights),
            )
                .and_then(|b| validation_accuracy(&b, &val_feats, &val_labels))
            {
                Ok(acc) => {
                    let (wlb, pass) = wilson_pass(acc);
                    (Some(acc), wlb, pass)
                }
                Err(e) => {
                    warn!(
                        symbol = %sym_short,
                        fold = fold_idx,
                        family = "lightgbm",
                        error = %e,
                        "gate4_fold_train_failed"
                    );
                    (None, 0.0, false)
                }
            };

        let (lr_acc, lr_wlb, lr_fold_pass) =
            match lr_l1::train_lr_l1_with_weights(&train_feats, &train_labels, &train_weights) {
                Ok(model) => {
                    let acc = lr_l1::lr_l1_accuracy(&model, &val_feats, &val_labels);
                    let (wlb, pass) = wilson_pass(acc);
                    (Some(acc), wlb, pass)
                }
                Err(e) => {
                    warn!(
                        symbol = %sym_short,
                        fold = fold_idx,
                        family = "lr_l1",
                        error = %e,
                        "gate4_fold_train_failed"
                    );
                    (None, 0.0, false)
                }
            };

        if lgbm_fold_pass {
            lgbm_passed += 1;
        }
        if lr_fold_pass {
            lr_passed += 1;
        }
        if let Some(a) = lgbm_acc {
            lgbm_accs.push(a);
            lgbm_maj_sum += fold_baseline;
        }
        if let Some(a) = lr_acc {
            lr_accs.push(a);
            lr_maj_sum += fold_baseline;
        }

        info!(
            symbol = %sym_short,
            fold = fold_idx,
            n_train = train_idx.len(),
            n_val = val_idx.len(),
            lgbm_val_acc = format!("{:.4}", lgbm_acc.unwrap_or(f64::NAN)),
            lgbm_wilson_lb = format!("{:.4}", lgbm_wlb),
            lgbm_passed = lgbm_fold_pass,
            lr_l1_val_acc = format!("{:.4}", lr_acc.unwrap_or(f64::NAN)),
            lr_l1_wilson_lb = format!("{:.4}", lr_wlb),
            lr_l1_passed = lr_fold_pass,
            "gate4_fold_evaluated_both_families"
        );
    }

    let lgbm = finalize_gate4(lgbm_passed, &lgbm_accs, lgbm_maj_sum, sym_short, "lightgbm");
    let lr_l1 = finalize_gate4(lr_passed, &lr_accs, lr_maj_sum, sym_short, "lr_l1");
    Gate4BothResult { lgbm, lr_l1 }
}

fn finalize_gate4(
    folds_passed: usize,
    accs: &[f64],
    maj_sum: f64,
    sym_short: &str,
    family: &str,
) -> Gate4CvResult {
    let passed = folds_passed >= CV_MIN_FOLDS_PASSING;
    let denom = accs.len().max(1) as f64;
    let mean_val_acc = accs.iter().sum::<f64>() / denom;
    let mean_majority = maj_sum / denom;
    let median_val_acc = median(accs);
    info!(
        symbol = %sym_short,
        family = family,
        folds_passed,
        folds_required = CV_MIN_FOLDS_PASSING,
        folds_total = CV_K_FOLDS,
        scored_folds = accs.len(),
        median_val_acc = format!("{:.4}", median_val_acc),
        decision = if passed { "pass" } else { "reject" },
        "gate4_cv_decision"
    );
    Gate4CvResult {
        passed,
        folds_passed,
        mean_val_acc,
        mean_majority,
        median_val_acc,
    }
}

fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        (v[m - 1] + v[m]) / 2.0
    }
}

/// Select the winning model family by higher median fold val_acc; ties favour `"lightgbm"`.
pub fn select_winner_family(
    lgbm_median_val_acc: Option<f64>,
    lr_l1_median_val_acc: Option<f64>,
) -> Option<&'static str> {
    match (lgbm_median_val_acc, lr_l1_median_val_acc) {
        (None, None) => None,
        (Some(_), None) => Some("lightgbm"),
        (None, Some(_)) => Some("lr_l1"),
        (Some(lgbm), Some(lr)) => Some(if lr > lgbm { "lr_l1" } else { "lightgbm" }),
    }
}

/// Train one 2-class LightGBM booster. `train_sample_weights` applies optional
/// exponential-decay weights to the training Dataset; val stays unweighted.
pub fn train_booster(
    train_features: &[f64],
    train_labels: &[f32],
    val_features: &[f64],
    val_labels: &[f32],
    num_trees: u32,
    train_sample_weights: Option<&[f32]>,
) -> Result<Booster> {
    let mut train_dataset = Dataset::from_slice(
        train_features,
        train_labels,
        TOTAL_FEATURES as i32,
        /*is_row_major=*/ true,
    )
    .map_err(|e| anyhow!("Dataset::from_slice (train): {e}"))?;

    if let Some(w) = train_sample_weights {
        if w.len() != train_labels.len() {
            return Err(anyhow!(
                "train_sample_weights len {} != train_labels len {}",
                w.len(),
                train_labels.len()
            ));
        }
        train_dataset
            .set_weights(w)
            .map_err(|e| anyhow!("Dataset::set_weights (train): {e}"))?;
    }

    // Val Dataset must reference train Dataset — otherwise LightGBM rejects mismatched bin mappers.
    let val_dataset = if val_labels.is_empty() {
        None
    } else {
        Some(
            Dataset::from_slice_with_reference(
                val_features,
                val_labels,
                TOTAL_FEATURES as i32,
                true,
                Some(&train_dataset),
            )
            .map_err(|e| anyhow!("Dataset::from_slice_with_reference (val): {e}"))?,
        )
    };

    let params = json!({
        "objective": "multiclass",
        "num_class": 2,
        "metric": "multi_logloss",
        "learning_rate": 0.05,

        "num_leaves": 5,
        "max_depth": 6,
        "min_data_in_leaf": 60,
        "num_iterations": num_trees,
        "lambda_l2": 5.0,
        "min_gain_to_split": 0.01,
        "early_stopping_rounds": 8,
        "first_metric_only": true,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,

        "verbose": -1,
    });

    Booster::train_with_valid(train_dataset, val_dataset, &params)
        .map_err(|e| anyhow!("Booster::train_with_valid: {e}"))
}

/// Serialise booster + BetaCal trailer (3×f64 + `BETA` magic).
fn serialise_booster(booster: &Booster, cal: BetaCal) -> Result<Vec<u8>> {
    let text = booster
        .save_string()
        .map_err(|e| anyhow!("save_string: {e}"))?;
    let mut bytes = text.into_bytes();
    bytes.extend_from_slice(&cal.a.to_le_bytes());
    bytes.extend_from_slice(&cal.b.to_le_bytes());
    bytes.extend_from_slice(&cal.c.to_le_bytes());
    bytes.extend_from_slice(calibration::BETA_MAGIC);
    Ok(bytes)
}

/// Fit beta calibration from booster val predictions. Empty val → IDENTITY.
fn fit_beta_calibration_from_booster(
    booster: &Booster,
    val_features: &[f64],
    val_labels: &[f32],
) -> Result<calibration::BetaCalFit> {
    if val_labels.is_empty() {
        return Ok(calibration::BetaCalFit {
            cal: calibration::IDENTITY,
            p_up: Vec::new(),
            y: Vec::new(),
        });
    }
    let preds = booster
        .predict::<f64>(val_features, TOTAL_FEATURES as i32, true)
        .map_err(|e| anyhow!("Booster::predict for beta fit: {e}"))?;
    let n = val_labels.len();
    if preds.len() != n * 2 {
        return Err(anyhow!(
            "predict returned {} values, expected {} (n_val × 2)",
            preds.len(),
            n * 2
        ));
    }
    let mut p_up = Vec::<f64>::with_capacity(n);
    let mut y = Vec::<f64>::with_capacity(n);
    for (i, &l) in val_labels.iter().enumerate() {
        p_up.push(preds[i * 2]);
        y.push(if (l as u8) == 0 { 1.0 } else { 0.0 });
    }
    let cal = calibration::fit_beta_calibration(&p_up, &y);
    Ok(calibration::BetaCalFit { cal, p_up, y })
}

/// Multiclass cross-entropy of `booster` on the given rows.
fn training_logloss(booster: &Booster, features_flat: &[f64], labels: &[f32]) -> Result<f64> {
    let preds = booster
        .predict::<f64>(features_flat, TOTAL_FEATURES as i32, true)
        .map_err(|e| anyhow!("Booster::predict for logloss: {e}"))?;
    let n = labels.len();
    let k = 2usize;
    if preds.len() != n * k {
        return Err(anyhow!(
            "predict returned {} values, expected {} ({} rows × {} classes)",
            preds.len(),
            n * k,
            n,
            k
        ));
    }
    let eps = 1e-15_f64;
    let mut sum = 0.0_f64;
    for (i, &l) in labels.iter().enumerate() {
        let c = l as usize;
        if c >= k {
            return Err(anyhow!("unexpected label value {} at row {}", l, i));
        }
        let p = preds[i * k + c].max(eps);
        sum -= p.ln();
    }
    Ok(sum / n as f64)
}

/// Argmax accuracy of `booster` on a (features, labels) slice.
pub fn validation_accuracy(
    booster: &Booster,
    features_flat: &[f64],
    labels: &[f32],
) -> Result<f64> {
    if labels.is_empty() {
        return Ok(0.0);
    }
    let preds = booster
        .predict::<f64>(features_flat, TOTAL_FEATURES as i32, true)
        .map_err(|e| anyhow!("Booster::predict for val accuracy: {e}"))?;
    let n = labels.len();
    let k = 2usize;
    if preds.len() != n * k {
        return Err(anyhow!(
            "predict returned {} values, expected {} ({} rows × {} classes)",
            preds.len(),
            n * k,
            n,
            k
        ));
    }
    let mut correct = 0usize;
    for (i, &l) in labels.iter().enumerate() {
        let truth = l as u8;
        if (truth as usize) >= k {
            return Err(anyhow!("unexpected label value {} at row {}", l, i));
        }
        let p0 = preds[i * k];
        let p1 = preds[i * k + 1];
        let pred: u8 = if p0 >= p1 { 0 } else { 1 };
        if pred == truth {
            correct += 1;
        }
    }
    Ok(correct as f64 / n as f64)
}

/// Build a `PerpSample` for `inst_id` at `bucket_ts`. Returns None if funding or index_close unavailable.
/// Funding is forward-filled (most recent settlement ≤ bucket_ts).
fn build_perp_sample(
    inst_id: &str,
    bucket_ts: i64,
    funding: &HashMap<String, BTreeMap<i64, f64>>,
    index: &HashMap<String, BTreeMap<i64, f64>>,
) -> Option<PerpSample> {
    let f = funding.get(inst_id)?;
    let (settled_ts, rate) = f.range(..=bucket_ts).next_back()?;
    let index_inst = inst_id.strip_suffix("-SWAP")?;
    let i = index.get(index_inst)?;
    let idx_close = *i.get(&bucket_ts)?;
    Some(PerpSample {
        funding_rate:          *rate,
        funding_settled_at_ms: *settled_ts,
        index_close:           idx_close,
    })
}

/// Pooled training set: union of all per-symbol rows, timestamp-sorted, with `symbol_id` appended.
pub struct PooledTrainingSet {
    /// Row-major 90-wide features (89-wide row + `symbol_id` at slot `TOTAL_FEATURES`).
    pub features_flat_90: Vec<f64>,
    pub labels: Vec<f32>,
    /// `symbol_id` (BTC=0, ETH=1, XRP=2, SOL=3) for each union row.
    pub symbol_ids: Vec<u8>,
    pub timestamps_ms: Vec<i64>,
    pub per_symbol_row_counts: Vec<(Symbol, i32)>,
}

/// Build the pooled training set by union-ing all per-symbol inputs with `symbol_id` appended,
/// stable-sorted by timestamp.
pub fn build_pooled_training_set(
    inputs: &HashMap<Symbol, PerSymbolPoolInput>,
) -> PooledTrainingSet {
    let n_union: usize = inputs.values().map(|p| p.labels.len()).sum();
    let mut rows: Vec<(i64, u8, [usize; 2])> = Vec::with_capacity(n_union);
    // Sort by symbol_id for deterministic order across HashMap iterations.
    let mut per_symbol_row_counts: Vec<(Symbol, i32)> = inputs
        .iter()
        .map(|(&sym, p)| (sym, p.labels.len() as i32))
        .collect();
    per_symbol_row_counts.sort_by_key(|(sym, _)| symbol_id(*sym));

    let mut sym_inputs: Vec<(Symbol, &PerSymbolPoolInput)> = inputs.iter().map(|(&s, p)| (s, p)).collect();
    sym_inputs.sort_by_key(|(sym, _)| symbol_id(*sym));
    for (sym_pos, (_sym, input)) in sym_inputs.iter().enumerate() {
        let sid = symbol_id(input.sym) as u8;
        for i in 0..input.labels.len() {
            rows.push((input.timestamps_ms[i], sid, [sym_pos, i]));
        }
    }
    rows.sort_by_key(|(ts, _, _)| *ts);

    let mut features_flat_90: Vec<f64> = Vec::with_capacity(n_union * POOLED_INPUT_DIM);
    let mut labels: Vec<f32> = Vec::with_capacity(n_union);
    let mut symbol_ids: Vec<u8> = Vec::with_capacity(n_union);
    let mut timestamps_ms: Vec<i64> = Vec::with_capacity(n_union);
    for (ts, sid, [sym_pos, i]) in rows {
        let input = sym_inputs[sym_pos].1;
        let row89 = &input.features_flat[i * TOTAL_FEATURES..(i + 1) * TOTAL_FEATURES];
        features_flat_90.extend_from_slice(row89);
        features_flat_90.push(sid as f64);
        labels.push(input.labels[i]);
        symbol_ids.push(sid);
        timestamps_ms.push(ts);
    }

    PooledTrainingSet {
        features_flat_90,
        labels,
        symbol_ids,
        timestamps_ms,
        per_symbol_row_counts,
    }
}

/// Purged k-fold index generator for the pooled union with per-symbol embargo.
/// Embargo is applied per symbol_id so each symbol's 96-row rolling window
/// boundary is respected independently.
pub fn purged_kfold_indices_pooled(
    symbol_ids: &[u8],
    k: usize,
    embargo: usize,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let n_union = symbol_ids.len();
    if k == 0 || n_union == 0 {
        return Vec::new();
    }
    let n_symbols = (*symbol_ids.iter().max().unwrap_or(&0) as usize) + 1;
    let mut subseries: Vec<Vec<usize>> = vec![Vec::new(); n_symbols];
    let mut pos_in_subseries: Vec<usize> = vec![0; n_union];
    for (i, &s) in symbol_ids.iter().enumerate() {
        let s = s as usize;
        pos_in_subseries[i] = subseries[s].len();
        subseries[s].push(i);
    }

    let base = n_union / k;
    let mut folds = Vec::with_capacity(k);
    let mut start = 0usize;
    for fold in 0..k {
        let end = if fold == k - 1 { n_union } else { start + base };
        let val_idx: Vec<usize> = (start..end).collect();

        let mut excluded = vec![false; n_union];
        for &v in &val_idx {
            excluded[v] = true;
            let s = symbol_ids[v] as usize;
            let pos = pos_in_subseries[v];
            let lo = pos.saturating_sub(embargo);
            let hi = (pos + embargo + 1).min(subseries[s].len());
            for q in lo..hi {
                excluded[subseries[s][q]] = true;
            }
        }
        let train_idx: Vec<usize> = (0..n_union).filter(|&i| !excluded[i]).collect();
        folds.push((train_idx, val_idx));
        start = end;
    }
    folds
}

/// Train pooled multi-task LightGBM booster (90-wide rows, `symbol_id` as categorical feature).
pub fn train_booster_pooled(
    train_features: &[f64],
    train_labels: &[f32],
    val_features: &[f64],
    val_labels: &[f32],
    num_trees: u32,
    train_sample_weights: Option<&[f32]>,
) -> Result<Booster> {
    let mut train_dataset = Dataset::from_slice(
        train_features,
        train_labels,
        POOLED_INPUT_DIM as i32,
        /*is_row_major=*/ true,
    )
    .map_err(|e| anyhow!("Dataset::from_slice (pooled train): {e}"))?;

    if let Some(w) = train_sample_weights {
        if w.len() != train_labels.len() {
            return Err(anyhow!(
                "pooled train_sample_weights len {} != train_labels len {}",
                w.len(),
                train_labels.len()
            ));
        }
        train_dataset
            .set_weights(w)
            .map_err(|e| anyhow!("Dataset::set_weights (pooled train): {e}"))?;
    }

    let val_dataset = if val_labels.is_empty() {
        None
    } else {
        Some(
            Dataset::from_slice_with_reference(
                val_features,
                val_labels,
                POOLED_INPUT_DIM as i32,
                true,
                Some(&train_dataset),
            )
            .map_err(|e| anyhow!("Dataset::from_slice_with_reference (pooled val): {e}"))?,
        )
    };

    let params = json!({
        "objective": "multiclass",
        "num_class": 2,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 5,
        "max_depth": 6,
        "min_data_in_leaf": 60,
        "num_iterations": num_trees,
        "lambda_l2": 5.0,
        "min_gain_to_split": 0.01,
        "early_stopping_rounds": 8,
        "first_metric_only": true,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "categorical_feature": [TOTAL_FEATURES],
    });

    Booster::train_with_valid(train_dataset, val_dataset, &params)
        .map_err(|e| anyhow!("Booster::train_with_valid (pooled): {e}"))
}

/// Serialise pooled booster + per-symbol BetaCals (12×f64 LE + `BETA_POOLED_MAGIC`).
pub fn serialise_booster_pooled(
    booster: &Booster,
    betas: [BetaCal; INSTRUMENT_COUNT],
) -> Result<Vec<u8>> {
    let text = booster
        .save_string()
        .map_err(|e| anyhow!("save_string (pooled): {e}"))?;
    let mut bytes = text.into_bytes();
    for cal in &betas {
        bytes.extend_from_slice(&cal.a.to_le_bytes());
        bytes.extend_from_slice(&cal.b.to_le_bytes());
        bytes.extend_from_slice(&cal.c.to_le_bytes());
    }
    bytes.extend_from_slice(calibration::BETA_POOLED_MAGIC);
    Ok(bytes)
}

pub struct PooledGate4PerSym {
    pub pooled_median_val_acc: f64,
    pub baseline_median_val_acc: f64,
    pub pooled_folds_wilson_pass: usize,
}

/// Output of `run_purged_cv_pooled_gate4`.
pub struct PooledGate4Result {
    pub per_sym: [PooledGate4PerSym; INSTRUMENT_COUNT],
    /// Out-of-fold P(UP) for every union row (used to fit per-symbol beta calibration).
    pub oof_p_up: Vec<f64>,
    pub oof_labels: Vec<f32>,
    pub oof_symbol_ids: Vec<u8>,
}

/// Purged k-fold CV for the pooled multi-task booster, scored vs per-symbol baselines.
/// Returns per-symbol Wilson-pass counts + OOF predictions for beta calibration fitting.
fn run_purged_cv_pooled_gate4(
    set: &PooledTrainingSet,
    num_trees: u32,
    half_life_days: f64,
) -> Result<PooledGate4Result> {
    let n_union = set.labels.len();
    let folds = purged_kfold_indices_pooled(&set.symbol_ids, CV_K_FOLDS, CV_EMBARGO_ROWS);

    let full_weights = exponential_decay_weights(&set.timestamps_ms, half_life_days);
    let full_ess = effective_sample_size(&full_weights);
    info!(
        k_folds = CV_K_FOLDS,
        half_life_days,
        n_rows = n_union,
        full_set_ess = format!("{:.0}", full_ess),
        "retrain:cv_pooled weights enabled, full-set ESS reference for fold-context"
    );

    let mut pooled_accs: [Vec<f64>; INSTRUMENT_COUNT] = Default::default();
    let mut baseline_accs: [Vec<f64>; INSTRUMENT_COUNT] = Default::default();
    let mut pooled_wilson_pass: [usize; INSTRUMENT_COUNT] = [0; INSTRUMENT_COUNT];

    let mut oof_p_up: Vec<f64> = vec![f64::NAN; n_union];
    let mut oof_labels: Vec<f32> = vec![0.0; n_union];
    let mut oof_symbol_ids: Vec<u8> = vec![0; n_union];

    for (fold_idx, (train_idx, val_idx)) in folds.iter().enumerate() {
        let mut train_feats_90 = Vec::with_capacity(train_idx.len() * POOLED_INPUT_DIM);
        let mut train_labels: Vec<f32> = Vec::with_capacity(train_idx.len());
        let mut train_weights: Vec<f32> = Vec::with_capacity(train_idx.len());
        for &i in train_idx {
            train_feats_90.extend_from_slice(
                &set.features_flat_90[i * POOLED_INPUT_DIM..(i + 1) * POOLED_INPUT_DIM],
            );
            train_labels.push(set.labels[i]);
            train_weights.push(full_weights[i]);
        }
        let mut val_feats_90 = Vec::with_capacity(val_idx.len() * POOLED_INPUT_DIM);
        let mut val_labels: Vec<f32> = Vec::with_capacity(val_idx.len());
        let mut val_symbol_ids: Vec<u8> = Vec::with_capacity(val_idx.len());
        for &i in val_idx {
            val_feats_90.extend_from_slice(
                &set.features_flat_90[i * POOLED_INPUT_DIM..(i + 1) * POOLED_INPUT_DIM],
            );
            val_labels.push(set.labels[i]);
            val_symbol_ids.push(set.symbol_ids[i]);
        }

        let pooled_booster = match train_booster_pooled(
            &train_feats_90,
            &train_labels,
            &val_feats_90,
            &val_labels,
            num_trees,
            Some(&train_weights),
        ) {
            Ok(b) => b,
            Err(e) => {
                warn!(
                    fold = fold_idx,
                    error = %e,
                    "[retrain:pooled] pooled fold train failed"
                );
                continue;
            }
        };

        let pooled_preds = pooled_booster
            .predict::<f64>(&val_feats_90, POOLED_INPUT_DIM as i32, true)
            .map_err(|e| anyhow!("pooled booster val predict (fold {fold_idx}): {e}"))?;
        if pooled_preds.len() != val_labels.len() * 2 {
            return Err(anyhow!(
                "pooled predict shape mismatch: expected {} got {}",
                val_labels.len() * 2,
                pooled_preds.len()
            ));
        }
        for (j, &uidx) in val_idx.iter().enumerate() {
            oof_p_up[uidx] = pooled_preds[j * 2];
            oof_labels[uidx] = val_labels[j];
            oof_symbol_ids[uidx] = val_symbol_ids[j];
        }

        for s in 0..INSTRUMENT_COUNT {
            let s_u8 = s as u8;
            let val_local: Vec<usize> = (0..val_labels.len())
                .filter(|&j| val_symbol_ids[j] == s_u8)
                .collect();
            if val_local.is_empty() {
                continue;
            }

            let mut pooled_correct = 0usize;
            for &j in &val_local {
                let p0 = pooled_preds[j * 2];
                let p1 = pooled_preds[j * 2 + 1];
                let pred: f32 = if p0 >= p1 { 0.0 } else { 1.0 };
                if (pred - val_labels[j]).abs() < f32::EPSILON {
                    pooled_correct += 1;
                }
            }
            let n_val_s = val_local.len();
            let pooled_acc_s = pooled_correct as f64 / n_val_s as f64;

            let mut up_count = 0usize;
            for &j in &val_local {
                if (val_labels[j] as u8) == 0 {
                    up_count += 1;
                }
            }
            let maj = (up_count.max(n_val_s - up_count)) as f64 / n_val_s as f64;
            let wlb = wilson_lower_bound_one_sided(
                pooled_correct as u32,
                n_val_s as u32,
                WILSON_ALPHA,
            );
            let wilson_fold_pass = wlb > maj + GATE4_BASELINE_MARGIN;

            let train_local: Vec<usize> = (0..train_labels.len())
                .filter(|&j| {
                    let union_i = train_idx[j];
                    set.symbol_ids[union_i] == s_u8
                })
                .collect();
            if train_local.len() < MIN_ROWS_FOR_GATE {
                continue; // too few per-symbol train rows in this fold
            }

            // Drop the symbol_id column at slot TOTAL_FEATURES for the 89-wide baseline.
            // Baseline weights use the same pooled union-max weights so comparator is paired.
            let mut bl_train_feats = Vec::with_capacity(train_local.len() * TOTAL_FEATURES);
            let mut bl_train_labels: Vec<f32> = Vec::with_capacity(train_local.len());
            let mut bl_train_weights: Vec<f32> = Vec::with_capacity(train_local.len());
            for &j in &train_local {
                let off = j * POOLED_INPUT_DIM;
                bl_train_feats
                    .extend_from_slice(&train_feats_90[off..off + TOTAL_FEATURES]);
                bl_train_labels.push(train_labels[j]);
                bl_train_weights.push(train_weights[j]);
            }
            let mut bl_val_feats = Vec::with_capacity(val_local.len() * TOTAL_FEATURES);
            let mut bl_val_labels: Vec<f32> = Vec::with_capacity(val_local.len());
            for &j in &val_local {
                let off = j * POOLED_INPUT_DIM;
                bl_val_feats.extend_from_slice(&val_feats_90[off..off + TOTAL_FEATURES]);
                bl_val_labels.push(val_labels[j]);
            }

            let baseline_acc_s = match train_booster(
                &bl_train_feats,
                &bl_train_labels,
                &bl_val_feats,
                &bl_val_labels,
                num_trees,
                Some(&bl_train_weights),
            )
            .and_then(|b| validation_accuracy(&b, &bl_val_feats, &bl_val_labels))
            {
                Ok(acc) => acc,
                Err(e) => {
                    warn!(
                        fold = fold_idx,
                        symbol_id = s,
                        error = %e,
                        "[retrain:pooled] baseline fold train/score failed"
                    );
                    continue;
                }
            };

            pooled_accs[s].push(pooled_acc_s);
            baseline_accs[s].push(baseline_acc_s);
            if wilson_fold_pass {
                pooled_wilson_pass[s] += 1;
            }

            info!(
                fold = fold_idx,
                symbol_id = s,
                n_val = n_val_s,
                pooled_acc = format!("{:.4}", pooled_acc_s),
                baseline_acc = format!("{:.4}", baseline_acc_s),
                wilson_lb = format!("{:.4}", wlb),
                wilson_pass = wilson_fold_pass,
                "[retrain:pooled] fold_evaluated"
            );
        }
    }

    let median = |v: &[f64]| -> f64 {
        if v.is_empty() {
            return f64::NAN;
        }
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        s[s.len() / 2]
    };
    let per_sym: [PooledGate4PerSym; INSTRUMENT_COUNT] = std::array::from_fn(|s| {
        PooledGate4PerSym {
            pooled_median_val_acc: median(&pooled_accs[s]),
            baseline_median_val_acc: median(&baseline_accs[s]),
            pooled_folds_wilson_pass: pooled_wilson_pass[s],
        }
    });

    Ok(PooledGate4Result {
        per_sym,
        oof_p_up,
        oof_labels,
        oof_symbol_ids,
    })
}

/// Read-only wrapper for the pooled CV evaluation for diagnostic tooling.
pub fn evaluate_pooled_cv(
    per_symbol: Vec<(Symbol, Vec<f64>, Vec<f32>, Vec<i64>)>,
    num_trees: u32,
    half_life_days: f64,
) -> Result<PooledGate4Result> {
    let mut inputs: HashMap<Symbol, PerSymbolPoolInput> = HashMap::new();
    for (sym, features_flat, labels, timestamps_ms) in per_symbol {
        inputs.insert(
            sym,
            PerSymbolPoolInput {
                sym,
                features_flat,
                labels,
                timestamps_ms,
                per_symbol_version_id: 0,
            },
        );
    }
    let set = build_pooled_training_set(&inputs);
    run_purged_cv_pooled_gate4(&set, num_trees, half_life_days)
}

/// Fit one `BetaCal` per symbol from OOF P(UP) predictions. IDENTITY fallback for empty/single-class slices.
fn fit_per_symbol_beta_from_oof(
    oof_p_up: &[f64],
    oof_labels: &[f32],
    oof_symbol_ids: &[u8],
) -> [BetaCal; INSTRUMENT_COUNT] {
    let mut betas: [BetaCal; INSTRUMENT_COUNT] = [calibration::IDENTITY; INSTRUMENT_COUNT];
    for s in 0..INSTRUMENT_COUNT {
        let s_u8 = s as u8;
        let mut p_up: Vec<f64> = Vec::new();
        let mut y: Vec<f64> = Vec::new();
        for i in 0..oof_p_up.len() {
            if oof_symbol_ids[i] != s_u8 || !oof_p_up[i].is_finite() {
                continue;
            }
            p_up.push(oof_p_up[i]);
            y.push(if (oof_labels[i] as u8) == 0 { 1.0 } else { 0.0 });
        }
        betas[s] = calibration::fit_beta_calibration(&p_up, &y);
    }
    betas
}
