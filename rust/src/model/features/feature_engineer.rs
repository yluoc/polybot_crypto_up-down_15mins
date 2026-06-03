// src/feature_engine/feature_engineer.rs
//
// Per-instrument feature computation. Indices [0..8] are OHLC-derived,
// [9..12] are perp-specific (funding ×3 + basis), [13..17] are
// multi-timescale momentum, [18..20] are v14 microstructure. The trailing
// GLOBAL block holds macro features (DXY/VIX/yields).
//
// Internal math is f64 throughout; the final write to FeatureRow.features
// is `as f32`.

use std::collections::{HashMap, VecDeque};

use super::types::{
    short_symbol, Candle, FeatureRow, MacroSnapshot, MicroSample, PerpSample, FEATURE_DIM, FUNDING_LOOKBACK,
    INSTRUMENT_COUNT, INSTRUMENT_ORDER, REQUIRED_MACRO_SERIES, TOTAL_FEATURES,
};

/// `true` → each instrument emits a row when independently ready;
/// `false` → legacy all-4-aligned global gate (`all_ready` + `push_candle_global_aligned`).
pub const USE_PER_SYMBOL_ALIGNMENT: bool = true;

pub const MAX_CANDLES: usize = 96;
pub const MIN_HISTORY: usize = 96;
pub const RSI_PERIOD: usize = 14;
pub const SMA_PERIOD: usize = 20;
pub const VOL_PERIOD: usize = 5;
pub const RANK_WINDOW: usize = 20;
pub const RSI_LOSS_EPS: f64 = 1e-12;

/// `(close - SMA_4) / SMA_4`; 4 × 15m = 1h window.
const FEAT_IDX_MOM_1H: usize = 13;
/// `(close - SMA_16) / SMA_16`; 16 × 15m = 4h window.
const FEAT_IDX_MOM_4H: usize = 14;
/// `(close - SMA_96) / SMA_96`; 96 × 15m = 1d window. Drives MIN_HISTORY = 96.
const FEAT_IDX_MOM_1D: usize = 15;
/// sign(mom_1h) + sign(mom_4h) + sign(mom_1d); integer in {-3..+3}.
const FEAT_IDX_MOM_ALIGN: usize = 16;

const SMA_1H_PERIOD: usize = 4;
const SMA_4H_PERIOD: usize = 16;
const SMA_1D_PERIOD: usize = 96;

/// sign(funding_change_1) - sign(price_change_8h); integer in {-2..+2}.
/// +2 → funding rising while price falling; -2 → funding falling while price rising.
const FEAT_IDX_FUNDING_PRICE_DIV: usize = 17;
/// 32 × 15m = 8h price-change window, matching one OKX funding settle period.
const FUNDING_PRICE_DIV_LOOKBACK: usize = 32;

/// sign(x): +1 / 0 / -1. Used by `mom_align` and `funding_price_div`.
fn sign_f64(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Per-instrument perp state. `current` is bound to the current bucket;
/// `funding_settlements` is the deduped settlement deque for z-score and change-1.
struct PerpState {
    current:             Option<(i64, PerpSample)>,
    funding_settlements: VecDeque<(i64, f64)>,    // (settlement_ts, rate); cap FUNDING_LOOKBACK
}

impl PerpState {
    fn new() -> Self {
        Self {
            current:             None,
            funding_settlements: VecDeque::with_capacity(FUNDING_LOOKBACK + 1),
        }
    }
}

/// v14 microstructure state per instrument.
struct MicroState {
    oi_history:    VecDeque<(i64, f64)>,
    liq_history:   VecDeque<(i64, f64, f64)>,
    current_taker: Option<(i64, f64, f64)>,    // (bucket_ts, buy_vol, sell_vol)
}

/// OI history depth: needs oi[t] and oi[t-4], so 5 entries.
const MICRO_OI_HISTORY: usize = 5;

/// Liquidation window for `liq_imbalance_4`.
const MICRO_LIQ_WINDOW: usize = 4;

impl MicroState {
    fn new() -> Self {
        Self {
            oi_history:    VecDeque::with_capacity(MICRO_OI_HISTORY + 1),
            liq_history:   VecDeque::with_capacity(MICRO_LIQ_WINDOW + 1),
            current_taker: None,
        }
    }
}

pub struct FeatureEngineer {
    history: HashMap<String, VecDeque<Candle>>,
    perp:    HashMap<String, PerpState>,
    /// v14 microstructure rolling state per instrument. Absent entry → NaN for slots [18..20].
    micro: HashMap<String, MicroState>,
    /// Latest macro snapshot; None blocks row emission.
    macro_snapshot: Option<MacroSnapshot>,
    /// Per-instrument block cache: most recent FEATURE_DIM block per instrument tagged by bucket ts.
    block_cache: HashMap<String, (i64, [f32; FEATURE_DIM])>,
}

impl FeatureEngineer {
    pub fn new() -> Self {
        Self {
            history: HashMap::with_capacity(INSTRUMENT_COUNT),
            perp:    HashMap::with_capacity(INSTRUMENT_COUNT),
            micro:   HashMap::with_capacity(INSTRUMENT_COUNT),
            macro_snapshot: None,
            block_cache: HashMap::with_capacity(INSTRUMENT_COUNT),
        }
    }

    /// Push v14 microstructure inputs for one (instrument, bucket). Idempotent per bucket.
    /// Must be called before `push_candle` for the same bucket to populate slots [18..20].
    pub fn push_micro_sample(&mut self, inst_id: &str, bucket_ts_ms: i64, sample: &MicroSample) {
        let st = self.micro.entry(inst_id.to_string()).or_insert_with(MicroState::new);

        // Only populated when both buy and sell are present.
        st.current_taker = match (sample.taker_buy_vol, sample.taker_sell_vol) {
            (Some(b), Some(s)) if b.is_finite() && s.is_finite() => Some((bucket_ts_ms, b, s)),
            _ => None,
        };

        // OI: append to history, deduped by ts, capped at MICRO_OI_HISTORY.
        if let Some(oi) = sample.oi_usd {
            if oi.is_finite() {
                if let Some(last) = st.oi_history.back_mut() {
                    if last.0 == bucket_ts_ms {
                        last.1 = oi;
                    } else if bucket_ts_ms > last.0 {
                        st.oi_history.push_back((bucket_ts_ms, oi));
                    }
                    // else: older bucket arriving out-of-order; ignore.
                } else {
                    st.oi_history.push_back((bucket_ts_ms, oi));
                }
                while st.oi_history.len() > MICRO_OI_HISTORY {
                    st.oi_history.pop_front();
                }
            }
        }

        // Liquidations: same dedup-and-cap logic at MICRO_LIQ_WINDOW.
        // None means "not queried"; Some(0.0) means "queried, none found".
        let long_v  = sample.long_liq_usd.unwrap_or(0.0);
        let short_v = sample.short_liq_usd.unwrap_or(0.0);
        if long_v.is_finite() && short_v.is_finite()
            && (sample.long_liq_usd.is_some() || sample.short_liq_usd.is_some())
        {
            if let Some(last) = st.liq_history.back_mut() {
                if last.0 == bucket_ts_ms {
                    last.1 = long_v;
                    last.2 = short_v;
                } else if bucket_ts_ms > last.0 {
                    st.liq_history.push_back((bucket_ts_ms, long_v, short_v));
                }
            } else {
                st.liq_history.push_back((bucket_ts_ms, long_v, short_v));
            }
            while st.liq_history.len() > MICRO_LIQ_WINDOW {
                st.liq_history.pop_front();
            }
        }
    }

    /// Replace the stored macro snapshot (forward-filled until the next push).
    pub fn push_macro_snapshot(&mut self, snapshot: &MacroSnapshot) {
        self.macro_snapshot = Some(snapshot.clone());
    }

    /// Push the perp snapshot for one (instrument, bucket). Idempotent per bucket;
    /// settlement deque only grows when `funding_settled_at_ms` is a new event.
    /// Does not emit rows — `push_candle` is the driver.
    pub fn push_perp_sample(&mut self, inst_id: &str, bucket_ts_ms: i64, sample: PerpSample) {
        let st = self.perp.entry(inst_id.to_string()).or_insert_with(PerpState::new);
        st.current = Some((bucket_ts_ms, sample));

        let push_settlement = match st.funding_settlements.back() {
            None => true,
            Some((last_ts, _)) => sample.funding_settled_at_ms > *last_ts,
        };
        if push_settlement {
            st.funding_settlements.push_back((sample.funding_settled_at_ms, sample.funding_rate));
            while st.funding_settlements.len() > FUNDING_LOOKBACK {
                st.funding_settlements.pop_front();
            }
        }
    }

    /// Feed one candle. Returns `Some(FeatureRow)` when emission criteria are met, else None.
    pub fn push_candle(&mut self, candle: Candle) -> Option<FeatureRow> {
        let ts = candle.open_ts_ms;
        let inst_id = candle.inst_id.clone();
        let dq = self
            .history
            .entry(inst_id.clone())
            .or_insert_with(|| VecDeque::with_capacity(MAX_CANDLES));
        dq.push_back(candle);
        if dq.len() > MAX_CANDLES {
            dq.pop_front();
        }

        if USE_PER_SYMBOL_ALIGNMENT {
            self.push_candle_per_symbol(&inst_id, ts)
        } else {
            self.push_candle_global_aligned(ts)
        }
    }

    /// Legacy path (`USE_PER_SYMBOL_ALIGNMENT = false`): emits only when all 4 instruments align.
    fn push_candle_global_aligned(&mut self, ts: i64) -> Option<FeatureRow> {
        if !self.all_ready(ts) {
            return None;
        }

        let mut row = FeatureRow::empty(ts);
        for (i, inst) in INSTRUMENT_ORDER.iter().enumerate() {
            let dq = self.history.get(*inst)?;
            let st = self.perp.get(*inst)?;
            let base = i * FEATURE_DIM;
            if !compute_features(dq, st, &mut row.features, base) {
                return None;
            }
            // Micro slots [18..20]; NaN-tolerant, never blocks emission.
            compute_micro_features(self.micro.get(*inst), &mut row.features, base);
        }

        let snap = self
            .macro_snapshot
            .as_ref()
            .expect("push_candle: macro_snapshot None after all_ready()");
        let global_base = INSTRUMENT_COUNT * FEATURE_DIM;
        if !compute_globals(snap, ts, &mut row.features, global_base) {
            return None;
        }

        row.valid = true;
        Some(row)
    }

    /// Per-symbol emission path. Computes `inst_id`'s block, caches it, assembles the full row
    /// using cached blocks for other instruments (NaN if not aligned at `ts`).
    fn push_candle_per_symbol(&mut self, inst_id: &str, ts: i64) -> Option<FeatureRow> {
        if !self.macro_ready() {
            tracing::trace!(window_ts = ts, "feature_row_blocked_macro_not_ready");
            return None;
        }
        if !self.instrument_ready(inst_id, ts) {
            return None;
        }

        let dq = self.history.get(inst_id)?;
        let st = self.perp.get(inst_id)?;
        let mut block = [f32::NAN; FEATURE_DIM];
        if !compute_features(dq, st, &mut block, 0) {
            return None;
        }
        // Micro slots at offset 0; NaN-tolerant, never blocks emission.
        compute_micro_features(self.micro.get(inst_id), &mut block, 0);
        self.block_cache.insert(inst_id.to_string(), (ts, block));

        // Assemble from cache; NaN for instruments not aligned at this bucket.
        let mut row = FeatureRow::empty(ts);
        row.features = [f32::NAN; TOTAL_FEATURES];
        let mut n_other_present = 0usize;
        for (i, inst) in INSTRUMENT_ORDER.iter().enumerate() {
            if let Some((cached_ts, cached)) = self.block_cache.get(*inst) {
                if *cached_ts == ts {
                    let base = i * FEATURE_DIM;
                    row.features[base..base + FEATURE_DIM].copy_from_slice(cached);
                    if *inst != inst_id {
                        n_other_present += 1;
                    }
                }
            }
        }

        let snap = self
            .macro_snapshot
            .as_ref()
            .expect("push_candle_per_symbol: macro_snapshot None after macro_ready()");
        let global_base = INSTRUMENT_COUNT * FEATURE_DIM;
        if !compute_globals(snap, ts, &mut row.features, global_base) {
            return None;
        }

        tracing::info!(
            symbol = %short_symbol(inst_id),
            window_ts = ts,
            n_other_instruments_present = n_other_present,
            macro_present = true,
            "feature_row_emitted"
        );
        row.valid = true;
        Some(row)
    }

    /// True when macro snapshot is loaded and contains every required FRED series.
    fn macro_ready(&self) -> bool {
        let Some(snap) = self.macro_snapshot.as_ref() else {
            return false;
        };
        REQUIRED_MACRO_SERIES.iter().all(|s| snap.contains_key(*s))
    }

    /// True when `inst` has ≥ MIN_HISTORY candles, its latest candle is at `ts_ms`,
    /// and it has a PerpSample bound to `ts_ms`.
    fn instrument_ready(&self, inst: &str, ts_ms: i64) -> bool {
        let Some(dq) = self.history.get(inst) else {
            return false;
        };
        if dq.len() < MIN_HISTORY {
            return false;
        }
        match dq.back() {
            Some(c) if c.open_ts_ms == ts_ms => {}
            _ => return false,
        }
        matches!(
            self.perp.get(inst).and_then(|s| s.current),
            Some((bucket, _)) if bucket == ts_ms
        )
    }

    fn all_ready(&self, ts_ms: i64) -> bool {
        if self.history.len() < INSTRUMENT_COUNT {
            return false;
        }
        let Some(snap) = self.macro_snapshot.as_ref() else {
            return false;
        };
        for s in REQUIRED_MACRO_SERIES.iter() {
            if !snap.contains_key(*s) {
                return false;
            }
        }
        for inst in INSTRUMENT_ORDER.iter() {
            let dq = match self.history.get(*inst) {
                Some(d) => d,
                None => return false,
            };
            if dq.len() < MIN_HISTORY {
                return false;
            }
            let Some(back) = dq.back() else { return false };
            if back.open_ts_ms != ts_ms {
                return false;
            }
            let st = match self.perp.get(*inst) {
                Some(s) => s,
                None => return false,
            };
            match st.current {
                Some((bucket, _)) if bucket == ts_ms => {}
                _ => return false,
            }
        }
        true
    }
}

impl Default for FeatureEngineer {
    fn default() -> Self {
        Self::new()
    }
}

fn compute_features(
    candles: &VecDeque<Candle>,
    perp:    &PerpState,
    out:     &mut [f32],
    offset:  usize,
) -> bool {
    if candles.len() < MIN_HISTORY {
        return false;
    }
    let n = candles.len();
    let close_at = |k: usize| candles[n - 1 - k].close;
    let (c0, c1, c3, c5) = (close_at(0), close_at(1), close_at(3), close_at(5));
    if c0 <= 0.0 || c1 <= 0.0 || c3 <= 0.0 || c5 <= 0.0 {
        return false;
    }
    let last = &candles[n - 1];

    out[offset]     = (c0 / c1).ln() as f32;
    out[offset + 1] = (c0 / c3).ln() as f32;
    out[offset + 2] = (c0 / c5).ln() as f32;
    out[offset + 3] = ((last.high - last.low) / c0) as f32;
    out[offset + 4] = rolling_std_log_return(candles, VOL_PERIOD) as f32;
    out[offset + 5] = (rsi14(candles) / 100.0) as f32;

    let s20 = sma(candles, SMA_PERIOD);
    out[offset + 6] = if s20 > 0.0 {
        ((c0 - s20) / s20) as f32
    } else {
        0.0
    };
    out[offset + 7] = (c0 / c3 - 1.0) as f32;

    // [8] hl_pct_rank
    let cur_range = (last.high - last.low) / c0;
    out[offset + 8] = percentile_rank(cur_range, candles, RANK_WINDOW, |c| {
        if c.close > 0.0 {
            (c.high - c.low) / c.close
        } else {
            0.0
        }
    }) as f32;

    // perp features [9..12]
    let (_bucket, sample) = perp.current.expect("compute_features called without perp.current");

    // [9] funding_rate_now — forward-filled latest settlement rate
    out[offset + 9] = sample.funding_rate as f32;
    // [10] funding z-score over last FUNDING_LOOKBACK unique settlements
    out[offset + 10] = funding_zscore(&perp.funding_settlements) as f32;
    // [11] latest settlement rate minus prior
    out[offset + 11] = funding_change_1(&perp.funding_settlements) as f32;
    // [12] basis_pct
    out[offset + 12] = if sample.index_close > 0.0 {
        ((c0 - sample.index_close) / sample.index_close) as f32
    } else {
        0.0
    };

    // multi-timescale momentum [13..16]
    let sma_1h = match trailing_sma(candles, SMA_1H_PERIOD) {
        Some(v) if v > 0.0 => v,
        _ => return false,
    };
    let sma_4h = match trailing_sma(candles, SMA_4H_PERIOD) {
        Some(v) if v > 0.0 => v,
        _ => return false,
    };
    let sma_1d = match trailing_sma(candles, SMA_1D_PERIOD) {
        Some(v) if v > 0.0 => v,
        _ => return false,
    };
    let mom_1h = (c0 - sma_1h) / sma_1h;
    let mom_4h = (c0 - sma_4h) / sma_4h;
    let mom_1d = (c0 - sma_1d) / sma_1d;
    out[offset + FEAT_IDX_MOM_1H] = mom_1h as f32;
    out[offset + FEAT_IDX_MOM_4H] = mom_4h as f32;
    out[offset + FEAT_IDX_MOM_1D] = mom_1d as f32;
    // sign(0) = 0 (vs f64::signum which returns ±1 for ±0)
    out[offset + FEAT_IDX_MOM_ALIGN] =
        (sign_f64(mom_1h) + sign_f64(mom_4h) + sign_f64(mom_1d)) as f32;

    // funding/price divergence [17]; 0.0 when history is insufficient
    let funding_delta = funding_change_1(&perp.funding_settlements);
    let price_div = if candles.len() > FUNDING_PRICE_DIV_LOOKBACK {
        let prior = candles[candles.len() - 1 - FUNDING_PRICE_DIV_LOOKBACK].close;
        if prior > 0.0 {
            let price_change = (c0 - prior) / prior;
            sign_f64(funding_delta) - sign_f64(price_change)
        } else {
            0.0
        }
    } else {
        0.0
    };
    out[offset + FEAT_IDX_FUNDING_PRICE_DIV] = price_div as f32;

    true
}

/// Fill microstructure slots [18..20]. Never blocks row emission; NaN = missing data.
///   [offset+18] taker_buy / (buy+sell); NaN if unavailable
///   [offset+19] (oi[t] − oi[t-4]) / oi[t-4]; NaN if < MICRO_OI_HISTORY entries
///   [offset+20] (Σ long − Σ short) / (Σ long + Σ short + ε); NaN if empty
fn compute_micro_features(state: Option<&MicroState>, out: &mut [f32], offset: usize) {
    let mut taker_ratio: f32 = f32::NAN;
    let mut oi_change_pct: f32 = f32::NAN;
    let mut liq_imbalance: f32 = f32::NAN;

    if let Some(s) = state {
        if let Some((_ts, buy, sell)) = s.current_taker {
            let denom = buy + sell;
            if denom > 0.0 {
                taker_ratio = (buy / denom) as f32;
            }
        }
        // [19] oi_change_pct_4: needs MICRO_OI_HISTORY=5 entries
        if s.oi_history.len() >= MICRO_OI_HISTORY {
            let last = s.oi_history.back().expect("non-empty by len check").1;
            let four_back = s.oi_history.front().expect("non-empty").1;
            if four_back > 0.0 {
                oi_change_pct = ((last - four_back) / four_back) as f32;
            }
        }
        // [20] liq_imbalance_4: partial window is OK (sparse upstream is common)
        if !s.liq_history.is_empty() {
            let (sum_l, sum_s): (f64, f64) = s
                .liq_history
                .iter()
                .fold((0.0, 0.0), |(a, b), (_ts, l, sh)| (a + l, b + sh));
            const EPS: f64 = 1e-9;
            let denom = sum_l + sum_s + EPS;
            liq_imbalance = ((sum_l - sum_s) / denom) as f32;
        }
    }

    out[offset + 18] = taker_ratio;
    out[offset + 19] = oi_change_pct;
    out[offset + 20] = liq_imbalance;
}

/// Fill the global tail block starting at `offset`. Returns false if any required series is missing.
/// DXY/VIX/yield change columns are event-gated: non-zero only on US macro release days.
fn compute_globals(
    snap: &MacroSnapshot,
    bucket_ts_ms: i64,
    out: &mut [f32],
    offset: usize,
) -> bool {
    let dxy   = match snap.get("DTWEXBGS") { Some(s) => s, None => return false };
    let vix   = match snap.get("VIXCLS")   { Some(s) => s, None => return false };
    let dgs10 = match snap.get("DGS10")    { Some(s) => s, None => return false };

    let bucket_secs = bucket_ts_ms / 1000;
    let bucket_dt = chrono::DateTime::<chrono::Utc>::from_timestamp(bucket_secs, 0)
        .expect("compute_globals: bucket_ts_ms outside chrono range — shouldn't happen for sane timestamps");
    let bucket_date = bucket_dt.date_naive();
    let is_release = crate::macro_calendar::is_release_day(bucket_date);
    let gate: f64 = if is_release { 1.0 } else { 0.0 };

    let dxy_chg = if dxy.prev > 0.0 && dxy.current > 0.0 {
        (dxy.current / dxy.prev).ln()
    } else {
        0.0
    };
    let vix_chg = vix.current - vix.prev;
    let y10_chg_bps = (dgs10.current - dgs10.prev) * 100.0; // FRED yields are in percent

    out[offset]     = gate as f32;                    // [0] is_macro_release_day
    out[offset + 1] = (dxy_chg * gate) as f32;        // [1] dxy_change_event_gated
    out[offset + 2] = (vix_chg * gate) as f32;        // [2] vix_change_event_gated
    out[offset + 3] = (y10_chg_bps * gate) as f32;    // [3] yield_10y_change_event_gated_bps
    out[offset + 4] = vix.current as f32;              // [4] vix_level (always on)

    true
}

fn funding_zscore(settlements: &VecDeque<(i64, f64)>) -> f64 {
    let n = settlements.len();
    if n < 2 {
        return 0.0;
    }
    let mut sum = 0.0;
    for (_, r) in settlements.iter() {
        sum += *r;
    }
    let mean = sum / n as f64;
    let mut var = 0.0;
    for (_, r) in settlements.iter() {
        let d = *r - mean;
        var += d * d;
    }
    let std = (var / n as f64).sqrt();
    if std == 0.0 {
        return 0.0;
    }
    let last = settlements.back().unwrap().1;
    (last - mean) / std
}

fn funding_change_1(settlements: &VecDeque<(i64, f64)>) -> f64 {
    let n = settlements.len();
    if n < 2 {
        return 0.0;
    }
    settlements[n - 1].1 - settlements[n - 2].1
}

fn rsi14(candles: &VecDeque<Candle>) -> f64 {
    if candles.len() < RSI_PERIOD + 1 {
        return 50.0;
    }
    let n = candles.len();
    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;
    for i in (n - RSI_PERIOD)..n {
        let diff = candles[i].close - candles[i - 1].close;
        if diff > 0.0 {
            avg_gain += diff;
        } else {
            avg_loss -= diff;
        }
    }
    avg_gain /= RSI_PERIOD as f64;
    avg_loss /= RSI_PERIOD as f64;
    if avg_loss < RSI_LOSS_EPS {
        return 100.0;
    }
    100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
}

fn sma(candles: &VecDeque<Candle>, period: usize) -> f64 {
    if candles.len() < period {
        return candles.back().map(|c| c.close).unwrap_or(0.0);
    }
    let sz = candles.len();
    let mut sum = 0.0;
    for i in 0..period {
        sum += candles[sz - 1 - i].close;
    }
    sum / period as f64
}

/// SMA over the last `period` closes. Returns None on insufficient history (no fallback).
fn trailing_sma(candles: &VecDeque<Candle>, period: usize) -> Option<f64> {
    if candles.len() < period {
        return None;
    }
    let sz = candles.len();
    let mut sum = 0.0;
    for i in 0..period {
        sum += candles[sz - 1 - i].close;
    }
    Some(sum / period as f64)
}

fn rolling_std_log_return(candles: &VecDeque<Candle>, period: usize) -> f64 {
    if candles.len() < period + 1 {
        return 0.0;
    }
    const MAX_VOL_WINDOW: usize = 32;
    if period > MAX_VOL_WINDOW {
        return 0.0;
    }
    let sz = candles.len();
    let mut rets = [0.0_f64; MAX_VOL_WINDOW];
    for i in 0..period {
        let cc = candles[sz - 1 - i].close;
        let cp = candles[sz - 2 - i].close;
        rets[i] = if cc > 0.0 && cp > 0.0 {
            (cc / cp).ln()
        } else {
            0.0
        };
    }
    let mut sum = 0.0;
    for r in rets.iter().take(period) {
        sum += r;
    }
    let mean = sum / period as f64;
    let mut var = 0.0;
    for r in rets.iter().take(period) {
        let d = r - mean;
        var += d * d;
    }
    (var / period as f64).sqrt()
}

fn percentile_rank<F>(value: f64, candles: &VecDeque<Candle>, n: usize, extract: F) -> f64
where
    F: Fn(&Candle) -> f64,
{
    let sz = candles.len();
    let cnt = n.min(sz);
    if cnt == 0 {
        return 0.5;
    }
    let mut below = 0;
    for i in 0..cnt {
        if extract(&candles[sz - 1 - i]) < value {
            below += 1;
        }
    }
    below as f64 / cnt as f64
}
