// src/feature_engine/types.rs
//
// Core data types and constants for the feature pipeline.

/// 15-minute bucket size in milliseconds.
pub const CANDLE_INTERVAL_MS: i64 = 15 * 60 * 1000;

/// Four traded symbols. Order is load-bearing: feature slots are `i × FEATURE_DIM`.
pub const INSTRUMENT_ORDER: [&str; 4] = [
    "BTC-USDT-SWAP",
    "ETH-USDT-SWAP",
    "XRP-USDT-SWAP",
    "SOL-USDT-SWAP",
];

pub const INSTRUMENT_COUNT: usize = INSTRUMENT_ORDER.len();
/// Per-instrument feature count. Bumping this is a load-bearing change — any
/// persisted model trained against a different value must be rejected by the
/// format_version gate in ModelHub.
///   [0..8]   OHLC-derived
///   [9..12]  perp (3 funding + basis)
///   [13..16] multi-timescale momentum (1h/4h/1d SMA-distance + alignment score)
///   [17]     funding/price divergence
///   [18..20] microstructure (taker ratio, OI change, liq imbalance); NaN-tolerant
pub const FEATURE_DIM: usize = 21;

/// Global (macro) feature count, stored as a trailing tail block after per-instrument slots.
pub const GLOBAL_FEATURE_DIM: usize = 5;
pub const TOTAL_FEATURES: usize = INSTRUMENT_COUNT * FEATURE_DIM + GLOBAL_FEATURE_DIM;

/// Per-instrument feature names, indexed by slot 0..FEATURE_DIM.
/// Order must match the layout in the FeatureRow doc; persisted into `model_importance`.
pub const FEATURE_NAMES: [&str; FEATURE_DIM] = [
    "log_return_1", "log_return_3", "log_return_5",
    "hl_range", "volatility_5", "rsi_14",
    "price_vs_sma20", "momentum_3", "hl_pct_rank",
    "funding_rate_now", "funding_z_90", "funding_change_1",
    "basis_pct",
    "mom_1h", "mom_4h", "mom_1d", "mom_align",
    "funding_price_div",
    "taker_buy_sell_ratio", // OKX taker buy / (buy+sell), per 15-min bucket
    "oi_change_pct_4",      // (oi_t − oi_{t-4}) / oi_{t-4}, Coinalyze
    "liq_imbalance_4",      // (Σ4 long_liq − Σ4 short_liq) / (Σ4 long + Σ4 short + ε)
];

/// Global feature names (tail slots). Sources: DTWEXBGS, VIXCLS, DGS10 (FRED).
/// Change columns are event-gated: non-zero only on US macro release days.
pub const GLOBAL_FEATURE_NAMES: [&str; GLOBAL_FEATURE_DIM] = [
    "is_macro_release_day",            // 1.0 if today ∈ macro_calendar::RELEASE_DATES else 0.0
    "dxy_change_event_gated",          // dxy_log_return_1d × is_macro_release_day
    "vix_change_event_gated",          // vix_change_1d × is_macro_release_day
    "yield_10y_change_event_gated_bps",// (DGS10_today − DGS10_prev) × 100 × is_macro_release_day
    "vix_level",                       // VIX raw level (always on; regime indicator)
];

/// FRED series IDs persisted to `macro_daily`. SP500 and T10Y2Y are persisted but unused as features.
pub const MACRO_SERIES_IDS: [&str; 6] = [
    "DTWEXBGS",  // Nominal Broad U.S. Dollar Index
    "SP500",     // S&P 500 (persisted; not a feature)
    "VIXCLS",    // CBOE Volatility Index
    "DGS10",     // 10-Year Treasury Constant Maturity, %
    "DGS2",      // 2-Year Treasury Constant Maturity, %
    "T10Y2Y",    // 10y - 2y, % (persisted; not a feature)
];

/// FRED series required at emit time. Missing any one blocks the row.
pub const REQUIRED_MACRO_SERIES: [&str; 3] = [
    "DTWEXBGS", "VIXCLS", "DGS10",
];

/// Unique funding settlements retained per instrument for the z-score at index [11].
/// Count-based (not time-based) so it's comparable across settlement cadences.
pub const FUNDING_LOOKBACK: usize = 90;

/// `"BTC-USDT-SWAP"` → `"BTC"`.
pub fn short_symbol(inst_id: &str) -> &str {
    match inst_id.find('-') {
        Some(i) => &inst_id[..i],
        None => inst_id,
    }
}

/// Human-readable name for feature slot `j`: `"BTC-USDT-SWAP:log_return_1"` or `"GLOBAL:vix_level"`.
/// Panics if `j >= TOTAL_FEATURES`.
pub fn feature_full_name(j: usize) -> String {
    assert!(
        j < TOTAL_FEATURES,
        "feature_full_name: j {j} >= TOTAL_FEATURES {TOTAL_FEATURES}"
    );
    let global_base = INSTRUMENT_COUNT * FEATURE_DIM;
    if j < global_base {
        let inst = INSTRUMENT_ORDER[j / FEATURE_DIM];
        let feat = FEATURE_NAMES[j % FEATURE_DIM];
        format!("{inst}:{feat}")
    } else {
        let feat = GLOBAL_FEATURE_NAMES[j - global_base];
        format!("GLOBAL:{feat}")
    }
}

#[derive(Debug, Clone)]
pub struct RawTick {
    pub inst_id: String,
    pub mark_px: f64,
    pub ts_ms: i64,
}

/// Perp snapshot bound to one (instrument, 15m bucket).
/// `funding_settled_at_ms` deduplicates forward-filled settlement events.
#[derive(Debug, Clone, Copy)]
pub struct PerpSample {
    pub funding_rate:           f64,
    pub funding_settled_at_ms:  i64,
    pub index_close:            f64,
}

/// Microstructure inputs for one (instrument, 15m bucket). All fields are Option so
/// partial availability is handled per-slot; missing inputs emit NaN, never block emission.
#[derive(Debug, Clone, Copy, Default)]
pub struct MicroSample {
    /// Taker buy volume (per OKX) summed over the 15-min bucket.
    pub taker_buy_vol:  Option<f64>,
    /// Taker sell volume (per OKX) summed over the 15-min bucket.
    pub taker_sell_vol: Option<f64>,
    /// Aggregated open-interest USD notional (Coinalyze close-of-bucket).
    pub oi_usd:         Option<f64>,
    /// Long liquidations USD (Coinalyze; sparse — absent in DB ⇒ pass Some(0.0), not None).
    pub long_liq_usd:   Option<f64>,
    /// Short-position liquidations USD in this bucket.
    pub short_liq_usd:  Option<f64>,
}

#[derive(Debug, Clone)]
pub struct Candle {
    pub inst_id: String,
    pub open_ts_ms: i64,
    pub close_ts_ms: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub tick_count: u32,
}

/// Per-instrument feature layout (index within one instrument's FEATURE_DIM slice).
///   [0]  log_return_1      ln(close[t]/close[t-1])
///   [1]  log_return_3      ln(close[t]/close[t-3])
///   [2]  log_return_5      ln(close[t]/close[t-5])
///   [3]  hl_range          (high-low)/close
///   [4]  volatility_5      rolling std of log_return_1 over 5 candles
///   [5]  rsi_14            RSI(14)/100
///   [6]  price_vs_sma20    (close-SMA20)/SMA20
///   [7]  momentum_3        close[t]/close[t-3]-1
///   [8]  hl_pct_rank       percentile rank of hl_range in last 20 candles
///   [9]  funding_rate_now  latest settled funding rate (forward-filled)
///   [10] funding_z_90      z-score over last FUNDING_LOOKBACK unique settlements
///   [11] funding_change_1  latest settlement rate minus prior
///   [12] basis_pct         (mark_close − index_close) / index_close
///   [13] mom_1h            (close − SMA_4) / SMA_4; 4 × 15m = 1h
///   [14] mom_4h            (close − SMA_16) / SMA_16; 16 × 15m = 4h
///   [15] mom_1d            (close − SMA_96) / SMA_96; 96 × 15m = 1d
///   [16] mom_align         sign(mom_1h) + sign(mom_4h) + sign(mom_1d); integer in {-3..+3}
///   [17] funding_price_div sign(funding_change_1) - sign(price_change_8h); integer in {-2..+2}
#[derive(Debug, Clone)]
pub struct FeatureRow {
    pub candle_ts_ms: i64,
    pub features: [f32; TOTAL_FEATURES],
    pub valid: bool,
}

impl FeatureRow {
    pub fn empty(candle_ts_ms: i64) -> Self {
        Self {
            candle_ts_ms,
            features: [0.0; TOTAL_FEATURES],
            valid: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MacroSeries {
    pub date_utc: chrono::NaiveDate, // observation date of `current`
    pub current:  f64,                // FRED-native units
    pub prev:     f64,                // value at the immediately-previous obs date
}

/// Snapshot keyed by FRED series_id (e.g. "DGS10" → MacroSeries).
pub type MacroSnapshot = std::collections::HashMap<String, MacroSeries>;
