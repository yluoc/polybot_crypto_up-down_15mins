//! Rust Structure: each struct maps 1:1 to a table row.

use chrono::{DateTime, Utc};

#[derive(Debug, sqlx::FromRow)]
pub struct ErrorRow {
    pub id:          i64,
    pub source:      String,   // 'cpp', 'rust', or 'python'
    pub context:     String,   // e.g. 'okx_ws', 'order_submit', 'on_signal'
    pub detail:      String,
    pub happened_at: Option<DateTime<Utc>>,
}

#[derive(Debug, sqlx::FromRow)]
pub struct SignalRow {
    pub id:         i64,
    pub error_id:   Option<i64>,
    pub ts_ms:      i64,
    pub signal:     i16,               // 1=UP, 2=DOWN (legacy rows may have 0=HOLD)
    pub confidence: f64,
    pub market_id:  Option<String>,
    pub created_at: Option<DateTime<Utc>>,
}

#[derive(Debug, sqlx::FromRow)]
pub struct OrderRow {
    pub id:           i64,
    pub error_id:     Option<i64>,
    pub signal_id:    Option<i64>,
    pub market_id:    String,
    pub token_id:     String,
    pub side:         String,          // 'BUY' or 'SELL'
    pub usdc:         f64,
    pub price:        f64,
    pub order_id:     Option<String>,  // null if submission failed
    pub order_status: String,          // PENDING | MATCHED | CANCELLED | FAILED
    pub created_at:   Option<DateTime<Utc>>,
    pub updated_at:   Option<DateTime<Utc>>,
}

#[derive(Debug, sqlx::FromRow)]
pub struct PositionRow {
    pub id:              i64,
    pub error_id:        Option<i64>,
    pub market_id:       String,
    pub token_id:        String,
    pub side:            String,
    pub usdc:            f64,
    pub avg_entry_price: f64,
    pub opened_at:       DateTime<Utc>,
    pub closed_at:       Option<DateTime<Utc>>,  // NULL = still open
    pub position_status: String,                 // OPEN | CLOSED
}

#[derive(Debug, sqlx::FromRow)]
pub struct TradeRow {
    pub id:          i64,
    pub error_id:    Option<i64>,
    pub market_id:   String,
    pub side:        String,
    pub entry_price: f64,
    pub exit_price:  f64,
    pub usdc:        f64,
    pub pnl:         f64,
    pub pnl_pct:     f64,
    pub opened_at:   DateTime<Utc>,
    pub closed_at:   DateTime<Utc>,
}

/// Dry-run signal row returned by `get_unresolved_dry_run_signals`.
#[derive(Debug, sqlx::FromRow)]
pub struct DryRunPendingRow {
    pub signal_id:    i64,
    pub market_id:    String,
    pub signal:       i16,    // predicted action: 1=UP, 2=DOWN
    pub confidence:   f64,
    pub entry_price:  Option<f64>,
    pub shares:       Option<f64>,
    pub fee_rate_bps: Option<i32>,
    pub skip_reason:  Option<String>,  // None = entered; Some = reason trader bailed pre-entry
}

#[derive(Debug, sqlx::FromRow)]
pub struct ModelVersionRow {
    pub id:         i64,
    pub symbol:     String,         // "BTC", "ETH", ...
    pub is_current: bool,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct CandleRow {
    pub inst_id:    String,         // "BTC-USDT-SWAP"
    pub ts_ms:      i64,            // bucket open, 15-min aligned
    pub open:       f64,
    pub high:       f64,
    pub low:        f64,
    pub close:      f64,
    pub tick_count: i32,
}

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct FundingRateRow {
    pub inst_id:            String,         // swap form, e.g. "BTC-USDT-SWAP"
    pub ts_ms:              i64,            // settlement time
    pub rate:               f64,            // signed fraction
    pub settle_period_secs: Option<i32>,    // 1h / 2h / 4h / 8h; None on first observed row
}

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct OpenInterestRow {
    pub inst_id: String,                    // swap form
    pub ts_ms:   i64,                       // 15m bucket open
    pub oi_ccy:  f64,
    pub oi_usd:  f64,
}

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct IndexCandleRow {
    pub inst_id: String,                    // index form, e.g. "BTC-USDT"
    pub ts_ms:   i64,                       // 15m bucket open
    pub open:    f64,
    pub high:    f64,
    pub low:     f64,
    pub close:   f64,
}

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct MacroDailyRow {
    pub series_id: String,                  // FRED series, e.g. "DGS10"
    pub date_utc:  chrono::NaiveDate,
    pub value:     f64,
}

#[derive(Debug, sqlx::FromRow)]
pub struct ModelBlobRow {
    pub id:               i64,
    pub model_version_id: i64,
    pub model_bytes:      Vec<u8>,
    pub byte_size:        i32,
    pub sha256_hex:       String,
    pub format_version:   i16,
    pub created_at:       DateTime<Utc>,
}

#[derive(Debug, sqlx::FromRow)]
pub struct BotRunRow {
    pub id:               i64,
    pub error_id:         Option<i64>,
    pub model_path:       Option<String>,
    pub conf_threshold:   Option<f64>,
    pub data_source:      Option<String>,
    pub started_at:       Option<DateTime<Utc>>,
    pub stopped_at:       Option<DateTime<Utc>>,
    pub notes:            Option<String>,
    pub model_version_id: Option<i64>,
}
