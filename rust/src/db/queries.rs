// src/db/queries.rs — Insert / update / select helpers, one function per table operation.

use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Utc};
use sqlx::PgPool;
use anyhow::Result;

use super::models::{
    CandleRow, DryRunPendingRow, FundingRateRow, IndexCandleRow, MacroDailyRow, ModelVersionRow,
    OpenInterestRow, PositionRow,
};
use crate::inference::model_hub::ModelBlobEntry;

pub async fn insert_error(
    pool: &PgPool,
    source: &str,
    context: &str,
    detail: &str,
) -> Result<i64> {
    let (id,): (i64,) = sqlx::query_as(
        "INSERT INTO errors (source, context, detail) VALUES ($1, $2, $3) RETURNING id",
    )
    .bind(source)
    .bind(context)
    .bind(detail)
    .fetch_one(pool)
    .await?;
    Ok(id)
}

pub async fn insert_signal(
    pool: &PgPool,
    error_id: Option<i64>,
    ts_ms: i64,
    signal: i16,
    confidence: f64,
    market_id: Option<&str>,
) -> Result<i64> {
    let (id,): (i64,) = sqlx::query_as(
        r#"INSERT INTO signals (error_id, ts_ms, signal, confidence, market_id)
           VALUES ($1, $2, $3, $4, $5)
           RETURNING id"#,
    )
    .bind(error_id)
    .bind(ts_ms)
    .bind(signal)
    .bind(confidence)
    .bind(market_id)
    .fetch_one(pool)
    .await?;
    Ok(id)
}

/// Attach a resolved market slug to an already-inserted signal row.
pub async fn update_signal_market_id(
    pool: &PgPool,
    signal_id: i64,
    market_id: &str,
) -> Result<()> {
    sqlx::query("UPDATE signals SET market_id=$1 WHERE id=$2")
        .bind(market_id)
        .bind(signal_id)
        .execute(pool)
        .await?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub async fn insert_order(
    pool: &PgPool,
    error_id: Option<i64>,
    signal_id: Option<i64>,
    market_id: &str,
    token_id: &str,
    side: &str,
    usdc: f64,
    price: f64,
    fee_rate_bps: Option<i32>,
) -> Result<i64> {
    let (id,): (i64,) = sqlx::query_as(
        r#"INSERT INTO orders
           (error_id, signal_id, market_id, token_id, side, usdc, price, fee_rate_bps, order_status)
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'PENDING')
           RETURNING id"#,
    )
    .bind(error_id)
    .bind(signal_id)
    .bind(market_id)
    .bind(token_id)
    .bind(side)
    .bind(usdc)
    .bind(price)
    .bind(fee_rate_bps)
    .fetch_one(pool)
    .await?;
    Ok(id)
}

pub async fn update_order_status(
    pool: &PgPool,
    id: i64,
    order_status: &str,
    order_id: Option<&str>,
    error_id: Option<i64>,
) -> Result<()> {
    sqlx::query(
        "UPDATE orders SET order_status=$1, order_id=$2, error_id=$3, updated_at=NOW() WHERE id=$4",
    )
    .bind(order_status)
    .bind(order_id)
    .bind(error_id)
    .bind(id)
    .execute(pool)
    .await?;
    Ok(())
}

/// Update only the `order_status` column, leaving all other columns untouched.
pub async fn update_order_status_only(
    pool: &PgPool,
    id: i64,
    order_status: &str,
) -> Result<()> {
    sqlx::query(
        "UPDATE orders SET order_status=$1, updated_at=NOW() WHERE id=$2",
    )
    .bind(order_status)
    .bind(id)
    .execute(pool)
    .await?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub async fn insert_position(
    pool: &PgPool,
    error_id: Option<i64>,
    market_id: &str,
    token_id: &str,
    side: &str,
    usdc: f64,
    avg_entry_price: f64,
    opened_at: DateTime<Utc>,
) -> Result<i64> {
    let (id,): (i64,) = sqlx::query_as(
        r#"INSERT INTO positions
           (error_id, market_id, token_id, side, usdc, avg_entry_price, opened_at)
           VALUES ($1, $2, $3, $4, $5, $6, $7)
           RETURNING id"#,
    )
    .bind(error_id)
    .bind(market_id)
    .bind(token_id)
    .bind(side)
    .bind(usdc)
    .bind(avg_entry_price)
    .bind(opened_at)
    .fetch_one(pool)
    .await?;
    Ok(id)
}

pub async fn close_position(
    pool: &PgPool,
    id: i64,
    closed_at: DateTime<Utc>,
) -> Result<()> {
    sqlx::query(
        "UPDATE positions SET position_status='CLOSED', closed_at=$1 WHERE id=$2",
    )
    .bind(closed_at)
    .bind(id)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn get_open_position(
    pool: &PgPool,
    market_id: &str,
    token_id: &str,
) -> Result<Option<PositionRow>> {
    let row = sqlx::query_as::<_, PositionRow>(
        "SELECT * FROM positions WHERE market_id=$1 AND token_id=$2 AND position_status='OPEN'",
    )
    .bind(market_id)
    .bind(token_id)
    .fetch_optional(pool)
    .await?;
    Ok(row)
}

pub async fn get_all_open_positions(pool: &PgPool) -> Result<Vec<PositionRow>> {
    let rows = sqlx::query_as::<_, PositionRow>(
        "SELECT * FROM positions WHERE position_status='OPEN'",
    )
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

#[allow(clippy::too_many_arguments)]
pub async fn insert_trade(
    pool: &PgPool,
    error_id: Option<i64>,
    market_id: &str,
    side: &str,
    entry_price: f64,
    exit_price: f64,
    usdc: f64,
    pnl: f64,
    pnl_pct: f64,
    opened_at: DateTime<Utc>,
    closed_at: DateTime<Utc>,
) -> Result<i64> {
    let (id,): (i64,) = sqlx::query_as(
        r#"INSERT INTO trades
           (error_id, market_id, side, entry_price, exit_price, usdc, pnl, pnl_pct, opened_at, closed_at)
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
           RETURNING id"#,
    )
    .bind(error_id)
    .bind(market_id)
    .bind(side)
    .bind(entry_price)
    .bind(exit_price)
    .bind(usdc)
    .bind(pnl)
    .bind(pnl_pct)
    .bind(opened_at)
    .bind(closed_at)
    .fetch_one(pool)
    .await?;
    Ok(id)
}

pub async fn insert_bot_run(
    pool: &PgPool,
    error_id: Option<i64>,
    model_path: Option<&str>,
    conf_threshold: Option<f64>,
    data_source: Option<&str>,
    notes: Option<&str>,
    model_version_id: Option<i64>,
) -> Result<i64> {
    let (id,): (i64,) = sqlx::query_as(
        r#"INSERT INTO bot_runs
           (error_id, model_path, conf_threshold, data_source, notes, model_version_id)
           VALUES ($1, $2, $3, $4, $5, $6)
           RETURNING id"#,
    )
    .bind(error_id)
    .bind(model_path)
    .bind(conf_threshold)
    .bind(data_source)
    .bind(notes)
    .bind(model_version_id)
    .fetch_one(pool)
    .await?;
    Ok(id)
}

pub async fn stop_bot_run(pool: &PgPool, id: i64) -> Result<()> {
    sqlx::query("UPDATE bot_runs SET stopped_at=NOW() WHERE id=$1")
        .bind(id)
        .execute(pool)
        .await?;
    Ok(())
}

/// Highest stored bucket timestamp for an instrument, or None if empty.
pub async fn get_max_candle_ts(pool: &PgPool, inst_id: &str) -> Result<Option<i64>> {
    let row: Option<(i64,)> = sqlx::query_as(
        "SELECT MAX(ts_ms) FROM candles WHERE inst_id=$1 AND ts_ms IS NOT NULL",
    )
    .bind(inst_id)
    .fetch_optional(pool)
    .await?;
    Ok(row.map(|(ts,)| ts))
}

/// Bulk-insert candles; idempotent via ON CONFLICT DO NOTHING.
pub async fn insert_candles_batch(pool: &PgPool, rows: &[CandleRow]) -> Result<u64> {
    if rows.is_empty() {
        return Ok(0);
    }
    let inst_ids:    Vec<&str> = rows.iter().map(|c| c.inst_id.as_str()).collect();
    let ts_mss:      Vec<i64> = rows.iter().map(|c| c.ts_ms).collect();
    let opens:       Vec<f64> = rows.iter().map(|c| c.open).collect();
    let highs:       Vec<f64> = rows.iter().map(|c| c.high).collect();
    let lows:        Vec<f64> = rows.iter().map(|c| c.low).collect();
    let closes:      Vec<f64> = rows.iter().map(|c| c.close).collect();
    let tick_counts: Vec<i32> = rows.iter().map(|c| c.tick_count).collect();

    let res = sqlx::query(
        r#"INSERT INTO candles (inst_id, ts_ms, open, high, low, close, tick_count)
           SELECT * FROM UNNEST(
               $1::varchar[], $2::bigint[], $3::float8[], $4::float8[],
               $5::float8[], $6::float8[], $7::int[]
           )
           ON CONFLICT (inst_id, ts_ms) DO NOTHING"#,
    )
    .bind(&inst_ids)
    .bind(&ts_mss)
    .bind(&opens)
    .bind(&highs)
    .bind(&lows)
    .bind(&closes)
    .bind(&tick_counts)
    .execute(pool)
    .await?;
    Ok(res.rows_affected())
}

/// Most recent `limit_per_inst` candles per instrument in chronological order.
/// When `before_ts_ms` is `Some(cutoff)`, restricts to rows with `ts_ms < cutoff`.
pub async fn select_recent_candles_for_warmup(
    pool: &PgPool,
    limit_per_inst: usize,
    before_ts_ms: Option<i64>,
) -> Result<Vec<CandleRow>> {
    let rows = sqlx::query_as::<_, CandleRow>(
        r#"WITH ranked AS (
               SELECT inst_id, ts_ms, open, high, low, close, tick_count,
                      ROW_NUMBER() OVER (PARTITION BY inst_id ORDER BY ts_ms DESC) AS rn
                 FROM candles
                WHERE $2::bigint IS NULL OR ts_ms < $2::bigint
           )
           SELECT inst_id, ts_ms, open, high, low, close, tick_count
             FROM ranked
            WHERE rn <= $1
            ORDER BY ts_ms ASC, inst_id ASC"#,
    )
    .bind(limit_per_inst as i64)
    .bind(before_ts_ms)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// All candles with `ts_ms >= cutoff_ts_ms`, ordered `(ts_ms, inst_id)`.
pub async fn select_candles_since(
    pool: &PgPool,
    cutoff_ts_ms: i64,
) -> Result<Vec<CandleRow>> {
    let rows = sqlx::query_as::<_, CandleRow>(
        r#"SELECT inst_id, ts_ms, open, high, low, close, tick_count
           FROM candles
           WHERE ts_ms >= $1
           ORDER BY ts_ms, inst_id"#,
    )
    .bind(cutoff_ts_ms)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

pub async fn get_max_funding_ts(pool: &PgPool, inst_id: &str) -> Result<Option<i64>> {
    let row: Option<(i64,)> = sqlx::query_as(
        "SELECT MAX(ts_ms) FROM funding_rates WHERE inst_id=$1 AND ts_ms IS NOT NULL",
    )
    .bind(inst_id)
    .fetch_optional(pool)
    .await?;
    Ok(row.map(|(ts,)| ts))
}

pub async fn insert_funding_batch(pool: &PgPool, rows: &[FundingRateRow]) -> Result<u64> {
    if rows.is_empty() {
        return Ok(0);
    }
    let inst_ids: Vec<&str> = rows.iter().map(|r| r.inst_id.as_str()).collect();
    let ts_mss:   Vec<i64>  = rows.iter().map(|r| r.ts_ms).collect();
    let rates:    Vec<f64>  = rows.iter().map(|r| r.rate).collect();
    let periods:  Vec<Option<i32>> = rows.iter().map(|r| r.settle_period_secs).collect();

    let res = sqlx::query(
        r#"INSERT INTO funding_rates (inst_id, ts_ms, rate, settle_period_secs)
           SELECT * FROM UNNEST(
               $1::varchar[], $2::bigint[], $3::float8[], $4::int[]
           )
           ON CONFLICT (inst_id, ts_ms) DO NOTHING"#,
    )
    .bind(&inst_ids)
    .bind(&ts_mss)
    .bind(&rates)
    .bind(&periods)
    .execute(pool)
    .await?;
    Ok(res.rows_affected())
}

pub async fn select_funding_since(
    pool: &PgPool,
    cutoff_ts_ms: i64,
) -> Result<Vec<FundingRateRow>> {
    let rows = sqlx::query_as::<_, FundingRateRow>(
        r#"SELECT inst_id, ts_ms, rate, settle_period_secs
             FROM funding_rates
            WHERE ts_ms >= $1
            ORDER BY ts_ms, inst_id"#,
    )
    .bind(cutoff_ts_ms)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

pub async fn get_max_oi_ts(pool: &PgPool, inst_id: &str) -> Result<Option<i64>> {
    let row: Option<(i64,)> = sqlx::query_as(
        "SELECT MAX(ts_ms) FROM open_interest WHERE inst_id=$1 AND ts_ms IS NOT NULL",
    )
    .bind(inst_id)
    .fetch_optional(pool)
    .await?;
    Ok(row.map(|(ts,)| ts))
}

pub async fn insert_oi_batch(pool: &PgPool, rows: &[OpenInterestRow]) -> Result<u64> {
    if rows.is_empty() {
        return Ok(0);
    }
    let inst_ids: Vec<&str> = rows.iter().map(|r| r.inst_id.as_str()).collect();
    let ts_mss:   Vec<i64>  = rows.iter().map(|r| r.ts_ms).collect();
    let oi_ccys:  Vec<f64>  = rows.iter().map(|r| r.oi_ccy).collect();
    let oi_usds:  Vec<f64>  = rows.iter().map(|r| r.oi_usd).collect();

    let res = sqlx::query(
        r#"INSERT INTO open_interest (inst_id, ts_ms, oi_ccy, oi_usd)
           SELECT * FROM UNNEST(
               $1::varchar[], $2::bigint[], $3::float8[], $4::float8[]
           )
           ON CONFLICT (inst_id, ts_ms) DO NOTHING"#,
    )
    .bind(&inst_ids)
    .bind(&ts_mss)
    .bind(&oi_ccys)
    .bind(&oi_usds)
    .execute(pool)
    .await?;
    Ok(res.rows_affected())
}

pub async fn select_oi_since(
    pool: &PgPool,
    cutoff_ts_ms: i64,
) -> Result<Vec<OpenInterestRow>> {
    let rows = sqlx::query_as::<_, OpenInterestRow>(
        r#"SELECT inst_id, ts_ms, oi_ccy, oi_usd
             FROM open_interest
            WHERE ts_ms >= $1
            ORDER BY ts_ms, inst_id"#,
    )
    .bind(cutoff_ts_ms)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

pub async fn get_max_index_ts(pool: &PgPool, inst_id: &str) -> Result<Option<i64>> {
    let row: Option<(i64,)> = sqlx::query_as(
        "SELECT MAX(ts_ms) FROM index_candles WHERE inst_id=$1 AND ts_ms IS NOT NULL",
    )
    .bind(inst_id)
    .fetch_optional(pool)
    .await?;
    Ok(row.map(|(ts,)| ts))
}

pub async fn insert_index_candles_batch(
    pool: &PgPool,
    rows: &[IndexCandleRow],
) -> Result<u64> {
    if rows.is_empty() {
        return Ok(0);
    }
    let inst_ids: Vec<&str> = rows.iter().map(|r| r.inst_id.as_str()).collect();
    let ts_mss:   Vec<i64>  = rows.iter().map(|r| r.ts_ms).collect();
    let opens:    Vec<f64>  = rows.iter().map(|r| r.open).collect();
    let highs:    Vec<f64>  = rows.iter().map(|r| r.high).collect();
    let lows:     Vec<f64>  = rows.iter().map(|r| r.low).collect();
    let closes:   Vec<f64>  = rows.iter().map(|r| r.close).collect();

    let res = sqlx::query(
        r#"INSERT INTO index_candles (inst_id, ts_ms, open, high, low, close)
           SELECT * FROM UNNEST(
               $1::varchar[], $2::bigint[], $3::float8[], $4::float8[],
               $5::float8[], $6::float8[]
           )
           ON CONFLICT (inst_id, ts_ms) DO NOTHING"#,
    )
    .bind(&inst_ids)
    .bind(&ts_mss)
    .bind(&opens)
    .bind(&highs)
    .bind(&lows)
    .bind(&closes)
    .execute(pool)
    .await?;
    Ok(res.rows_affected())
}

pub async fn select_index_candles_since(
    pool: &PgPool,
    cutoff_ts_ms: i64,
) -> Result<Vec<IndexCandleRow>> {
    let rows = sqlx::query_as::<_, IndexCandleRow>(
        r#"SELECT inst_id, ts_ms, open, high, low, close
             FROM index_candles
            WHERE ts_ms >= $1
            ORDER BY ts_ms, inst_id"#,
    )
    .bind(cutoff_ts_ms)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

pub async fn get_max_macro_date(
    pool: &PgPool,
    series_id: &str,
) -> Result<Option<chrono::NaiveDate>> {
    let row: Option<(chrono::NaiveDate,)> = sqlx::query_as(
        "SELECT MAX(date_utc) FROM macro_daily WHERE series_id=$1 AND date_utc IS NOT NULL",
    )
    .bind(series_id)
    .fetch_optional(pool)
    .await?;
    Ok(row.map(|(d,)| d))
}

pub async fn insert_macro_batch(pool: &PgPool, rows: &[MacroDailyRow]) -> Result<u64> {
    if rows.is_empty() {
        return Ok(0);
    }
    let series_ids: Vec<&str> = rows.iter().map(|r| r.series_id.as_str()).collect();
    let dates:      Vec<chrono::NaiveDate> = rows.iter().map(|r| r.date_utc).collect();
    let values:     Vec<f64> = rows.iter().map(|r| r.value).collect();

    let res = sqlx::query(
        r#"INSERT INTO macro_daily (series_id, date_utc, value)
           SELECT * FROM UNNEST(
               $1::varchar[], $2::date[], $3::float8[]
           )
           ON CONFLICT (series_id, date_utc) DO NOTHING"#,
    )
    .bind(&series_ids)
    .bind(&dates)
    .bind(&values)
    .execute(pool)
    .await?;
    Ok(res.rows_affected())
}

/// Upsert aggregated open-interest rows; idempotent via ON CONFLICT DO NOTHING.
pub async fn insert_oi_aggregated_batch(
    pool: &PgPool,
    rows: &[crate::coinalyze::OiRow],
) -> Result<u64> {
    if rows.is_empty() {
        return Ok(0);
    }
    let symbols: Vec<&str> = rows.iter().map(|r| r.symbol.as_str()).collect();
    let ts_mss:  Vec<i64> = rows.iter().map(|r| r.ts_ms).collect();
    let oi_usds: Vec<f64> = rows.iter().map(|r| r.oi_usd).collect();
    let res = sqlx::query(
        r#"INSERT INTO open_interest_aggregated (symbol, ts_ms, oi_usd)
           SELECT * FROM UNNEST(
               $1::varchar[], $2::int8[], $3::float8[]
           )
           ON CONFLICT (symbol, ts_ms) DO NOTHING"#,
    )
    .bind(&symbols)
    .bind(&ts_mss)
    .bind(&oi_usds)
    .execute(pool)
    .await?;
    Ok(res.rows_affected())
}

/// Upsert aggregated liquidation rows; idempotent via ON CONFLICT DO NOTHING.
pub async fn insert_liq_aggregated_batch(
    pool: &PgPool,
    rows: &[crate::coinalyze::LiqRow],
) -> Result<u64> {
    if rows.is_empty() {
        return Ok(0);
    }
    let symbols: Vec<&str> = rows.iter().map(|r| r.symbol.as_str()).collect();
    let ts_mss:  Vec<i64> = rows.iter().map(|r| r.ts_ms).collect();
    let longs:   Vec<f64> = rows.iter().map(|r| r.long_liq_usd).collect();
    let shorts:  Vec<f64> = rows.iter().map(|r| r.short_liq_usd).collect();
    let res = sqlx::query(
        r#"INSERT INTO liquidations_aggregated (symbol, ts_ms, long_liq_usd, short_liq_usd)
           SELECT * FROM UNNEST(
               $1::varchar[], $2::int8[], $3::float8[], $4::float8[]
           )
           ON CONFLICT (symbol, ts_ms) DO NOTHING"#,
    )
    .bind(&symbols)
    .bind(&ts_mss)
    .bind(&longs)
    .bind(&shorts)
    .execute(pool)
    .await?;
    Ok(res.rows_affected())
}

/// Upsert OKX 15-min-aggregated taker buy/sell volume rows.
pub async fn insert_taker_volume_batch(
    pool: &PgPool,
    rows: &[crate::okx_taker_volume::TakerVolRow],
) -> Result<u64> {
    if rows.is_empty() {
        return Ok(0);
    }
    let inst_ids: Vec<&str> = rows.iter().map(|r| r.inst_id.as_str()).collect();
    let ts_mss:   Vec<i64> = rows.iter().map(|r| r.ts_ms).collect();
    let buys:     Vec<f64> = rows.iter().map(|r| r.taker_buy_vol).collect();
    let sells:    Vec<f64> = rows.iter().map(|r| r.taker_sell_vol).collect();
    let res = sqlx::query(
        r#"INSERT INTO taker_volume_15m (inst_id, ts_ms, taker_buy_vol, taker_sell_vol)
           SELECT * FROM UNNEST(
               $1::varchar[], $2::int8[], $3::float8[], $4::float8[]
           )
           ON CONFLICT (inst_id, ts_ms) DO NOTHING"#,
    )
    .bind(&inst_ids)
    .bind(&ts_mss)
    .bind(&buys)
    .bind(&sells)
    .execute(pool)
    .await?;
    Ok(res.rows_affected())
}

/// Fetch aggregated OI for `symbol` over `[from_ms, to_ms]`, as a `BTreeMap<ts_ms, oi_usd>`.
pub async fn select_oi_aggregated_range(
    pool: &PgPool,
    symbol: &str,
    from_ms: i64,
    to_ms: i64,
) -> Result<std::collections::BTreeMap<i64, f64>> {
    let rows: Vec<(i64, f64)> = sqlx::query_as(
        r#"SELECT ts_ms, oi_usd
             FROM open_interest_aggregated
            WHERE symbol = $1 AND ts_ms BETWEEN $2 AND $3
            ORDER BY ts_ms"#,
    )
    .bind(symbol)
    .bind(from_ms)
    .bind(to_ms)
    .fetch_all(pool)
    .await?;
    Ok(rows.into_iter().collect())
}

/// Fetch `(long_liq_usd, short_liq_usd)` for `symbol` over `[from_ms, to_ms]`.
pub async fn select_liq_aggregated_range(
    pool: &PgPool,
    symbol: &str,
    from_ms: i64,
    to_ms: i64,
) -> Result<std::collections::BTreeMap<i64, (f64, f64)>> {
    let rows: Vec<(i64, f64, f64)> = sqlx::query_as(
        r#"SELECT ts_ms, long_liq_usd, short_liq_usd
             FROM liquidations_aggregated
            WHERE symbol = $1 AND ts_ms BETWEEN $2 AND $3
            ORDER BY ts_ms"#,
    )
    .bind(symbol)
    .bind(from_ms)
    .bind(to_ms)
    .fetch_all(pool)
    .await?;
    Ok(rows.into_iter().map(|(t, l, s)| (t, (l, s))).collect())
}

/// Fetch `(taker_buy_vol, taker_sell_vol)` for `inst_id` over `[from_ms, to_ms]`.
pub async fn select_taker_volume_range(
    pool: &PgPool,
    inst_id: &str,
    from_ms: i64,
    to_ms: i64,
) -> Result<std::collections::BTreeMap<i64, (f64, f64)>> {
    let rows: Vec<(i64, f64, f64)> = sqlx::query_as(
        r#"SELECT ts_ms, taker_buy_vol, taker_sell_vol
             FROM taker_volume_15m
            WHERE inst_id = $1 AND ts_ms BETWEEN $2 AND $3
            ORDER BY ts_ms"#,
    )
    .bind(inst_id)
    .bind(from_ms)
    .bind(to_ms)
    .fetch_all(pool)
    .await?;
    Ok(rows.into_iter().map(|(t, b, s)| (t, (b, s))).collect())
}

pub async fn select_macro_since(
    pool: &PgPool,
    cutoff_date_utc: chrono::NaiveDate,
) -> Result<Vec<MacroDailyRow>> {
    let rows = sqlx::query_as::<_, MacroDailyRow>(
        r#"SELECT series_id, date_utc, value
             FROM macro_daily
            WHERE date_utc >= $1
            ORDER BY date_utc, series_id"#,
    )
    .bind(cutoff_date_utc)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// Current model blob for `symbol`, or None if not yet promoted.
/// Returns `(model_bytes, format_version, model_version_id, model_family)`.
pub async fn get_current_model_bytes(
    pool: &PgPool,
    symbol: &str,
) -> Result<Option<(Vec<u8>, i16, i64, String)>> {
    let row: Option<(Vec<u8>, i16, i64, String)> = sqlx::query_as(
        r#"SELECT m.model_bytes, m.format_version, v.id, v.model_family
           FROM models m
           JOIN model_versions v ON v.id = m.model_version_id
           WHERE v.symbol = $1 AND v.is_current"#,
    )
    .bind(symbol)
    .fetch_optional(pool)
    .await?;
    Ok(row)
}

/// Set of symbols with `is_current=true` and `format_version = $1`.
pub async fn load_current_model_symbols(
    pool: &PgPool,
    format_version: i16,
) -> Result<HashSet<String>> {
    let rows: Vec<(String,)> = sqlx::query_as(
        r#"SELECT v.symbol
             FROM model_versions v
             JOIN models m ON m.model_version_id = v.id
            WHERE v.is_current
              AND m.format_version = $1"#,
    )
    .bind(format_version)
    .fetch_all(pool)
    .await?;
    Ok(rows.into_iter().map(|(s,)| s).collect())
}

/// Map of symbol → `ModelBlobEntry` for every currently-promoted model.
pub async fn load_current_models(
    pool: &PgPool,
) -> Result<HashMap<String, ModelBlobEntry>> {
    let rows: Vec<(String, Vec<u8>, i16, i64, String, String)> = sqlx::query_as(
        r#"SELECT v.symbol, m.model_bytes, m.format_version, v.id, v.model_family, m.sha256_hex
           FROM models m
           JOIN model_versions v ON v.id = m.model_version_id
           WHERE v.is_current"#,
    )
    .fetch_all(pool)
    .await?;
    Ok(rows
        .into_iter()
        .map(|(symbol, bytes, format_version, model_version_id, model_family, sha256_hex)| {
            (
                symbol,
                ModelBlobEntry {
                    bytes,
                    format_version,
                    model_version_id,
                    model_family,
                    sha256_hex,
                },
            )
        })
        .collect())
}

pub async fn get_current_model_version(
    pool: &PgPool,
    symbol: &str,
) -> Result<Option<ModelVersionRow>> {
    let row = sqlx::query_as::<_, ModelVersionRow>(
        "SELECT id, symbol, is_current, created_at FROM model_versions
         WHERE symbol=$1 AND is_current",
    )
    .bind(symbol)
    .fetch_optional(pool)
    .await?;
    Ok(row)
}

/// Most recent `labeled_row_count` for `symbol`, or None if no prior version exists.
pub async fn prior_labeled_row_count(
    pool: &PgPool,
    symbol: &str,
) -> std::result::Result<Option<i32>, sqlx::Error> {
    let row: Option<(Option<i32>,)> = sqlx::query_as(
        r#"SELECT labeled_row_count
             FROM model_versions
            WHERE symbol = $1
            ORDER BY id DESC
            LIMIT 1"#,
    )
    .bind(symbol)
    .fetch_optional(pool)
    .await?;
    Ok(row.and_then(|(c,)| c))
}

/// Most recent `(labeled_row_count, format_version)` for the last promoted model of `symbol`.
pub async fn prior_promoted_meta(
    pool: &PgPool,
    symbol: &str,
) -> std::result::Result<Option<(Option<i32>, i16)>, sqlx::Error> {
    let row: Option<(Option<i32>, i16)> = sqlx::query_as(
        r#"SELECT v.labeled_row_count, m.format_version
             FROM model_versions v
             JOIN models m ON m.model_version_id = v.id
            WHERE v.symbol = $1
            ORDER BY v.id DESC
            LIMIT 1"#,
    )
    .bind(symbol)
    .fetch_optional(pool)
    .await?;
    Ok(row)
}

/// Atomically promote a newly-trained model: insert version + blob rows, flip
/// `is_current`, and NOTIFY `model_updated`. Returns the new `model_versions.id`.
pub async fn promote_model(
    pool: &PgPool,
    symbol: &str,
    model_bytes: &[u8],
    labeled_row_count: i32,
    format_version: i16,
    model_family: &str,
) -> Result<i64> {
    let mut tx = pool.begin().await?;

    let (new_version_id,): (i64,) = sqlx::query_as(
        "INSERT INTO model_versions (symbol, is_current, labeled_row_count, model_family)
         VALUES ($1, false, $2, $3) RETURNING id",
    )
    .bind(symbol)
    .bind(labeled_row_count)
    .bind(model_family)
    .fetch_one(&mut *tx)
    .await?;

    let byte_size: i32 = model_bytes.len().try_into()?;
    sqlx::query(
        r#"INSERT INTO models (model_version_id, model_bytes, byte_size, sha256_hex, format_version)
           VALUES ($1, $2, $3, encode(digest($2, 'sha256'), 'hex'), $4)"#,
    )
    .bind(new_version_id)
    .bind(model_bytes)
    .bind(byte_size)
    .bind(format_version)
    .execute(&mut *tx)
    .await?;

    sqlx::query(
        "UPDATE model_versions SET is_current=false WHERE symbol=$1 AND is_current",
    )
    .bind(symbol)
    .execute(&mut *tx)
    .await?;

    sqlx::query("UPDATE model_versions SET is_current=true WHERE id=$1")
        .bind(new_version_id)
        .execute(&mut *tx)
        .await?;

    sqlx::query("SELECT pg_notify('model_updated', $1)")
        .bind(symbol)
        .execute(&mut *tx)
        .await?;

    tx.commit().await?;
    Ok(new_version_id)
}

/// Atomically promote a pooled multi-symbol model blob for all entries in
/// `per_symbol_row_counts` in a single transaction. Returns new `model_versions.id`
/// per entry in the same order.
pub async fn promote_pooled_model(
    pool: &PgPool,
    model_bytes: &[u8],
    per_symbol_row_counts: &[(&str, i32)],
    format_version: i16,
) -> Result<Vec<i64>> {
    let mut tx = pool.begin().await?;
    let byte_size: i32 = model_bytes.len().try_into()?;

    let mut new_ids: Vec<i64> = Vec::with_capacity(per_symbol_row_counts.len());

    for (symbol, labeled_row_count) in per_symbol_row_counts {
        let (vid,): (i64,) = sqlx::query_as(
            "INSERT INTO model_versions (symbol, is_current, labeled_row_count, model_family)
             VALUES ($1, false, $2, 'lightgbm_pooled') RETURNING id",
        )
        .bind(symbol)
        .bind(labeled_row_count)
        .fetch_one(&mut *tx)
        .await?;

        sqlx::query(
            r#"INSERT INTO models (model_version_id, model_bytes, byte_size, sha256_hex, format_version)
               VALUES ($1, $2, $3, encode(digest($2, 'sha256'), 'hex'), $4)"#,
        )
        .bind(vid)
        .bind(model_bytes)
        .bind(byte_size)
        .bind(format_version)
        .execute(&mut *tx)
        .await?;

        new_ids.push(vid);
    }

    // Flip is_current per symbol after all inserts; new rows stay is_current=false
    // until their final UPDATE to satisfy the partial unique index.
    for ((symbol, _), &vid) in per_symbol_row_counts.iter().zip(new_ids.iter()) {
        sqlx::query(
            "UPDATE model_versions SET is_current=false WHERE symbol=$1 AND is_current AND id<>$2",
        )
        .bind(symbol)
        .bind(vid)
        .execute(&mut *tx)
        .await?;

        sqlx::query("UPDATE model_versions SET is_current=true WHERE id=$1")
            .bind(vid)
            .execute(&mut *tx)
            .await?;
    }

    for (symbol, _) in per_symbol_row_counts {
        sqlx::query("SELECT pg_notify('model_updated', $1)")
            .bind(symbol)
            .execute(&mut *tx)
            .await?;
    }

    tx.commit().await?;
    Ok(new_ids)
}

/// Dry-run signals that have settled but lack a `dry_run_results` row.
pub async fn get_unresolved_dry_run_signals(
    pool: &PgPool,
    before_ts_ms: i64,
) -> Result<Vec<DryRunPendingRow>> {
    let rows = sqlx::query_as::<_, DryRunPendingRow>(
        r#"SELECT s.id AS signal_id, s.market_id AS market_id,
                  s.signal, s.confidence,
                  p.entry_price, p.shares, p.fee_rate_bps, p.skip_reason
             FROM signals s
             LEFT JOIN dry_run_results r  ON r.signal_id = s.id
             LEFT JOIN dry_run_pending p  ON p.signal_id = s.id
            WHERE s.market_id IS NOT NULL
              AND s.signal IN (1, 2)
              AND s.ts_ms <= $1
              AND r.id IS NULL"#,
    )
    .bind(before_ts_ms)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// Stash at-decision-time price/size for a dry-run signal.
/// `fee_rate_bps` is NULL in dry-run; `skip_reason` is set when the trader bailed before pricing.
pub async fn insert_dry_run_pending(
    pool: &PgPool,
    signal_id: i64,
    entry_price: Option<f64>,
    shares: Option<f64>,
    fee_rate_bps: Option<i32>,
    skip_reason: Option<&str>,
) -> Result<()> {
    sqlx::query(
        r#"INSERT INTO dry_run_pending (signal_id, entry_price, shares, fee_rate_bps, skip_reason)
           VALUES ($1, $2, $3, $4, $5)
           ON CONFLICT (signal_id) DO NOTHING"#,
    )
    .bind(signal_id)
    .bind(entry_price)
    .bind(shares)
    .bind(fee_rate_bps)
    .bind(skip_reason)
    .execute(pool)
    .await?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub async fn insert_dry_run_result(
    pool: &PgPool,
    signal_id: i64,
    market_id: &str,
    predicted_action: i16,
    confidence: f64,
    actual_outcome: i16,
    correct: bool,
    entry_price: Option<f64>,
    shares: Option<f64>,
    fee_rate_bps: Option<i32>,
    skip_reason: Option<&str>,
) -> Result<i64> {
    let (id,): (i64,) = sqlx::query_as(
        r#"INSERT INTO dry_run_results
           (signal_id, market_id, predicted_action, confidence, actual_outcome, correct,
            entry_price, shares, fee_rate_bps, skip_reason)
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
           RETURNING id"#,
    )
    .bind(signal_id)
    .bind(market_id)
    .bind(predicted_action)
    .bind(confidence)
    .bind(actual_outcome)
    .bind(correct)
    .bind(entry_price)
    .bind(shares)
    .bind(fee_rate_bps)
    .bind(skip_reason)
    .fetch_one(pool)
    .await?;
    Ok(id)
}

/// Upsert a resolved outcome. Idempotent.
pub async fn upsert_window_outcome(
    pool: &PgPool,
    symbol: &str,
    window_ts_secs: i64,
    outcome: i16,
    slug: &str,
) -> Result<()> {
    sqlx::query(
        r#"INSERT INTO window_outcomes (symbol, window_ts_secs, outcome, slug)
           VALUES ($1, $2, $3, $4)
           ON CONFLICT (symbol, window_ts_secs) DO UPDATE
           SET outcome = EXCLUDED.outcome, slug = EXCLUDED.slug"#,
    )
    .bind(symbol)
    .bind(window_ts_secs)
    .bind(outcome)
    .bind(slug)
    .execute(pool)
    .await?;
    Ok(())
}

/// Bulk-load outcomes for a symbol over a window-ts range as a `BTreeMap<window_ts_secs, outcome>`.
pub async fn load_window_outcomes(
    pool: &PgPool,
    symbol: &str,
    from_ts_secs: i64,
    to_ts_secs: i64,
) -> Result<std::collections::BTreeMap<i64, i16>> {
    let rows: Vec<(i64, i16)> = sqlx::query_as(
        r#"SELECT window_ts_secs, outcome FROM window_outcomes
           WHERE symbol=$1 AND window_ts_secs >= $2 AND window_ts_secs <= $3"#,
    )
    .bind(symbol)
    .bind(from_ts_secs)
    .bind(to_ts_secs)
    .fetch_all(pool)
    .await?;
    Ok(rows.into_iter().collect())
}

/// Return the set of `window_ts_secs` in `[from, to]` that already have a row.
pub async fn existing_window_ts(
    pool: &PgPool,
    symbol: &str,
    from_ts_secs: i64,
    to_ts_secs: i64,
) -> Result<HashSet<i64>> {
    let rows: Vec<(i64,)> = sqlx::query_as(
        r#"SELECT window_ts_secs FROM window_outcomes
           WHERE symbol=$1 AND window_ts_secs >= $2 AND window_ts_secs <= $3"#,
    )
    .bind(symbol)
    .bind(from_ts_secs)
    .bind(to_ts_secs)
    .fetch_all(pool)
    .await?;
    Ok(rows.into_iter().map(|(t,)| t).collect())
}

/// Drop non-current versions older than `keep_days`; `models` rows cascade.
pub async fn prune_old_versions(
    pool: &PgPool,
    symbol: &str,
    keep_days: i32,
) -> Result<u64> {
    let res = sqlx::query(
        r#"DELETE FROM model_versions
           WHERE symbol=$1 AND NOT is_current
             AND created_at < NOW() - make_interval(days => $2)"#,
    )
    .bind(symbol)
    .bind(keep_days)
    .execute(pool)
    .await?;
    Ok(res.rows_affected())
}

/// Bulk-insert per-prediction feature importance rows for one signal (slot features + GLOBAL tail).
pub async fn insert_model_importance_batch(
    pool: &PgPool,
    signal_id: i64,
    model_version_id: i64,
    symbol: &str,
    features: &[f32],
    importances: &[f64],
) -> Result<u64> {
    use crate::feature_engine::{
        short_symbol, FEATURE_DIM, FEATURE_NAMES, GLOBAL_FEATURE_DIM, GLOBAL_FEATURE_NAMES,
        INSTRUMENT_COUNT, INSTRUMENT_ORDER, TOTAL_FEATURES,
    };

    if features.len() != TOTAL_FEATURES || importances.len() != TOTAL_FEATURES {
        return Err(anyhow::anyhow!(
            "insert_model_importance_batch: expected len={} (features={}, importances={})",
            TOTAL_FEATURES, features.len(), importances.len()
        ));
    }

    let slot = INSTRUMENT_ORDER
        .iter()
        .position(|&s| short_symbol(s) == symbol)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "insert_model_importance_batch: symbol {:?} not in INSTRUMENT_ORDER",
                symbol
            )
        })?;
    let inst_name = INSTRUMENT_ORDER[slot];

    let n = FEATURE_DIM + GLOBAL_FEATURE_DIM;
    let mut signal_ids  = Vec::with_capacity(n);
    let mut version_ids = Vec::with_capacity(n);
    let mut symbols     = Vec::with_capacity(n);
    let mut inst_names  = Vec::with_capacity(n);
    let mut feat_names  = Vec::with_capacity(n);
    let mut feat_values = Vec::with_capacity(n);
    let mut imps        = Vec::with_capacity(n);
    for (feat, name) in FEATURE_NAMES.iter().enumerate().take(FEATURE_DIM) {
        let idx = slot * FEATURE_DIM + feat;
        signal_ids.push(signal_id);
        version_ids.push(model_version_id);
        symbols.push(symbol);
        inst_names.push(inst_name);
        feat_names.push(*name);
        feat_values.push(features[idx] as f64);
        imps.push(importances[idx]);
    }
    let global_base = INSTRUMENT_COUNT * FEATURE_DIM;
    for (g, name) in GLOBAL_FEATURE_NAMES.iter().enumerate().take(GLOBAL_FEATURE_DIM) {
        let idx = global_base + g;
        signal_ids.push(signal_id);
        version_ids.push(model_version_id);
        symbols.push(symbol);
        inst_names.push("GLOBAL");
        feat_names.push(*name);
        feat_values.push(features[idx] as f64);
        imps.push(importances[idx]);
    }

    let res = sqlx::query(
        r#"INSERT INTO model_importance
           (signal_id, model_version_id, symbol, instrument_name, feature_name, feature_value, importance)
           SELECT * FROM UNNEST(
               $1::bigint[], $2::bigint[], $3::varchar[],
               $4::varchar[], $5::varchar[], $6::float8[], $7::float8[]
           )
           ON CONFLICT (signal_id, instrument_name, feature_name) DO NOTHING"#,
    )
    .bind(&signal_ids)
    .bind(&version_ids)
    .bind(&symbols)
    .bind(&inst_names)
    .bind(&feat_names)
    .bind(&feat_values)
    .bind(&imps)
    .execute(pool)
    .await?;
    Ok(res.rows_affected())
}

#[allow(clippy::too_many_arguments)]
pub async fn insert_retrain_diagnostic(
    pool: &PgPool,
    symbol: &str,
    labeled_rows: i32,
    train_acc: f64,
    val_acc: Option<f64>,
    majority_baseline: Option<f64>,
    agreement_rate: Option<f64>,
    logloss: f64,
    iterations_used: i32,
    promoted: bool,
    gate_rejected: Option<&str>,
) -> Result<()> {
    sqlx::query(
        r#"INSERT INTO retrain_diagnostics
           (symbol, labeled_rows, train_acc, val_acc, majority_baseline,
            agreement_rate, logloss, iterations_used, promoted, gate_rejected)
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)"#,
    )
    .bind(symbol)
    .bind(labeled_rows)
    .bind(train_acc)
    .bind(val_acc)
    .bind(majority_baseline)
    .bind(agreement_rate)
    .bind(logloss)
    .bind(iterations_used)
    .bind(promoted)
    .bind(gate_rejected)
    .execute(pool)
    .await?;
    Ok(())
}
