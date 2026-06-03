// src/feature_engine/candle_builder.rs
//
// Builds 15-minute OHLC candles from accumulated ticks.
// Polls TickBuffer once per second via a tokio task; emits on bucket rollover.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::time;

use super::tick_buffer::TickBuffer;
use super::types::{Candle, CANDLE_INTERVAL_MS};

#[inline]
pub fn bucket_start(ts_ms: i64) -> i64 {
    (ts_ms / CANDLE_INTERVAL_MS) * CANDLE_INTERVAL_MS
}

/// Returns the UNIX epoch time in ms.
fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Spawns a task that watches `buffer` and emits a Candle on `tx` whenever
/// the current 15-minute bucket closes. `cancel` (shared) flipped to true
/// ends the task at the next tick.
pub fn spawn(
    buffer: Arc<Mutex<TickBuffer>>,
    tx: mpsc::UnboundedSender<Candle>,
    cancel: Arc<std::sync::atomic::AtomicBool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut current_bucket: i64 = -1;
        let mut ticker = time::interval(Duration::from_secs(1));
        loop {
            ticker.tick().await;
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            let bucket = bucket_start(now_ms());
            if current_bucket < 0 {
                current_bucket = bucket;
                continue;
            }
            if bucket == current_bucket {
                continue;
            }

            let inst_id;
            let ticks;
            {
                let mut buf = match buffer.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(), // poisoned — continue with recovered guard
                };
                inst_id = buf.inst_id().to_string();
                ticks = buf.drain_range(current_bucket, current_bucket + CANDLE_INTERVAL_MS);
            }

            if let Some(candle) = build_candle(inst_id, current_bucket, &ticks) {
                if tx.send(candle).is_err() {
                    break; // receiver gone
                }
            }
            current_bucket = bucket;
        }
    })
}

/// Returns None if there were no ticks for this bucket — the caller can
/// decide whether to emit a warning.
pub fn build_candle(inst_id: String, open_ts_ms: i64, ticks: &[super::types::RawTick]) -> Option<Candle> {
    let first = ticks.first()?;
    let last = ticks.last()?;
    let mut high = f64::MIN;
    let mut low = f64::MAX;
    for t in ticks {
        if t.mark_px > high {
            high = t.mark_px;
        }
        if t.mark_px < low {
            low = t.mark_px;
        }
    }
    Some(Candle {
        inst_id,
        open_ts_ms,
        close_ts_ms: last.ts_ms,
        open: first.mark_px,
        close: last.mark_px,
        high,
        low,
        tick_count: ticks.len() as u32,
    })
}
