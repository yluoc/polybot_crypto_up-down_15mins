// src/feature_engine/tick_buffer.rs
//
// Fixed-capacity ring buffer of RawTick, one per instrument.
// Wrapped in Arc<Mutex<_>> by the live pipeline; drained by CandleBuilder at bucket rollover.

use std::collections::VecDeque;

use super::types::RawTick;

pub const TICK_BUFFER_CAPACITY: usize = 2048;

pub struct TickBuffer {
    inst_id: String,
    buf: VecDeque<RawTick>, // FIFO; oldest tick dropped on overflow
}

impl TickBuffer {
    pub fn new(inst_id: impl Into<String>) -> Self {
        Self {
            inst_id: inst_id.into(),
            buf: VecDeque::with_capacity(TICK_BUFFER_CAPACITY),
        }
    }

    pub fn inst_id(&self) -> &str {
        &self.inst_id
    }

    pub fn push(&mut self, tick: RawTick) {
        if self.buf.len() == TICK_BUFFER_CAPACITY {
            self.buf.pop_front();
        }
        self.buf.push_back(tick);
    }

    /// Drain ticks with `start_ms <= ts_ms < end_ms`. Ticks older than
    /// `start_ms` are dropped (stale); ticks at or past `end_ms` are left
    /// for the next drain. Mirrors C++ TickBuffer::drain_range.
    pub fn drain_range(&mut self, start_ms: i64, end_ms: i64) -> Vec<RawTick> {
        let mut out = Vec::new();
        while let Some(t) = self.buf.pop_front() {
            if t.ts_ms >= end_ms {
                self.buf.push_front(t);          // future tick — leave for next drain
                break;
            }
            if t.ts_ms >= start_ms {
                out.push(t);
            }
            // else: stale, drop
        }
        out
    }

    pub fn latest(&self) -> Option<&RawTick> {
        self.buf.back()
    }
}
