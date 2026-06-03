// src/feature_engine/pipeline.rs
//
// Thin wiring layer: FeatureEngineer → Normalizer.
// Stays pure (no WS code) so retrain can drive it directly from stored candles.

use super::feature_engineer::FeatureEngineer;
use super::normalizer::Normalizer;
use super::types::{Candle, FeatureRow, MacroSnapshot, PerpSample};

pub struct Pipeline {
    engineer: FeatureEngineer,
    normalizer: Normalizer,
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            engineer: FeatureEngineer::new(),
            normalizer: Normalizer::new(),
        }
    }

    /// Push a candle; returns a normalised FeatureRow when emission criteria are met, else None.
    pub fn push_candle(&mut self, candle: Candle) -> Option<FeatureRow> {
        let raw = self.engineer.push_candle(candle)?;
        self.normalizer.push(&raw)
    }

    /// Push a candle and return the raw (un-normalised) row; used by retrain.
    pub fn push_candle_raw(&mut self, candle: Candle) -> Option<FeatureRow> {
        self.engineer.push_candle(candle)
    }

    /// Forward a perp snapshot into the engineer; rows are emitted by `push_candle`.
    pub fn push_perp_sample(&mut self, inst_id: &str, bucket_ts_ms: i64, sample: PerpSample) {
        self.engineer.push_perp_sample(inst_id, bucket_ts_ms, sample);
    }

    /// Forward the macro snapshot; missing macro data blocks row emission.
    pub fn push_macro_snapshot(&mut self, snapshot: &MacroSnapshot) {
        self.engineer.push_macro_snapshot(snapshot);
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}
