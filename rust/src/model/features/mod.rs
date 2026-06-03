// src/feature_engine/mod.rs

pub mod candle_builder;
pub mod feature_engineer;
pub mod normalizer;
pub mod pipeline;
pub mod tick_buffer;
pub mod types;

pub use candle_builder::{bucket_start, build_candle};
pub use feature_engineer::{FeatureEngineer, MIN_HISTORY};
pub use normalizer::{Normalizer, NORM_WINDOW};
pub use pipeline::Pipeline;
pub use tick_buffer::TickBuffer;
pub use types::{
    feature_full_name, short_symbol, Candle, FeatureRow, MacroSeries, MacroSnapshot, MicroSample,
    PerpSample, RawTick, CANDLE_INTERVAL_MS, FEATURE_DIM, FEATURE_NAMES, FUNDING_LOOKBACK,
    GLOBAL_FEATURE_DIM, GLOBAL_FEATURE_NAMES, INSTRUMENT_COUNT, INSTRUMENT_ORDER,
    MACRO_SERIES_IDS, REQUIRED_MACRO_SERIES, TOTAL_FEATURES,
};
