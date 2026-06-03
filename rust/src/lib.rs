pub mod cli;
pub mod config;
pub mod data;
pub mod db;
pub mod markets;
pub mod model;
pub mod runtime;

// data/
pub use data::backfill_candles as backfill;
pub use data::backfill_macro;
pub use data::backfill_outcomes as outcome_backfill;
pub use data::coinalyze;
pub use data::macro_calendar;
pub use data::macro_poll;
pub use data::okx_taker as okx_taker_volume;
pub use data::perp_poll;
pub use data::ws;

// markets/
pub use markets::book_stream;
pub use markets::order_manager;
pub use markets::resolution;
pub use markets::resolver as market_resolver;

// config/
pub use config::bootstrap;
pub use config::preflight;

// cli/
pub use cli::cron;

// model/
pub use model::calibration;
pub use model::features as feature_engine;
pub use model::inference;
pub use model::retrain;
pub use model::training;

// runtime/
pub use runtime::run;
pub use runtime::signal;
pub use runtime::trader;
pub use runtime::warmup;
