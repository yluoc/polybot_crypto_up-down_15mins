
use anyhow::{bail, Result};

use crate::feature_engine::INSTRUMENT_ORDER;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    BuyUp,
    BuyDown,
}

impl Action {
    /// DB encoding: 1=UP, 2=DOWN.
    pub fn as_i16(self) -> i16 {
        match self {
            Action::BuyUp   => 1,
            Action::BuyDown => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Symbol {
    Btc,
    Eth,
    Sol,
    Xrp,
}

impl Symbol {
    /// Lowercase Polymarket slug (e.g. `"btc"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Symbol::Btc => "btc",
            Symbol::Eth => "eth",
            Symbol::Sol => "sol",
            Symbol::Xrp => "xrp",
        }
    }

    pub fn from_str_ci(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "btc" => Ok(Symbol::Btc),
            "eth" => Ok(Symbol::Eth),
            "sol" => Ok(Symbol::Sol),
            "xrp" => Ok(Symbol::Xrp),
            other => bail!("Unknown crypto symbol: '{other}'"),
        }
    }

    /// Uppercase DB key and NOTIFY payload (e.g. `"BTC"`).
    pub fn short(self) -> &'static str {
        match self {
            Symbol::Btc => "BTC",
            Symbol::Eth => "ETH",
            Symbol::Sol => "SOL",
            Symbol::Xrp => "XRP",
        }
    }

    /// OKX instrument ID matching `feature_engine::INSTRUMENT_ORDER`.
    pub fn inst_id(self) -> &'static str {
        match self {
            Symbol::Btc => "BTC-USDT-SWAP",
            Symbol::Eth => "ETH-USDT-SWAP",
            Symbol::Sol => "SOL-USDT-SWAP",
            Symbol::Xrp => "XRP-USDT-SWAP",
        }
    }
}

/// Categorical feature id: position of the symbol's instrument in `INSTRUMENT_ORDER`.
/// Pinned by the `symbol_id_pinning` integration test — reordering `INSTRUMENT_ORDER` will fail CI.
pub fn symbol_id(sym: Symbol) -> i32 {
    INSTRUMENT_ORDER
        .iter()
        .position(|&s| s == sym.inst_id())
        .expect("Symbol::inst_id must be present in INSTRUMENT_ORDER") as i32
}

#[derive(Debug, Clone, Copy)]
pub struct TradingSignal {
    pub candle_ts_ms: i64,
    pub action: Action,
    pub symbol: Symbol,
    pub confidence: f32,
    #[allow(dead_code)]
    pub raw_score: f32,
    /// `signals.id` inserted upstream; `None` on DB transient (order still proceeds).
    pub signal_id: Option<i64>,
}
