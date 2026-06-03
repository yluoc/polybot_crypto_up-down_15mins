// All configuration loaded from environment / .env file.

use anyhow::{Context, Result};
use polymarket_client_sdk_v2::types::Address;
use rust_decimal::Decimal;
use std::collections::HashSet;
use std::str::FromStr;

use crate::signal::Symbol;

#[derive(Clone, Debug)]
pub struct Config {
    /// Polymarket CLOB base URL
    pub polymarket_api_url: String,

    /// Polymarket Gamma API base URL (no auth required)
    pub gamma_api_url: String,

    /// EOA private key (0x-prefixed hex) for signing orders
    pub private_key: String,

    /// Polymarket proxy wallet address (holds USDC.e). The signer's EOA
    /// controls this proxy — orders are signed with `SignatureType::Proxy`.
    pub funder: Address,

    /// Max USDC notional per order — if `price * shares` exceeds this, the cycle is skipped.
    pub order_max_usdc: Decimal,

    /// Min model confidence to act on.
    pub min_confidence: f32,

    /// Seconds before window boundary to place order (e.g. 10)
    pub entry_offset_secs: u64,

    /// Polymarket WSS market-channel endpoint.
    pub wss_url: String,

    /// Max age of a cached book entry before it's considered stale (ms).
    pub book_staleness_ms: u64,

    /// Seconds before window end at which to pre-subscribe to the next window.
    pub prewarm_lead_secs: u64,

    /// Cryptos this bot is trading (parsed from CRYPTOS=btc,eth,sol,...).
    /// A signal whose symbol is not in this set is dropped.
    pub cryptos: HashSet<Symbol>,

    /// Postgres connection string (e.g. "postgres://user:pass@localhost/polybot")
    pub database_url: String,

    /// OKX public-channel WS endpoint (mark-price stream source for live inference).
    pub okx_ws_url: String,

    /// Kill switch for the startup wallet pre-flight check. Default true.
    pub preflight_enabled: bool,

    /// Paper-trading mode: pipeline runs but no orders are placed. Default false.
    pub dry_run: bool,

    /// On startup, run backfill + retrain if any symbol lacks a model row. Default true.
    pub bootstrap_on_empty: bool,

    /// FRED API key for the macro (DXY/SPX/VIX/yields) feature block.
    pub fred_api_key: String,

    /// Minimum EV per share (dollars, net of fee) for the EV gate. Default 0.02.
    pub ev_min_edge: f64,

    /// Maximum top-of-book spread in basis points of mid. Default 300.
    pub max_spread_bps: u32,

    /// Kill switch for the EV gate. Default true.
    pub ev_gate_enabled: bool,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        dotenvy::dotenv().ok();

        let cryptos: HashSet<Symbol> = var("CRYPTOS")
            .unwrap_or_else(|_| "btc,eth,sol,xrp".into())
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(Symbol::from_str_ci)
            .collect::<Result<_>>()?;
        if cryptos.is_empty() {
            anyhow::bail!("CRYPTOS must list at least one symbol");
        }

        Ok(Self {
            polymarket_api_url: var("POLYMARKET_API_URL")
                .unwrap_or_else(|_| "https://clob.polymarket.com".into()),
            gamma_api_url: var("GAMMA_API_URL")
                .unwrap_or_else(|_| "https://gamma-api.polymarket.com".into()),
            private_key: var("POLYMARKET_PRIVATE_KEY")?,
            funder: Address::from_str(&var("POLYMARKET_FUNDER")?)
                .context("POLYMARKET_FUNDER must be a 0x-prefixed checksum address")?,
            order_max_usdc: Decimal::from_str(&var("ORDER_MAX_NOTIONAL_USD")?)
                .context("ORDER_MAX_NOTIONAL_USD must be a decimal number")?,
            min_confidence: var("MIN_CONFIDENCE")
                .unwrap_or_else(|_| "0.62".into())
                .parse()
                .context("MIN_CONFIDENCE must be a float")?,
            entry_offset_secs: var("ENTRY_OFFSET_SECS")
                .unwrap_or_else(|_| "10".into())
                .parse()
                .context("ENTRY_OFFSET_SECS must be an integer")?,
            wss_url: var("WSS_URL")
                .unwrap_or_else(|_| "wss://ws-subscriptions-clob.polymarket.com/ws/market".into()),
            book_staleness_ms: var("BOOK_STALENESS_MS")
                .unwrap_or_else(|_| "2000".into())
                .parse()
                .context("BOOK_STALENESS_MS must be an integer")?,
            prewarm_lead_secs: var("PREWARM_LEAD_SECS")
                .unwrap_or_else(|_| "30".into())
                .parse()
                .context("PREWARM_LEAD_SECS must be an integer")?,
            cryptos,
            database_url: var("DATABASE_URL")?,
            okx_ws_url: var("OKX_WS_URL")
                .unwrap_or_else(|_| "wss://ws.okx.com:8443/ws/v5/public".into()),
            preflight_enabled: var("PREFLIGHT_ENABLED")
                .unwrap_or_else(|_| "true".into())
                .parse()
                .context("PREFLIGHT_ENABLED must be a boolean (true/false)")?,
            dry_run: var("DRY_RUN")
                .unwrap_or_else(|_| "false".into())
                .parse()
                .context("DRY_RUN must be a boolean (true/false)")?,
            bootstrap_on_empty: var("BOOTSTRAP_ON_EMPTY")
                .unwrap_or_else(|_| "true".into())
                .parse()
                .context("BOOTSTRAP_ON_EMPTY must be a boolean (true/false)")?,
            fred_api_key: var("FRED_API_KEY")?,
            ev_min_edge: var("EV_MIN_EDGE")
                .unwrap_or_else(|_| "0.02".into())
                .parse()
                .context("EV_MIN_EDGE must be a float")?,
            max_spread_bps: var("MAX_SPREAD_BPS")
                .unwrap_or_else(|_| "300".into())
                .parse()
                .context("MAX_SPREAD_BPS must be an integer")?,
            ev_gate_enabled: var("EV_GATE_ENABLED")
                .unwrap_or_else(|_| "true".into())
                .parse()
                .context("EV_GATE_ENABLED must be a boolean (true/false)")?,
        })
    }
}

fn var(key: &str) -> Result<String> {
    std::env::var(key).with_context(|| format!("Missing env var: {key}"))
}
