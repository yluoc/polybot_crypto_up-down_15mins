// Deterministic slug computation + Gamma API lookup.

use anyhow::{bail, Context, Result};
use rust_decimal::Decimal;
use serde::{Deserialize, Deserializer};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;
use tracing::{info, warn};

/// Polymarket crypto market window duration (e.g. 900 for 15-min, 300 for 5-min).
pub const WINDOW_SECS: u64 = 900;
const WINDOW_MINS: u64 = WINDOW_SECS / 60;
/// Drop cached entries older than this many seconds, so the cache stays bounded.
const CACHE_RETENTION_SECS: u64 = 7_200;

/// Outcome token pair for a single market window.
#[derive(Debug, Clone)]
pub struct MarketTokens {
    pub window_ts: u64, // Unix seconds of window start
    pub slug: String,
    pub up_token_id: String,
    pub down_token_id: String,
    /// Market-required minimum order size, in shares.
    pub min_order_size: Decimal,
}

/// Minimal Gamma API market response shape.
#[derive(Debug, Deserialize)]
struct GammaMarket {
    slug: String,
    // `clobTokenIds` is returned as a JSON-encoded string, not a raw array.
    #[serde(rename = "clobTokenIds", default, deserialize_with = "de_stringified_vec")]
    clob_token_ids: Option<Vec<String>>,
    #[serde(rename = "orderMinSize")]
    order_min_size: Option<Decimal>,
    #[allow(dead_code)]
    active: Option<bool>,
    closed: Option<bool>,
}

pub fn de_stringified_vec<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrVec {
        Vec(Vec<String>),
        Str(String),
    }

    match Option::<StringOrVec>::deserialize(deserializer)? {
        None => Ok(None),
        Some(StringOrVec::Vec(v)) => Ok(Some(v)),
        Some(StringOrVec::Str(s)) => serde_json::from_str(&s)
            .map(Some)
            .map_err(serde::de::Error::custom),
    }
}

pub struct MarketResolver {
    gamma_base: String,
    http: reqwest::Client,
    crypto: String, // e.g. "btc"
    /// Per-window-ts token cache.
    cache: Mutex<HashMap<u64, Arc<MarketTokens>>>,
}

impl MarketResolver {
    pub fn new(http: reqwest::Client, gamma_base: &str, crypto: &str) -> Self {
        Self {
            gamma_base: gamma_base.to_string(),
            http,
            crypto: crypto.to_string(),
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Return tokens for the current window, falling back to the next window if needed.
    pub async fn current_tokens(&self) -> Result<Arc<MarketTokens>> {
        let window_ts = current_window_ts();

        if let Some(cached) = self.cache.lock().await.get(&window_ts).cloned() {
            return Ok(cached);
        }

        let tokens = Arc::new(self.fetch_tokens(window_ts).await?);
        info!(
            "[MarketResolver] window={} slug={} up={:.8}... down={:.8}...",
            window_ts,
            tokens.slug,
            &tokens.up_token_id[..8.min(tokens.up_token_id.len())],
            &tokens.down_token_id[..8.min(tokens.down_token_id.len())],
        );
        self.store_in_cache(window_ts, tokens.clone()).await;
        Ok(tokens)
    }

    #[allow(dead_code)]
    pub async fn invalidate(&self) {
        self.cache.lock().await.clear();
    }

    /// Fetch tokens for a specific window_ts (exact-slug, no fallback). Cached.
    pub async fn tokens_for_window(&self, window_ts: u64) -> Result<Arc<MarketTokens>> {
        if let Some(cached) = self.cache.lock().await.get(&window_ts).cloned() {
            return Ok(cached);
        }
        let slug = format!("{}-updown-{}m-{}", self.crypto, WINDOW_MINS, window_ts);
        let tokens = Arc::new(self.fetch_by_slug(&slug).await?);
        self.store_in_cache(window_ts, tokens.clone()).await;
        Ok(tokens)
    }

    async fn store_in_cache(&self, key: u64, tokens: Arc<MarketTokens>) {
        let mut cache = self.cache.lock().await;
        let now = current_window_ts();
        cache.retain(|&k, _| k + CACHE_RETENTION_SECS > now);
        cache.insert(key, tokens);
    }

    async fn fetch_tokens(&self, window_ts: u64) -> Result<MarketTokens> {
        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut ts = window_ts;
        loop {
            let slug = format!("{}-updown-{}m-{}", self.crypto, WINDOW_MINS, ts);
            match self.fetch_by_slug(&slug).await {
                Ok(tokens) => return Ok(tokens),
                Err(e) => {
                    let msg = format!("{:#}", e);
                    if msg.contains("No market found") || msg.contains("already closed") {
                        ts += WINDOW_SECS;
                        if ts > now_secs + WINDOW_SECS {
                            bail!(
                                "no open market found for {} after walking forward to ts={}",
                                self.crypto,
                                ts
                            );
                        }
                        warn!(
                            "[MarketResolver] slug={} not found/closed, trying ts={}",
                            slug, ts
                        );
                        continue;
                    }
                    return Err(e);
                }
            }
        }
    }

    async fn fetch_by_slug(&self, slug: &str) -> Result<MarketTokens> {
        let url = format!("{}/markets?slug={}", self.gamma_base, slug);

        let response = self
            .http
            .get(&url)
            .send()
            .await
            .context("Gamma API request failed")?;

        if !response.status().is_success() {
            bail!("Gamma API returned status {}", response.status());
        }

        let markets: Vec<GammaMarket> = response
            .json()
            .await
            .context("Gamma API JSON parse failed")?;

        let market = markets
            .into_iter()
            .find(|m| m.slug == slug)
            .with_context(|| format!("No market found for slug={slug}"))?;

        if market.closed.unwrap_or(false) {
            bail!("Market slug={slug} is already closed");
        }

        let token_ids = market
            .clob_token_ids
            .filter(|v| v.len() >= 2)
            .with_context(|| format!("No clob_token_ids for slug={slug}"))?;

        let min_order_size = market
            .order_min_size
            .with_context(|| format!("No orderMinSize for slug={slug}"))?;

        Ok(MarketTokens {
            window_ts: slug_timestamp(slug),
            slug: slug.to_string(),
            up_token_id: token_ids[0].clone(),
            down_token_id: token_ids[1].clone(),
            min_order_size,
        })
    }
}

/// Deterministic slug for a (crypto, window_ts) pair, e.g. `btc-updown-15m-1776040500`.
pub fn slug_for(crypto: &str, window_ts: u64) -> String {
    format!("{}-updown-{}m-{}", crypto, WINDOW_MINS, window_ts)
}

/// Compute the current window start timestamp (Unix seconds).
pub fn current_window_ts() -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    now - (now % WINDOW_SECS)
}

/// How many seconds remain in the current window.
pub fn secs_until_window_end() -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    WINDOW_SECS - (now % WINDOW_SECS)
}

/// Extract the numeric timestamp suffix from a slug like "btc-updown-15m-1776040500".
pub fn slug_timestamp(slug: &str) -> u64 {
    slug.rsplit('-')
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// Extract everything before the trailing timestamp, e.g.
/// "sol-updown-15m-1776636900" → "sol-updown-15m"
pub fn slug_prefix(slug: &str) -> &str {
    match slug.rfind('-') {
        Some(idx) => &slug[..idx],
        None => slug,
    }
}
