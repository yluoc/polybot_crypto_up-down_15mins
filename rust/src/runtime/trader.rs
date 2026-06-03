
use std::collections::HashMap;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use chrono::Utc;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal::Decimal;
use sqlx::PgPool;
use tracing::{info, warn};

use crate::book_stream::BookStream;
use crate::config::Config;
use crate::db::queries;
use crate::market_resolver::{current_window_ts, secs_until_window_end, MarketResolver};
use crate::order_manager::OrderManager;
use crate::signal::{Action, Symbol, TradingSignal};

/// Dry-run fee assumption (bps) used for EV computation; `fee_rate_bps` stays NULL in dry-run P&L records.
const DRY_RUN_FEE_BPS_ESTIMATE: u32 = 100;

/// Outcome of the spread filter (Gate 4) and EV gate (Gate 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateOutcome {
    Pass,
    SpreadFilter,
    EvGate,
}

/// Spread in basis points of mid: `(ask - bid) / mid * 10_000`. Returns `None` for a degenerate book.
pub fn spread_bps(best_bid: Decimal, best_ask: Decimal) -> Option<f64> {
    let mid = ((best_bid + best_ask) / Decimal::from(2)).to_f64()?;
    if mid <= 0.0 {
        return None;
    }
    let bid = best_bid.to_f64()?;
    let ask = best_ask.to_f64()?;
    Some((ask - bid) / mid * 10_000.0)
}

/// Expected value per share: `EV = p_calibrated - best_ask - fee_frac`.
pub fn ev_per_share(p: f64, best_ask: f64, fee_frac: f64) -> f64 {
    p - best_ask - fee_frac
}

/// Evaluate spread filter then EV gate; returns the first failing gate or `Pass`.
pub fn evaluate_gates(
    spread_bps: f64,
    max_spread_bps: u32,
    ev: f64,
    ev_min_edge: f64,
    ev_gate_enabled: bool,
) -> GateOutcome {
    if spread_bps > max_spread_bps as f64 {
        return GateOutcome::SpreadFilter;
    }
    if ev_gate_enabled && ev < ev_min_edge {
        return GateOutcome::EvGate;
    }
    GateOutcome::Pass
}

/// Insert a dry-run pending skip row. No-op when `signal_id` is `None`.
async fn record_dry_run_skip(pool: &PgPool, signal_id: Option<i64>, reason: &str) {
    if let Some(sid) = signal_id {
        db_try(
            "insert_dry_run_pending",
            queries::insert_dry_run_pending(pool, sid, None, None, None, Some(reason)),
        )
        .await;
    }
}

/// Non-fatal DB write wrapper — logs failures and returns `None` instead of propagating.
async fn db_try<T>(label: &str, fut: impl std::future::Future<Output = Result<T>>) -> Option<T> {
    match fut.await {
        Ok(v) => Some(v),
        Err(e) => {
            warn!("[DB:{}] non-fatal: {:#}", label, e);
            None
        }
    }
}

pub type ResolverMap = Arc<HashMap<Symbol, Arc<MarketResolver>>>;

pub struct Trader {
    resolvers: ResolverMap,
    /// `None` in dry-run mode.
    orders: Option<Arc<OrderManager>>,
    book_stream: Arc<BookStream>,
    config: Arc<Config>,
    pub pool: PgPool,
    pub run_id: i64,

    /// (symbol → window_ts of last placed order). Prevents double-buying.
    last_traded: StdMutex<HashMap<Symbol, u64>>,
}

impl Trader {
    pub fn new(
        resolvers: ResolverMap,
        orders: Option<Arc<OrderManager>>,
        book_stream: Arc<BookStream>,
        config: Arc<Config>,
        pool: PgPool,
        run_id: i64,
    ) -> Arc<Self> {
        Arc::new(Self {
            resolvers,
            orders,
            book_stream,
            config,
            pool,
            run_id,
            last_traded: StdMutex::new(HashMap::new()),
        })
    }

    pub async fn on_signal(&self, sig: TradingSignal) -> Result<()> {
        // Gate 0: symbol is one we're configured to trade
        if !self.config.cryptos.contains(&sig.symbol) {
            warn!("[Trader] signal for unsupported symbol {:?} — skipped", sig.symbol);
            return Ok(());
        }

        // Gate 1: confidence threshold
        if sig.confidence < self.config.min_confidence {
            warn!(
                "[Trader:{}] confidence {:.3} below threshold {:.3} — skipped",
                sig.symbol.as_str(), sig.confidence, self.config.min_confidence
            );
            return Ok(());
        }

        let signal_id = sig.signal_id;

        if self.config.dry_run {
            let window_ts = current_window_ts();
            let resolver = self.resolvers.get(&sig.symbol).ok_or_else(|| {
                anyhow!("no MarketResolver for symbol {:?}", sig.symbol)
            })?;
            let tokens = resolver.current_tokens().await?;

            if let Some(sid) = signal_id {
                db_try(
                    "update_signal_market_id",
                    queries::update_signal_market_id(&self.pool, sid, &tokens.slug),
                )
                .await;
            }

            let asset_id = match sig.action {
                Action::BuyUp   => &tokens.up_token_id,
                Action::BuyDown => &tokens.down_token_id,
            };

            let (best_bid, best_ask) = match self.book_stream.best_bid_ask(asset_id).await {
                Some((bid, ask, book_ts)) => {
                    let now_ms = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as u64;
                    let age_ms = now_ms.saturating_sub(book_ts);
                    if age_ms > self.config.book_staleness_ms {
                        warn!(
                            "[Trader:{}] DRY-RUN book stale for {} ({} ms > {} ms) — \
                             recording pending row with skip_reason=book_stale",
                            sig.symbol.as_str(), asset_id, age_ms, self.config.book_staleness_ms
                        );
                        record_dry_run_skip(&self.pool, signal_id, "book_stale").await;
                        return Ok(());
                    }
                    (bid, ask)
                }
                None => {
                    warn!(
                        "[Trader:{}] DRY-RUN book not yet populated for {} — \
                         recording pending row with skip_reason=book_missing",
                        sig.symbol.as_str(), asset_id
                    );
                    record_dry_run_skip(&self.pool, signal_id, "book_missing").await;
                    return Ok(());
                }
            };

            let Some(spread) = spread_bps(best_bid, best_ask) else {
                info!(
                    "[Trader:{}] DRY-RUN skip reason=\"degenerate_book\" \
                     bid={} ask={}",
                    sig.symbol.as_str(), best_bid, best_ask
                );
                record_dry_run_skip(&self.pool, signal_id, "degenerate_book").await;
                return Ok(());
            };
            let fee_frac = DRY_RUN_FEE_BPS_ESTIMATE as f64 / 1e4;
            let p = sig.confidence as f64; // already calibrated P(chosen class)
            let Some(ask_f) = best_ask.to_f64() else {
                warn!(
                    "[Trader:{}] DRY-RUN skip reason=\"ask_unrepresentable\" ask={}",
                    sig.symbol.as_str(), best_ask
                );
                record_dry_run_skip(&self.pool, signal_id, "ask_unrepresentable").await;
                return Ok(());
            };
            let ev = ev_per_share(p, ask_f, fee_frac);

            match evaluate_gates(
                spread,
                self.config.max_spread_bps,
                ev,
                self.config.ev_min_edge,
                self.config.ev_gate_enabled,
            ) {
                GateOutcome::SpreadFilter => {
                    info!(
                        "[Trader:{}] DRY-RUN skip reason=\"spread_filter\" \
                         spread_bps={:.0} max={} window={}",
                        sig.symbol.as_str(), spread, self.config.max_spread_bps, window_ts
                    );
                    record_dry_run_skip(&self.pool, signal_id, "spread_filter").await;
                    return Ok(());
                }
                GateOutcome::EvGate => {
                    info!(
                        "[Trader:{}] DRY-RUN skip reason=\"ev_gate\" ev={:.4} \
                         p={:.4} ask={:.4} fee_frac={:.4} min_edge={:.4} window={}",
                        sig.symbol.as_str(), ev, p, ask_f, fee_frac,
                        self.config.ev_min_edge, window_ts
                    );
                    record_dry_run_skip(&self.pool, signal_id, "ev_gate").await;
                    return Ok(());
                }
                GateOutcome::Pass => {}
            }

            // fee_rate_bps stays NULL in dry-run pending rows
            let shares_dec = tokens.min_order_size;
            let stake = best_ask * shares_dec;

            let entry_price = Some(ask_f);
            let shares = shares_dec.to_f64();
            if let Some(sid) = signal_id {
                db_try(
                    "insert_dry_run_pending",
                    queries::insert_dry_run_pending(
                        &self.pool, sid, entry_price, shares, None, None,
                    ),
                )
                .await;
            }

            info!(
                "[Trader:{}] DRY-RUN {:?} @ slug={} window={} conf={:.3} \
                 entry_price={:?} shares={:?} ev={:.4} spread={:.0}bps \
                 stake=${:.4} — recorded, no order",
                sig.symbol.as_str(), sig.action, tokens.slug, window_ts,
                sig.confidence, entry_price, shares, ev, spread, stake
            );
            return Ok(());
        }

        // Gate 2: timing
        let remaining = secs_until_window_end();
        let window_ts = current_window_ts();

        if remaining < self.config.entry_offset_secs {
            warn!(
                "[Trader:{}] only {}s left in window — skipped",
                sig.symbol.as_str(), remaining
            );
            return Ok(());
        }

        // Gate 3: one trade per (symbol, window)
        {
            let lt = self
                .last_traded
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if lt.get(&sig.symbol) == Some(&window_ts) {
                info!(
                    "[Trader:{}] already traded in window={} — skipped",
                    sig.symbol.as_str(), window_ts
                );
                return Ok(());
            }
        }

        let resolver = self.resolvers.get(&sig.symbol).ok_or_else(|| {
            anyhow!("no MarketResolver for symbol {:?}", sig.symbol)
        })?;
        let tokens = resolver.current_tokens().await?;

        info!(
            "[Trader:{}] signal={:?} conf={:.3} | window={} slug={}",
            sig.symbol.as_str(), sig.action, sig.confidence, tokens.window_ts, tokens.slug
        );

        let orders = self.orders.as_ref().expect("live mode: orders set");
        let book_stream = &self.book_stream;

        let asset_id = match sig.action {
            Action::BuyUp   => &tokens.up_token_id,
            Action::BuyDown => &tokens.down_token_id,
        };

        let (price, book_ts) = book_stream
            .best_ask(asset_id)
            .await
            .ok_or_else(|| anyhow!("book not yet populated for {}", asset_id))?;

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let age_ms = now_ms.saturating_sub(book_ts);
        if age_ms > self.config.book_staleness_ms {
            warn!(
                "[Trader:{}] book stale for {} ({} ms > {} ms) — skipped",
                sig.symbol.as_str(), asset_id, age_ms, self.config.book_staleness_ms
            );
            return Ok(());
        }

        // Gate 4: spread filter
        let (best_bid, gate_ask, _gate_book_ts) = book_stream
            .best_bid_ask(asset_id)
            .await
            .ok_or_else(|| anyhow!("book not yet populated for {}", asset_id))?;
        let Some(spread) = spread_bps(best_bid, gate_ask) else {
            warn!(
                "[Trader:{}] degenerate book (bid={} ask={}) — skipped",
                sig.symbol.as_str(), best_bid, gate_ask
            );
            return Ok(());
        };

        // Gate 5: EV gate — fee fetch failure skips the trade (EV uncomputable)
        let fee_rate_bps = match orders.fee_rate_bps(asset_id).await {
            Ok(bps) => bps,
            Err(e) => {
                warn!(
                    "[Trader:{}] fee_rate_bps({}) failed — skipping trade (EV uncomputable): {:#}",
                    sig.symbol.as_str(), &asset_id[..16.min(asset_id.len())], e
                );
                return Ok(());
            }
        };
        let fee_frac = fee_rate_bps as f64 / 1e4;
        let p = sig.confidence as f64; // calibrated P(chosen class)
        let Some(ask_f) = gate_ask.to_f64() else {
            warn!("[Trader:{}] best_ask {} not representable as f64 — skipped",
                  sig.symbol.as_str(), gate_ask);
            return Ok(());
        };
        let ev = ev_per_share(p, ask_f, fee_frac);

        match evaluate_gates(
            spread,
            self.config.max_spread_bps,
            ev,
            self.config.ev_min_edge,
            self.config.ev_gate_enabled,
        ) {
            GateOutcome::SpreadFilter => {
                warn!(
                    "[Trader:{}] spread {:.0}bps > {}bps — skipped (spread_filter)",
                    sig.symbol.as_str(), spread, self.config.max_spread_bps
                );
                return Ok(());
            }
            GateOutcome::EvGate => {
                warn!(
                    "[Trader:{}] EV gate: ev={:.4} p={:.4} ask={:.4} fee_frac={:.4} \
                     < min_edge={:.4} — skipped (ev_gate)",
                    sig.symbol.as_str(), ev, p, ask_f, fee_frac, self.config.ev_min_edge
                );
                return Ok(());
            }
            GateOutcome::Pass => {}
        }

        let shares = tokens.min_order_size;
        let stake = price * shares;

        let label = match sig.action {
            Action::BuyUp   => "UP",
            Action::BuyDown => "DOWN",
        };
        info!(
            "[Trader:{}] → BUY {} token {} @ price={} shares={} ev={:.4} spread={:.0}bps \
             stake=${:.4} (book {}ms old)",
            sig.symbol.as_str(), label,
            &asset_id[..16.min(asset_id.len())], price, shares, ev, spread,
            stake, age_ms
        );

        let fee_bps = Some(fee_rate_bps as i32);
        let notional = price * shares;
        let db_order_id = db_try("insert_order", queries::insert_order(
            &self.pool,
            None,                                       // error_id
            signal_id,                                  // FK from insert_signal
            &tokens.slug,                               // market_id
            asset_id,                                   // token_id
            "BUY",                                      // side (always BUY)
            notional.to_f64().unwrap_or(0.0),
            price.to_f64().unwrap_or(0.0),
            fee_bps,
        )).await;

        let result = match sig.action {
            Action::BuyUp   => orders.buy_up(asset_id, price, shares).await,
            Action::BuyDown => orders.buy_down(asset_id, price, shares).await,
        };

        match &result {
            Ok(Some(order_result)) => {
                if let Some(row_id) = db_order_id {
                    db_try("update_order_status", queries::update_order_status(
                        &self.pool,
                        row_id,
                        &order_result.status,
                        Some(order_result.order_id.as_str()),
                        None,
                    )).await;

                    if is_non_terminal(&order_result.status) {
                        spawn_order_status_recheck(
                            Arc::clone(orders),
                            self.pool.clone(),
                            row_id,
                            order_result.order_id.clone(),
                            sig.symbol.as_str().to_string(),
                        );
                    }
                }
                if order_result.status == "MATCHED" {
                    let existing = db_try("get_open_position", queries::get_open_position(
                        &self.pool,
                        &tokens.slug,
                        asset_id,
                    )).await.flatten();
                    if existing.is_none() {
                        db_try("insert_position", queries::insert_position(
                            &self.pool,
                            None,
                            &tokens.slug,
                            asset_id,
                            "BUY",
                            notional.to_f64().unwrap_or(0.0),
                            price.to_f64().unwrap_or(0.0),
                            Utc::now(),
                        )).await;
                    }
                }
                self.last_traded
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(sig.symbol, window_ts);
            }
            Ok(None) => {
                // Notional exceeded max_usdc
                if let Some(row_id) = db_order_id {
                    db_try("update_order_status", queries::update_order_status(
                        &self.pool,
                        row_id,
                        "SKIPPED",
                        None,
                        None,
                    )).await;
                }
            }
            Err(e) => {
                let err_id = db_try("insert_error", queries::insert_error(
                    &self.pool,
                    "rust",
                    "order_submit",
                    &format!("{:#}", e),
                )).await;
                if let Some(row_id) = db_order_id {
                    db_try("update_order_status", queries::update_order_status(
                        &self.pool,
                        row_id,
                        "FAILED",
                        None,
                        err_id,
                    )).await;
                }
            }
        }

        result.map(|_| ())
    }
}

/// Returns true for non-terminal Polymarket order status strings.
fn is_non_terminal(status: &str) -> bool {
    matches!(status, "LIVE" | "DELAYED")
}

/// Poll up to 3 times at 2-second intervals to resolve a non-terminal order. Best-effort — failures leave the row as-is.
fn spawn_order_status_recheck(
    orders: Arc<OrderManager>,
    pool: PgPool,
    row_id: i64,
    order_id: String,
    symbol_label: String,
) {
    tokio::spawn(async move {
        for attempt in 1..=3 {
            tokio::time::sleep(Duration::from_secs(2)).await;
            match orders.check_order(&order_id).await {
                Ok(resp) => {
                    let status = resp.status.to_string();
                    if is_non_terminal(&status) {
                        continue;
                    }
                    info!(
                        "[Trader:{}] re-poll #{}: order {} → {}",
                        symbol_label, attempt, order_id, status
                    );
                    if let Err(e) = queries::update_order_status_only(
                        &pool, row_id, &status,
                    ).await {
                        warn!(
                            "[DB:update_order_status_only] non-fatal: {:#}", e
                        );
                    }
                    return;
                }
                Err(e) => {
                    warn!(
                        "[Trader:{}] re-poll #{}: check_order({}) failed: {:#}",
                        symbol_label, attempt, order_id, e
                    );
                    return;
                }
            }
        }
        warn!(
            "[Trader:{}] re-poll exhausted for order {} — leaving DB as-is",
            symbol_label, order_id
        );
    });
}
