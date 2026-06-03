// Persistent WSS connection to Polymarket's public market channel.
// Maintains an in-memory top-of-book map fed by `book` snapshots and `price_change` deltas.

use anyhow::{anyhow, Result};
use futures::{SinkExt, StreamExt};
use rust_decimal::Decimal;
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio_tungstenite::{
    connect_async,
    tungstenite::{self, Message},
};
use tracing::{debug, info, warn};

/// Polymarket's WS endpoint drops connections without a closing handshake on a
/// regular cadence; treat that family of disconnects as informational rather
/// than as a warning so real errors stay visible.
fn is_benign_disconnect(err: &anyhow::Error) -> bool {
    for cause in err.chain() {
        if let Some(te) = cause.downcast_ref::<tungstenite::Error>() {
            return match te {
                tungstenite::Error::ConnectionClosed | tungstenite::Error::AlreadyClosed => true,
                tungstenite::Error::Protocol(
                    tungstenite::error::ProtocolError::ResetWithoutClosingHandshake,
                ) => true,
                tungstenite::Error::Io(io) => matches!(
                    io.kind(),
                    std::io::ErrorKind::ConnectionReset
                        | std::io::ErrorKind::UnexpectedEof
                        | std::io::ErrorKind::BrokenPipe
                ),
                _ => false,
            };
        }
        if let Some(io) = cause.downcast_ref::<std::io::Error>() {
            if matches!(
                io.kind(),
                std::io::ErrorKind::ConnectionReset
                    | std::io::ErrorKind::UnexpectedEof
                    | std::io::ErrorKind::BrokenPipe
            ) {
                return true;
            }
        }
    }
    false
}

#[derive(Debug, Clone)]
pub struct BookState {
    pub best_bid: Decimal,
    pub best_ask: Decimal,
    pub last_update_ms: u64,
}

type StateMap = Arc<RwLock<HashMap<String, BookState>>>;
type SubSet = Arc<Mutex<HashSet<String>>>;

enum Cmd {
    Subscribe(Vec<String>),
    Unsubscribe(Vec<String>),
}

pub struct BookStream {
    state: StateMap,
    subscribed: SubSet,
    cmd_tx: mpsc::Sender<Cmd>,
}

impl BookStream {
    /// Spawn the driver task and return a shared handle.
    pub fn spawn(url: String) -> Arc<Self> {
        let state: StateMap = Arc::new(RwLock::new(HashMap::new()));
        let subscribed: SubSet = Arc::new(Mutex::new(HashSet::new()));
        let (cmd_tx, cmd_rx) = mpsc::channel::<Cmd>(64);

        let handle = Arc::new(Self {
            state: state.clone(),
            subscribed: subscribed.clone(),
            cmd_tx,
        });

        tokio::spawn(driver_loop(url, state, subscribed, cmd_rx));
        handle
    }

    /// Diff the desired set against the current subscriptions and send subscribe/unsubscribe frames.
    pub async fn ensure_subscribed(&self, ids: &[String]) {
        let desired_set: HashSet<&str> = ids.iter().map(String::as_str).collect();
        let mut current = self.subscribed.lock().await;

        let to_add: Vec<String> = ids
            .iter()
            .filter(|id| !current.contains(id.as_str()))
            .cloned()
            .collect();
        let to_remove: Vec<String> = current
            .iter()
            .filter(|id| !desired_set.contains(id.as_str()))
            .cloned()
            .collect();

        if to_add.is_empty() && to_remove.is_empty() {
            return;
        }

        for id in &to_add {
            current.insert(id.clone());
        }
        for id in &to_remove {
            current.remove(id);
        }
        drop(current);

        if !to_add.is_empty() {
            let _ = self.cmd_tx.send(Cmd::Subscribe(to_add)).await;
        }
        if !to_remove.is_empty() {
            let _ = self.cmd_tx.send(Cmd::Unsubscribe(to_remove)).await;
        }
    }

    /// Returns `(best_ask, last_update_ms)` for an asset, or `None` if not yet populated.
    pub async fn best_ask(&self, asset_id: &str) -> Option<(Decimal, u64)> {
        self.state
            .read()
            .await
            .get(asset_id)
            .map(|b| (b.best_ask, b.last_update_ms))
    }

    /// Returns `(best_bid, best_ask, last_update_ms)` for an asset, or `None` if not yet populated.
    pub async fn best_bid_ask(&self, asset_id: &str) -> Option<(Decimal, Decimal, u64)> {
        self.state
            .read()
            .await
            .get(asset_id)
            .map(|b| (b.best_bid, b.best_ask, b.last_update_ms))
    }
}

async fn driver_loop(url: String, state: StateMap, subscribed: SubSet, mut cmd_rx: mpsc::Receiver<Cmd>) {
    let mut backoff_ms: u64 = 250;
    loop {
        match run_session(&url, &state, &subscribed, &mut cmd_rx).await {
            Ok(()) => {
                info!("[BookStream] session ended; reconnecting");
                backoff_ms = 250;
            }
            Err(e) => {
                if is_benign_disconnect(&e) {
                    info!("[BookStream] disconnected ({}); reconnecting in {}ms", e, backoff_ms);
                } else {
                    warn!("[BookStream] session error: {:#}; reconnecting in {}ms", e, backoff_ms);
                }
                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms * 2).min(5000);
            }
        }
    }
}

async fn run_session(
    url: &str,
    state: &StateMap,
    subscribed: &SubSet,
    cmd_rx: &mut mpsc::Receiver<Cmd>,
) -> Result<()> {
    let (ws, _resp) = connect_async(url).await?;
    let (mut write, mut read) = ws.split();
    info!("[BookStream] connected to {}", url);

    let initial: Vec<String> = subscribed.lock().await.iter().cloned().collect();
    if !initial.is_empty() {
        let frame = json!({
            "assets_ids": initial,
            "type": "market",
            "level": 1,
            "initial_dump": true,
        })
        .to_string();
        write.send(Message::Text(frame)).await?;
        info!("[BookStream] sent initial subscribe for {} asset(s)", initial.len());
    }

    let mut ping_tick = tokio::time::interval(Duration::from_secs(10));
    ping_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    ping_tick.tick().await; // consume the immediate tick

    loop {
        tokio::select! {
            _ = ping_tick.tick() => {
                write.send(Message::Text("PING".to_string())).await?;
            }

            Some(cmd) = cmd_rx.recv() => {
                match cmd {
                    Cmd::Subscribe(ids) => {
                        let frame = json!({
                            "operation": "subscribe",
                            "assets_ids": ids,
                            "level": 1,
                        })
                        .to_string();
                        write.send(Message::Text(frame)).await?;
                        info!("[BookStream] subscribed: {:?}", ids);
                    }
                    Cmd::Unsubscribe(ids) => {
                        let frame = json!({
                            "operation": "unsubscribe",
                            "assets_ids": ids,
                        })
                        .to_string();
                        write.send(Message::Text(frame)).await?;
                        let still_subscribed = subscribed.lock().await;
                        let mut s = state.write().await;
                        for id in &ids {
                            if !still_subscribed.contains(id) {
                                s.remove(id);
                            }
                        }
                        info!("[BookStream] unsubscribed: {:?}", ids);
                    }
                }
            }

            frame = read.next() => {
                let frame = frame.ok_or_else(|| anyhow!("WSS stream closed"))??;
                match frame {
                    Message::Text(text) => handle_text(text.as_ref(), state).await,
                    Message::Binary(_) => {},
                    Message::Ping(data) => { write.send(Message::Pong(data)).await?; }
                    Message::Pong(_) => {},
                    Message::Close(_) => return Err(anyhow!("server closed connection")),
                    Message::Frame(_) => {},
                }
            }
        }
    }
}

async fn handle_text(text: &str, state: &StateMap) {
    if text == "PONG" {
        return;
    }

    let val: serde_json::Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(e) => {
            warn!("[BookStream] parse error: {} raw={}", e, text);
            return;
        }
    };

    let events: Vec<serde_json::Value> = match val {
        serde_json::Value::Array(a) => a,
        v => vec![v],
    };

    for ev in events {
        handle_event(ev, state).await;
    }
}

async fn handle_event(ev: serde_json::Value, state: &StateMap) {
    let event_type = ev.get("event_type").and_then(|v| v.as_str()).unwrap_or("");
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    match event_type {
        "book" => {
            let Some(asset_id) = ev.get("asset_id").and_then(|v| v.as_str()).map(str::to_owned) else {
                return;
            };
            let bids = ev.get("bids").and_then(|v| v.as_array());
            let asks = ev.get("asks").and_then(|v| v.as_array());
            let best_bid = bids.and_then(|a| top_price(a, true));
            let best_ask = asks.and_then(|a| top_price(a, false));
            if let (Some(bb), Some(ba)) = (best_bid, best_ask) {
                state.write().await.insert(
                    asset_id.clone(),
                    BookState { best_bid: bb, best_ask: ba, last_update_ms: now_ms },
                );
                debug!("[BookStream] book {} bid={} ask={}", short(&asset_id), bb, ba);
            }
        }

        "price_change" => {
            let Some(changes) = ev.get("price_changes").and_then(|v| v.as_array()) else { return; };
            let mut s = state.write().await;
            for c in changes {
                let Some(asset_id) = c.get("asset_id").and_then(|v| v.as_str()).map(str::to_owned) else { continue; };
                let bb = c.get("best_bid").and_then(|v| v.as_str()).and_then(|s| Decimal::from_str(s).ok());
                let ba = c.get("best_ask").and_then(|v| v.as_str()).and_then(|s| Decimal::from_str(s).ok());
                if let (Some(bb), Some(ba)) = (bb, ba) {
                    s.insert(asset_id, BookState { best_bid: bb, best_ask: ba, last_update_ms: now_ms });
                }
            }
        }

        "tick_size_change" | "last_trade_price" => {}
        other if !other.is_empty() => {
            debug!("[BookStream] unhandled event_type: {}", other);
        }
        _ => {}
    }
}

/// Best price across level entries. Bids: max. Asks: min.
fn top_price(levels: &[serde_json::Value], is_bid: bool) -> Option<Decimal> {
    let mut best: Option<Decimal> = None;
    for level in levels {
        let Some(p) = level.get("price").and_then(|v| v.as_str()).and_then(|s| Decimal::from_str(s).ok()) else { continue; };
        match best {
            None => best = Some(p),
            Some(cur) => {
                if is_bid && p > cur { best = Some(p); }
                if !is_bid && p < cur { best = Some(p); }
            }
        }
    }
    best
}

fn short(s: &str) -> &str {
    &s[..16.min(s.len())]
}
