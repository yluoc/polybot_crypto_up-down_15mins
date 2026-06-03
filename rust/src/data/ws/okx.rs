// OKX public v5 WebSocket client. Subscribes to mark-price for all
// instruments in INSTRUMENT_ORDER; reconnects with exponential backoff.
// Keepalive: sends "ping" every 25s (OKX closes idle connections after 30s).
// Protocol reference: https://www.okx.com/docs-v5/en/#websocket-api

use std::time::Duration;

use anyhow::{Context, Result};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio::time::{self, Instant};
use tokio_tungstenite::{
    connect_async,
    tungstenite::protocol::Message,
};
use tracing::{debug, error, info, warn};

use crate::feature_engine::{RawTick, INSTRUMENT_ORDER};

/// Reconnect backoff bounds; capped to avoid silently skipping a 15-min bucket.
const BACKOFF_INITIAL: Duration = Duration::from_secs(1);
const BACKOFF_MAX:     Duration = Duration::from_secs(30);
const PING_INTERVAL:   Duration = Duration::from_secs(25);

#[derive(Serialize)]
struct SubscribeArg<'a> {
    channel: &'a str,
    #[serde(rename = "instId")]
    inst_id: &'a str,
}

#[derive(Serialize)]
struct SubscribeReq<'a> {
    op: &'a str,
    args: Vec<SubscribeArg<'a>>,
}

#[derive(Deserialize)]
struct MarkPxMsg {
    #[serde(default)]
    data: Vec<MarkPxDatum>,
}

#[derive(Deserialize)]
struct MarkPxDatum {
    #[serde(rename = "instId")]
    inst_id: String,
    #[serde(rename = "markPx")]
    mark_px: String,
    ts: String,
}

/// Spawn the OKX WS loop; runs until the receiver is dropped.
pub fn spawn(url: String, tx: mpsc::UnboundedSender<RawTick>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut backoff = BACKOFF_INITIAL;
        loop {
            match run_once(&url, &tx).await {
                Ok(()) => {
                    // Normal close (receiver dropped) — exit.
                    info!("[ws/okx] connection closed cleanly, stopping");
                    return;
                }
                Err(e) => {
                    error!("[ws/okx] session failed: {e:#}");
                    if tx.is_closed() {
                        return;
                    }
                }
            }
            warn!("[ws/okx] reconnecting in {:?}", backoff);
            time::sleep(backoff).await;
            backoff = (backoff * 2).min(BACKOFF_MAX);
        }
    })
}

async fn run_once(url: &str, tx: &mpsc::UnboundedSender<RawTick>) -> Result<()> {
    info!("[ws/okx] connecting to {url}");
    let (mut socket, _resp) = connect_async(url)
        .await
        .with_context(|| format!("connect_async {url}"))?;

    let args: Vec<SubscribeArg> = INSTRUMENT_ORDER
        .iter()
        .map(|id| SubscribeArg { channel: "mark-price", inst_id: id })
        .collect();
    let req = SubscribeReq { op: "subscribe", args };
    let req_json = serde_json::to_string(&req)?;
    socket.send(Message::Text(req_json)).await?;
    info!("[ws/okx] subscribed to mark-price for {} instruments", INSTRUMENT_ORDER.len());

    let mut ping_timer = time::interval_at(Instant::now() + PING_INTERVAL, PING_INTERVAL);

    loop {
        tokio::select! {
            _ = ping_timer.tick() => {
                socket.send(Message::Text("ping".into())).await
                    .context("ws send ping failed")?;
            }
            msg = socket.next() => {
                let Some(msg) = msg else {
                    anyhow::bail!("ws stream ended");
                };
                let msg = msg.context("ws recv error")?;
                match msg {
                    Message::Text(txt) => {
                        let s: &str = txt.as_str();
                        if s == "pong" { continue; }
                        if let Err(e) = dispatch_text(s, tx) {
                            debug!("[ws/okx] skipped frame: {e} | raw={s}");
                        }
                    }
                    Message::Ping(payload) => {
                        socket.send(Message::Pong(payload)).await.ok();
                    }
                    Message::Close(frame) => {
                        anyhow::bail!("ws closed by server: {frame:?}");
                    }
                    _ => {}  // binary / pong / frame — ignore
                }
                if tx.is_closed() {
                    return Ok(());
                }
            }
        }
    }
}

fn dispatch_text(txt: &str, tx: &mpsc::UnboundedSender<RawTick>) -> Result<()> {
    let parsed: MarkPxMsg = serde_json::from_str(txt)
        .context("bad mark-price frame")?;
    for d in parsed.data {
        let ts_ms: i64 = d.ts.parse().context("parse ts")?;
        let mark_px: f64 = d.mark_px.parse().context("parse markPx")?;
        let tick = RawTick { inst_id: d.inst_id, mark_px, ts_ms };
        if tx.send(tick).is_err() {
            return Err(anyhow::anyhow!("tick receiver dropped"));
        }
    }
    Ok(())
}
