
use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use sqlx::PgPool;
use tokio::sync::{mpsc, watch, RwLock};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::book_stream::BookStream;
use crate::bootstrap;
use crate::config::Config;
use crate::db;
use crate::db::queries;
use crate::feature_engine::{
    candle_builder, short_symbol, Candle, FeatureEngineer, Normalizer, RawTick, TickBuffer,
    FEATURE_DIM, GLOBAL_FEATURE_DIM, INSTRUMENT_ORDER, TOTAL_FEATURES,
};
use crate::feature_engine::feature_engineer::USE_PER_SYMBOL_ALIGNMENT;
use crate::inference::model_hub::CURRENT_FORMAT_VERSION;
use crate::inference::{predict_one, ModelHub};
use crate::macro_poll::{self};
use crate::feature_engine::MacroSnapshot;
use crate::market_resolver::{
    current_window_ts, secs_until_window_end, MarketResolver, WINDOW_SECS,
};
use crate::order_manager::OrderManager;
use crate::perp_poll::{self, PerpSnapshot};
use crate::preflight;
use crate::resolution;
use crate::signal::{Symbol, TradingSignal};
use crate::trader::{ResolverMap, Trader};
use crate::warmup;
use crate::ws::okx;

/// Symbols whose signals are blocked from order dispatch.
const DISABLED_SYMBOLS: &[Symbol] = &[Symbol::Xrp];

/// Per-prediction feature-importance record shipped to the importance worker.
struct ImportanceRecord {
    signal_id:        i64,
    model_version_id: i64,
    symbol:           &'static str,
    features:         [f32; TOTAL_FEATURES],
    importances:      Vec<f64>,
}

pub async fn run(cfg: Arc<Config>) -> Result<()> {
    info!(
        "[startup] feature layout: format_version={} FEATURE_DIM={} GLOBAL_FEATURE_DIM={} TOTAL_FEATURES={}",
        CURRENT_FORMAT_VERSION, FEATURE_DIM, GLOBAL_FEATURE_DIM, TOTAL_FEATURES
    );
    info!(
        "[run] config loaded: okx_ws={} cryptos={:?}",
        cfg.okx_ws_url,
        cfg.cryptos.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
    );

    let cancel_token = CancellationToken::new();
    spawn_shutdown_listener(cancel_token.clone());

    let db_pool: PgPool = db::init_pool(&cfg.database_url).await?;
    info!("[run] DB pool ready, migrations applied");

    bootstrap::bootstrap_if_needed(&db_pool, &cfg.cryptos, &cfg).await?;

    let model_map = db::queries::load_current_models(&db_pool).await?;
    if model_map.is_empty() {
        warn!(
            "[run] no current models in DB — predictions will be skipped for every symbol \
             until `polybot retrain` promotes at least one"
        );
    }
    let model_hub = Arc::new(RwLock::new(ModelHub::from_bytes_map(model_map)?));

    spawn_model_updated_listener(db_pool.clone(), Arc::clone(&model_hub));

    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()?;
    let resolvers: ResolverMap = Arc::new(
        cfg.cryptos
            .iter()
            .map(|s| {
                (
                    *s,
                    Arc::new(MarketResolver::new(
                        http.clone(),
                        &cfg.gamma_api_url,
                        s.as_str(),
                    )),
                )
            })
            .collect(),
    );

    if cfg.dry_run {
        info!("[run] DRY-RUN mode — no orders will be placed at startup");
    }
    let book_stream: Arc<BookStream> = BookStream::spawn(cfg.wss_url.clone());
    info!("[run] BookStream driver spawned → {}", cfg.wss_url);
    spawn_subscription_task(book_stream.clone(), Arc::clone(&resolvers), Arc::clone(&cfg));

    resolution::spawn(db_pool.clone(), http.clone(), cfg.gamma_api_url.clone());
    info!("[run] resolution checker spawned (60s interval)");

    if cfg.dry_run {
        resolution::spawn_dry_run(db_pool.clone(), http.clone(), cfg.gamma_api_url.clone());
        info!("[run] dry-run outcome checker spawned (60s interval)");
    }

    // dry-run skips OrderManager entirely
    let orders: Option<Arc<OrderManager>> = if cfg.dry_run {
        None
    } else {
        let o = Arc::new(
            OrderManager::new(
                &cfg.polymarket_api_url,
                &cfg.private_key,
                cfg.funder,
                cfg.order_max_usdc,
            )
            .await?,
        );

        preflight::run(&o, &cfg, &db_pool).await?;

        // Reap open orders left behind by a previous crash (best-effort).
        match o.cancel_all_orders().await {
            Ok((canceled, not_canceled)) => {
                if canceled.is_empty() && not_canceled == 0 {
                    info!("[run] startup cancel-all: nothing to reap");
                } else {
                    info!(
                        "[run] startup cancel-all: canceled={} not_canceled={}",
                        canceled.len(), not_canceled
                    );
                }
            }
            Err(e) => {
                warn!("[run] startup cancel-all failed (non-fatal): {:#}", e);
            }
        }
        Some(o)
    };

    let run_id = db::queries::insert_bot_run(
        &db_pool,
        None,
        None,                            // model_path (DB-backed now)
        Some(cfg.min_confidence as f64), // conf_threshold
        Some(&cfg.okx_ws_url),           // data_source
        None,
        None,
    )
    .await?;
    info!("[run] bot_run #{} started", run_id);

    let trader = Trader::new(
        resolvers,
        orders.clone(),
        Arc::clone(&book_stream),
        Arc::clone(&cfg),
        db_pool.clone(),
        run_id,
    );

    let mut engineer = FeatureEngineer::new();
    let mut normalizer = Normalizer::new();
    warmup::warm_up_pipeline(&db_pool, &mut engineer, &mut normalizer, None).await?;

    let perp_rx = perp_poll::spawn();
    let macro_rx = macro_poll::spawn(db_pool.clone(), cfg.fred_api_key.clone());

    // Unbounded so the inference path never back-pressures on DB latency.
    let (imp_tx, imp_rx) = mpsc::unbounded_channel::<ImportanceRecord>();
    spawn_importance_worker(db_pool.clone(), imp_rx);

    let (sig_tx, mut sig_rx) = mpsc::channel::<TradingSignal>(128);
    spawn_live_pipeline(
        Arc::clone(&cfg),
        Arc::clone(&model_hub),
        engineer,
        normalizer,
        perp_rx,
        macro_rx,
        sig_tx,
        imp_tx,
        db_pool.clone(),
        cancel_token.clone(),
    );

    loop {
        tokio::select! {
            biased;
            _ = cancel_token.cancelled() => {
                info!("[run] shutdown signal received — draining");
                break;
            }
            maybe_sig = sig_rx.recv() => {
                let Some(sig) = maybe_sig else {
                    warn!("[run] signal channel closed — exiting main loop");
                    break;
                };
                info!(
                    "[run] signal: sym={} action={:?} conf={:.3} ts={}",
                    sig.symbol.as_str(), sig.action, sig.confidence, sig.candle_ts_ms
                );
                let t = trader.clone();
                tokio::spawn(async move {
                    if let Err(e) = t.on_signal(sig).await {
                        error!("[run:{}] trader error: {:#}", sig.symbol.as_str(), e);
                        let _ = db::queries::insert_error(
                            &t.pool,
                            "rust",
                            "on_signal",
                            &format!("[{}] {:#}", sig.symbol.as_str(), e),
                        )
                        .await;
                    }
                });
            }
        }
    }

    if let Err(e) = db::queries::stop_bot_run(&db_pool, run_id).await {
        warn!("[run] stop_bot_run failed during shutdown: {e:#}");
    }
    info!("[run] bot_run #{} stopped cleanly", run_id);
    Ok(())
}

/// Flips `cancel` on Ctrl-C or SIGTERM.
fn spawn_shutdown_listener(cancel: CancellationToken) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            let mut sigterm = match signal(SignalKind::terminate()) {
                Ok(s) => s,
                Err(e) => {
                    error!("[run] failed to install SIGTERM handler: {e:#}");
                    cancel.cancel();
                    return;
                }
            };
            tokio::select! {
                res = tokio::signal::ctrl_c() => {
                    if let Err(e) = res {
                        error!("[run] ctrl_c install failed: {e:#}");
                    } else {
                        info!("[run] received SIGINT");
                    }
                }
                _ = sigterm.recv() => {
                    info!("[run] received SIGTERM");
                }
            }
        }
        #[cfg(not(unix))]
        {
            if let Err(e) = tokio::signal::ctrl_c().await {
                error!("[run] ctrl_c install failed: {e:#}");
            } else {
                info!("[run] received SIGINT");
            }
        }
        cancel.cancel();
    })
}

fn spawn_model_updated_listener(
    pool: PgPool,
    model_hub: Arc<RwLock<ModelHub>>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut listener = match db::notify::ModelUpdateListener::connect(&pool).await {
            Ok(l) => l,
            Err(e) => {
                error!("[run] could not start model_updated listener: {e:#}");
                return;
            }
        };
        info!("[run] LISTEN model_updated started");
        loop {
            let symbol = match listener.recv().await {
                Ok(s) => s,
                Err(e) => {
                    error!("[run] model_updated recv failed: {e:#}");
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    continue;
                }
            };
            let sym = match Symbol::from_str_ci(&symbol) {
                Ok(s) => s,
                Err(e) => {
                    warn!("[run] NOTIFY payload '{symbol}' unparseable: {e}");
                    continue;
                }
            };
            let (bytes, fmt, version_id, family) =
                match db::queries::get_current_model_bytes(&pool, &symbol).await {
                    Ok(Some(b)) => b,
                    Ok(None) => {
                        warn!("[run] NOTIFY {symbol} but no current model row found");
                        continue;
                    }
                    Err(e) => {
                        error!("[run] fetching current model for {symbol}: {e:#}");
                        continue;
                    }
                };
            let mut hub = model_hub.write().await;
            if let Err(e) = hub.reload(sym, &bytes, fmt, version_id, &family).await {
                error!("[run] ModelHub::reload({symbol}): {e:#}");
            }
        }
    })
}

/// Drain importance records and batch-insert into `model_importance`. Best-effort — failures are logged and dropped.
fn spawn_importance_worker(
    pool: PgPool,
    mut rx: mpsc::UnboundedReceiver<ImportanceRecord>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        while let Some(rec) = rx.recv().await {
            if let Err(e) = queries::insert_model_importance_batch(
                &pool,
                rec.signal_id,
                rec.model_version_id,
                rec.symbol,
                &rec.features,
                &rec.importances,
            )
            .await
            {
                warn!(
                    "[run/importance] insert_model_importance_batch(sig={}, sym={}) non-fatal: {:#}",
                    rec.signal_id, rec.symbol, e
                );
            }
        }
        info!("[run/importance] channel closed — worker exiting");
    })
}

#[allow(clippy::too_many_arguments)]
fn spawn_live_pipeline(
    cfg: Arc<Config>,
    model_hub: Arc<RwLock<ModelHub>>,
    engineer: FeatureEngineer,
    normalizer: Normalizer,
    perp_rx: watch::Receiver<PerpSnapshot>,
    macro_rx: watch::Receiver<MacroSnapshot>,
    sig_tx: mpsc::Sender<TradingSignal>,
    imp_tx: mpsc::UnboundedSender<ImportanceRecord>,
    pool: PgPool,
    shutdown: CancellationToken,
) {
    let (tick_tx, mut tick_rx) = mpsc::unbounded_channel::<RawTick>();
    okx::spawn(cfg.okx_ws_url.clone(), tick_tx);

    let mut buffers: HashMap<&'static str, Arc<Mutex<TickBuffer>>> =
        HashMap::with_capacity(INSTRUMENT_ORDER.len());
    for inst in INSTRUMENT_ORDER.iter() {
        buffers.insert(*inst, Arc::new(Mutex::new(TickBuffer::new(*inst))));
    }

    let (candle_tx, candle_rx) = mpsc::unbounded_channel::<Candle>();
    let builder_cancel = Arc::new(AtomicBool::new(false));
    {
        let shutdown = shutdown.clone();
        let builder_cancel = Arc::clone(&builder_cancel);
        tokio::spawn(async move {
            shutdown.cancelled().await;
            builder_cancel.store(true, std::sync::atomic::Ordering::Relaxed);
        });
    }
    for inst in INSTRUMENT_ORDER.iter() {
        let buf = Arc::clone(buffers.get(inst).expect("buffer for inst"));
        let tx = candle_tx.clone();
        let c = Arc::clone(&builder_cancel);
        candle_builder::spawn(buf, tx, c);
    }
    drop(candle_tx); // only the builders hold it now

    let buffers_for_router = buffers.clone();
    let router_shutdown = shutdown.clone();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                biased;
                _ = router_shutdown.cancelled() => {
                    info!("[run/router] shutdown — exiting");
                    return;
                }
                maybe_tick = tick_rx.recv() => {
                    let Some(tick) = maybe_tick else {
                        info!("[run/router] tick stream ended");
                        return;
                    };
                    let Some(buf) = buffers_for_router.get(tick.inst_id.as_str()) else {
                        debug!("[run/router] unknown inst_id {}", tick.inst_id);
                        continue;
                    };
                    let mut guard = match buf.lock() {
                        Ok(g) => g,
                        Err(p) => p.into_inner(),
                    };
                    guard.push(tick);
                }
            }
        }
    });

    tokio::spawn(feature_inference_loop(
        cfg, model_hub, engineer, normalizer, perp_rx, macro_rx, candle_rx, sig_tx, imp_tx, pool,
        shutdown,
    ));
}

#[allow(clippy::too_many_arguments)]
async fn feature_inference_loop(
    cfg: Arc<Config>,
    model_hub: Arc<RwLock<ModelHub>>,
    mut engineer: FeatureEngineer,
    mut normalizer: Normalizer,
    perp_rx: watch::Receiver<PerpSnapshot>,
    macro_rx: watch::Receiver<MacroSnapshot>,
    mut candle_rx: mpsc::UnboundedReceiver<Candle>,
    sig_tx: mpsc::Sender<TradingSignal>,
    imp_tx: mpsc::UnboundedSender<ImportanceRecord>,
    pool: PgPool,
    shutdown: CancellationToken,
) {
    loop {
        let candle = tokio::select! {
            biased;
            _ = shutdown.cancelled() => {
                info!("[run/infer] shutdown — exiting");
                return;
            }
            maybe = candle_rx.recv() => {
                let Some(c) = maybe else {
                    info!("[run/infer] candle stream ended");
                    return;
                };
                c
            }
        };
        let ts_ms = candle.open_ts_ms;

        {
            let snap = perp_rx.borrow();
            if let Some(sample) = snap.get(&candle.inst_id) {
                engineer.push_perp_sample(&candle.inst_id, ts_ms, *sample);
            } else {
                debug!(
                    "[run/infer] no perp snapshot for {} yet — row will block until poller catches up",
                    candle.inst_id
                );
            }
        }

        {
            let snap = macro_rx.borrow();
            if snap.is_empty() {
                debug!("[run/infer] no macro snapshot yet — row will block until poller catches up");
            } else {
                engineer.push_macro_snapshot(&snap);
            }
        }

        let arrived_sym: Option<Symbol> =
            Symbol::from_str_ci(short_symbol(&candle.inst_id)).ok();

        let Some(raw) = engineer.push_candle(candle) else {
            continue;
        };
        let Some(normed) = normalizer.push(&raw) else {
            continue;
        };

        let predict_syms: Vec<Symbol> = if USE_PER_SYMBOL_ALIGNMENT {
            match arrived_sym {
                Some(s) => vec![s],
                None => continue,
            }
        } else {
            cfg.cryptos.iter().copied().collect()
        };

        let hub = model_hub.read().await;
        for sym in predict_syms.into_iter() {
            if !cfg.cryptos.contains(&sym) {
                continue;
            }
            if DISABLED_SYMBOLS.contains(&sym) {
                debug!("[run/infer] {} is in DISABLED_SYMBOLS — skipping order path", sym.as_str());
                continue;
            }
            let Some(slot) = hub.get(sym) else {
                debug!("[run/infer] no model for {} — skipping", sym.as_str());
                continue;
            };
            let booster = slot.read().await;
            let (probs, calibration) = match booster.predict_up_down(&normed.features) {
                Ok(pc) => pc,
                Err(e) => {
                    error!("[run/infer:{}] predict failed: {e:#}", sym.as_str());
                    continue;
                }
            };
            let pred = predict_one(&probs, calibration);
            if pred.confidence < cfg.min_confidence {
                debug!(
                    "[run/infer:{}] gated: action={:?} conf={:.3} < {:.3}",
                    sym.as_str(),
                    pred.action,
                    pred.confidence,
                    cfg.min_confidence
                );
                continue;
            }

            let signal_id = match queries::insert_signal(
                &pool,
                None,
                ts_ms,
                pred.action.as_i16(),
                pred.confidence as f64,
                None,
            )
            .await
            {
                Ok(id) => Some(id),
                Err(e) => {
                    warn!(
                        "[run/infer:{}] insert_signal failed (non-fatal): {:#}",
                        sym.as_str(), e
                    );
                    None
                }
            };

            if let Some(sid) = signal_id {
                let rec = ImportanceRecord {
                    signal_id:        sid,
                    model_version_id: booster.model_version_id(),
                    symbol:           sym.short(),
                    features:         normed.features,
                    importances:      booster.feature_importances().to_vec(),
                };
                if imp_tx.send(rec).is_err() {
                    warn!("[run/infer] importance channel closed — recording disabled");
                }
            }

            let sig = TradingSignal {
                candle_ts_ms: ts_ms,
                action: pred.action,
                symbol: sym,
                confidence: pred.confidence,
                raw_score: pred.raw_score,
                signal_id,
            };
            if sig_tx.send(sig).await.is_err() {
                warn!("[run/infer] signal channel closed, stopping inference loop");
                return;
            }
        }
    }
}

fn spawn_subscription_task(book_stream: Arc<BookStream>, resolvers: ResolverMap, cfg: Arc<Config>) {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_secs(10));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            let current_ts = current_window_ts();
            let next_ts = current_ts + WINDOW_SECS;
            let include_next = secs_until_window_end() < cfg.prewarm_lead_secs;

            let mut ids: Vec<String> = Vec::with_capacity(resolvers.len() * 4);
            for (sym, resolver) in resolvers.iter() {
                match resolver.tokens_for_window(current_ts).await {
                    Ok(t) => {
                        ids.push(t.up_token_id.clone());
                        ids.push(t.down_token_id.clone());
                    }
                    Err(e) => warn!(
                        "[subscribe_task:{}] current window {} unresolvable: {}",
                        sym.as_str(),
                        current_ts,
                        e
                    ),
                }
                if include_next {
                    if let Ok(t) = resolver.tokens_for_window(next_ts).await {
                        ids.push(t.up_token_id.clone());
                        ids.push(t.down_token_id.clone());
                    } else {
                        debug!(
                            "[subscribe_task:{}] next window {} not yet indexed",
                            sym.as_str(),
                            next_ts
                        );
                    }
                }
            }
            if !ids.is_empty() {
                book_stream.ensure_subscribed(&ids).await;
            }
        }
    });
}
