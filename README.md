# polybot — project overview

A single Rust binary that trades 15-minute UP/DOWN crypto prediction markets on Polymarket, using machine learning models trained on OKX exchange candle data.

---

## What it does, in one paragraph

Every 15 minutes, Polymarket lists a market like *"Will BTC be higher 15 minutes from now?"*. polybot watches BTC (and 3 other cryptos) live on OKX, runs the incoming price data through a trained model, and if the model is confident enough about the direction, it places a small order on the corresponding Polymarket market before the window closes.

---

## The 4 cryptos it trades

BTC, ETH, SOL, XRP. (DOGE, HYPE, and BNB were dropped due to insufficient Polymarket label history.)

Each one has its own model — either a LightGBM booster or an L1-regularized logistic regression (LR-L1) baseline. Every retrain trains *both* families per symbol, runs both through the same promotion gates, and promotes the per-symbol winner; the model family is recorded on the `model_versions` row. The models share the same 89-feature input row (21 per-instrument features × 4 instruments + 5 global macro features from FRED, event-gated around US macro releases), but each model predicts only *its own* symbol's direction.

---

## The binary has 8 subcommands

| Command | What it does |
|---|---|
| `polybot run` | The live trader. Connects to OKX, builds candles, runs models, places orders. Runs forever. |
| `polybot backfill` | Downloads historical candles from OKX into the `candles` table. |
| `polybot backfill-outcomes` | Walks Polymarket markets to record how each past 15-min window actually resolved (UP or DOWN). These become training labels. |
| `polybot backfill-macro` | Pulls daily macro series (DXY, SPX, VIX, yields) from FRED into the `macro_daily` table. |
| `polybot backfill-coinalyze` | Pulls aggregated open-interest + liquidation history from Coinalyze into `open_interest_aggregated` / `liquidations_aggregated` (needs `COINALYZE_API_KEY`). |
| `polybot backfill-taker` | Pulls OKX taker buy/sell volume into `taker_volume_15m` (public endpoint, no key). |
| `polybot retrain` | Trains a fresh model per symbol on the last 180 days of candles + outcomes. Promotes new models into the DB. |
| `polybot cron` | Runs the data backfills then retrain in sequence (backfill → backfill-outcomes → backfill-macro → backfill-coinalyze → backfill-taker → retrain). Fires every night at 00:10 UTC. The two microstructure backfills are soft-fails — they warn and skip rather than aborting the chain. |

---

## How the live pipeline works

```
OKX WebSocket (live prices)
   ↓
Candle builder (15-min OHLC per instrument)
   ↓
Feature engineer (89 features, z-score normalized)
   ↓
ModelHub (4 models — LightGBM or LR-L1 per symbol — picks the one for the target symbol)
   ↓
Confidence filter (skip if below MIN_CONFIDENCE threshold)
   ↓
Market resolver (find the matching Polymarket market via Gamma API)
   ↓
Order manager (FOK limit buy, notional capped at $25)
   ↓
Polymarket CLOB (place the order)
```

---

## Where state lives

**Everything is in Postgres.** No local files, no in-memory-only state, no Python, no ZMQ.

Key tables:

- `candles` — raw OHLC data from OKX
- `funding_rates` / `open_interest` / `index_candles` — perp feature inputs (funding rate, OI, basis)
- `open_interest_aggregated` / `liquidations_aggregated` / `taker_volume_15m` — microstructure feature inputs (Coinalyze + OKX)
- `macro_daily` — daily FRED macro series (DXY, SPX, VIX, yields) shared across all instruments
- `window_outcomes` — how each past 15-min window actually settled on Polymarket
- `model_versions` + `models` — trained model blobs, one "current" row per symbol
- `model_importance` — per-prediction feature-importance fingerprint
- `retrain_diagnostics` — per-symbol retrain quality metrics (accuracy, agreement, gate outcome)
- `signals` — every prediction the model has made
- `orders` / `trades` — every order placed
- `cron_runs` — lifecycle ledger for the nightly job
- `dry_run_results` / `dry_run_pending` — tracks predictions vs actual outcomes even when not placing real orders

---

## How models get updated (the magic part)

1. Nightly cron runs `polybot retrain`.
2. For each symbol, **two fresh models** are trained from scratch on the last 180 days (no fine-tuning — stale crypto market regimes would hurt): a LightGBM booster and an L1-regularized logistic regression (LR-L1) baseline.
3. Both families run through the same sanity gates. Whichever family clears Gate 4 is a promotion candidate; if both clear it, the higher median cross-validation accuracy wins. The winner is promoted in a DB transaction that fires `NOTIFY model_updated`. If neither clears Gate 4, nothing is promoted and the previous model keeps serving.
4. The already-running `polybot run` process is `LISTEN`ing for that notification. It reloads the new model **in place**, under a lock, without restarting — dispatching on the `model_family` column to deserialize it correctly.

The 4 gates that prevent bad models from getting promoted:

1. **Class distribution** — reject if UP or DOWN has 0 samples or one class is > 98%.
2. **Row count stability** — reject if this run has less than half the training rows of the previous one (catches backfill gaps).
3. **Training convergence** — reject if the model can't even learn its own training data.
4. **Validation accuracy (purged 5-fold CV + Wilson lower bound)** — fit five contiguous chronological folds with the training rows purged of any window whose label horizon overlaps the val fold (de Prado, AFML Ch. 7). Per fold, compute the Wilson 95% one-sided lower bound on val accuracy and require it to clear that fold's own majority baseline by 0.5pp. The model passes Gate 4 only if ≥ 3 of 5 folds clear.

---

## Where it runs

- **Process**: one long-running `polybot run` process plus a nightly `polybot cron` invocation (see `ops/` for sample systemd units and a crontab line).
- **Database**: Postgres.

---

## Project status

This is an experimental research project, not a proven-profitable strategy. Where the current implementation stands:

- Single Rust binary; all state in Postgres (no Python, C++, or ZMQ at runtime).
- Daily rolling retrain (`polybot cron`) on a 180-day window, labelled from Polymarket's actual settlements.
- Two model families per symbol — a LightGBM booster and an L1-logistic-regression (LR-L1) baseline — both trained each cycle, both run through the 4 promotion gates, and the per-symbol winner (higher median CV accuracy) promoted. `model_family` on `model_versions` records which is serving; `polybot run` hot-reloads on `NOTIFY model_updated` and dispatches on that column.
- A pooled multi-task LightGBM (with `symbol_id` as a categorical feature) is trained alongside the per-symbol models; when it clears Gate 4 for *every* symbol it overlays the per-symbol path, which stays as the rollback.
- Validation-based early stopping; beta calibration on top of the raw model probability.
- 89-feature row: 21 per-instrument features (OHLC-derived, funding, basis, multi-timescale momentum, and microstructure) × 4 symbols + 5 event-gated global macro features. `CURRENT_FORMAT_VERSION = 16`.

**Known limitations:**
- Model accuracy is weak; the ceiling is likely feature quality.
- The microstructure slots (`taker_buy_sell_ratio`, `oi_change_pct_4`, `liq_imbalance_4`) are consumed by the training path but not yet fed in the live `run` path, so live inference currently falls back to NaN for those three slots (LightGBM treats NaN as a learnable split direction).

---

## Key files to know

| File | What it is |
|---|---|
| `rust/src/main.rs` | Entry point |
| `rust/src/cli/` | Clap subcommand dispatch (`args.rs`) + nightly pipeline wrapper (`cron.rs`) |
| `rust/src/runtime/run.rs` | Live trading loop |
| `rust/src/runtime/trader.rs` | Signal → order decision gates |
| `rust/src/model/retrain.rs` | Training logic + the 4 gates + per-symbol model-family winner selection |
| `rust/src/model/training/lr_l1.rs` | L1-regularized logistic regression baseline (hand-rolled coordinate descent) |
| `rust/src/model/features/` | Candle builder + 89-feature computation (per-instrument blocks + global macro tail) |
| `rust/src/model/inference/` | ModelHub — loads and hot-swaps models |
| `rust/src/data/` | Backfills + live pollers (OKX candles, perp, FRED macro, Coinalyze, taker volume) |
| `rust/src/markets/order_manager.rs` | Polymarket order placement |
| `rust/src/migrations/001_init.sql` | The full Postgres schema (applied on first connect) |
| `ops/crontab` | The one line that fires `polybot cron` at 00:10 UTC daily |

---

## The mental model

Think of it as three separate concerns sharing one binary:

1. **Data pipeline** — pulls candles from OKX and outcomes from Polymarket into Postgres (`backfill`, `backfill-outcomes`).
2. **Model factory** — turns that data into one model per symbol (`retrain`), picking per symbol between a LightGBM booster and an LR-L1 baseline.
3. **Trader** — uses those models live to place Polymarket orders (`run`).

The three are loosely coupled through Postgres. The trader doesn't know or care how models got there; it just hot-reloads whatever's current. The retrainer doesn't know whether anyone's listening; it just promotes and fires a notification.