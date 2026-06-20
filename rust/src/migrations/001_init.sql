-- Consolidated schema for polybot_crypto_15mins.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ── errors ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS errors (
    id          BIGSERIAL PRIMARY KEY,
    source      VARCHAR(20) NOT NULL,
    context     TEXT NOT NULL,
    detail      TEXT NOT NULL,
    happened_at TIMESTAMPTZ DEFAULT NOW()
);

-- ── model_versions ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_versions (
    id                BIGSERIAL   PRIMARY KEY,
    symbol            VARCHAR(20) NOT NULL,
    is_current        BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    labeled_row_count INTEGER,
    model_family      VARCHAR(16) NOT NULL DEFAULT 'lightgbm'
);

CREATE UNIQUE INDEX IF NOT EXISTS model_versions_one_current_per_symbol
    ON model_versions (symbol) WHERE is_current;

-- ── models ───────────────────────────────────────────────────────────────────
-- model_bytes: raw lightgbm3 Booster::save_to_string output.
-- sha256_hex: encode(digest(model_bytes, 'sha256'), 'hex'), computed server-side.
CREATE TABLE IF NOT EXISTS models (
    id                BIGSERIAL   PRIMARY KEY,
    model_version_id  BIGINT      NOT NULL UNIQUE
                                  REFERENCES model_versions(id) ON DELETE CASCADE,
    model_bytes       BYTEA       NOT NULL,
    byte_size         INTEGER     NOT NULL,
    sha256_hex        VARCHAR(64) NOT NULL,
    format_version    SMALLINT    NOT NULL DEFAULT 2,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── candles (15-minute OHLC buckets per instrument) ──────────────────────────-
CREATE TABLE IF NOT EXISTS candles (
    inst_id    VARCHAR(20)      NOT NULL,
    ts_ms      BIGINT           NOT NULL,       -- bucket open time, 15-min aligned
    open       DOUBLE PRECISION NOT NULL,
    high       DOUBLE PRECISION NOT NULL,
    low        DOUBLE PRECISION NOT NULL,
    close      DOUBLE PRECISION NOT NULL,
    tick_count INTEGER          NOT NULL,
    PRIMARY KEY (inst_id, ts_ms)
);

CREATE INDEX IF NOT EXISTS candles_ts_ms_idx ON candles (ts_ms);

-- ── signals ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS signals (
    id         BIGSERIAL PRIMARY KEY,
    error_id   BIGINT REFERENCES errors(id),
    ts_ms      BIGINT NOT NULL,
    signal     SMALLINT NOT NULL,               -- HOLD=0, BUY=1, SELL=2
    confidence FLOAT NOT NULL,
    market_id  VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ── orders ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS orders (
    id           BIGSERIAL PRIMARY KEY,
    error_id     BIGINT REFERENCES errors(id),
    signal_id    BIGINT REFERENCES signals(id),
    market_id    VARCHAR(100) NOT NULL,
    token_id     VARCHAR(100) NOT NULL,
    side         VARCHAR(4) NOT NULL,            -- 'BUY' or 'SELL'
    usdc         DOUBLE PRECISION NOT NULL,
    price        DOUBLE PRECISION NOT NULL,
    fee_rate_bps INTEGER,                        -- null if fee fetch failed
    order_id     VARCHAR(100),                   -- null if failed
    order_status VARCHAR(20) NOT NULL,           -- 'PENDING', 'MATCHED', 'CANCELLED', 'FAILED'
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    updated_at   TIMESTAMPTZ DEFAULT NOW()
);

-- ── positions ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS positions (
    id              BIGSERIAL PRIMARY KEY,
    error_id        BIGINT REFERENCES errors(id),
    market_id       VARCHAR(100) NOT NULL,
    token_id        VARCHAR(100) NOT NULL,
    side            VARCHAR(4) NOT NULL,
    usdc            DOUBLE PRECISION NOT NULL,
    avg_entry_price DOUBLE PRECISION NOT NULL,
    opened_at       TIMESTAMPTZ NOT NULL,
    closed_at       TIMESTAMPTZ,                 -- NULL = still open
    position_status VARCHAR(100) NOT NULL DEFAULT 'OPEN'
);

CREATE UNIQUE INDEX IF NOT EXISTS positions_one_open_per_token
    ON positions (market_id, token_id)
    WHERE position_status = 'OPEN';

-- ── trades ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trades (
    id          BIGSERIAL PRIMARY KEY,
    error_id    BIGINT REFERENCES errors(id),
    market_id   VARCHAR(100) NOT NULL,
    side        VARCHAR(4) NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    exit_price  DOUBLE PRECISION NOT NULL,
    usdc        DOUBLE PRECISION NOT NULL,
    pnl         DOUBLE PRECISION NOT NULL,       -- in USDC
    pnl_pct     DOUBLE PRECISION NOT NULL,       -- return on entry, e.g. 0.375 = 37.5%
    opened_at   TIMESTAMPTZ NOT NULL,
    closed_at   TIMESTAMPTZ NOT NULL
);

-- ── bot_runs ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS bot_runs (
    id               BIGSERIAL PRIMARY KEY,
    error_id         BIGINT REFERENCES errors(id),
    model_path       VARCHAR(255),
    conf_threshold   FLOAT,
    data_source      VARCHAR(200),
    started_at       TIMESTAMPTZ DEFAULT NOW(),
    stopped_at       TIMESTAMPTZ,
    notes            TEXT,
    model_version_id BIGINT REFERENCES model_versions(id)
);

-- ── window_outcomes (Polymarket resolved outcome per symbol+window) ──────────--
CREATE TABLE IF NOT EXISTS window_outcomes (
    symbol         VARCHAR(20) NOT NULL,
    window_ts_secs BIGINT      NOT NULL,         -- Unix-second start of the 15-min window
    outcome        SMALLINT    NOT NULL CHECK (outcome IN (1, 2)),   -- 1=UP, 2=DOWN
    slug           TEXT        NOT NULL,
    resolved_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol, window_ts_secs)
);

-- ── cron_runs (nightly chain lifecycle ledger) ───────────────────────────────-
CREATE TABLE IF NOT EXISTS cron_runs (
    id          BIGSERIAL   PRIMARY KEY,
    command     TEXT        NOT NULL,
    stage       TEXT        NOT NULL DEFAULT 'chain',
    started_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,                     -- NULL = running or crashed
    exit_code   INTEGER,                         -- NULL iff finished_at IS NULL
    host        TEXT,
    summary     JSONB       NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_cron_runs_started_at
    ON cron_runs (started_at DESC);
CREATE INDEX IF NOT EXISTS idx_cron_runs_unfinished
    ON cron_runs (started_at DESC)
    WHERE finished_at IS NULL;

-- ── perp feature inputs (OKX public endpoints) ───────────────────────────────-
-- ts_ms on all three is 15-min bucket OPEN time, joinable to candles.ts_ms.
CREATE TABLE IF NOT EXISTS funding_rates (
    inst_id            VARCHAR(20)      NOT NULL,
    ts_ms              BIGINT           NOT NULL,    -- OKX fundingTime (settlement)
    rate               DOUBLE PRECISION NOT NULL,    -- signed fraction; 0.0001 == 1bp
    settle_period_secs INTEGER,                      -- NULL if unknown
    PRIMARY KEY (inst_id, ts_ms)
);
CREATE INDEX IF NOT EXISTS funding_rates_ts_ms_idx ON funding_rates (ts_ms);

CREATE TABLE IF NOT EXISTS open_interest (
    inst_id VARCHAR(20)      NOT NULL,            -- swap form, e.g. "BTC-USDT-SWAP"
    ts_ms   BIGINT           NOT NULL,
    oi_ccy  DOUBLE PRECISION NOT NULL,            -- OI in coin units
    oi_usd  DOUBLE PRECISION NOT NULL,            -- OI in USD
    PRIMARY KEY (inst_id, ts_ms)
);
CREATE INDEX IF NOT EXISTS open_interest_ts_ms_idx ON open_interest (ts_ms);

CREATE TABLE IF NOT EXISTS index_candles (
    inst_id VARCHAR(20)      NOT NULL,            -- index form, e.g. "BTC-USDT"
    ts_ms   BIGINT           NOT NULL,
    open    DOUBLE PRECISION NOT NULL,
    high    DOUBLE PRECISION NOT NULL,
    low     DOUBLE PRECISION NOT NULL,
    close   DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (inst_id, ts_ms)
);
CREATE INDEX IF NOT EXISTS index_candles_ts_ms_idx ON index_candles (ts_ms);

-- ── model_importance (per-prediction feature fingerprint) ────────────────────-
CREATE TABLE IF NOT EXISTS model_importance (
    id                BIGSERIAL        PRIMARY KEY,
    signal_id         BIGINT           NOT NULL
                                       REFERENCES signals(id) ON DELETE CASCADE,
    model_version_id  BIGINT           NOT NULL
                                       REFERENCES model_versions(id),
    symbol            VARCHAR(20)      NOT NULL,
    instrument_name   VARCHAR(20)      NOT NULL,
    feature_name      VARCHAR(30)      NOT NULL,
    feature_value     DOUBLE PRECISION NOT NULL,
    importance        DOUBLE PRECISION NOT NULL,
    created_at        TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    CONSTRAINT model_importance_signal_inst_feat_key
        UNIQUE (signal_id, instrument_name, feature_name)
);
CREATE INDEX IF NOT EXISTS model_importance_signal_idx
    ON model_importance (signal_id);
CREATE INDEX IF NOT EXISTS model_importance_model_symbol_idx
    ON model_importance (model_version_id, symbol);

-- ── retrain_diagnostics (per-symbol retrain quality metrics) ─────────────────-
CREATE TABLE IF NOT EXISTS retrain_diagnostics (
    id                BIGSERIAL        PRIMARY KEY,
    symbol            VARCHAR(10)      NOT NULL,
    retrain_date      DATE             NOT NULL DEFAULT CURRENT_DATE,
    labeled_rows      INTEGER,
    train_acc         DOUBLE PRECISION,
    val_acc           DOUBLE PRECISION,           -- NULL when do_holdout=false
    majority_baseline DOUBLE PRECISION,           -- NULL when do_holdout=false
    agreement_rate    DOUBLE PRECISION,           -- NULL when agree+disagree=0
    logloss           DOUBLE PRECISION,
    iterations_used   INTEGER,
    promoted          BOOLEAN          NOT NULL,
    gate_rejected     VARCHAR(40),                -- NULL when promoted=true
    created_at        TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS retrain_diagnostics_symbol_date_idx
    ON retrain_diagnostics (symbol, retrain_date);

-- ── dry-run paper trading ────────────────────────────────────────────────────
-- predicted_action: 1=BUY/UP, 2=SELL/DOWN.  actual_outcome: 1=UP won, 2=DOWN won.
CREATE TABLE IF NOT EXISTS dry_run_results (
    id               BIGSERIAL PRIMARY KEY,
    signal_id        BIGINT NOT NULL REFERENCES signals(id),
    market_id        VARCHAR(100) NOT NULL,
    predicted_action SMALLINT NOT NULL,
    confidence       FLOAT NOT NULL,
    actual_outcome   SMALLINT NOT NULL,
    correct          BOOLEAN NOT NULL,
    resolved_at      TIMESTAMPTZ DEFAULT NOW(),
    entry_price      DOUBLE PRECISION,
    fee_rate_bps     INTEGER,
    shares           DOUBLE PRECISION,
    skip_reason      TEXT
);
CREATE INDEX IF NOT EXISTS dry_run_signal_idx ON dry_run_results (signal_id);

-- At-decision-time price/size, joined by the resolver when it materialises
-- the dry_run_results row.
CREATE TABLE IF NOT EXISTS dry_run_pending (
    signal_id    BIGINT           PRIMARY KEY REFERENCES signals(id) ON DELETE CASCADE,
    entry_price  DOUBLE PRECISION,
    shares       DOUBLE PRECISION,
    fee_rate_bps INTEGER,
    created_at   TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    skip_reason  TEXT
);

-- ── macro_daily (global FRED series) ─────────────────────────────────────────-
-- value is stored in FRED-native units; unit transforms happen at feature-emit.
CREATE TABLE IF NOT EXISTS macro_daily (
    series_id VARCHAR(20)      NOT NULL,          -- FRED series, e.g. "DGS10", "VIXCLS"
    date_utc  DATE             NOT NULL,          -- FRED observation date (UTC trading day)
    value     DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (series_id, date_utc)
);
CREATE INDEX IF NOT EXISTS macro_daily_date_idx ON macro_daily (date_utc);

-- ── Coinalyze aggregated OI + liquidations ───────────────────────────────────-
CREATE TABLE IF NOT EXISTS open_interest_aggregated (
    symbol VARCHAR(20)      NOT NULL,             -- polybot short form: "BTC"/"ETH"/...
    ts_ms  BIGINT           NOT NULL,
    oi_usd DOUBLE PRECISION NOT NULL,             -- close-of-bucket aggregated OI, USD
    PRIMARY KEY (symbol, ts_ms)
);
CREATE INDEX IF NOT EXISTS oi_agg_ts_idx ON open_interest_aggregated (ts_ms);

CREATE TABLE IF NOT EXISTS liquidations_aggregated (
    symbol        VARCHAR(20)      NOT NULL,
    ts_ms         BIGINT           NOT NULL,
    long_liq_usd  DOUBLE PRECISION NOT NULL,
    short_liq_usd DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (symbol, ts_ms)
);
CREATE INDEX IF NOT EXISTS liq_agg_ts_idx ON liquidations_aggregated (ts_ms);

-- ── taker_volume_15m (OKX taker buy/sell volume per bucket) ──────────────────-
CREATE TABLE IF NOT EXISTS taker_volume_15m (
    inst_id        VARCHAR(20)      NOT NULL,     -- "BTC-USDT-SWAP" / ...
    ts_ms          BIGINT           NOT NULL,     -- 15-min bucket-open in UTC ms
    taker_buy_vol  DOUBLE PRECISION NOT NULL,
    taker_sell_vol DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (inst_id, ts_ms)
);
CREATE INDEX IF NOT EXISTS taker_volume_15m_ts_idx ON taker_volume_15m (ts_ms);
