-- Migration 001: Stalker + Paper-Engine Schema (21.05.2026)
-- IDEMPOTENT — läuft auf jedem Server beliebig oft ohne Schaden.
-- Hält Test ↔ Live strukturell synchron (Volker-Direktive: 1:1, nur Daten+URL differieren).
-- analyser_app DB.

-- 1. open_predictions: Skip-Flag (UI-hidden, Predictor lernt trotzdem) + invalid-Status
ALTER TABLE open_predictions ADD COLUMN IF NOT EXISTS auto_trade_skipped BOOLEAN NOT NULL DEFAULT FALSE;
CREATE INDEX IF NOT EXISTS idx_op_auto_trade_skipped ON open_predictions(auto_trade_skipped) WHERE auto_trade_skipped=TRUE;

ALTER TABLE open_predictions DROP CONSTRAINT IF EXISTS open_predictions_status_check;
ALTER TABLE open_predictions ADD CONSTRAINT open_predictions_status_check
  CHECK (status = ANY (ARRAY['open','win','loss','timeout','auto_close_failsafe','auto_trade_failed','modify_close','invalid']));

-- 2. Stalker-Baseline (pro Coin × hour × weekday × btc_regime Quantile)
CREATE TABLE IF NOT EXISTS coin_baseline_profile (
    symbol      TEXT NOT NULL,
    hour_of_day SMALLINT NOT NULL CHECK (hour_of_day BETWEEN 0 AND 23),
    weekday     SMALLINT NOT NULL CHECK (weekday BETWEEN 0 AND 6),
    btc_regime  TEXT NOT NULL CHECK (btc_regime IN ('bull','bear','range')),
    metric      TEXT NOT NULL,
    n_samples   INT NOT NULL DEFAULT 0,
    p25 DOUBLE PRECISION, p50 DOUBLE PRECISION, p75 DOUBLE PRECISION, mad DOUBLE PRECISION,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol, hour_of_day, weekday, btc_regime, metric)
);
CREATE INDEX IF NOT EXISTS idx_cbp_symbol ON coin_baseline_profile(symbol);
CREATE INDEX IF NOT EXISTS idx_cbp_updated ON coin_baseline_profile(updated_at);

-- 3. Paper-Trading-Engine (virtuelle Positionen + Wallet)
CREATE TABLE IF NOT EXISTS paper_positions (
    id              SERIAL PRIMARY KEY,
    prediction_id   INT,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL CHECK (side IN ('long','short')),
    entry_px        DOUBLE PRECISION NOT NULL,
    qty             DOUBLE PRECISION NOT NULL,
    leverage        INT NOT NULL,
    margin_usd      DOUBLE PRECISION NOT NULL,
    tp_px           DOUBLE PRECISION NOT NULL,
    sl_px           DOUBLE PRECISION NOT NULL,
    opened_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    closed_at       TIMESTAMPTZ,
    exit_px         DOUBLE PRECISION,
    pnl_pct         DOUBLE PRECISION,
    pnl_usd         DOUBLE PRECISION,
    status          TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','win','loss','timeout')),
    close_reason    TEXT,
    timeout_enabled BOOLEAN NOT NULL DEFAULT TRUE
);
CREATE INDEX IF NOT EXISTS idx_paper_pos_status ON paper_positions(status) WHERE status='open';
CREATE INDEX IF NOT EXISTS idx_paper_pos_opened ON paper_positions(opened_at DESC);

CREATE TABLE IF NOT EXISTS paper_wallet_state (
    id            INT PRIMARY KEY DEFAULT 1,
    balance       DOUBLE PRECISION NOT NULL,
    peak_balance  DOUBLE PRECISION NOT NULL,
    start_balance DOUBLE PRECISION NOT NULL,
    n_trades      INT NOT NULL DEFAULT 0,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (id = 1)
);
INSERT INTO paper_wallet_state (id, balance, peak_balance, start_balance)
VALUES (1, 1000, 1000, 1000) ON CONFLICT (id) DO NOTHING;
