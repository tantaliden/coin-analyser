-- GBM-Predictor Paper-Shadow: eigene Welt, getrennt vom alten Predictor.
-- Idempotent. DB: analyser_app (app).

CREATE TABLE IF NOT EXISTS gbm_paper_positions (
  id           BIGSERIAL PRIMARY KEY,
  symbol       TEXT NOT NULL,
  side         TEXT NOT NULL,                      -- 'long' | 'short'
  entry_px     DOUBLE PRECISION NOT NULL,
  tp_px        DOUBLE PRECISION NOT NULL,
  sl_px        DOUBLE PRECISION NOT NULL,
  qty          DOUBLE PRECISION NOT NULL,
  margin_usd   DOUBLE PRECISION NOT NULL,
  dir_conf     DOUBLE PRECISION NOT NULL,          -- Richtungs-Confidence (nach Vorzeichen)
  p_resolve    DOUBLE PRECISION NOT NULL,          -- Gate: P(sauberer Move)
  regime_sign  SMALLINT NOT NULL,                  -- +1 = Modell direkt, -1 = invertiert
  opened_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  closed_at    TIMESTAMPTZ,
  exit_px      DOUBLE PRECISION,
  pnl_pct      DOUBLE PRECISION,
  pnl_usd      DOUBLE PRECISION,
  status       TEXT NOT NULL DEFAULT 'open',        -- 'open'|'win'|'loss'|'timeout'
  close_reason TEXT
);
CREATE INDEX IF NOT EXISTS gbm_pp_status ON gbm_paper_positions(status);
CREATE INDEX IF NOT EXISTS gbm_pp_symbol_open ON gbm_paper_positions(symbol) WHERE status='open';

CREATE TABLE IF NOT EXISTS gbm_paper_wallet_state (
  id            INT PRIMARY KEY DEFAULT 1,
  balance       DOUBLE PRECISION NOT NULL,
  peak_balance  DOUBLE PRECISION NOT NULL,
  start_balance DOUBLE PRECISION NOT NULL,
  n_trades      INT NOT NULL DEFAULT 0,
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Regime-Detektor-Zustand: rollende Korrektheit der zuletzt geschlossenen Trades.
CREATE TABLE IF NOT EXISTS gbm_regime_state (
  id           INT PRIMARY KEY DEFAULT 1,
  current_sign SMALLINT NOT NULL DEFAULT 1,
  recent       JSONB NOT NULL DEFAULT '[]',          -- Liste 0/1 (Basis-Modell-Richtung korrekt?)
  n_closed     INT NOT NULL DEFAULT 0,
  hit_rate     DOUBLE PRECISION,
  updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
