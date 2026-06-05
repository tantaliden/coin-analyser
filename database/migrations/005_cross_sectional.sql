-- Cross-Sectional Momentum Paper-Shadow (marktneutral). Idempotent. DB: analyser_app.
CREATE TABLE IF NOT EXISTS xs_paper_positions (
  id          BIGSERIAL PRIMARY KEY,
  symbol      TEXT NOT NULL,
  side        TEXT NOT NULL,
  entry_px    DOUBLE PRECISION NOT NULL,
  qty         DOUBLE PRECISION NOT NULL,
  margin_usd  DOUBLE PRECISION NOT NULL,
  opened_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  closed_at   TIMESTAMPTZ,
  exit_px     DOUBLE PRECISION,
  pnl_usd     DOUBLE PRECISION,
  status      TEXT NOT NULL DEFAULT 'open'
);
CREATE INDEX IF NOT EXISTS xs_pp_status ON xs_paper_positions(status);

CREATE TABLE IF NOT EXISTS xs_paper_wallet_state (
  id            INT PRIMARY KEY DEFAULT 1,
  balance       DOUBLE PRECISION NOT NULL,
  peak_balance  DOUBLE PRECISION NOT NULL,
  start_balance DOUBLE PRECISION NOT NULL,
  n_rebalances  INT NOT NULL DEFAULT 0,
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
