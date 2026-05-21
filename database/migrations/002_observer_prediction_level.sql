-- 002_observer_prediction_level.sql  (idempotent)
-- Observer v2 (21.05.2026): Observer arbeitet auf Prediction-Ebene (open_predictions)
-- statt auf paper_positions. Anti-Loop-State pro Prediction:
--   observer_adj_count   = wie oft wurde diese Prediction vom Observer angefasst
--   observer_last_adj_at = wann zuletzt (fuer reeval_cooldown_minutes)
-- DB: analyser_app
-- Anwenden auf BEIDEN Servern (Test + Live).

ALTER TABLE open_predictions
  ADD COLUMN IF NOT EXISTS observer_adj_count   INTEGER     NOT NULL DEFAULT 0;

ALTER TABLE open_predictions
  ADD COLUMN IF NOT EXISTS observer_last_adj_at TIMESTAMPTZ;
