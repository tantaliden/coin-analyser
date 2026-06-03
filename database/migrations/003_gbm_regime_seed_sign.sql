-- GBM-Regime-Detektor: Start-Vorzeichen vom Live-Fenster trennen.
-- Seed setzt seed_sign (Start-Vorzeichen), das laufende `recent` zählt nur echte
-- Live-Closes -> schnelle Adaption (Seed erstickt die Anpassung nicht mehr).
-- Idempotent. DB: analyser_app (app).

ALTER TABLE gbm_regime_state ADD COLUMN IF NOT EXISTS seed_sign SMALLINT;
