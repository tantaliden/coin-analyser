"""
Multi-Head Predictor Model (v5 - 2026-05-17)
============================================

Drei Online-lernende Köpfe mit gemeinsamer Feature-Basis:

  1. DirectionClassifier  — 3-Klassen ('long','short','skip') mit P-Output
  2. MagnitudeRegressor(long)  — predicted TP-Magnitude in % (Aufwärtsbewegung bei Long)
                                 + predicted SL-Magnitude in % (Abwärts-Pullback bei Long)
  3. MagnitudeRegressor(short) — spiegelbildlich

Pro Side ein Regressor-Paar (TP, SL), weil die Markt-Asymmetrie sonst nicht
gelernt wird (Long-Aufwärts != Short-Abwärts).

Online-Learning auf Live-Closes:
  - WIN  → DirectionClassifier.learn(features, label=predicted_side, weight=1.0)
           + MagnitudeRegressor_<side>.tp.learn(features, target=actual_peak_pct)
           + MagnitudeRegressor_<side>.sl.learn(features, target=actual_trough_pct)
  - LOSS → DirectionClassifier.learn(features, label=opposite_side, weight=1.0)
           (hart bestrafen: Modell soll lernen die GEGENRICHTUNG wäre richtig gewesen)
  - TIMEOUT mit drift in Predict-Richtung > drift in Gegenrichtung:
           → label=predicted_side, weight=0.3 (richtige Richtung aber zu schwach)
  - TIMEOUT mit drift gegen Predict-Richtung:
           → label=opposite_side, weight=1.0 (falsche Richtung)
  - TIMEOUT flach (|peak|,|trough| < 0.3%):
           → label='skip', weight=1.0 (Setup war eigentlich nichts)

State-Persistenz via Pickle, atomic via tempfile + os.replace.

NO-FALLBACK-POLICY (CARDINAL RULE_6):
  predict_proba() liefert kalibrierte P nur wenn Modell genug gesehen hat.
  Bei n_obs < min_n_for_predict: returns None — Caller muss skip.
  predict() der Regressoren returns None bei Cold-Start.
"""

import os
import pickle
from river import forest, tree


class DirectionClassifier:
    """3-Klassen-Klassifikator (long / short / skip) auf River ARFClassifier.

    Output ist kalibrierte P pro Klasse, summiert zu 1.
    Sample-Weight via Mehrfach-learn_one (River ARF kennt keinen weight-Param).
    """

    LABELS = ('long', 'short', 'skip')

    def __init__(self, n_models=10, grace_period=50, seed=42):
        self.model = forest.ARFClassifier(
            n_models=n_models,
            seed=seed,
            grace_period=grace_period,
            delta=1e-3,
        )
        self.n_obs = 0
        self.n_per_label = {l: 0 for l in self.LABELS}

    def predict_proba(self, features: dict, min_n_for_predict: int = 30):
        """Returns dict {'long':p, 'short':p, 'skip':p} or None bei Cold-Start.
        None = Caller muss skip (kein silent fallback).
        """
        if self.n_obs < min_n_for_predict:
            return None
        try:
            p = self.model.predict_proba_one(features)
        except Exception:
            return None
        if not p:
            return None
        out = {l: float(p.get(l, 0.0)) for l in self.LABELS}
        s = sum(out.values())
        if s <= 0:
            return None
        return {l: v / s for l, v in out.items()}

    def learn(self, features: dict, label: str, weight: float = 1.0):
        if label not in self.LABELS:
            raise ValueError(f"label must be in {self.LABELS}, got {label!r}")
        if weight <= 0:
            return
        n_repeats = max(1, min(5, int(round(weight))))
        for _ in range(n_repeats):
            self.model.learn_one(features, label)
        self.n_obs += 1
        self.n_per_label[label] = self.n_per_label.get(label, 0) + 1


class MagnitudeRegressor:
    """Online-Regressor für Bewegungs-Magnitude in % (nicht-negativ).

    Lernt direkt auf realisierte Peak- oder Trough-Werte (in Prozent).
    Caller wandelt prediction in TP/SL via Safety-Faktor um.
    """

    def __init__(self, grace_period=50, seed=42):
        self.model = tree.HoeffdingAdaptiveTreeRegressor(
            grace_period=grace_period,
            seed=seed,
        )
        self.n_obs = 0

    def predict(self, features: dict, min_n_for_predict: int = 30):
        """Returns float (in %) or None bei Cold-Start / Fehler.
        None = Caller muss skip (kein silent fallback).
        """
        if self.n_obs < min_n_for_predict:
            return None
        try:
            v = self.model.predict_one(features)
        except Exception:
            return None
        if v is None:
            return None
        return max(0.0, float(v))

    def learn(self, features: dict, target: float):
        if target is None:
            return
        t = float(target)
        if t < 0:
            t = 0.0
        self.model.learn_one(features, t)
        self.n_obs += 1


class MultiHeadPredictor:
    """Bündel: DirectionClassifier + 2× MagnitudeRegressor (TP) + 2× MagnitudeRegressor (SL).

    Speicher-Layout (pickle):
      direction
      tp_long  / tp_short
      sl_long  / sl_short
      version, created_at, last_save_at
    """

    VERSION = 5

    def __init__(self, seed=42, n_models_direction=10, grace_period=50):
        from datetime import datetime, timezone
        self.direction = DirectionClassifier(
            n_models=n_models_direction,
            grace_period=grace_period,
            seed=seed,
        )
        self.tp_long  = MagnitudeRegressor(grace_period=grace_period, seed=seed + 11)
        self.tp_short = MagnitudeRegressor(grace_period=grace_period, seed=seed + 12)
        self.sl_long  = MagnitudeRegressor(grace_period=grace_period, seed=seed + 21)
        self.sl_short = MagnitudeRegressor(grace_period=grace_period, seed=seed + 22)
        self.version = self.VERSION
        self.created_at = datetime.now(timezone.utc)
        self.last_save_at = None

    def predict_proba(self, features: dict, min_n_for_predict: int = 30):
        return self.direction.predict_proba(features, min_n_for_predict=min_n_for_predict)

    def predict_tp(self, features: dict, side: str, min_n_for_predict: int = 30):
        if side == 'long':
            return self.tp_long.predict(features, min_n_for_predict=min_n_for_predict)
        if side == 'short':
            return self.tp_short.predict(features, min_n_for_predict=min_n_for_predict)
        raise ValueError(f"side must be long/short, got {side!r}")

    def predict_sl(self, features: dict, side: str, min_n_for_predict: int = 30):
        if side == 'long':
            return self.sl_long.predict(features, min_n_for_predict=min_n_for_predict)
        if side == 'short':
            return self.sl_short.predict(features, min_n_for_predict=min_n_for_predict)
        raise ValueError(f"side must be long/short, got {side!r}")

    def learn_close(self, features: dict, predicted_side: str, status: str,
                    peak_pct: float, trough_pct: float,
                    timeout_flat_threshold_pct: float = 0.3,
                    timeout_correct_weight: float = 0.3,
                    timeout_wrong_weight: float = 1.0,
                    loss_weight: float = 1.0,
                    win_weight: float = 1.0):
        """Verarbeitet einen geschlossenen Trade.

        features: Feature-Dict zum Zeitpunkt des Open
        predicted_side: 'long' oder 'short' (was der Predictor entschieden hat)
        status: 'win' | 'loss' | 'timeout'
        peak_pct: maximale Aufwärtsbewegung in % (positiv) während Trade-Lebensdauer
        trough_pct: maximale Abwärtsbewegung in % (positiv) während Trade-Lebensdauer

        Schreibt:
          - DirectionClassifier-Update mit korrekter Label
          - TP-Regressor für die Side mit peak_pct (Long) bzw. trough_pct (Short)
          - SL-Regressor für die Side mit trough_pct (Long) bzw. peak_pct (Short)
        """
        if predicted_side not in ('long', 'short'):
            raise ValueError(f"predicted_side must be long/short, got {predicted_side!r}")
        if status not in ('win', 'loss', 'timeout'):
            raise ValueError(f"status must be win/loss/timeout, got {status!r}")

        opposite = 'short' if predicted_side == 'long' else 'long'

        # 1) Direction-Klassifikator
        if status == 'win':
            self.direction.learn(features, predicted_side, weight=win_weight)
        elif status == 'loss':
            self.direction.learn(features, opposite, weight=loss_weight)
        else:
            drift_in_predict = peak_pct if predicted_side == 'long' else trough_pct
            drift_against    = trough_pct if predicted_side == 'long' else peak_pct
            if max(drift_in_predict, drift_against) < timeout_flat_threshold_pct:
                self.direction.learn(features, 'skip', weight=1.0)
            elif drift_against > drift_in_predict:
                self.direction.learn(features, opposite, weight=timeout_wrong_weight)
            else:
                self.direction.learn(features, predicted_side, weight=timeout_correct_weight)

        # 2) Magnitude-Regressoren
        if predicted_side == 'long':
            self.tp_long.learn(features, peak_pct)
            self.sl_long.learn(features, trough_pct)
        else:
            self.tp_short.learn(features, trough_pct)
            self.sl_short.learn(features, peak_pct)

    def save(self, path: str):
        from datetime import datetime, timezone
        self.last_save_at = datetime.now(timezone.utc)
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not MultiHeadPredictor: {type(obj).__name__}")
        return obj

    def stats(self) -> dict:
        return {
            'version': self.version,
            'direction_n_obs': self.direction.n_obs,
            'direction_n_per_label': dict(self.direction.n_per_label),
            'tp_long_n_obs': self.tp_long.n_obs,
            'tp_short_n_obs': self.tp_short.n_obs,
            'sl_long_n_obs': self.sl_long.n_obs,
            'sl_short_n_obs': self.sl_short.n_obs,
            'created_at': str(self.created_at) if self.created_at else None,
            'last_save_at': str(self.last_save_at) if self.last_save_at else None,
        }
