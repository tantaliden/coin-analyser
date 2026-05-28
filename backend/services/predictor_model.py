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
    """3-Klassen-Klassifikator (long / short / skip).

    v6.3 (26.05.2026, Volker): Decision Trees (ARF/Hoeffding) ersetzt durch
    StandardScaler → OneVsRest(LogisticRegression). Trees splitten per Design — pro Knoten
    EIN Feature, andere ignoriert. Volker will dass JEDES der 172 Features in JEDE einzelne
    Prediction einfließt. LogReg leistet das: jedes Feature hat ein Gewicht, Prediction =
    softmax(Σ wᵢ·xᵢ), alle Gewichte werden bei jedem Sample via SGD angepasst — nichts wird
    ignoriert. StandardScaler standardisiert online (running mean/std), damit Features mit
    großen Beträgen (CVD, Volumen) die Gewichte nicht dominieren. n_models/grace_period
    sind Legacy (API-Kompat), werden ignoriert.
    """

    LABELS = ('long', 'short', 'skip')

    def __init__(self, n_models=10, grace_period=50, seed=42, model_type='logreg'):
        """model_type:
        - 'logreg': StandardScaler → OneVsRest(LogisticRegression) — jedes Feature
          mit eigenem Gewicht, ALLE in jeder Prediction. (v6.3 Standard.)
        - 'arf_full': ARFClassifier mit max_features=None — Tree-Ensemble, jeder Knoten
          kann aus ALLEN 172 Features wählen (kein zufälliges Subsampling). Pro
          einzelner Prediction werden trotzdem nur die Features auf dem Wurzel-zu-Blatt-
          Pfad genutzt — andere Architektur als LogReg. (Volker 27.05.: A/B-Vergleich
          ob random_subsampling allein der Übeltäter im alten ARF war.)
        """
        self.model_type = model_type
        if model_type == 'logreg':
            from river import linear_model, multiclass, preprocessing, compose
            self.model = compose.Pipeline(
                preprocessing.StandardScaler(),
                multiclass.OneVsRestClassifier(linear_model.LogisticRegression()),
            )
        elif model_type == 'arf_full':
            from river import forest
            self.model = forest.ARFClassifier(
                n_models=n_models, seed=seed, grace_period=grace_period,
                delta=1e-3, max_features=None,
            )
        else:
            raise ValueError(f"unknown direction_model_type: {model_type!r} (erlaubt: 'logreg' / 'arf_full')")
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

    VERSION = 6

    def __init__(self, seed=42, n_models_direction=10, grace_period=50,
                 direction_model_type='logreg'):
        from datetime import datetime, timezone
        self.direction = DirectionClassifier(
            n_models=n_models_direction,
            grace_period=grace_period,
            seed=seed,
            model_type=direction_model_type,
        )
        self.tp_long  = MagnitudeRegressor(grace_period=grace_period, seed=seed + 11)
        self.tp_short = MagnitudeRegressor(grace_period=grace_period, seed=seed + 12)
        self.sl_long  = MagnitudeRegressor(grace_period=grace_period, seed=seed + 21)
        self.sl_short = MagnitudeRegressor(grace_period=grace_period, seed=seed + 22)
        self.version = self.VERSION
        self.created_at = datetime.now(timezone.utc)
        self.last_save_at = None
        # TP-Erreichbarkeit (v6, 2026-05-25, Volker): TP wird nicht mehr per festem
        # Safety-Faktor gesetzt, sondern per selbst-kalibriertem Faktor je Side. Ein
        # Robbins-Monro-Regelkreis justiert den Faktor so, dass das gesetzte TP in
        # target-Quote der RICHTUNGSRICHTIGEN Trades erreicht wird ("Trefferquote führt").
        # Start neutral = 1.0 (TP = voller predicted Peak); der Regelkreis senkt auf den
        # tatsächlich erreichbaren Wert.
        self.tp_reach_factor = {'long': 1.0, 'short': 1.0}
        self.tp_reach_rate   = {'long': 0.0, 'short': 0.0}  # EWMA Erreichungsquote (nur Metrik)
        self.tp_reach_n      = {'long': 0, 'short': 0}      # Anzahl richtungsrichtiger TP-Bewertungen
        self.dir_correct     = {'long': 0, 'short': 0}      # davon Richtung gestimmt (Metrik)
        self.dir_total       = {'long': 0, 'short': 0}      # alle bewerteten Trades je Side
        # SL-Überlebensgrenze (v6.1, 2026-05-26, Volker): SL lernt aus ZURÜCKGEKEHRTEN
        # Trades (die das TP erreichten) ihren max-Drawdown; ein Robbins-Monro-Faktor setzt
        # das SL knapp unter die Grenze, ab der kein Return mehr kommt (survive_target der
        # Gewinner überlebt das SL). KEIN künstlich weites SL mehr (fester sl_safety raus).
        self.sl_reach_factor = {'long': 1.0, 'short': 1.0}
        self.sl_reach_rate   = {'long': 0.0, 'short': 0.0}  # EWMA Stop-Quote der Gewinner (Metrik)
        self.sl_reach_n      = {'long': 0, 'short': 0}
        # Entry-Timing: zählt "zu früh gesetzt" (bester Peak erst in der Nachphase nach Close)
        self.early_n     = 0
        self.early_total = 0

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

    def get_tp_reach_factor(self, side: str) -> float:
        """Selbst-kalibrierter Faktor: gesetztes TP-% = predicted_peak * factor."""
        return float(self.tp_reach_factor.get(side, 1.0))

    def register_tp_reach(self, side, set_tp_pct, realized_move_pct, direction_correct,
                          target_rate, step, min_factor, max_factor, ewma_alpha):
        """Robbins-Monro-Update des TP-Erreichbarkeits-Faktors.

        NUR für richtungsrichtige Trades: ein Trade mit falscher Richtung sagt nichts
        über die erreichbare TP-Höhe bei korrekter Richtung. hit = die realisierte
        Bewegung in Gewinnrichtung hat das gesetzte TP erreicht. Im Gleichgewicht gilt
        E[Δfactor]=0 ⇔ E[hit]=target_rate, d.h. der Faktor konvergiert exakt auf die
        gewünschte Trefferquote (Volker: 80%).
        """
        self.dir_total[side] = self.dir_total.get(side, 0) + 1
        if not direction_correct:
            return
        self.dir_correct[side] = self.dir_correct.get(side, 0) + 1
        if set_tp_pct is None or set_tp_pct <= 0:
            return
        hit = 1.0 if realized_move_pct >= set_tp_pct else 0.0
        a = float(ewma_alpha)
        self.tp_reach_rate[side] = (1.0 - a) * self.tp_reach_rate.get(side, 0.0) + a * hit
        f = self.tp_reach_factor.get(side, 1.0) + float(step) * (hit - float(target_rate))
        self.tp_reach_factor[side] = max(float(min_factor), min(float(max_factor), f))
        self.tp_reach_n[side] = self.tp_reach_n.get(side, 0) + 1

    def get_sl_reach_factor(self, side: str) -> float:
        """Selbst-kalibrierter Faktor: gesetztes SL-% = sl_raw (Gewinner-Drawdown) * factor."""
        return float(self.sl_reach_factor.get(side, 1.0))

    def register_sl_reach(self, side, set_sl_pct, trade_adverse_pct, returned,
                          survive_target, step, min_factor, max_factor, ewma_alpha):
        """SL-Überlebens-Regelkreis. Lernt NUR aus zurückgekehrten Trades (die das TP
        erreichten) — nur sie zeigen, wie tief ein Trade fallen darf und trotzdem einen
        Return macht. hit_sl = das gesetzte SL wäre bei diesem Gewinner ausgelöst worden
        (hätte ihn fälschlich ausgestoppt). E[Δ]=0 ⇔ E[hit_sl]=1-survive_target → das SL
        pendelt sich knapp unter die Return-Grenze ein (survive_target der Gewinner
        überleben), ohne künstlich weit zu sein.
        """
        if not returned or set_sl_pct is None or set_sl_pct <= 0 or trade_adverse_pct is None:
            return
        hit_sl = 1.0 if trade_adverse_pct >= set_sl_pct else 0.0
        a = float(ewma_alpha)
        self.sl_reach_rate[side] = (1.0 - a) * self.sl_reach_rate.get(side, 0.0) + a * hit_sl
        target_stop = 1.0 - float(survive_target)
        f = self.sl_reach_factor.get(side, 1.0) + float(step) * (hit_sl - target_stop)
        self.sl_reach_factor[side] = max(float(min_factor), min(float(max_factor), f))
        self.sl_reach_n[side] = self.sl_reach_n.get(side, 0) + 1

    def learn_close(self, features: dict, predicted_side: str, status: str,
                    peak_pct: float, trough_pct: float,
                    timeout_flat_threshold_pct: float = 0.3,
                    timeout_correct_weight: float = 0.3,
                    timeout_wrong_weight: float = 1.0,
                    loss_weight: float = 1.0,
                    win_weight: float = 1.0,
                    set_tp_pct: float = None,
                    tp_reach: dict = None,
                    peak_trade: float = None,
                    trough_trade: float = None,
                    peak_after_gain: float = None,
                    set_sl_pct: float = None,
                    sl_reach: dict = None,
                    early_cfg: dict = None):
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

        # Bewegungsgrößen:
        #   peak_pct/trough_pct      = VOLLE Range (Magnitude-Schätzung, Anti-Zirkularität)
        #   peak_trade/trough_trade  = nur TRADE-ZEIT (Entry..Close) -> Trefferquote + SL-Drawdown
        #   peak_after_gain          = Gewinnrichtungs-Peak der NACHPHASE (Close..+30m) -> Timing
        if predicted_side == 'long':
            full_gain, full_adverse = peak_pct, trough_pct
            trade_gain    = peak_trade    if peak_trade    is not None else peak_pct
            trade_adverse = trough_trade  if trough_trade  is not None else trough_pct
        else:
            full_gain, full_adverse = trough_pct, peak_pct
            trade_gain    = trough_trade  if trough_trade  is not None else trough_pct
            trade_adverse = peak_trade    if peak_trade    is not None else peak_pct

        direction_correct = full_gain > full_adverse
        # "Return" = der Trade erreichte in der Trade-Zeit sein TP (kam in Gewinnrichtung zurück)
        returned = (set_tp_pct is not None and set_tp_pct > 0 and trade_gain >= set_tp_pct)

        # 2) Magnitude-Regressoren
        # TP: volle Gewinnbewegung, NUR aus richtungsrichtigen Trades (v6).
        # SL: Trade-Zeit-Drawdown, NUR aus ZURÜCKGEKEHRTEN Trades (v6.1, Volker) — nur Gewinner
        #     zeigen, wie tief man fallen darf und trotzdem einen Return macht.
        if predicted_side == 'long':
            if direction_correct:
                self.tp_long.learn(features, full_gain)
            if returned:
                self.sl_long.learn(features, trade_adverse)
        else:
            if direction_correct:
                self.tp_short.learn(features, full_gain)
            if returned:
                self.sl_short.learn(features, trade_adverse)

        # 3) TP-Erreichbarkeit: Trefferquote über die TRADE-ZEIT (Volker: TP muss VOR Close
        #    erreicht werden, nicht erst in der Nachphase) -> trade_gain statt voller Range.
        if tp_reach is not None:
            self.register_tp_reach(
                side=predicted_side, set_tp_pct=set_tp_pct,
                realized_move_pct=trade_gain, direction_correct=direction_correct,
                target_rate=tp_reach['target'], step=tp_reach['step'],
                min_factor=tp_reach['min'], max_factor=tp_reach['max'],
                ewma_alpha=tp_reach['alpha'],
            )

        # 4) SL-Überlebensgrenze justieren (nur aus zurückgekehrten Trades, Volker)
        if sl_reach is not None:
            self.register_sl_reach(
                side=predicted_side, set_sl_pct=set_sl_pct,
                trade_adverse_pct=trade_adverse, returned=returned,
                survive_target=sl_reach['survive'], step=sl_reach['step'],
                min_factor=sl_reach['min'], max_factor=sl_reach['max'],
                ewma_alpha=sl_reach['alpha'],
            )

        # 5) Entry-Timing: war das Event "zu früh"? Bester Gewinn-Peak erst in der Nachphase
        #    (nach Close). Nur für NICHT-Gewinner relevant (ein Win lief in der Trade-Zeit).
        #    Wenn ja -> DirectionClassifier lernt 'skip' (Modell wird beim Timing wählerischer).
        if early_cfg is not None and status != 'win' and peak_after_gain is not None:
            self.early_total += 1
            if peak_after_gain > trade_gain + float(early_cfg['extra_pct']):
                self.direction.learn(features, 'skip', weight=float(early_cfg['skip_weight']))
                self.early_n += 1

    def save(self, path: str):
        from datetime import datetime, timezone
        self.last_save_at = datetime.now(timezone.utc)
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str, direction_model_type: str = 'logreg'):
        """Load + Migrationen.

        direction_model_type: Soll-Typ des DirectionClassifier ('logreg' oder 'arf_full').
        Wenn das geladene Modell einen anderen Direction-Typ hat, wird es ersetzt
        (n_obs=0). Magnitude/Reach-Faktoren bleiben erhalten.
        """
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not MultiHeadPredictor: {type(obj).__name__}")
        import copy
        MIN_COMPATIBLE_VERSION = 5
        if getattr(obj, 'version', 0) >= MIN_COMPATIBLE_VERSION:
            for attr, default in (('tp_reach_factor', {'long': 1.0, 'short': 1.0}),
                                  ('tp_reach_rate',   {'long': 0.0, 'short': 0.0}),
                                  ('tp_reach_n',      {'long': 0, 'short': 0}),
                                  ('dir_correct',     {'long': 0, 'short': 0}),
                                  ('dir_total',       {'long': 0, 'short': 0}),
                                  ('sl_reach_factor', {'long': 1.0, 'short': 1.0}),
                                  ('sl_reach_rate',   {'long': 0.0, 'short': 0.0}),
                                  ('sl_reach_n',      {'long': 0, 'short': 0}),
                                  ('early_n', 0),
                                  ('early_total', 0)):
                if not hasattr(obj, attr):
                    setattr(obj, attr, copy.deepcopy(default))

            # Direction-Modell-Typ migrieren wenn gewünschter Typ abweicht.
            current_type = getattr(obj.direction, 'model_type', None)
            if current_type is None:
                # Alte DirectionClassifier-Instanzen hatten kein model_type-Attribut;
                # aus der Modell-Klasse ableiten:
                from river import compose as _rc, forest as _rf, tree as _rt
                if isinstance(obj.direction.model, _rc.Pipeline):
                    current_type = 'logreg'
                elif isinstance(obj.direction.model, _rf.ARFClassifier):
                    current_type = 'arf_legacy'  # alter ARF mit max_features != None
                elif isinstance(obj.direction.model, _rt.HoeffdingAdaptiveTreeClassifier):
                    current_type = 'hoeffding_legacy'
                else:
                    current_type = 'unknown'
                obj.direction.model_type = current_type
            if current_type != direction_model_type:
                obj.direction = DirectionClassifier(
                    n_models=10, grace_period=50, seed=42,
                    model_type=direction_model_type,
                )

            obj.version = cls.VERSION
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
            'tp_reach_factor': dict(getattr(self, 'tp_reach_factor', {})),
            'tp_reach_rate': dict(getattr(self, 'tp_reach_rate', {})),
            'tp_reach_n': dict(getattr(self, 'tp_reach_n', {})),
            'sl_reach_factor': dict(getattr(self, 'sl_reach_factor', {})),
            'sl_reach_rate': dict(getattr(self, 'sl_reach_rate', {})),
            'sl_reach_n': dict(getattr(self, 'sl_reach_n', {})),
            'early_skip_rate': (self.early_n / self.early_total) if getattr(self, 'early_total', 0) > 0 else None,
            'early_total': getattr(self, 'early_total', 0),
            'dir_accuracy': {
                s: (self.dir_correct.get(s, 0) / self.dir_total[s])
                for s in ('long', 'short')
                if getattr(self, 'dir_total', {}).get(s, 0) > 0
            },
            'created_at': str(self.created_at) if self.created_at else None,
            'last_save_at': str(self.last_save_at) if self.last_save_at else None,
        }
