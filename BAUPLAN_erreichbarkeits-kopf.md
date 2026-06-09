# Bauplan: Erreichbarkeits-Kopf für den Predictor

**Status:** KONZEPT — noch nicht gebaut (28.05.2026). Erst Wirkung der aktuellen Änderungen
abwarten, dann bauen. Volker-Vision, mehrfach präzisiert.

---

## Das Ziel (Volkers Vorgabe, 1:1)

Der Predictor soll **keine Mikrogewinne** mehr produzieren und sich an eine **Erreichbarkeits-
Schwelle halten** statt blind zu hoffen. Zwei Stufen:

1. **TP-Höhe:** Der Predictor berechnet den erwarteten **High-Peak** (Magnitude-Regressor).
   Das TP wird auf **80 % dieses Peaks** gesetzt. (Z. B. erwarteter Peak +10 % → TP = 8 %.)
   → *Das ist seit 28.05. gebaut:* `tp_pct = tp_raw * tp_peak_share` (tp_peak_share = 0,8).

2. **Erreichbarkeits-Filter (FEHLT NOCH — das ist dieser Bauplan):** Eine Prediction wird
   **nur getradet, wenn bei vergleichbaren früheren Fällen in ≥80 % BEIDE Bedingungen galten:**
   - **(a) Getroffen:** Der reale High-Peak des damaligen Falls ≥ unser gesetztes TP (0,8×Peak)
     → das TP wäre erreicht worden.
   - **(b) Nah dran:** Unser TP lag bei **≥80 % des damaligen realen Peaks** (oder näher)
     → der berechnete Peak stimmte mit dem real erreichten überein (real ∈ [0,8×Peak, Peak]).
   Sonst → skip.

   > Volker (28.05., wörtlich): „hätte sie getroffen UND wäre die Prediktion mindestens 80 %
   > vom High-Peak entfernt oder sogar noch näher dran?" Beide Lecks geschlossen: kein zu hohes
   > TP (trifft nicht) UND kein zu niedriges (Mikrogewinn, schöpft real-Peak nicht ab).
   >
   > „es soll nur getradet werden, wenn diese Prediktion bei anderen Prediktionen mindestens zu
   > 80 % möglich gewesen wäre … ob es damals so gemacht wurde ist eine andere Sache, aber
   > zukünftig soll es so sein." → **Retrospektive Counterfactual-Prüfung** mit dem *jetzigen* TP.

3. **Stop-Loss = „Point of no Return":** Nicht der Punkt, wo noch 80 % zurückkommen (das ist
   zu früh/flach — das war v6.7), sondern der Drawdown-Punkt, ab dem **80 % der vergleichbaren
   Fälle KEINEN Return mehr machen** und es nur noch schlimmer wird. Dort — und erst dort —
   aussteigen. (Dreht v6.7 um: recover-target 0,8 → 0,2.)

**Wichtig (Volker):** Die Anzahl der Predictions ist EGAL. Es geht NICHT um Quantitäts-Drosselung
(`min_tp` ist NICHT der richtige Hebel — wurde 28.05. als Hebel verworfen). Es geht einzig darum,
dass die getradeten Predictions **zutreffen**. Wenige verlässliche statt viele mittelmäßige.

---

## Warum das aktuelle System das NICHT kann

Der aktuelle Multi-Head prüft die Erreichbarkeit **gar nicht**. Ablauf heute:
- DirectionClassifier → Side (long/short) + Richtungs-Confidence (`score`)
- Magnitude-Regressor → erwarteter Peak (`tp_raw`)
- TP = 0,8 × tp_raw, deploy wenn Richtungs-Confidence ≥ Schwelle

→ Es setzt das TP und **hofft**. Ob der Coin in solchen Lagen historisch überhaupt bis dahin
läuft, fragt es nie. Der alte `tp_reach_factor`-Regelkreis war ein **globaler** Faktor (ein Wert
pro Side), kein **pro-Setup-Filter** — und er kollabierte auf das Minimum (0,2), was das TP auf
den Mikro-Floor (0,30 %) drückte. Deshalb „hält sich der Predictor nicht an die Schwelle": er hat
keine.

---

## Die Datenbasis (Volker-Vorgabe)

- **Grunddaten aus der Coin-Historie**, NICHT nur aus den vergangenen Predictions. Begründung:
  Predictions sind ein verzerrtes, dünnes Sample — man sieht nur, wo getradet wurde. Die ehrliche
  Häufigkeit („wie oft trat das Muster wirklich auf und ging auf") steht in den Kursdaten.
- **Fenster: letzte 10–14 Tage** des jeweiligen Coins (mehr nicht — Volker). Macht den Live-
  Lookup performant.
- **Live im scan_pass** (nicht Batch-vorberechnet) — Volker-Entscheidung.

### Zirkelschluss-Vermeidung (kritisch)
- Gematcht wird auf die **Vorboten** = die Features/Dynamik VOR dem Move (aus klines/aggs).
- Gemessen wird das Ergebnis mit **kline_metrics** (= „wie viel % Bewegung kam danach").
- **Nie** auf der kline_metrics des laufenden Trades matchen — die enthält die Zukunft, die man
  vorhersagen will. (= Volkers altes „nie auf metrics matchen"-Prinzip.)
- `kline_metrics.pct_Xm` ist **rückwärts** gerechnet (`(close[t] − open[t−Xm])`). Die Forward-
  Bewegung ab T ist dieselbe Metrik zeitversetzt: `pct_Xm[T+X]`.

---

## Das Vorboten-Profil (aus `event_finder.py` übernommen)

Volkers „Vorboten" = Dynamik der ~120 min vor dem Move. Aus `analyze_pre_event()`:
- `vol_trend` — Volumen-Trend (2. Hälfte / 1. Hälfte)
- `trades_trend` — Trade-Anzahl-Trend
- `vol_last30_ratio` — Volumen-Konzentration der letzten 30 min
- `avg_taker_ratio` + `taker_shift` — Taker-Buy-Verhältnis und dessen Kippen
- `cumulative_pct` + `last30_pct` — Momentum (gesamt + letzte 30 min)
- `range_expansion` — wird die Bewegung größer?

---

## Mechanik des neuen Kopfes (live, pro Coin, im scan_pass)

1. Berechne das Vorboten-Profil für **jetzt** (Coin C).
2. Gehe die letzten **10–14 Tage** von C durch, finde Momente mit **ähnlichem Profil**.
   - **Ähnlichkeit:** Distanz-Schwelle (normalisierte Distanz < ε) **mit Mindest-Trefferzahl**
     (z. B. „mind. 20 ähnliche Fälle, sonst skip — zu wenig Evidenz"). Am ehrlichsten zur
     „wie oft trat es wirklich ein"-Logik. (Alternative: Top-k — verworfen, zieht halb-ähnliche rein.)
3. Für jeden Treffer: schau die **Forward-Bewegung** (nächste timeout_h) und ermittle den
   **realen High-Peak** des Falls. Zähle den Fall als „Treffer" nur wenn BEIDE gelten:
   - (a) realer Peak ≥ gesetztes TP (0,8×Peak)  → getroffen
   - (b) gesetztes TP ≥ 0,8 × realer Peak       → nah dran (real ∈ [0,8×Peak, Peak])
   - separat: **point of no return** = Drawdown-Level, ab dem ≤ 20 % noch zurückkamen.
4. **Entscheidung:**
   - `Treffer-Quote (a∧b) ≥ 0,80` → weiter, sonst **skip**
   - `TP = 0,8 × Peak` (schon gebaut)
   - `SL = point of no return`

---

## event_finder als Blaupause

`database/services/event_finder.py` macht das Konzept bereits retrospektiv (Events aus
kline_metrics, Vorboten aus klines, Precision-Test „wie oft folgt der Move", pro Coin, Long/Short).
ABER: nur Text-Report (`/opt/coin/database/data/event_report_*.txt`), keine Predictor-Anbindung,
disabled seit Feb (letzter Report 23.03.).

**⚠️ event_finder.py NICHT anfassen** — es ist die manuelle Suche im Analyser. Für den neuen Kopf
eine **Kopie** der Logik nehmen, Original bleibt.

---

## Offene Entscheidungen (beim Bauen klären)

1. **Datenbasis final:** reine Coin-Historie (klines) oder counterfactual über frühere ähnliche
   Predictions? Volker tendiert zu Coin-Historie, hat es aber zuletzt als „andere Prediktionen"
   formuliert. Vor dem Bau bestätigen.
2. **Distanz-Metrik + ε + Mindest-Trefferzahl** empirisch kalibrieren.
3. **Klines-Auflösung** für den Durchlauf (1m vs 10s) — Performance vs Genauigkeit.
4. **Cold-Start:** Coins mit < 10–14 d Historie (neu gelistet) → wie behandeln?
5. **Interaktion mit Richtungs-Confidence:** ersetzt der Erreichbarkeits-Filter die Confidence-
   Schwelle oder kommt er obendrauf?

## Akzeptanzkriterium
Die getradeten Predictions erreichen ihr 0,8×Peak-TP tatsächlich zu ~80 % (über reale Trades
gemessen, nicht nur im Backtest). Wenn das hält → Volkers Vision erfüllt, egal wie viele Trades.
Offene empirische Frage: Gibt es überhaupt genug Setups, die die Doppelbedingung (80 % Peak +
80 % Trefferquote) erfüllen?
