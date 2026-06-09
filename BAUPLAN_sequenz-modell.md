# BAUPLAN — Sequenz-Modell (Trend-Event-Predictor)

**Stand:** 2026-06-02 · **Auslöser:** Schnappschuss-Predictor (LogReg/ARF) hat keinen Richtungs-Edge
(reine Richtung 36–39 % richtig, Drift ≈0, Confidence wertlos/invers — auf BEIDEN Servern, auch
voll trainiert mit 25.650 obs). Volkers These bestätigt: das Modell ist *strukturell* falsch —
Momentaufnahme statt Sequenz. Sein manueller Edge ist zeitlich (Trend-Beginn erkennen, früh rein,
dynamisch raus bevor er dreht). Genau das soll das neue Modell lernen.

## Ziel (Volkers Worte)
- **Wenige, größtenteils richtige** Events mit + nach Gebühren. 20–200 Predictions/Tag sind ok.
- **Kein** Millisekunden-/Microtrading. Events die sich *aufbauen*.
- Früher Einstieg, dynamischer Ausstieg bevor der Trend kippt.

## Architektur-Entscheidung: WIEDERVERWENDEN, nicht neu
- **Backbone:** `MultiTimeframeCNN` aus `/opt/training/scripts/train_cnn_progressive.py`
  (Multi-Timeframe-Conv1d, Fenster-basiert) — ist bereits sequenzfähig.
- **NEU:** Labeling, Daten-Pipeline (rohe klines statt 57 RL-Features), Skip-lastige Selektion,
  ehrlicher Backtest mit Gebühren.
- Der alte Schnappschuss-Predictor (`predictor_service`) bleibt unangetastet weiterlaufen
  (Vergleichsbasis), bis das Sequenz-Modell ihn nachweislich schlägt.

## Daten
- `coins.klines`, 1-Min, 30 Spalten (OHLCV + funding + OI + premium + bbo + book_imbalance/depth).
- Tiefe: ab **24.04.2026** (~39 Tage) × 230 Coins. **Ehrlich:** dünn für robuste Generalisierung,
  reicht für Proof-of-Concept. Falls PoC trägt → HL-Historie backfillen BEVOR wir vertrauen.
- Fenster (Vorschlag, anpassbar): 60×1m + 48×5m + 24×15m (Minuten bis Stunden = „aufbauende Events").
- Channels/Bar (Vorschlag): ~12 kuratiert (returns, range, volume, taker-imbalance, funding, OI-Δ,
  book_imbalance, spread) — **pro Fenster normalisiert** (kein Kurs-Level-Leak).

## Volker-Vorgaben (02.06.2026, VERBINDLICH)
- **Nur klare Moves: 3–5 %.** 0,5 %-Teile sind KEINE Events — gar nicht erst aufnehmen.
- **TP/SL müssen resolven:** jeder Trade endet an TP **oder** SL. Kein Rumdriften, kein
  Breakeven-Timeout. „TP/SL nicht nur zum Spaß da" — eines von beiden wird IMMER erreicht.

## Labeling (hier lebt oder stirbt es) — TP/SL-Resolution
- Entry an t. **TP = +3 % / SL = −1,5 %** (R:R 2:1, anpassbar). Halten bis eines hittet
  (max. Horizont 12 h; Ziel: Auflösungsquote ~100 %).
- `win` = TP vor SL erreicht (+3 % − Gebühr). `loss` = SL vor TP (−1,5 % − Gebühr).
- **Event** = Kandidat, an dem es einen klaren 3 %-Move (long ODER short) VOR dem 1,5 %-Gegen-Move
  gibt. Label = diese Richtung. Sonst `skip`.
- 3 Klassen `long` / `short` / `skip`, **skip-lastig** → Modell schweigt, feuert nur bei Überzeugung.
- Das Modell sagt also vorher: „hier kommt ein sauberer 3 %-Move in Richtung X, bevor es 1,5 %
  gegen mich läuft" — genau Volkers manuelles Vorgehen.

## Phasen mit ehrlichen GO/NO-GO-Gates
**Das Ziel des Plans ist FAIL-FAST — nicht noch 9 Monate Hoffnung.** An jedem Gate echte
Dollar-Zahlen, nicht Gefühl.

### Phase 0 — Daten- & Label-Pipeline (offline)
- Fenster-Extraktor + Label-Simulator bauen. Walk-forward-Split (früh=train, spät=test, KEIN Leak).
- Output: Datensatz-Statistik (wie viele long/short/skip, Move-Verteilung).
- **Gate 0:** Gibt es überhaupt genug ride-able Events (Netto > Marge)? Wenn die Marktstruktur fast
  nur skip liefert → frühes ehrliches Stopp.

### Phase 1 — Backtest-Harness mit Gebühren (VOR jedem Training)
- Realistische Kosten: HL-Taker ~0,045 %×2 + Slippage, Hebel berücksichtigt.
- Erst Baseline: was bringt das LABEL selbst (perfektes Modell) netto in $? Obergrenze des Möglichen.
- **Gate 1:** Wenn selbst das perfekte Label nach Gebühren nicht klar + ist → das Konzept trägt
  nicht, Stopp & Pivot (mechanische Edges/Funding).

### Phase 2 — Sequenz-Modell trainieren + ehrlich evaluieren
- `MultiTimeframeCNN` auf die Fenster/Labels. Out-of-sample (späte Tage).
- Kernmetrik — **die, an der der alte Predictor scheitert:** steigt die WR mit der Modell-Confidence?
  (Kalibrierungskurve). Plus: Netto-$-Expectancy, Trades/Tag.
- **Gate 2:** Nur weiter wenn (a) Kalibrierung monoton (high-conviction → höhere WR) UND
  (b) selektierte Trades netto + nach Gebühren, out-of-sample.

### Phase 3 — Paper-Shadow (live, kein Echtgeld)
- Über vorhandene Paper-Engine mitlaufen lassen, gegen Backtest-Erwartung prüfen
  (Realität = Backtest? sonst Leak/Bug).
- **Gate 3:** Paper-$ ≈ Backtest-$ über mind. 2 Wochen.

### Phase 4 — Live (nur nach Gate 3)
- Klein anfangen, dynamischer Exit, harte Stop-Bedingungen.

## Risiken (ehrlich)
- 39 Tage Daten sind dünn → Überanpassung. Walk-forward + ggf. Backfill nötig.
- Markt kann auf diesem Horizont schlicht nicht prognostizierbar sein — dann zeigen Gate 1/2 das
  früh und billig, statt nach Monaten.
- Dynamischer Exit ist ein zweites Lernproblem (RL-Agent kann hier andocken).

## Erster konkreter Schritt
Phase 0 + 1 zusammen: Pipeline + Backtest-Harness, und **Gate 1 rechnen** (perfektes Label nach
Gebühren). Das ist der erste ehrliche Wahrheitstest — bevor ein einziges Netz trainiert wird.
