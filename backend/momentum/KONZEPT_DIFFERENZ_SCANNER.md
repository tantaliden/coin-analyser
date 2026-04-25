# Differenz-Scanner — Konzept (von Volker)

## Übersicht
Finde Events (≥5% Moves), analysiere die 12 Stunden DAVOR auf Anomalien,
vergleiche diese Anomalien zwischen Long und Short Events, baue Indikator-Sets.

---

## Schritt 1: Event-Erkennung (strenger als bisher)

Suche über alle Coins: Welche Coins hatten innerhalb von 30min/1h ein +/-5% Move?

**Wichtige Bedingung:** Wenn VOR dem Erreichen der +5% bereits die -5% erreicht wurde,
ist das KEIN gültiges Long-Event (und umgekehrt). Nur Events wo die Gegenseite
NICHT vorher erreicht wurde.

Beispiel:
- Preis geht erst -6%, dann +5% → KEIN Long-Event (weil -5% vorher erreicht)
- Preis geht seitwärts, dann +5% → GÜLTIGES Long-Event

Datenquelle: kline_metrics (pct_30m, pct_60m) + Prüfung auf Gegenextrem via lowest/highest.

---

## Schritt 2: Rohdaten speichern (12h vor Event)

Für jedes gültige Event:
- ALLE verfügbaren Rohdaten der letzten 12 Stunden VOR dem Event speichern
- agg_5m: 144 Candles (OHLCV, Trades, Taker Buy Volume)
- agg_1h: 12 Candles
- agg_4h: 3 Candles
- Long/Short Label mitgespeichern

Das ist die Datenbasis: Nicht ein einzelner Indikatorwert pro Event,
sondern eine ZEITREIHE von 12 Stunden mit allen Datenströmen.

---

## Schritt 3: Anomalie-Erkennung (pro Event, pro Datenstrom, über 12h)

Für JEDES Event die 12-Stunden-Zeitreihe durchscannen und Auffälligkeiten finden:

### Was sind Anomalien?
- Volumen-Spike: Plötzlicher Anstieg (z.B. ≥50% über Durchschnitt)
- Volumen-Stillstand: Plötzlich kaum noch Volumen
- Preis-Spike: Schnelle Bewegung in kurzer Zeit
- Trade-Anomalie: Viele Trades → plötzlich keine mehr (oder umgekehrt)
- Taker-Ratio-Shift: Plötzlich viel mehr Käufer oder Verkäufer
- Candle-Anomalien: Ungewöhnlich große Wicks, Doji nach Trend, etc.
- Volatilitäts-Wechsel: Ruhiger Markt → plötzlich volatil (oder umgekehrt)
- Momentum-Wechsel: Trend bricht ab, kehrt um

### Pro Anomalie speichern:
- **Typ**: Was für eine Anomalie (volume_spike, trade_halt, price_spike, etc.)
- **Stärke/Magnitude**: Wie stark (z.B. 50% Volumenanstieg, 2% Preissprung)
- **Zeit vor Event**: Wann relativ zum Event (z.B. 240 Minuten vorher)
- **Dauer**: Wie lange hielt die Anomalie an (z.B. 20 Minuten)
- **Richtung**: Aufwärts/Abwärts (bei Preis/Momentum relevant)

### "Normal" definieren:
- Normal = Durchschnitt der Zeitreihe (oder gleitender Durchschnitt)
- Anomalie = Wert weicht signifikant vom lokalen Normal ab
- Verschiedene Zeitfenster testen: Innerhalb von 5min, 15min, 30min, 1h, 2h

---

## Schritt 4: Vergleich über alle Events (Long vs Short)

Jetzt hat jedes der ~41k Events eine Liste von Anomalien mit Typ, Stärke, Zeitpunkt, Dauer.

### Für Long-Events:
- Welche Art von Anomalie taucht bei VIELEN Long-Events auf?
- Z.B.: "Bei 60% aller Longs gab es 2-6h vor Event einen Volumenanstieg von ≥50% für 20+ Minuten"
- Der ZEITRAUM und die DAUER werden dabei mit bestimmt, nicht vorher festgelegt

### Für Short-Events:
- Dasselbe, unabhängig von Long

### Gegenprobe:
- Taucht ein Long-Muster auch bei Short-Events auf?
- Wenn ja bei >10% der Shorts → Signal ist zu schwach, muss verfeinert oder verworfen werden
- Verfeinern: Bereich einengen (Stärke, Zeitfenster, Dauer anpassen)

### Ergebnis:
- Liste von validierten Anomalie-Mustern, jeweils mit:
  - Typ, Stärke-Bereich, Zeitfenster vor Event, Dauer
  - Häufigkeit bei Long (z.B. 65%)
  - Häufigkeit bei Short (z.B. 3%) → geringe Contamination

---

## Schritt 5: Kombinieren zu Indikator-Sets

- Einzelne Anomalie-Muster kombinieren
- Ziel: Bei Long kein Short, bei Short kein Long
- Mindestens 20% der Events müssen noch abgedeckt sein (Coverage ≥ 20%)
- Übereinstimmung durch die Indikatoren bei mindestens 90% (Purity ≥ 90%)
- Mehrere Muster zusammen → Contamination sinkt, Purity steigt

---

## Schritt 6: Neue Suche mit den Sets (Live-Scanner)

Die gefundenen Indikator-Sets haben Zeitfenster (z.B. "2-6h vor Event passiert X").
Diese kann man umdrehen für eine Vorwärts-Suche:

- Suche: Zeige mir in den letzten 12h wo diese Indikatoren aufgetreten sind
- Wenn Indikator-Set matcht → Erwartung: In den nächsten 0-12h kommt ein Event
- Richtung (Long/Short) ist bekannt aus dem Set

---

## Schritt 7: Ergebnisse / Validierung

- Alle Ergebnisse über denselben Zeitraum wie die Event-Suche anzeigen
- Mit Long/Short Vorhersagen
- Backtest: Stimmen die Vorhersagen mit den tatsächlichen Events überein?

---

## Verfügbare Daten

- **agg_5m**: 5-Min-Candles (OHLCV, number_of_trades, taker_buy_base_asset_volume)
  - 144 Candles pro 12h-Fenster
  - 2.8M Rows/30d, 285 Symbole
- **agg_1h**: 1h-Candles
  - 12 Candles pro 12h-Fenster
- **agg_4h**: 4h-Candles
  - 3 Candles pro 12h-Fenster
- **agg_1d**: Tages-Candles (Kontext)
- **kline_metrics**: pct_30m bis pct_600m (für Event-Erkennung)

## Rechenaufwand

- ~41k Events × 144 5min-Candles × mehrere Datenströme × Anomalie-Erkennung = STUNDEN
- Vergleich der Anomalien über alle Events: STUNDEN
- Kombinations-Tests: STUNDEN
- Das ist gewollt und notwendig für gründliche Analyse
