// Candlestick Pattern Detection — korrekte Definitionen
import { CHART_SETTINGS } from '../../../config/chartSettings'

const T = CHART_SETTINGS.candlePatterns.thresholds

// === HELPER ===
function bodySize(c) { return Math.abs(c.close - c.open) }
function range(c) { const r = c.high - c.low; if (r <= 0) throw new Error(`invalid candle range at ${c.time}`); return r }
function bodyRatio(c) { return bodySize(c) / range(c) }
function isGreen(c) { return c.close > c.open }
function isRed(c) { return c.close < c.open }
function bodyTop(c) { return Math.max(c.open, c.close) }
function bodyBot(c) { return Math.min(c.open, c.close) }
function upperWick(c) { return c.high - bodyTop(c) }
function lowerWick(c) { return bodyBot(c) - c.low }
function bodyMid(c) { return (c.open + c.close) / 2 }

const DETECTORS = {
  // === SINGLE CANDLE ===

  doji: (candles, i) => {
    // Body < 5% der Range, Range muss existieren
    const c = candles[i]
    return range(c) > 0 && bodyRatio(c) < T.dojiBodyRatio
  },

  hammer: (candles, i) => {
    // Kleiner Body am oberen Ende, langer unterer Docht (>= 2x Body), kaum oberer Docht
    const c = candles[i]
    const body = bodySize(c)
    if (body <= 0 || range(c) === 0) return false
    return lowerWick(c) >= body * T.hammerWickRatio &&
      upperWick(c) <= body * 0.5 &&
      bodyRatio(c) > T.dojiBodyRatio // kein Doji
  },

  inverted_hammer: (candles, i) => {
    // Kleiner Body am unteren Ende, langer oberer Docht, kaum unterer Docht
    const c = candles[i]
    const body = bodySize(c)
    if (body <= 0 || range(c) === 0) return false
    return upperWick(c) >= body * T.hammerWickRatio &&
      lowerWick(c) <= body * 0.5 &&
      bodyRatio(c) > T.dojiBodyRatio
  },

  spinning_top: (candles, i) => {
    // Kleiner Body (< 30% Range), lange Dochte auf beiden Seiten (jeweils > Body)
    const c = candles[i]
    const body = bodySize(c)
    return bodyRatio(c) < 0.3 && bodyRatio(c) > T.dojiBodyRatio &&
      upperWick(c) > body && lowerWick(c) > body
  },

  marubozu_bull: (candles, i) => {
    // Gruene Kerze, fast keine Dochte (< 2% der Range)
    const c = candles[i]
    return isGreen(c) && range(c) > 0 &&
      upperWick(c) / range(c) < T.marubozuWickRatio &&
      lowerWick(c) / range(c) < T.marubozuWickRatio
  },

  marubozu_bear: (candles, i) => {
    // Rote Kerze, fast keine Dochte
    const c = candles[i]
    return isRed(c) && range(c) > 0 &&
      upperWick(c) / range(c) < T.marubozuWickRatio &&
      lowerWick(c) / range(c) < T.marubozuWickRatio
  },

  // === TWO CANDLE ===

  engulfing_bull: (candles, i) => {
    // Vorherige Kerze rot, aktuelle gruen
    // Gruener Body umschliesst komplett den roten Body
    if (i < 1) return false
    const prev = candles[i - 1], curr = candles[i]
    if (!isRed(prev) || !isGreen(curr)) return false
    return curr.open <= prev.close && curr.close >= prev.open &&
      bodySize(curr) > bodySize(prev) * 1.1 // mindestens 10% groesser
  },

  engulfing_bear: (candles, i) => {
    // Vorherige gruen, aktuelle rot
    // Roter Body umschliesst komplett den gruenen Body
    if (i < 1) return false
    const prev = candles[i - 1], curr = candles[i]
    if (!isGreen(prev) || !isRed(curr)) return false
    return curr.open >= prev.close && curr.close <= prev.open &&
      bodySize(curr) > bodySize(prev) * 1.1
  },

  harami_bull: (candles, i) => {
    // Grosse rote Kerze, dann kleine gruene die komplett im Body der roten liegt
    if (i < 1) return false
    const prev = candles[i - 1], curr = candles[i]
    if (!isRed(prev) || !isGreen(curr)) return false
    return bodyBot(curr) >= bodyBot(prev) && bodyTop(curr) <= bodyTop(prev) &&
      bodySize(curr) < bodySize(prev) * 0.6
  },

  harami_bear: (candles, i) => {
    // Grosse gruene Kerze, dann kleine rote die komplett im Body der gruenen liegt
    if (i < 1) return false
    const prev = candles[i - 1], curr = candles[i]
    if (!isGreen(prev) || !isRed(curr)) return false
    return bodyBot(curr) >= bodyBot(prev) && bodyTop(curr) <= bodyTop(prev) &&
      bodySize(curr) < bodySize(prev) * 0.6
  },

  tweezer_top: (candles, i) => {
    // Erste Kerze gruen (Aufwaertsbewegung), zweite rot (Umkehr)
    // Beide haben (fast) gleiches High
    // Beide Kerzen haben signifikanten Body
    if (i < 1) return false
    const prev = candles[i - 1], curr = candles[i]
    if (!isGreen(prev) || !isRed(curr)) return false
    if (bodyRatio(prev) < 0.15 || bodyRatio(curr) < 0.15) return false
    const tolerance = Math.max(prev.high, curr.high) * T.tweezerTolerance
    return Math.abs(prev.high - curr.high) <= tolerance
  },

  tweezer_bottom: (candles, i) => {
    // Erste Kerze rot (Abwaertsbewegung), zweite gruen (Umkehr)
    // Beide haben (fast) gleiches Low
    // Beide Kerzen haben signifikanten Body
    if (i < 1) return false
    const prev = candles[i - 1], curr = candles[i]
    if (!isRed(prev) || !isGreen(curr)) return false
    if (bodyRatio(prev) < 0.15 || bodyRatio(curr) < 0.15) return false
    const tolerance = Math.max(prev.low, curr.low) * T.tweezerTolerance
    return Math.abs(prev.low - curr.low) <= tolerance
  },

  piercing: (candles, i) => {
    // Rote Kerze, dann gruene die unter dem Low der roten oeffnet
    // und ueber der Mitte des roten Body schliesst (aber nicht darueber)
    if (i < 1) return false
    const prev = candles[i - 1], curr = candles[i]
    if (!isRed(prev) || !isGreen(curr)) return false
    const prevMid = bodyMid(prev)
    return curr.open <= prev.close &&
      curr.close > prevMid &&
      curr.close < prev.open // schliesst nicht ueber dem roten Open
  },

  dark_cloud: (candles, i) => {
    // Gruene Kerze, dann rote die ueber dem High der gruenen oeffnet
    // und unter der Mitte des gruenen Body schliesst (aber nicht darunter)
    if (i < 1) return false
    const prev = candles[i - 1], curr = candles[i]
    if (!isGreen(prev) || !isRed(curr)) return false
    const prevMid = bodyMid(prev)
    return curr.open >= prev.close &&
      curr.close < prevMid &&
      curr.close > prev.open // schliesst nicht unter dem gruenen Open
  },

  // === THREE CANDLE ===

  morning_star: (candles, i) => {
    // 1. Grosse rote Kerze
    // 2. Kleine Kerze (Star) — Body liegt unter dem Body der ersten (Gap down)
    // 3. Grosse gruene Kerze die ueber der Mitte der ersten schliesst
    if (i < 2) return false
    const a = candles[i - 2], b = candles[i - 1], c = candles[i]
    if (!isRed(a) || !isGreen(c)) return false
    if (bodyRatio(a) < 0.3 || bodyRatio(c) < 0.3) return false // a und c muessen signifikant sein
    if (bodyRatio(b) > 0.3) return false // Star muss klein sein
    // Gap: Star-Body-Top liegt unter dem Body-Bottom der ersten Kerze
    // Bei Crypto selten echte Gaps, daher: Star-Body-Mid unter Body-Bot von A
    if (bodyMid(b) > bodyBot(a)) return false
    // C schliesst ueber Mitte von A
    return c.close > bodyMid(a)
  },

  evening_star: (candles, i) => {
    // 1. Grosse gruene Kerze
    // 2. Kleine Kerze (Star) — Body liegt ueber dem Body der ersten (Gap up)
    // 3. Grosse rote Kerze die unter der Mitte der ersten schliesst
    if (i < 2) return false
    const a = candles[i - 2], b = candles[i - 1], c = candles[i]
    if (!isGreen(a) || !isRed(c)) return false
    if (bodyRatio(a) < 0.3 || bodyRatio(c) < 0.3) return false
    if (bodyRatio(b) > 0.3) return false
    // Star-Body-Mid ueber Body-Top von A
    if (bodyMid(b) < bodyTop(a)) return false
    return c.close < bodyMid(a)
  },

  three_white: (candles, i) => {
    // Drei aufeinanderfolgende gruene Kerzen
    // Jede schliesst hoeher als die vorherige
    // Jede oeffnet innerhalb des Body der vorherigen (nicht darueber)
    // Alle haben signifikanten Body
    if (i < 2) return false
    const a = candles[i - 2], b = candles[i - 1], c = candles[i]
    if (!isGreen(a) || !isGreen(b) || !isGreen(c)) return false
    if (bodyRatio(a) < 0.3 || bodyRatio(b) < 0.3 || bodyRatio(c) < 0.3) return false
    return b.close > a.close && c.close > b.close &&
      b.open >= a.open && b.open <= a.close && // B oeffnet im Body von A
      c.open >= b.open && c.open <= b.close    // C oeffnet im Body von B
  },

  three_black: (candles, i) => {
    // Drei aufeinanderfolgende rote Kerzen
    // Jede schliesst tiefer als die vorherige
    // Jede oeffnet innerhalb des Body der vorherigen
    // Alle haben signifikanten Body
    if (i < 2) return false
    const a = candles[i - 2], b = candles[i - 1], c = candles[i]
    if (!isRed(a) || !isRed(b) || !isRed(c)) return false
    if (bodyRatio(a) < 0.3 || bodyRatio(b) < 0.3 || bodyRatio(c) < 0.3) return false
    return b.close < a.close && c.close < b.close &&
      b.open <= a.open && b.open >= a.close && // B oeffnet im Body von A
      c.open <= b.open && c.open >= b.close    // C oeffnet im Body von B
  },
}

// Detect patterns in candle data
export function detectPatterns(candles, patternIds = null) {
  const results = {}
  const toCheck = patternIds || Object.keys(DETECTORS)
  for (const pid of toCheck) {
    const fn = DETECTORS[pid]
    if (!fn) continue
    results[pid] = []
    for (let i = 0; i < candles.length; i++) {
      if (fn(candles, i)) {
        results[pid].push({ index: i, candle: candles[i] })
      }
    }
  }
  return results
}

// Detect all patterns at a specific candle index
export function detectPatternsAt(candles, index) {
  const found = []
  for (const [pid, fn] of Object.entries(DETECTORS)) {
    if (fn(candles, index)) {
      const meta = CHART_SETTINGS.candlePatterns.available.find(p => p.id === pid)
      found.push({ id: pid, label: meta?.label || pid, description: meta?.description || '' })
    }
  }
  return found
}
