// Indikator-Berechnungen — geprueft und korrigiert

function sma(data, period) {
  const result = []
  for (let i = period - 1; i < data.length; i++) {
    let sum = 0
    for (let j = i - period + 1; j <= i; j++) sum += data[j].close
    result.push({ idx: i, value: sum / period })
  }
  return result
}

function ema(data, period) {
  const result = []
  const k = 2 / (period + 1)
  let prev = null
  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) continue
    if (prev === null) {
      let sum = 0
      for (let j = 0; j < period; j++) sum += data[j].close
      prev = sum / period
    } else {
      prev = data[i].close * k + prev * (1 - k)
    }
    result.push({ idx: i, value: prev })
  }
  return result
}

// EMA aus beliebigen Werten (nicht nur close) — fuer MACD Signal
function emaFromValues(values, period) {
  if (values.length < period) return []
  const k = 2 / (period + 1)
  let prev = null
  const result = []
  for (let i = 0; i < values.length; i++) {
    if (i < period - 1) continue
    if (prev === null) {
      let sum = 0
      for (let j = 0; j < period; j++) sum += values[j].value
      prev = sum / period
    } else {
      prev = values[i].value * k + prev * (1 - k)
    }
    result.push({ idx: values[i].idx, value: prev })
  }
  return result
}

function rsi(data, period) {
  // Wilder's RSI mit korrekter Edge-Case Behandlung
  const result = []
  let avgGain = 0, avgLoss = 0
  for (let i = 1; i < data.length; i++) {
    const change = data[i].close - data[i - 1].close
    const gain = change > 0 ? change : 0
    const loss = change < 0 ? -change : 0

    if (i <= period) {
      avgGain += gain
      avgLoss += loss
      if (i === period) {
        avgGain /= period
        avgLoss /= period
        // Edge cases
        if (avgGain === 0 && avgLoss === 0) {
          result.push({ idx: i, value: 50 }) // Kein Movement = neutral
        } else if (avgLoss === 0) {
          result.push({ idx: i, value: 100 }) // Nur Gains = max bullish
        } else {
          result.push({ idx: i, value: 100 - 100 / (1 + avgGain / avgLoss) })
        }
      }
    } else {
      // Wilder's Smoothing
      avgGain = (avgGain * (period - 1) + gain) / period
      avgLoss = (avgLoss * (period - 1) + loss) / period
      if (avgGain === 0 && avgLoss === 0) {
        result.push({ idx: i, value: 50 })
      } else if (avgLoss === 0) {
        result.push({ idx: i, value: 100 })
      } else {
        result.push({ idx: i, value: 100 - 100 / (1 + avgGain / avgLoss) })
      }
    }
  }
  return result
}

function macd(data, fast, slow, signal) {
  // MACD = Fast EMA - Slow EMA
  const fastEma = ema(data, fast)
  const slowEma = ema(data, slow)

  // Align: beide Arrays auf gleiche Laenge bringen (vom Ende her)
  const minLen = Math.min(fastEma.length, slowEma.length)
  const fa = fastEma.slice(fastEma.length - minLen)
  const sa = slowEma.slice(slowEma.length - minLen)

  const macdLine = fa.map((f, i) => ({
    idx: f.idx,
    value: f.value - sa[i].value,
  }))

  // Signal = EMA der MACD-Linie (NICHT SMA — das war der Bug)
  const signalLine = emaFromValues(macdLine, signal)

  return { values: macdLine, signal: signalLine }
}

function bollinger(data, period, stddev) {
  const mid = sma(data, period)
  const upper = [], lower = []
  for (let i = 0; i < mid.length; i++) {
    const dataIdx = mid[i].idx
    let sumSq = 0
    for (let j = dataIdx - period + 1; j <= dataIdx; j++) {
      sumSq += (data[j].close - mid[i].value) ** 2
    }
    const std = Math.sqrt(sumSq / period) // Population variance (Standard fuer Bollinger)
    upper.push({ idx: dataIdx, value: mid[i].value + stddev * std })
    lower.push({ idx: dataIdx, value: mid[i].value - stddev * std })
  }
  return { upper, middle: mid, lower }
}

function vwap(data) {
  const result = []
  let cumVP = 0, cumV = 0
  for (let i = 0; i < data.length; i++) {
    const tp = (data[i].high + data[i].low + data[i].close) / 3
    cumVP += tp * data[i].volume
    cumV += data[i].volume
    result.push({ idx: i, value: cumV > 0 ? cumVP / cumV : tp })
  }
  return result
}

function atr(data, period) {
  // Wilder's ATR: erster Wert = SMA von True Range, danach Wilder's Smoothing
  const result = []
  const trs = []
  let prevATR = null

  for (let i = 0; i < data.length; i++) {
    const c = data[i]
    const pc = i > 0 ? data[i - 1].close : c.open
    const tr = Math.max(c.high - c.low, Math.abs(c.high - pc), Math.abs(c.low - pc))
    trs.push(tr)

    if (i === period - 1) {
      // Erster ATR = SMA der True Ranges
      let sum = 0
      for (let j = 0; j < period; j++) sum += trs[j]
      prevATR = sum / period
      result.push({ idx: i, value: prevATR })
    } else if (i >= period) {
      // Wilder's Smoothing: ATR = (prevATR * (period-1) + TR) / period
      prevATR = (prevATR * (period - 1) + tr) / period
      result.push({ idx: i, value: prevATR })
    }
  }
  return result
}

function stochastic(data, kPeriod, dPeriod) {
  const kValues = []
  for (let i = kPeriod - 1; i < data.length; i++) {
    let high = -Infinity, low = Infinity
    for (let j = i - kPeriod + 1; j <= i; j++) {
      if (data[j].high > high) high = data[j].high
      if (data[j].low < low) low = data[j].low
    }
    const range = high - low || 1
    kValues.push({ idx: i, value: ((data[i].close - low) / range) * 100 })
  }
  // %D = SMA of %K (Standard)
  const dValues = []
  for (let i = dPeriod - 1; i < kValues.length; i++) {
    let sum = 0
    for (let j = i - dPeriod + 1; j <= i; j++) sum += kValues[j].value
    dValues.push({ idx: kValues[i].idx, value: sum / dPeriod })
  }
  return { values: kValues, signal: dValues }
}

function volumeSma(data, period) {
  const result = []
  for (let i = period - 1; i < data.length; i++) {
    let sum = 0
    for (let j = i - period + 1; j <= i; j++) sum += data[j].volume
    result.push({ idx: i, value: sum / period })
  }
  return result
}

function tradesLine(data) {
  return data.map((c, i) => { if (c.trades == null) throw new Error(`trades missing at candle ${i}`); return { idx: i, value: c.trades } })
}

export function computeIndicator(indicator, candles) {
  if (!candles || candles.length < 2) return null
  const p = indicator.params || {}
  switch (indicator.type) {
    case 'sma': { if (p.period == null) throw new Error('sma requires period'); return { values: sma(candles, p.period) } }
    case 'ema': { if (p.period == null) throw new Error('ema requires period'); return { values: ema(candles, p.period) } }
    case 'rsi': { if (p.period == null) throw new Error('rsi requires period'); return { values: rsi(candles, p.period) } }
    case 'macd': { if (p.fast == null || p.slow == null || p.signal == null) throw new Error('macd requires fast/slow/signal'); return macd(candles, p.fast, p.slow, p.signal) }
    case 'bollinger': { if (p.period == null || p.stddev == null) throw new Error('bollinger requires period/stddev'); return bollinger(candles, p.period, p.stddev) }
    case 'vwap': return { values: vwap(candles) }
    case 'atr': { if (p.period == null) throw new Error('atr requires period'); return { values: atr(candles, p.period) } }
    case 'volume_sma': { if (p.period == null) throw new Error('volume_sma requires period'); return { values: volumeSma(candles, p.period) } }
    case 'trades': return { values: tradesLine(candles) }
    case 'stochastic': { if (p.kPeriod == null || p.dPeriod == null) throw new Error('stochastic requires kPeriod/dPeriod'); return stochastic(candles, p.kPeriod, p.dPeriod) }
    default: return null
  }
}
