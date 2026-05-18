// Candle-Daten Utilities

export const aggregateCandles = (candles, tfMinutes) => {
  if (tfMinutes <= 1) return candles
  const aggregated = []
  for (let i = 0; i < candles.length; i += tfMinutes) {
    const chunk = candles.slice(i, i + tfMinutes)
    if (chunk.length === 0) continue
    aggregated.push({
      time: chunk[0].time,
      relativeTime: chunk[0].relativeTime,
      open: chunk[0].open,
      high: Math.max(...chunk.map(c => c.high)),
      low: Math.min(...chunk.map(c => c.low)),
      close: chunk[chunk.length - 1].close,
      volume: chunk.reduce((sum, c) => sum + c.volume, 0),
      trades: chunk.reduce((sum, c) => sum + (c.trades || 0), 0),
    })
  }
  return aggregated
}

export const calculateDerivedValues = (candles) => {
  return candles.map(c => ({
    ...c,
    volatility: c.high - c.low,
    body: c.close - c.open,
    bodyPercent: c.open !== 0 ? ((c.close - c.open) / c.open) * 100 : 0,
    range: c.high - c.low,
    upperWick: c.close >= c.open ? c.high - c.close : c.high - c.open,
    lowerWick: c.close >= c.open ? c.open - c.low : c.close - c.low,
  }))
}

export const normalizeToPercent = (candles, field = 'close') => {
  if (!candles.length) return []
  const base = candles[0]?.[field] || 1
  return candles.map(c => ({
    time: c.relativeTime,
    value: base !== 0 ? ((c[field] - base) / Math.abs(base)) * 100 : 0,
  }))
}
