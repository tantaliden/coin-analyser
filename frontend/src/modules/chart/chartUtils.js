// Chart Utility Functions

// Felder die relativ zum Startwert normalisiert werden (Preise)
export const PRICE_FIELDS = ['close', 'open', 'high', 'low', 'priceChange']

// Value Fields für Overlay-Charts
export const VALUE_FIELDS = [
  { key: 'close', label: 'Close', color: '#3b82f6', type: 'base' },
  { key: 'volume', label: 'Volume', color: '#f59e0b', type: 'base' },
  { key: 'trades', label: 'Trades', color: '#22c55e', type: 'base' },
  { key: 'volatility', label: 'Volatility', color: '#ef4444', type: 'calc' },
  { key: 'body', label: 'Body', color: '#84cc16', type: 'calc' },
  { key: 'bodyPercent', label: 'Body%', color: '#14b8a6', type: 'calc' },
]

// Timeframes für Candle-Aggregation
export const CANDLE_TIMEFRAMES = [
  { key: '1m', label: '1m', minutes: 1 },
  { key: '5m', label: '5m', minutes: 5 },
  { key: '15m', label: '15m', minutes: 15 },
  { key: '30m', label: '30m', minutes: 30 },
  { key: '1h', label: '1h', minutes: 60 },
]

// Normalisierung: Preise -> % vom Start, andere -> Faktor vom Durchschnitt  
export const normalizeData = (aggData, field) => {
  if (!aggData.length) return []
  const isPriceField = PRICE_FIELDS.includes(field)
  if (isPriceField) {
    const baseValue = aggData[0]?.[field] || 1
    return aggData.map(c => ({ 
      time: c.relativeTime, 
      value: baseValue !== 0 ? ((c[field] - baseValue) / Math.abs(baseValue)) * 100 : 0 
    }))
  } else {
    const avg = aggData.reduce((sum, c) => sum + (c[field] || 0), 0) / aggData.length
    return aggData.map(c => ({ 
      time: c.relativeTime, 
      value: avg !== 0 ? (c[field] || 0) / avg : 0 
    }))
  }
}

// Candles aggregieren (z.B. 1m -> 5m)
export const aggregateCandles = (candles, tfMinutes) => {
  if (tfMinutes === 1) return candles
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

// Berechnete Werte (Volatility, Body, etc.)
export const calculateDerivedValues = (candles, avgPeriod = 15) => {
  return candles.map((c, idx) => {
    const startIdx = Math.max(0, idx - avgPeriod)
    const windowCandles = candles.slice(startIdx, idx + 1)
    const avgVolume = windowCandles.reduce((sum, x) => sum + x.volume, 0) / windowCandles.length
    const avgClose = windowCandles.reduce((sum, x) => sum + x.close, 0) / windowCandles.length
    
    return {
      ...c,
      volatility: c.high - c.low,
      body: c.close - c.open,
      bodyPercent: c.open !== 0 ? ((c.close - c.open) / c.open) * 100 : 0,
    }
  })
}

// Event-Farben
export const EVENT_COLORS = [
  '#3b82f6', '#22c55e', '#f59e0b', '#a855f7', '#ef4444',
  '#06b6d4', '#f97316', '#ec4899', '#84cc16', '#14b8a6',
  '#f43f5e', '#6366f1', '#78716c', '#0ea5e9', '#d946ef'
]

export const getEventColor = (index) => EVENT_COLORS[index % EVENT_COLORS.length]
