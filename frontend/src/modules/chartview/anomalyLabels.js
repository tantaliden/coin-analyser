// Labels fuer Anomalie-Metriken und Candle-Muster. Einzige Quelle fuer UI-Beschriftung.
export const ANOMALY_BUCKET_OPTIONS = [1, 2, 5, 10, 15, 30, 60, 120]

export const METRIC_LABELS = {
  volume: 'Volume', trades: 'Trades',
  body_pct: 'Body %', range_pct: 'Range %',
  upper_wick_pct: 'Oberer Docht', lower_wick_pct: 'Unterer Docht',
  close_delta_pct: 'Close Δ %',
  rsi_14: 'RSI (14)', macd_hist: 'MACD Histogram',
  bollinger_pos: 'Bollinger-Position', atr_14: 'ATR (14)',
}

export const PATTERN_LABELS = {
  doji: 'Doji', hammer: 'Hammer', inverted_hammer: 'Inverted Hammer',
  spinning_top: 'Spinning Top', marubozu_bull: 'Marubozu Bull', marubozu_bear: 'Marubozu Bear',
  engulfing_bull: 'Engulfing Bull', engulfing_bear: 'Engulfing Bear',
  harami_bull: 'Harami Bull', harami_bear: 'Harami Bear',
  piercing: 'Piercing', dark_cloud: 'Dark Cloud',
  tweezer_top: 'Tweezer Top', tweezer_bottom: 'Tweezer Bottom',
  three_white: 'Three White Soldiers', three_black: 'Three Black Crows',
  morning_star: 'Morning Star', evening_star: 'Evening Star',
}

// Window-Familien (Praefix -> lesbarer Name)
const WINDOW_FAMILY_LABELS = {
  volume_sum: 'Volume-Summe',
  trades_sum: 'Trades-Summe',
  close_pct: 'Close-%-Sprung',
  rsi_slope: 'RSI-Steigung',
  macd_slope: 'MACD-Steigung',
}

function parseWindowMetric(metric) {
  // z.B. 'volume_sum_5' -> {family: 'volume_sum', window: '5'}
  for (const fam of Object.keys(WINDOW_FAMILY_LABELS)) {
    const prefix = fam + '_'
    if (metric.startsWith(prefix)) {
      const window = metric.slice(prefix.length)
      if (/^\d+$/.test(window)) return { family: fam, window }
    }
  }
  return null
}

export function labelFor(metric) {
  if (metric.startsWith('pattern:')) {
    const pid = metric.slice(8)
    return `Pattern · ${PATTERN_LABELS[pid] || pid}`
  }
  if (metric.startsWith('anomaly:')) return labelFor(metric.slice(8))
  const win = parseWindowMetric(metric)
  if (win) return `${WINDOW_FAMILY_LABELS[win.family]} (${win.window})`
  return METRIC_LABELS[metric] || metric
}
