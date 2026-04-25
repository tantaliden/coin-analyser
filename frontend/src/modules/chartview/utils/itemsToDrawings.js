// Rekonstruiert Drawing-Objekte aus geladenen indicator_items.
// Wird vom Load-Flow aufgerufen — laedt Set -> Drawings erscheinen im Chart.
// Keine Fallbacks — fehlende Felder werden als explizite Fehler in der Konsole gezeigt.

export function itemsToDrawings(loadedDrawings, candles, dayOpen) {
  if (!Array.isArray(loadedDrawings)) throw new Error('loadedDrawings muss Array sein')
  if (!candles || candles.length === 0) throw new Error('candles fehlen')
  if (dayOpen == null) throw new Error('dayOpen fehlt')

  const result = []
  for (const item of loadedDrawings) {
    const d = itemToDrawing(item, candles, dayOpen)
    if (d) result.push(d)
  }
  return result
}

function itemToDrawing(item, candles, dayOpen) {
  const timeStart = item.time_start_minutes
  const timeEnd = item.time_end_minutes

  // Zeit-Offset (Minuten) -> candleIdx
  const idxStart = findCandleIdxByOffset(candles, timeStart)
  const idxEnd = findCandleIdxByOffset(candles, timeEnd)

  if (idxStart < 0) return null

  // Initialpunkt
  if (item.is_initial_point) {
    return {
      type: 'initialPoint',
      candleIdx: idxStart,
      color: '#fbbf24',
      width: 2,
      loaded_item_id: item.item_id,
      fuzzy_config: item.fuzzy_config,
    }
  }

  // Candle-Pattern
  if (item.indicator_type === 'candle_pattern') {
    return {
      type: 'candleHighlight',
      candleIdx: idxStart,
      color: '#a855f7',
      width: 2,
      pattern_data: item.pattern_data,
      loaded_item_id: item.item_id,
      fuzzy_config: item.fuzzy_config,
    }
  }

  // Slope (Anstieg zwischen zwei Punkten)
  if (item.condition_operator === 'slope' || (idxStart !== idxEnd && item.condition_value != null)) {
    return {
      type: 'trendline',
      idx1: idxStart, idx2: idxEnd,
      price1: candles[idxStart].close, price2: candles[idxEnd]?.close || candles[idxStart].close,
      color: '#3b82f6', width: 2,
      loaded_item_id: item.item_id,
      fuzzy_config: item.fuzzy_config,
    }
  }

  // Range (between) -> priceRange
  if (item.condition_operator === 'between') {
    const pct1 = item.condition_value
    const pct2 = item.condition_value2
    const price1 = dayOpen * (1 + pct1 / 100)
    const price2 = dayOpen * (1 + pct2 / 100)
    return {
      type: 'priceRange',
      idx1: idxStart,
      price1, price2,
      color: '#22c55e', width: 2,
      loaded_item_id: item.item_id,
      fuzzy_config: item.fuzzy_config,
    }
  }

  // Value -> Horizontale Linie
  if (item.condition_value != null) {
    const price = dayOpen * (1 + item.condition_value / 100)
    return {
      type: 'hline',
      price, color: '#f59e0b', width: 2,
      loaded_item_id: item.item_id,
      fuzzy_config: item.fuzzy_config,
    }
  }

  console.warn('itemsToDrawings: unbekannter Typ fuer Item', item)
  return null
}

function findCandleIdxByOffset(candles, offsetMinutes) {
  // relativeTime ist in Sekunden. offsetMinutes * 60 = Sekunden.
  const targetSec = offsetMinutes * 60
  let bestIdx = -1, bestDiff = Infinity
  for (let i = 0; i < candles.length; i++) {
    const diff = Math.abs(candles[i].relativeTime - targetSec)
    if (diff < bestDiff) { bestDiff = diff; bestIdx = i }
  }
  return bestIdx
}
