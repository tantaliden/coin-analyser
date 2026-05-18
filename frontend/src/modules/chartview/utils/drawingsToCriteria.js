// Konvertiert Chart-Zeichnungen in Suchkriterien fuer die Gegensuche.
// Werte werden RELATIV zum Tagesanfangswert (00:00 Berlin) gespeichert.
// So sind Coins mit unterschiedlichen Preis-Groessenordnungen vergleichbar.

function idxToTimeOffset(candleIdx, candles) {
  if (!candles?.length || candleIdx < 0 || candleIdx >= candles.length) {
    throw new Error(`candleIdx ${candleIdx} out of range`)
  }
  return Math.round(candles[candleIdx].relativeTime / 60)
}

function pctFromDayOpen(price, dayOpen) {
  if (!dayOpen) throw new Error('dayOpen missing')
  return ((price - dayOpen) / dayOpen) * 100
}

/**
 * Extrahiert Kriterien aus einer Liste von Zeichnungen eines Charts.
 * Braucht zusaetzlich dayOpen fuer die % Berechnung.
 */
export function drawingsToCriteria(drawings, candles, dayOpen) {
  if (!drawings || !candles) throw new Error('drawings and candles required')
  if (!dayOpen) throw new Error('dayOpen required for relative criteria')
  if (drawings.length === 0 || candles.length === 0) return []
  const criteria = []

  for (const d of drawings) {
    if (d.type === 'initialPoint') {
      if (d.candleIdx == null || !candles[d.candleIdx]) continue
      criteria.push({
        kind: 'value',
        field: 'close',
        value: 0,
        time_offset: idxToTimeOffset(d.candleIdx, candles),
        is_initial_point: true,
        initial_fixed_offset: idxToTimeOffset(d.candleIdx, candles),
        meta: { drawingType: 'initialPoint', label: 'Initialpunkt (Zeit-Anker)' },
      })
      continue
    }
    switch (d.type) {
      case 'buyMarker':
      case 'sellMarker': {
        // 2-Click Trade: speichert Anstieg + Zeitdifferenz (slope-Kriterium)
        if (d.startIdx == null || d.endIdx == null) break
        const startCandle = candles[d.startIdx]
        const endCandle = candles[d.endIdx]
        if (!startCandle || !endCandle) break
        const isBuy = d.type === 'buyMarker'
        const startPrice = isBuy ? startCandle.low : startCandle.high
        const endPrice = isBuy ? endCandle.high : endCandle.low
        const pctChange = ((endPrice - startPrice) / startPrice) * 100
        criteria.push({
          kind: 'slope',
          field: 'close',
          value: pctChange,
          time_offset: idxToTimeOffset(d.startIdx, candles),
          time_offset2: idxToTimeOffset(d.endIdx, candles),
          meta: {
            drawingType: d.type,
            label: `${isBuy ? 'Buy' : 'Sell'}-Trade ${pctChange >= 0 ? '+' : ''}${pctChange.toFixed(2)}%`,
          },
        })
        break
      }

      case 'hline':
      case 'support':
      case 'resistance': {
        // Preislinie: % vom Tagesanfangswert (zeit-unabhaengig)
        if (d.price == null) break
        const pct = pctFromDayOpen(d.price, dayOpen)
        criteria.push({
          kind: 'value',
          field: 'close_pct_dayopen',
          value: pct,
          time_offset: 0,
          meta: {
            drawingType: d.type,
            label: `${d.type === 'support' ? 'Support' : d.type === 'resistance' ? 'Resistance' : 'Linie'} ${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%`,
          },
        })
        break
      }

      case 'priceRange':
      case 'rect': {
        // Wertbereich in % vom Tagesanfang
        if (d.price1 == null || d.price2 == null) break
        const pct1 = pctFromDayOpen(d.price1, dayOpen)
        const pct2 = pctFromDayOpen(d.price2, dayOpen)
        const lo = Math.min(pct1, pct2)
        const hi = Math.max(pct1, pct2)
        const offset = d.idx1 != null ? idxToTimeOffset(d.idx1, candles) : 0
        criteria.push({
          kind: 'range',
          field: 'close_pct_dayopen',
          time_offset: offset,
          fuzzy: { useRange: true, rangeMin: lo, rangeMax: hi },
          meta: {
            drawingType: d.type,
            label: `Bereich ${lo.toFixed(2)}%..${hi.toFixed(2)}% vom Tagesanfang`,
          },
        })
        break
      }

      case 'trendline':
      case 'measure': {
        // Zwei-Punkt-Anstieg: nur Anstieg + Zeitdifferenz, kein Tagesbezug
        if (d.price1 == null || d.price2 == null || d.idx1 == null || d.idx2 == null) break
        const t1 = idxToTimeOffset(d.idx1, candles)
        const t2 = idxToTimeOffset(d.idx2, candles)
        const pct = ((d.price2 - d.price1) / d.price1) * 100
        criteria.push({
          kind: 'slope',
          field: 'close',
          value: pct,
          time_offset: t1,
          time_offset2: t2,
          meta: { drawingType: d.type, label: `Anstieg ${pct >= 0 ? '+' : ''}${pct.toFixed(2)}% in ${t2 - t1}m` },
        })
        break
      }

      case 'candleHigh':
      case 'candleLow': {
        if (d.candleIdx == null || !candles[d.candleIdx]) break
        const c = candles[d.candleIdx]
        const val = d.type === 'candleHigh' ? c.high : c.low
        const pct = pctFromDayOpen(val, dayOpen)
        criteria.push({
          kind: 'value',
          field: d.type === 'candleHigh' ? 'high_pct_dayopen' : 'low_pct_dayopen',
          value: pct,
          time_offset: idxToTimeOffset(d.candleIdx, candles),
          meta: { drawingType: d.type, label: `${d.type === 'candleHigh' ? 'High' : 'Low'} ${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%` },
        })
        break
      }

      case 'candleRange':
      case 'wickToWick': {
        if (d.startIdx == null || d.endIdx == null || !candles[d.startIdx] || !candles[d.endIdx]) break
        const c1 = candles[d.startIdx], c2 = candles[d.endIdx]
        const pct = ((c2.close - c1.close) / c1.close) * 100
        criteria.push({
          kind: 'slope',
          field: 'close',
          value: pct,
          time_offset: idxToTimeOffset(d.startIdx, candles),
          time_offset2: idxToTimeOffset(d.endIdx, candles),
          meta: { drawingType: d.type, label: `Candle-Range ${pct.toFixed(2)}%` },
        })
        break
      }
    }
  }

  return criteria
}

/**
 * Sammelt Kriterien aus allen Charts (oder nur aktivem).
 */
export function collectCriteriaFromAllCharts(drawingsMap, chartDataMap, activeChartId = null) {
  const result = []
  const chartIds = activeChartId ? [activeChartId] : Object.keys(drawingsMap)

  for (const cid of chartIds) {
    const drawings = drawingsMap[cid]
    const data = chartDataMap[cid]
    if (!drawings || !data?.candles || !data?.dayOpen) continue
    const criteria = drawingsToCriteria(drawings, data.candles, data.dayOpen)
    result.push(...criteria.map(c => ({ ...c, eventId: cid })))
  }

  return result
}

/**
 * Trennt Kriterien in Initialpunkte und Prae-Event-Kriterien.
 * Initialpunkte: is_initial_point === true (aus 'initialPoint' drawing).
 */
export function splitInitialAndCriteria(allCriteria) {
  const initialPoints = []
  const criteria = []
  for (const c of allCriteria) {
    if (c.is_initial_point) initialPoints.push(c)
    else criteria.push(c)
  }
  return { initialPoints, criteria }
}

