// Berechnet Mapping-Funktionen (Pixel <-> Chart-Koordinaten).
// Alle Layout-Werte aus CHART_LAYOUT (chartSettings.js).
import { CHART_LAYOUT } from '../../../config/chartSettings'

export function computeChartMappings(candles, width, height, showVolume, separateIndCount = 0) {
  if (!candles?.length || !width || !height) return null

  const L = CHART_LAYOUT
  const paneCount = (showVolume ? 1 : 0) + separateIndCount
  const paneH = paneCount > 0 ? Math.min(L.paneHMax, (height - L.timeAxisH) * L.paneHRatio) : 0
  const mainH = height - L.timeAxisH - paneH * paneCount
  const chartW = width - L.priceAxisW

  let minP = Infinity, maxP = -Infinity
  for (const c of candles) {
    if (c.low < minP) minP = c.low
    if (c.high > maxP) maxP = c.high
  }
  if (maxP === minP) throw new Error(`all candles have same price: ${minP}`)
  const pad = (maxP - minP) * L.pricePadding
  minP -= pad; maxP += pad
  const totalPR = maxP - minP

  const cStep = chartW / candles.length
  const cW = Math.max(1, cStep * L.candleBodyRatio)

  const pToY = (p) => mainH * (1 - (p - minP) / totalPR)
  const yToP = (y) => minP + (1 - y / mainH) * totalPR
  const iToX = (i) => i * cStep + cStep / 2
  const xToI = (x) => Math.max(0, Math.min(candles.length - 1, Math.round((x - cStep / 2) / cStep)))

  return {
    pToY, yToP, iToX, xToI,
    minP, maxP, totalPR,
    cStep, cW,
    chartW, chartH: mainH,
    paneH, paneCount,
    timeAxisH: L.timeAxisH, priceAxisW: L.priceAxisW,
  }
}
