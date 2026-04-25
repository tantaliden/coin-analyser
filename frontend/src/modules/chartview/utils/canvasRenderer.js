// Reiner Canvas-Renderer mit Sub-Panes fuer separate Indikatoren
import { CHART_SETTINGS, MARKER_COLORS, CHART_LAYOUT, CHART_FONTS } from '../../../config/chartSettings'

export function renderChart(ctx, candles, options) {
  const {
    width, height, chartType = 'candle', showVolume = true,
    eventColor = CHART_SETTINGS.eventColors[0], indicators = [],
    crosshairPos = null, patternHighlights = [],
    remoteCrosshairIdx = null,
  } = options

  if (!candles?.length || !width || !height) return

  const C = CHART_SETTINGS.colors
  const L = CHART_LAYOUT
  const timeAxisH = L.timeAxisH
  const priceAxisW = L.priceAxisW

  // Separate indicators get their own panes below main chart
  const separateInds = indicators.filter(i => i.visible && i.config?.separate)
  const overlayInds = indicators.filter(i => i.visible && !i.config?.separate)
  const paneCount = (showVolume ? 1 : 0) + separateInds.length
  const paneH = paneCount > 0 ? Math.min(L.paneHMax, (height - timeAxisH) * L.paneHRatio) : 0
  const mainH = height - timeAxisH - paneH * paneCount
  const chartW = width - priceAxisW

  ctx.clearRect(0, 0, width, height)

  // === PRICE RANGE ===
  let minP = Infinity, maxP = -Infinity, maxVol = 0
  for (const c of candles) {
    if (c.low < minP) minP = c.low
    if (c.high > maxP) maxP = c.high
    if (c.volume > maxVol) maxVol = c.volume
  }
  const pad = (maxP - minP || 1) * L.pricePadding
  minP -= pad; maxP += pad
  const totalPR = maxP - minP

  // Helpers
  const pToY = (p) => mainH * (1 - (p - minP) / totalPR)
  const yToP = (y) => minP + (1 - y / mainH) * totalPR
  const cStep = chartW / candles.length
  const cW = Math.max(1, cStep * L.candleBodyRatio)
  const iToX = (i) => i * cStep + cStep / 2
  const xToI = (x) => Math.max(0, Math.min(candles.length - 1, Math.round((x - cStep / 2) / cStep)))

  // Export mappings
  options._pToY = pToY; options._yToP = yToP
  options._iToX = iToX; options._xToI = xToI
  options._chartH = mainH; options._chartW = chartW

  // === MAIN CHART AREA ===

  // Grid
  ctx.strokeStyle = C.grid; ctx.lineWidth = 0.5
  for (let i = 0; i <= 6; i++) {
    const p = minP + (totalPR / 6) * i, y = pToY(p)
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(chartW, y); ctx.stroke()
    ctx.fillStyle = C.text; ctx.font = CHART_FONTS.priceLabel; ctx.textAlign = 'right'
    const dec = p > 100 ? 2 : p > 1 ? 4 : 6
    ctx.fillText(p.toFixed(dec), width - 4, y + 3)
  }
  const tStep = Math.max(1, Math.floor(candles.length / L.timeGridDiv))
  for (let i = 0; i < candles.length; i += tStep) {
    const x = iToX(i)
    ctx.strokeStyle = C.grid; ctx.lineWidth = 0.5
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height - timeAxisH); ctx.stroke()
    ctx.fillStyle = C.text; ctx.font = CHART_FONTS.axisSmall; ctx.textAlign = 'center'
    ctx.fillText(`${Math.round(candles[i].relativeTime / 60)}m`, x, height - 4)
  }

  // Event start (t=0)
  const z = candles.findIndex(c => c.relativeTime >= 0)
  if (z >= 0) {
    const x = iToX(z)
    ctx.strokeStyle = MARKER_COLORS.eventStart; ctx.lineWidth = 1; ctx.setLineDash([4, 4])
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height - timeAxisH); ctx.stroke()
    ctx.setLineDash([])
  }

  // Pattern highlights
  if (patternHighlights.length > 0) {
    ctx.fillStyle = MARKER_COLORS.patternHighlight
    for (const idx of patternHighlights) ctx.fillRect(iToX(idx) - cStep/2, 0, cStep, mainH)
    ctx.fillStyle = MARKER_COLORS.patternMarker
    for (const idx of patternHighlights) {
      const x = iToX(idx)
      ctx.beginPath(); ctx.moveTo(x-4,0); ctx.lineTo(x+4,0); ctx.lineTo(x,6); ctx.closePath(); ctx.fill()
    }
  }

  // Overlay indicators (SMA, EMA, Bollinger, VWAP — on main chart)
  for (const ind of overlayInds) {
    if (!ind.data) continue
    const ic = ind.color
    if (ind.type === 'bollinger' && ind.data.upper) {
      ctx.globalAlpha = 0.05; ctx.fillStyle = ic; ctx.beginPath()
      for (let j = 0; j < ind.data.upper.length; j++) {
        const x = iToX(ind.data.upper[j].idx), y = pToY(ind.data.upper[j].value)
        j === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
      }
      for (let j = ind.data.lower.length - 1; j >= 0; j--)
        ctx.lineTo(iToX(ind.data.lower[j].idx), pToY(ind.data.lower[j].value))
      ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1
      drawLine(ctx, ind.data.upper, ic, 1, iToX, pToY, [4,4])
      drawLine(ctx, ind.data.middle, ic, 1, iToX, pToY)
      drawLine(ctx, ind.data.lower, ic, 1, iToX, pToY, [4,4])
    } else if (ind.data.values) {
      drawLine(ctx, ind.data.values, ic, 1.5, iToX, pToY)
    }
  }

  // Candlesticks / Line
  if (chartType === 'candle') {
    for (let i = 0; i < candles.length; i++) {
      const c = candles[i], x = iToX(i), up = c.close >= c.open
      const col = up ? C.up : C.down
      const bTop = pToY(up ? c.close : c.open), bBot = pToY(up ? c.open : c.close)
      ctx.strokeStyle = col; ctx.lineWidth = 1
      ctx.beginPath(); ctx.moveTo(x, pToY(c.high)); ctx.lineTo(x, pToY(c.low)); ctx.stroke()
      ctx.fillStyle = col; ctx.fillRect(x - cW/2, bTop, cW, Math.max(1, bBot - bTop))
    }
  } else {
    ctx.strokeStyle = eventColor; ctx.lineWidth = 2; ctx.beginPath()
    for (let i = 0; i < candles.length; i++) {
      const x = iToX(i), y = pToY(candles[i].close)
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
    }
    ctx.stroke()
    ctx.globalAlpha = 0.05; ctx.fillStyle = eventColor
    ctx.lineTo(iToX(candles.length-1), mainH); ctx.lineTo(iToX(0), mainH)
    ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1
  }

  // Main chart border
  ctx.strokeStyle = C.grid; ctx.lineWidth = 1
  ctx.beginPath(); ctx.moveTo(0, mainH); ctx.lineTo(chartW, mainH); ctx.stroke()

  // === SUB-PANES ===
  let paneTop = mainH

  // Volume pane
  if (showVolume && maxVol > 0) {
    renderSubPane(ctx, 'Volume', paneTop, paneH, chartW, priceAxisW, width, candles, C, iToX, cW, (c, pH) => {
      const bH = (c.volume / maxVol) * (pH - 4)
      return { height: bH, color: c.close >= c.open ? C.volumeUp : C.volumeDown, type: 'bar' }
    }, maxVol, 0)
    paneTop += paneH
  }

  // Separate indicator panes (RSI, MACD, ATR, Stochastic, Trades, Volume SMA)
  for (const ind of separateInds) {
    if (!ind.data) continue
    const vals = ind.data.values || []
    if (vals.length === 0) continue

    let minV = Infinity, maxV = -Infinity
    for (const v of vals) { if (v.value < minV) minV = v.value; if (v.value > maxV) maxV = v.value }
    const vRange = maxV - minV || 1
    const vPad = vRange * 0.1
    minV -= vPad; maxV += vPad
    const localPToY = (v) => paneTop + 2 + (paneH - 4) * (1 - (v - minV) / (maxV - minV))

    // Pane background + border
    ctx.fillStyle = CHART_SETTINGS.colors.grid + CHART_LAYOUT.paneGridOpacity; ctx.fillRect(0, paneTop, chartW, paneH)
    ctx.strokeStyle = C.grid; ctx.lineWidth = 0.5
    ctx.beginPath(); ctx.moveTo(0, paneTop); ctx.lineTo(chartW, paneTop); ctx.stroke()

    // Label
    ctx.fillStyle = C.text; ctx.font = CHART_FONTS.axisSmall; ctx.textAlign = 'left'
    ctx.fillText(ind.label, 4, paneTop + 10)

    // Min/Max labels
    ctx.textAlign = 'right'; ctx.font = CHART_FONTS.axisTiny; ctx.fillStyle = MARKER_COLORS.axisLabel
    ctx.fillText(maxV.toFixed(1), width - 4, paneTop + 10)
    ctx.fillText(minV.toFixed(1), width - 4, paneTop + paneH - 2)

    // Threshold lines (RSI 30/70, Stochastic 20/80)
    if (ind.config?.thresholds) {
      for (const th of ind.config.thresholds) {
        const y = localPToY(th.value)
        ctx.strokeStyle = (th.color) + '40'; ctx.lineWidth = 1; ctx.setLineDash([3,3])
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(chartW, y); ctx.stroke()
        ctx.setLineDash([])
      }
    }

    // Line
    const ic = ind.color
    ctx.strokeStyle = ic; ctx.lineWidth = 1.5; ctx.beginPath()
    for (let j = 0; j < vals.length; j++) {
      const x = iToX(vals[j].idx), y = localPToY(vals[j].value)
      j === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
    }
    ctx.stroke()

    // MACD signal line
    if (ind.data.signal) {
      ctx.strokeStyle = ic + '80'; ctx.lineWidth = 1; ctx.setLineDash([3,2]); ctx.beginPath()
      for (let j = 0; j < ind.data.signal.length; j++) {
        const x = iToX(ind.data.signal[j].idx), y = localPToY(ind.data.signal[j].value)
        j === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
      }
      ctx.stroke(); ctx.setLineDash([])
    }

    paneTop += paneH
  }

  // === CROSSHAIR (full height) ===
  if (crosshairPos && crosshairPos.x >= 0 && crosshairPos.x <= chartW) {
    const totalH = height - timeAxisH
    ctx.strokeStyle = C.crosshair; ctx.lineWidth = 0.5; ctx.setLineDash([3,3])
    ctx.beginPath(); ctx.moveTo(crosshairPos.x, 0); ctx.lineTo(crosshairPos.x, totalH); ctx.stroke()
    if (crosshairPos.y >= 0 && crosshairPos.y <= mainH) {
      ctx.beginPath(); ctx.moveTo(0, crosshairPos.y); ctx.lineTo(chartW, crosshairPos.y); ctx.stroke()
      ctx.setLineDash([])
      const price = yToP(crosshairPos.y)
      ctx.fillStyle = MARKER_COLORS.crosshairBg; ctx.fillRect(chartW, crosshairPos.y-8, priceAxisW, L.crosshairHeaderH)
      ctx.fillStyle = MARKER_COLORS.textOnColor; ctx.font = CHART_FONTS.priceLabel; ctx.textAlign = 'right'
      ctx.fillText(price.toFixed(price > 100 ? 2 : price > 1 ? 4 : 6), width-4, crosshairPos.y+4)
    }
    ctx.setLineDash([])

    // OHLCV header
    const idx = xToI(crosshairPos.x)
    if (idx >= 0 && idx < candles.length) {
      const c = candles[idx]
      ctx.fillStyle = MARKER_COLORS.crosshairBg; ctx.fillRect(0, 0, L.crosshairHeaderW, L.crosshairHeaderH)
      ctx.fillStyle = C.text; ctx.font = CHART_FONTS.priceLabel; ctx.textAlign = 'left'
      ctx.fillText(`O:${c.open.toFixed(4)} H:${c.high.toFixed(4)} L:${c.low.toFixed(4)} C:${c.close.toFixed(4)} V:${Math.round(c.volume)} T:${c.trades||0}`, 4, 11)
    }
  }

  // === REMOTE CROSSHAIR (Sync von anderem Chart) ===
  if (remoteCrosshairIdx != null && remoteCrosshairIdx >= 0 && remoteCrosshairIdx < candles.length && !crosshairPos) {
    const rx = iToX(remoteCrosshairIdx)
    const totalH = height - timeAxisH
    ctx.strokeStyle = MARKER_COLORS.patternMarker || '#f59e0b'
    ctx.lineWidth = 1
    ctx.setLineDash([2, 4])
    ctx.globalAlpha = 0.6
    ctx.beginPath(); ctx.moveTo(rx, 0); ctx.lineTo(rx, totalH); ctx.stroke()
    ctx.setLineDash([])
    ctx.globalAlpha = 1
    // Kleine Zeit-Marke oben
    const c = candles[remoteCrosshairIdx]
    if (c) {
      const label = `sync ${Math.round(c.relativeTime / 60)}m`
      ctx.fillStyle = MARKER_COLORS.crosshairBg
      ctx.fillRect(rx - 32, 0, 64, L.crosshairHeaderH)
      ctx.fillStyle = MARKER_COLORS.patternMarker || '#f59e0b'
      ctx.font = CHART_FONTS.axisSmall
      ctx.textAlign = 'center'
      ctx.fillText(label, rx, 11)
    }
  }

  // Right axis border
  ctx.strokeStyle = C.grid; ctx.lineWidth = 1
  ctx.beginPath(); ctx.moveTo(chartW, 0); ctx.lineTo(chartW, height); ctx.stroke()
}

function renderSubPane(ctx, label, top, height, chartW, priceAxisW, totalW, candles, C, iToX, cW, getBar, maxVal, minVal) {
  ctx.fillStyle = CHART_SETTINGS.colors.grid + CHART_LAYOUT.paneGridOpacity; ctx.fillRect(0, top, chartW, height)
  ctx.strokeStyle = C.grid; ctx.lineWidth = 0.5
  ctx.beginPath(); ctx.moveTo(0, top); ctx.lineTo(chartW, top); ctx.stroke()
  ctx.fillStyle = C.text; ctx.font = CHART_FONTS.axisSmall; ctx.textAlign = 'left'
  ctx.fillText(label, 4, top + 10)

  for (let i = 0; i < candles.length; i++) {
    const bar = getBar(candles[i], height)
    const x = iToX(i)
    ctx.fillStyle = bar.color
    ctx.fillRect(x - cW/2, top + height - 2 - bar.height, cW, bar.height)
  }
}

function drawLine(ctx, pts, color, lw, iToX, pToY, dash = []) {
  if (!pts?.length) return
  ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash)
  ctx.beginPath()
  for (let i = 0; i < pts.length; i++) {
    const x = iToX(pts[i].idx), y = pToY(pts[i].value)
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
  }
  ctx.stroke(); ctx.setLineDash([])
}
