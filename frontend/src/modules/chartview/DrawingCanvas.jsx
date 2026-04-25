import { useRef, useEffect, useState, useCallback } from 'react'
import { CHART_SETTINGS, MARKER_COLORS, CHART_FONTS } from '../../config/chartSettings'

/**
 * Zeichnungen werden in CHART-KOORDINATEN gespeichert (price + candleIdx),
 * nicht in Pixeln. Mappings werden aus plain numbers intern berechnet,
 * NICHT als Funktions-Props uebergeben (vermeidet Closure-Timing-Probleme).
 */
export default function DrawingCanvas({
  width, height,
  activeTool, drawingColor, drawingWidth,
  drawings, onAddDrawing, onRemoveDrawing,
  candles, visible = true,
  // Plain numbers statt Funktionen
  minP, maxP, chartH, chartW,
  // Offset fuer Zoom: visibleCandles ist ein Slice von fullCandles
  indexOffset = 0, fullCandles,
}) {
  const canvasRef = useRef(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [startPoint, setStartPoint] = useState(null)
  const [currentPoint, setCurrentPoint] = useState(null)
  const [textInput, setTextInput] = useState(null)
  const [channelPhase, setChannelPhase] = useState(0)
  const [channelPoints, setChannelPoints] = useState([])
  // Buy/Sell/Trade two-click state: { start: {price, candleIdx} }
  const [pendingTrade, setPendingTrade] = useState(null)

  // Mappings intern berechnen. i/idx = absoluter Full-Chart-Index.
  // Visible Slice beginnt bei indexOffset; wir rechnen immer zu visible um.
  const totalPR = maxP !== undefined && minP !== undefined ? (maxP - minP) : 0
  const cStep = candles?.length && chartW ? chartW / candles.length : 0
  const pToY = (p) => chartH ? chartH * (1 - (p - minP) / totalPR) : 0
  const yToP = (y) => chartH ? minP + (1 - y / chartH) * totalPR : 0
  // iToX: nimmt ABSOLUTEN idx, rechnet auf visible um
  const iToX = (absIdx) => (absIdx - indexOffset + 0.5) * cStep
  // xToI: pixel -> ABSOLUTER idx (inkl. Offset)
  const xToI = (x) => {
    if (cStep === 0) return 0
    const vis = Math.round(x / cStep - 0.5)
    return Math.max(0, Math.min((fullCandles?.length || candles.length) - 1, vis + indexOffset))
  }
  // Check ob absoluter idx im visible Bereich liegt
  const isVisible = (absIdx) => absIdx >= indexOffset && absIdx < indexOffset + (candles?.length || 0)
  // Hole Candle aus fullCandles anhand des absoluten idx
  const getCandle = (absIdx) => fullCandles ? fullCandles[absIdx] : candles?.[absIdx - indexOffset]

  const ready = width > 0 && height > 0 && chartH > 0 && chartW > 0 && candles?.length > 0

  const pixelToChart = (px, py) => {
    if (!ready) return null
    return { price: yToP(py), candleIdx: xToI(px) }
  }

  const chartToPixel = (price, candleIdx) => {
    if (!ready) return null
    return { x: iToX(candleIdx), y: pToY(price) }
  }

  // Get drawing points in current pixel space
  const getPixelPoints = (d) => {
    const r = {}
    if (d.price1 !== undefined && d.idx1 !== undefined) {
      r.x1 = iToX(d.idx1); r.y1 = pToY(d.price1)
    }
    if (d.price2 !== undefined && d.idx2 !== undefined) {
      r.x2 = iToX(d.idx2); r.y2 = pToY(d.price2)
    }
    if (d.price3 !== undefined && d.idx3 !== undefined) {
      r.x3 = iToX(d.idx3); r.y3 = pToY(d.price3)
    }
    if (d.price !== undefined) r.y = pToY(d.price)
    if (d.candleIdx !== undefined) r.x = iToX(d.candleIdx)
    return r
  }

  const redraw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || !ready) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, width, height)
    if (!visible) return

    const cW = Math.max(1, cStep * 0.7)

    for (const d of drawings) {
      const px = getPixelPoints(d)
      ctx.save()
      ctx.strokeStyle = d.color
      ctx.lineWidth = d.width
      ctx.fillStyle = d.color
      ctx.setLineDash([])

      switch (d.type) {
        case 'trendline':
          if (px.x1 == null) break
          ctx.beginPath(); ctx.moveTo(px.x1, px.y1); ctx.lineTo(px.x2, px.y2); ctx.stroke()
          break

        case 'ray': {
          if (px.x1 == null) break
          const dx = px.x2 - px.x1, dy = px.y2 - px.y1
          const len = Math.sqrt(dx*dx + dy*dy)
          if (len === 0) break
          ctx.beginPath(); ctx.moveTo(px.x1, px.y1)
          ctx.lineTo(px.x1 + (dx/len)*3000, px.y1 + (dy/len)*3000); ctx.stroke()
          break
        }

        case 'hline':
          ctx.setLineDash([5,3]); ctx.beginPath(); ctx.moveTo(0, px.y); ctx.lineTo(width, px.y); ctx.stroke()
          if (d.price != null) {
            ctx.setLineDash([]); ctx.font = CHART_FONTS.priceLabel; ctx.textAlign = 'right'
            ctx.fillText(d.price.toFixed(4), width - 4, px.y - 4)
          }
          break

        case 'vline':
          ctx.setLineDash([5,3]); ctx.beginPath(); ctx.moveTo(px.x, 0); ctx.lineTo(px.x, height); ctx.stroke()
          break

        case 'support': case 'resistance': {
          if (px.y == null) break
          const sColor = d.type === 'support' ? MARKER_COLORS.supportLine : MARKER_COLORS.resistanceLine
          ctx.strokeStyle = sColor; ctx.lineWidth = 2
          ctx.beginPath(); ctx.moveTo(0, px.y); ctx.lineTo(width, px.y); ctx.stroke()
          ctx.globalAlpha = 0.06; ctx.fillStyle = sColor
          ctx.fillRect(0, px.y - 8, width, 16); ctx.globalAlpha = 1
          ctx.fillStyle = sColor; ctx.font = CHART_FONTS.markerLabel; ctx.textAlign = 'left'
          if (d.price == null) break
          ctx.fillText((d.type === 'support' ? 'S ' : 'R ') + d.price.toFixed(4), 4, px.y - 10)
          break
        }

        case 'channel':
          if (px.x1 == null || px.x3 == null) break
          ctx.beginPath(); ctx.moveTo(px.x1, px.y1); ctx.lineTo(px.x2, px.y2); ctx.stroke()
          const cdx = px.x2-px.x1, cdy = px.y2-px.y1
          ctx.beginPath(); ctx.moveTo(px.x3, px.y3); ctx.lineTo(px.x3+cdx, px.y3+cdy); ctx.stroke()
          break

        case 'rect':
          if (px.x1 == null) break
          ctx.globalAlpha = 0.12
          ctx.fillRect(Math.min(px.x1,px.x2), Math.min(px.y1,px.y2), Math.abs(px.x2-px.x1), Math.abs(px.y2-px.y1))
          ctx.globalAlpha = 1
          ctx.strokeRect(Math.min(px.x1,px.x2), Math.min(px.y1,px.y2), Math.abs(px.x2-px.x1), Math.abs(px.y2-px.y1))
          break

        case 'priceRange':
          if (px.y1 == null) break
          ctx.globalAlpha = 0.1; ctx.fillRect(0, Math.min(px.y1,px.y2), width, Math.abs(px.y2-px.y1))
          ctx.globalAlpha = 0.7; ctx.setLineDash([3,3])
          ctx.beginPath(); ctx.moveTo(0, px.y1); ctx.lineTo(width, px.y1); ctx.stroke()
          ctx.beginPath(); ctx.moveTo(0, px.y2); ctx.lineTo(width, px.y2); ctx.stroke()
          break

        case 'timeRange':
          if (px.x1 == null) break
          ctx.globalAlpha = 0.1; ctx.fillRect(Math.min(px.x1,px.x2), 0, Math.abs(px.x2-px.x1), height)
          ctx.globalAlpha = 0.7; ctx.setLineDash([3,3])
          ctx.beginPath(); ctx.moveTo(px.x1, 0); ctx.lineTo(px.x1, height); ctx.stroke()
          ctx.beginPath(); ctx.moveTo(px.x2, 0); ctx.lineTo(px.x2, height); ctx.stroke()
          break

        case 'fibonacci': case 'fibExtension': {
          if (px.y1 == null) break
          const yMin = Math.min(px.y1,px.y2), yMax = Math.max(px.y1,px.y2), rng = yMax-yMin
          const cfg = d.type === 'fibonacci' ? CHART_SETTINGS.drawing.fibonacci : CHART_SETTINGS.drawing.fibExtension
          cfg.levels.forEach((level, li) => {
            const y = yMax - rng * level
            ctx.strokeStyle = cfg.colors[li]; ctx.lineWidth = 1; ctx.setLineDash([3,3])
            ctx.beginPath(); ctx.moveTo(Math.min(px.x1,px.x2), y); ctx.lineTo(Math.max(px.x1,px.x2), y); ctx.stroke()
            ctx.setLineDash([]); ctx.fillStyle = ctx.strokeStyle; ctx.font = CHART_FONTS.priceLabel
            ctx.fillText(`${(level*100).toFixed(1)}%`, Math.max(px.x1,px.x2)+4, y+3)
          })
          break
        }

        case 'measure': {
          if (px.x1 == null) break
          ctx.setLineDash([3,3]); ctx.beginPath(); ctx.moveTo(px.x1,px.y1); ctx.lineTo(px.x2,px.y2); ctx.stroke()
          ctx.setLineDash([]); ctx.font = CHART_FONTS.measureLabel
          const pctChange = d.price1 && d.price2 ? ((d.price2 - d.price1)/d.price1*100).toFixed(2) : '?'
          const timeDiff = d.idx2 !== undefined && d.idx1 !== undefined ? d.idx2 - d.idx1 : '?'
          ctx.fillText(`${pctChange}% | ${timeDiff}`, (px.x1+px.x2)/2+6, (px.y1+px.y2)/2-6)
          break
        }

        case 'candleHighlight': {
          const cx = iToX(d.candleIdx)
          ctx.globalAlpha = 0.25; ctx.fillRect(cx-cStep/2, 0, cStep, chartH)
          ctx.globalAlpha = 1; ctx.lineWidth = 2; ctx.strokeRect(cx-cStep/2, 0, cStep, chartH)
          break
        }

        case 'candleHigh': case 'candleLow': {
          const c = getCandle(d.candleIdx)
          if (!c) break
          const cx = iToX(d.candleIdx), py2 = pToY(d.type === 'candleHigh' ? c.high : c.low)
          ctx.beginPath(); ctx.arc(cx, py2, 5, 0, Math.PI*2); ctx.stroke()
          ctx.setLineDash([3,3]); ctx.beginPath(); ctx.moveTo(0, py2); ctx.lineTo(width, py2); ctx.stroke()
          ctx.setLineDash([]); ctx.font = CHART_FONTS.priceLabel
          ctx.fillText((d.type==='candleHigh' ? c.high : c.low).toFixed(4), cx+8, py2-4)
          break
        }

        case 'candleRange': case 'wickToWick': {
          const c1 = getCandle(d.startIdx), c2 = getCandle(d.endIdx)
          if (!c1 || !c2) break
          const x1 = iToX(d.startIdx), x2 = iToX(d.endIdx)
          const isW = d.type === 'wickToWick'
          const y1 = pToY(isW ? (d.startIsHigh ? c1.high : c1.low) : Math.max(c1.open, c1.close))
          const y2 = pToY(isW ? (d.endIsHigh ? c2.high : c2.low) : Math.max(c2.open, c2.close))
          ctx.globalAlpha = 0.1; ctx.beginPath()
          ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.lineTo(x2,chartH); ctx.lineTo(x1,chartH)
          ctx.closePath(); ctx.fill()
          ctx.globalAlpha = 1; ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke()
          if (d.label) { ctx.font = CHART_FONTS.priceLabel; ctx.fillText(d.label, (x1+x2)/2, Math.min(y1,y2)-6) }
          break
        }

        case 'buyMarker': case 'sellMarker': {
          // Signal mit Anfang und Ende (2 Candles)
          if (d.startIdx == null) break
          const isBuy = d.type === 'buyMarker'
          const startCandle = getCandle(d.startIdx)
          const endCandle = d.endIdx != null ? getCandle(d.endIdx) : startCandle
          if (!startCandle) break
          const startPrice = isBuy ? startCandle.low : startCandle.high
          // Anderes Ende: Buy endet oben (high), Sell endet unten (low)
          const endPrice = endCandle ? (isBuy ? endCandle.high : endCandle.low) : startPrice
          const sx = iToX(d.startIdx)
          const ex = d.endIdx != null ? iToX(d.endIdx) : sx
          const sy = pToY(startPrice)
          const ey = pToY(endPrice)
          const color = isBuy ? MARKER_COLORS.buy : MARKER_COLORS.sell

          // Vertical drop from candle to marker
          ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.setLineDash([2,2])
          ctx.beginPath()
          ctx.moveTo(sx, isBuy ? sy - 4 : sy + 4)
          ctx.lineTo(sx, isBuy ? sy + 20 : sy - 20)
          ctx.stroke()
          if (d.endIdx != null) {
            ctx.beginPath()
            ctx.moveTo(ex, isBuy ? ey - 4 : ey + 4)
            ctx.lineTo(ex, isBuy ? ey + 20 : ey - 20)
            ctx.stroke()
          }
          ctx.setLineDash([])

          // Triangle marker
          ctx.fillStyle = color
          const drawTri = (x, y) => {
            ctx.beginPath()
            if (isBuy) {
              ctx.moveTo(x, y + 22); ctx.lineTo(x-8, y + 36); ctx.lineTo(x+8, y + 36)
            } else {
              ctx.moveTo(x, y - 22); ctx.lineTo(x-8, y - 36); ctx.lineTo(x+8, y - 36)
            }
            ctx.closePath(); ctx.fill()
            ctx.fillStyle = MARKER_COLORS.textOnColor; ctx.font = CHART_FONTS.markerLabel; ctx.textAlign = 'center'
            ctx.fillText(isBuy ? 'B' : 'S', x, isBuy ? y + 33 : y - 27)
            ctx.textAlign = 'start'; ctx.fillStyle = color
          }
          drawTri(sx, sy)
          if (d.endIdx != null) drawTri(ex, ey)

          // Connector line between start and end
          if (d.endIdx != null) {
            ctx.strokeStyle = color; ctx.lineWidth = 2
            ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); ctx.stroke()
            // Profit/Loss label
            const pct = ((endPrice - startPrice) / startPrice * 100) * (isBuy ? 1 : -1)
            ctx.fillStyle = pct >= 0 ? MARKER_COLORS.buy : MARKER_COLORS.sell
            ctx.font = CHART_FONTS.alertLabel
            ctx.fillText(`${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%`, (sx+ex)/2, (sy+ey)/2 - 8)
          }
          break
        }

        case 'initialPoint': {
          // Stern-Marker fuer Initialpunkt (Zeit-Anker)
          const c = data.candles[d.candleIdx]
          if (!c) break
          const x = iToX(d.candleIdx), y = pToY(c.close)
          ctx.fillStyle = '#fbbf24'
          ctx.beginPath()
          const r = 8
          for (let i = 0; i < 10; i++) {
            const a = (Math.PI / 5) * i - Math.PI / 2
            const rr = i % 2 === 0 ? r : r / 2.3
            ctx.lineTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr)
          }
          ctx.closePath()
          ctx.fill()
          ctx.fillStyle = '#1e293b'
          ctx.font = 'bold 8px Inter'
          ctx.textAlign = 'center'
          ctx.fillText('IP', x, y + 3)
          ctx.textAlign = 'start'
          break
        }
        case 'flag': {
          const p = chartToPixel(d.price, d.candleIdx)
          if (!p) break
          ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(p.x, p.y-20)
          ctx.lineTo(p.x+14, p.y-15); ctx.lineTo(p.x, p.y-10); ctx.closePath(); ctx.fill()
          ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(p.x, p.y-20); ctx.stroke()
          break
        }
        case 'alert': {
          const p = chartToPixel(d.price, d.candleIdx)
          if (!p) break
          ctx.fillStyle = MARKER_COLORS.alert
          ctx.beginPath(); ctx.moveTo(p.x, p.y-12); ctx.lineTo(p.x-8, p.y+4); ctx.lineTo(p.x+8, p.y+4); ctx.closePath(); ctx.fill()
          ctx.fillStyle = MARKER_COLORS.textOnAlert; ctx.font = CHART_FONTS.alertLabel; ctx.textAlign = 'center'; ctx.fillText('!', p.x, p.y+2); ctx.textAlign = 'start'
          break
        }
        case 'text': {
          const p = chartToPixel(d.price, d.candleIdx)
          if (!p) break
          if (!d.text) break
          ctx.font = CHART_FONTS.text; ctx.fillText(d.text, p.x, p.y)
          break
        }
      }
      ctx.restore()
    }

    // Pending Buy/Sell (waiting for end click)
    if (pendingTrade) {
      const c = getCandle(pendingTrade.startIdx)
      if (c) {
        const isBuy = pendingTrade.type === 'buyMarker'
        const x = iToX(pendingTrade.startIdx)
        const y = pToY(isBuy ? c.low : c.high)
        ctx.fillStyle = isBuy ? MARKER_COLORS.buy : MARKER_COLORS.sell
        ctx.globalAlpha = 0.5
        ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI*2); ctx.fill()
        ctx.globalAlpha = 1
      }
    }

    // In-progress line preview
    if (isDrawing && startPoint && currentPoint) {
      ctx.save()
      ctx.strokeStyle = drawingColor; ctx.lineWidth = drawingWidth; ctx.setLineDash([2,2]); ctx.globalAlpha = 0.6
      switch (activeTool) {
        case 'trendline': case 'ray': case 'measure':
          ctx.beginPath(); ctx.moveTo(startPoint.x, startPoint.y); ctx.lineTo(currentPoint.x, currentPoint.y); ctx.stroke(); break
        case 'rect': case 'fibonacci': case 'fibExtension':
          ctx.strokeRect(Math.min(startPoint.x,currentPoint.x), Math.min(startPoint.y,currentPoint.y),
            Math.abs(currentPoint.x-startPoint.x), Math.abs(currentPoint.y-startPoint.y)); break
        case 'priceRange':
          ctx.beginPath(); ctx.moveTo(0, startPoint.y); ctx.lineTo(width, startPoint.y); ctx.stroke()
          ctx.beginPath(); ctx.moveTo(0, currentPoint.y); ctx.lineTo(width, currentPoint.y); ctx.stroke(); break
        case 'timeRange':
          ctx.fillStyle = drawingColor; ctx.globalAlpha = 0.08
          ctx.fillRect(Math.min(startPoint.x,currentPoint.x), 0, Math.abs(currentPoint.x-startPoint.x), height); break
      }
      ctx.restore()
    }
  }, [drawings, isDrawing, startPoint, currentPoint, activeTool, drawingColor, drawingWidth,
      width, height, visible, candles, minP, maxP, chartH, chartW, pendingTrade, ready,
      indexOffset, fullCandles])

  useEffect(() => { redraw() }, [redraw])

  const getPos = (e) => {
    const rect = canvasRef.current.getBoundingClientRect()
    return { x: e.clientX - rect.left, y: e.clientY - rect.top }
  }

  const handleMouseDown = (e) => {
    if (!ready) return
    if (activeTool === 'cursor' || activeTool === 'crosshair') return
    const pos = getPos(e)

    // Eraser
    if (activeTool === 'eraser') {
      let minDist = 25, minIdx = -1
      drawings.forEach((d, idx) => {
        const px = getPixelPoints(d)
        let dist = Infinity
        if (['flag','alert','text','initialPoint'].includes(d.type)) {
          const p = chartToPixel(d.price, d.candleIdx)
          if (p) dist = Math.sqrt((p.x-pos.x)**2 + (p.y-pos.y)**2)
        } else if (['buyMarker','sellMarker'].includes(d.type)) {
          const c = getCandle(d.startIdx)
          if (c) {
            const x = iToX(d.startIdx)
            const y = pToY(d.type === 'buyMarker' ? c.low : c.high)
            dist = Math.sqrt((x-pos.x)**2 + (y-pos.y)**2)
          }
        } else if (d.type === 'hline' || d.type === 'support' || d.type === 'resistance') {
          if (px.y == null) return
          dist = Math.abs(px.y - pos.y)
        } else if (d.type === 'vline') {
          if (px.x == null) return
          dist = Math.abs(px.x - pos.x)
        } else if (['candleHighlight','candleHigh','candleLow'].includes(d.type)) {
          dist = Math.abs(iToX(d.candleIdx) - pos.x)
        } else if (px.x1 != null) {
          const ddx = px.x2-px.x1, ddy = px.y2-px.y1, l = Math.sqrt(ddx*ddx+ddy*ddy)||1
          const t = Math.max(0,Math.min(1,((pos.x-px.x1)*ddx+(pos.y-px.y1)*ddy)/(l*l)))
          dist = Math.sqrt((pos.x-px.x1-t*ddx)**2+(pos.y-px.y1-t*ddy)**2)
        }
        if (dist < minDist) { minDist = dist; minIdx = idx }
      })
      if (minIdx >= 0) onRemoveDrawing(minIdx)
      return
    }

    // Buy/Sell als 2-Click Trade mit Auto-Snap an Candle
    if (activeTool === 'buyMarker' || activeTool === 'sellMarker') {
      const candleIdx = xToI(pos.x)
      if (!pendingTrade) {
        // Erster Klick = Start
        setPendingTrade({ type: activeTool, startIdx: candleIdx })
      } else {
        // Zweiter Klick = Ende
        onAddDrawing({
          type: pendingTrade.type,
          startIdx: pendingTrade.startIdx,
          endIdx: candleIdx,
          color: pendingTrade.type === 'buyMarker' ? MARKER_COLORS.buy : MARKER_COLORS.sell,
          width: drawingWidth,
        })
        setPendingTrade(null)
      }
      return
    }

    // Single-click tools
    if (['flag','alert','initialPoint'].includes(activeTool)) {
      const c = pixelToChart(pos.x, pos.y)
      if (c) onAddDrawing({ type: activeTool, price: c.price, candleIdx: c.candleIdx, color: drawingColor, width: drawingWidth })
      return
    }
    if (activeTool === 'hline' || activeTool === 'support' || activeTool === 'resistance') {
      const c = pixelToChart(pos.x, pos.y)
      if (c) onAddDrawing({ type: activeTool, price: c.price, color: drawingColor, width: drawingWidth })
      return
    }
    if (activeTool === 'vline') {
      const c = pixelToChart(pos.x, pos.y)
      if (c) onAddDrawing({ type: 'vline', candleIdx: c.candleIdx, color: drawingColor, width: drawingWidth })
      return
    }
    if (activeTool === 'text') { setTextInput(pos); return }

    // Candle single-click tools
    if (['candleHighlight','candleHigh','candleLow'].includes(activeTool)) {
      const candleIdx = xToI(pos.x)
      onAddDrawing({ type: activeTool, candleIdx, color: drawingColor, width: drawingWidth })
      return
    }

    // Candle range tools (2-click)
    if (['candleRange','wickToWick'].includes(activeTool)) {
      if (!isDrawing) { setIsDrawing(true); setStartPoint(pos); setCurrentPoint(pos) }
      else {
        const si = xToI(startPoint.x), ei = xToI(pos.x)
        const ca = getCandle(si), cb = getCandle(ei)
        const pct = ca && cb ? ((cb.close-ca.close)/ca.close*100) : 0
        onAddDrawing({
          type: activeTool, startIdx: si, endIdx: ei,
          startIsHigh: startPoint.y < chartH/2, endIsHigh: pos.y < chartH/2,
          color: drawingColor, width: drawingWidth,
          label: `${pct>=0?'+':''}${pct.toFixed(2)}%`
        })
        setIsDrawing(false); setStartPoint(null); setCurrentPoint(null)
      }
      return
    }

    // Channel 3-click
    if (activeTool === 'channel') {
      if (channelPhase < 2) {
        setChannelPoints(p => [...p, pos])
        setChannelPhase(p => p+1)
        setIsDrawing(true); setStartPoint(pos); setCurrentPoint(pos)
      } else {
        const pts = [...channelPoints, pos]
        const c0 = pixelToChart(pts[0].x, pts[0].y)
        const c1 = pixelToChart(pts[1].x, pts[1].y)
        const c2 = pixelToChart(pts[2].x, pts[2].y)
        if (c0 && c1 && c2) {
          onAddDrawing({ type:'channel', price1:c0.price, idx1:c0.candleIdx,
            price2:c1.price, idx2:c1.candleIdx, price3:c2.price, idx3:c2.candleIdx,
            color:drawingColor, width:drawingWidth })
        }
        setChannelPhase(0); setChannelPoints([]); setIsDrawing(false); setStartPoint(null)
      }
      return
    }

    // Standard two-point drag tools
    setIsDrawing(true); setStartPoint(pos); setCurrentPoint(pos)
  }

  const handleMouseMove = (e) => { if (isDrawing) setCurrentPoint(getPos(e)) }

  const handleMouseUp = (e) => {
    if (!isDrawing || !startPoint) return
    if (['channel','candleRange','wickToWick'].includes(activeTool)) { setCurrentPoint(getPos(e)); return }
    const end = getPos(e)
    setIsDrawing(false)
    const c1 = pixelToChart(startPoint.x, startPoint.y)
    const c2 = pixelToChart(end.x, end.y)
    if (c1 && c2) {
      onAddDrawing({
        type: activeTool,
        price1: c1.price, idx1: c1.candleIdx,
        price2: c2.price, idx2: c2.candleIdx,
        color: drawingColor, width: drawingWidth,
      })
    }
    setStartPoint(null); setCurrentPoint(null)
  }

  const handleTextSubmit = (text) => {
    if (text && textInput) {
      const c = pixelToChart(textInput.x, textInput.y)
      if (c) onAddDrawing({ type:'text', price:c.price, candleIdx:c.candleIdx, text, color:drawingColor, width:drawingWidth })
    }
    setTextInput(null)
  }

  const isInteractive = activeTool !== 'cursor' && activeTool !== 'crosshair'

  return (
    <>
      <canvas ref={canvasRef} width={width} height={height} className="absolute top-0 left-0"
        style={{ pointerEvents: isInteractive ? 'auto' : 'none',
          cursor: activeTool === 'eraser' ? 'crosshair' : activeTool === 'text' ? 'text' : isInteractive ? 'crosshair' : 'default',
          zIndex: isInteractive ? 10 : 0 }}
        onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp}
        onMouseLeave={() => { if (isDrawing && !['channel','candleRange','wickToWick'].includes(activeTool)) { setIsDrawing(false); setStartPoint(null) } }}
      />
      {textInput && (
        <div className="absolute z-20" style={{ left: textInput.x, top: textInput.y }}>
          <input autoFocus className="bg-gray-900 border border-blue-500 text-white text-xs px-1 py-0.5 rounded w-40"
            placeholder="Text eingeben..."
            onKeyDown={e => { if (e.key==='Enter') handleTextSubmit(e.target.value); if (e.key==='Escape') setTextInput(null) }}
            onBlur={e => handleTextSubmit(e.target.value)} />
        </div>
      )}
    </>
  )
}
