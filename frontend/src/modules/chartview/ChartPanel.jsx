import { useEffect, useRef, useState, useCallback } from 'react'
import { CHART_SETTINGS } from '../../config/chartSettings'
import { renderChart } from './utils/canvasRenderer'
import { computeIndicator } from './utils/indicatorCalc'
import { computeChartMappings } from './utils/chartMappings'
import { detectPatterns } from './utils/patternDetect'
import DrawingCanvas from './DrawingCanvas'
import CandleInfoBar from './CandleInfoBar'

export default function ChartPanel({
  event, data, eventIndex, eventColor,
  chartType, showVolume, isGridMode, isActive, onClick, onDoubleClick,
  activeTool, drawingColor, drawingWidth,
  drawings, onAddDrawing, onRemoveDrawing,
  // drawings muss immer ein Array sein (wird in ChartViewModule initialisiert)
  activeIndicators, drawingsVisible = true,
  hoveredPattern,
  syncedHoverTime, onSyncHover,
}) {
  const canvasRef = useRef(null)
  const containerRef = useRef(null)
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 })
  const [crosshairPos, setCrosshairPos] = useState(null)
  const renderOpts = useRef({})
  const [hoveredCandle, setHoveredCandle] = useState(null)
  const height = isGridMode ? CHART_SETTINGS.display.gridChartHeight : CHART_SETTINGS.display.defaultChartHeight

  // Zoom/Pan state
  const [viewRange, setViewRange] = useState(null) // null = show all
  const totalCandles = data?.candles?.length || 0

  // Compute visible candles based on viewRange
  const visibleCandles = viewRange && data?.candles
    ? data.candles.slice(viewRange.start, viewRange.end)
    : data?.candles || []

  // Indicator data — berechnet auf rawCandles (1m) fuer maximale Datenmenge,
  // dann gemappt auf sichtbare aggregierte Candles
  const indicatorData = activeIndicators.map((ind, i) => {
    // Berechnung auf vollen 1m-Rohdaten (mehr Datenpunkte = genauere Indikatoren)
    const rawData = data?.rawCandles
    if (!rawData) throw new Error("rawCandles missing in chartData")
    const result = computeIndicator(ind, rawData)
    if (!result) return null

    // Mapping: raw index -> aggregierter index
    // Verhaeltnis raw zu aggregiert (z.B. 5m = jeder 5. raw-Punkt)
    const rawLen = rawData?.length || 0
    const aggLen = data?.candles?.length || 0
    const ratio = aggLen > 0 ? rawLen / aggLen : 1

    // Mappt raw-Indikator-Werte auf aggregierte Positionen.
    // Fuer jeden agg-Candle nehmen wir den letzten raw-Wert der zu diesem
    // Aggregat gehoert (entspricht dem close). So gibt es genau einen Wert
    // pro agg-Position, keine Duplikate, keine Zickzack-Linien.
    const mapToVisibleZoomed = (vals) => {
      if (!vals) return null
      const byRawIdx = new Map()
      for (const v of vals) byRawIdx.set(v.idx, v)
      const out = []
      for (let agg = 0; agg < aggLen; agg++) {
        // letzter raw-idx dieses agg-Candles
        const rawIdx = Math.min(rawLen - 1, Math.floor((agg + 1) * ratio) - 1)
        const v = byRawIdx.get(rawIdx)
        if (v) out.push({ ...v, idx: agg })
      }
      if (viewRange) {
        return out
          .filter(v => v.idx >= viewRange.start && v.idx < viewRange.end)
          .map(v => ({ ...v, idx: v.idx - viewRange.start }))
      }
      return out
    }

    return {
      ...ind,
      color: CHART_SETTINGS.indicators.defaultColors[i % CHART_SETTINGS.indicators.defaultColors.length],
      data: result.upper ? {
        upper: mapToVisibleZoomed(result.upper),
        middle: mapToVisibleZoomed(result.middle),
        lower: mapToVisibleZoomed(result.lower),
      } : { values: mapToVisibleZoomed(result.values || result) },
    }
  }).filter(Boolean)

  // Pattern highlights
  const patternHighlights = hoveredPattern && data?.candles
    ? (detectPatterns(data.candles, [hoveredPattern])[hoveredPattern] || []).map(h => h.index)
    : []

  // Resize
  useEffect(() => {
    if (!containerRef.current) return
    const obs = new ResizeObserver(entries => {
      for (const e of entries) setDimensions({ width: e.contentRect.width, height })
    })
    obs.observe(containerRef.current)
    setDimensions({ width: containerRef.current.clientWidth, height })
    return () => obs.disconnect()
  }, [height])

  // Mappings werden direkt berechnet (nicht aus Ref) damit DrawingCanvas sie synchron bekommt
  const separateIndCount = indicatorData.filter(i => i.config?.separate && i.visible !== false).length
  const mappings = computeChartMappings(visibleCandles, dimensions.width, dimensions.height, showVolume, separateIndCount)

  // Render
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !visibleCandles.length || dimensions.width === 0) return
    const ctx = canvas.getContext('2d')
    const opts = {
      width: dimensions.width, height: dimensions.height,
      chartType, showVolume, eventColor,
      indicators: indicatorData, crosshairPos,
      patternHighlights,
    }
    // Remote-Crosshair (Hover von anderem Chart): finde naechste Candle
    let remoteIdx = null
    if (syncedHoverTime != null && !crosshairPos && visibleCandles.length > 0) {
      let bestDiff = Infinity
      for (let i = 0; i < visibleCandles.length; i++) {
        const diff = Math.abs(visibleCandles[i].relativeTime - syncedHoverTime)
        if (diff < bestDiff) { bestDiff = diff; remoteIdx = i }
      }
    }
    opts.remoteCrosshairIdx = remoteIdx
    renderChart(ctx, visibleCandles, opts)
    renderOpts.current = opts
  }, [visibleCandles, dimensions, chartType, showVolume, eventColor, indicatorData, crosshairPos, patternHighlights, syncedHoverTime])

  // Mouse: crosshair
  const handleMouseMove = useCallback((e) => {
    if (activeTool !== 'cursor' && activeTool !== 'crosshair') { setCrosshairPos(null); setHoveredCandle(null); if (onSyncHover) onSyncHover(null); return }
    const rect = canvasRef.current?.getBoundingClientRect()
    if (!rect) return
    const x = e.clientX - rect.left
    setCrosshairPos({ x, y: e.clientY - rect.top })
    if (visibleCandles.length && dimensions.width) {
      const step = (dimensions.width - 70) / visibleCandles.length
      const idx = Math.round((x - step / 2) / step)
      if (idx >= 0 && idx < visibleCandles.length) {
        setHoveredCandle({ candle: visibleCandles[idx], index: idx })
        // Shared: relativeTime (Sek) an andere Charts melden
        if (onSyncHover && visibleCandles[idx]?.relativeTime != null) {
          onSyncHover(visibleCandles[idx].relativeTime)
        }
      } else { setHoveredCandle(null) }
    }
  }, [activeTool, visibleCandles, dimensions, onSyncHover])

  // Zoom: nur wenn Ctrl/Cmd gedrueckt — sonst normales Scrolling durch Container
  const handleWheel = useCallback((e) => {
    if (!(e.ctrlKey || e.metaKey)) return
    e.preventDefault()
    if (totalCandles < 10) return
    const current = viewRange || { start: 0, end: totalCandles }
    const visCount = current.end - current.start
    const zoomFactor = e.deltaY > 0 ? 1.15 : 0.87 // scroll down = zoom out
    let newCount = Math.round(visCount * zoomFactor)

    // Clamp: min 10 candles, max all candles
    newCount = Math.max(10, Math.min(totalCandles, newCount))

    if (newCount === totalCandles) { setViewRange(null); return }

    // Zoom centered on mouse position
    const rect = canvasRef.current?.getBoundingClientRect()
    const mouseRatio = rect ? (e.clientX - rect.left) / rect.width : 0.5
    const center = current.start + Math.round(visCount * mouseRatio)
    let newStart = Math.round(center - newCount * mouseRatio)
    let newEnd = newStart + newCount

    // Clamp to bounds — NO empty space
    if (newStart < 0) { newStart = 0; newEnd = newCount }
    if (newEnd > totalCandles) { newEnd = totalCandles; newStart = totalCandles - newCount }
    newStart = Math.max(0, newStart)

    setViewRange({ start: newStart, end: newEnd })
  }, [viewRange, totalCandles])

  // Pan: drag with middle mouse or shift+drag
  const [isPanning, setIsPanning] = useState(false)
  const [panStart, setPanStart] = useState(0)

  const handlePanStart = useCallback((e) => {
    if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
      setIsPanning(true); setPanStart(e.clientX); e.preventDefault()
    }
  }, [])

  const handlePanMove = useCallback((e) => {
    if (!isPanning || !viewRange) return
    const dx = e.clientX - panStart
    const rect = canvasRef.current?.getBoundingClientRect()
    if (!rect) return
    const candlesPerPx = (viewRange.end - viewRange.start) / rect.width
    const shift = Math.round(-dx * candlesPerPx)
    if (shift === 0) return

    let newStart = viewRange.start + shift
    let newEnd = viewRange.end + shift
    // Clamp
    if (newStart < 0) { newEnd -= newStart; newStart = 0 }
    if (newEnd > totalCandles) { newStart -= (newEnd - totalCandles); newEnd = totalCandles }
    newStart = Math.max(0, newStart)

    setViewRange({ start: newStart, end: newEnd })
    setPanStart(e.clientX)
  }, [isPanning, panStart, viewRange, totalCandles])

  const handlePanEnd = useCallback(() => setIsPanning(false), [])

  if (!data?.candles?.length) return null

  return (
    <div
      className={`bg-gray-800 rounded border transition-colors ${
        isActive ? 'border-blue-500 ring-1 ring-blue-500/30' : 'border-gray-700'
      }`}
      onClick={onClick}
      onDoubleClick={(e) => { e.stopPropagation(); onDoubleClick?.() }}
    >
      <div className="px-2 py-1 border-b border-gray-700 text-xs flex items-center gap-2"
        style={{ borderLeftColor: eventColor, borderLeftWidth: 3 }}>
        <span className="font-mono font-semibold">{event.symbol}</span>
        <span className={event.change_percent >= 0 ? 'text-green-400' : 'text-red-400'}>
          {event.change_percent >= 0 ? '+' : ''}{event.change_percent?.toFixed(2)}%
        </span>
        <span className="text-gray-600 text-[10px]">{event.duration_minutes}min</span>
        {viewRange && (
          <button onClick={(e) => { e.stopPropagation(); setViewRange(null) }}
            className="text-[10px] text-blue-400 hover:text-blue-300">Reset Zoom</button>
        )}
        {drawings.length > 0 && <span className="text-gray-500 ml-auto">{drawings.length} Zeichnung(en)</span>}
      </div>

      <div ref={containerRef} className="relative" style={{ height }}
        onWheel={handleWheel}
        onMouseDown={handlePanStart} onMouseMove={(e) => { handleMouseMove(e); handlePanMove(e) }}
        onMouseUp={handlePanEnd} onMouseLeave={() => { setCrosshairPos(null); handlePanEnd() }}>
        <canvas ref={canvasRef} width={dimensions.width} height={dimensions.height} className="w-full h-full" />
        <DrawingCanvas
          width={dimensions.width} height={dimensions.height}
          activeTool={activeTool} drawingColor={drawingColor} drawingWidth={drawingWidth}
          drawings={drawings} onAddDrawing={onAddDrawing} onRemoveDrawing={onRemoveDrawing}
          candles={visibleCandles} visible={drawingsVisible}
          minP={mappings?.minP} maxP={mappings?.maxP}
          chartH={mappings?.chartH} chartW={mappings?.chartW}
          indexOffset={viewRange?.start || 0}
          fullCandles={data?.candles}
        />
      </div>

      {/* Kennlinien bei Hover */}
      {hoveredCandle && (
        <CandleInfoBar
          candle={hoveredCandle.candle}
          candles={visibleCandles}
          index={hoveredCandle.index}
        />
      )}
    </div>
  )
}
