import { useEffect, useRef, useState, useCallback } from 'react'
import { X, RotateCcw, Maximize, Minimize } from 'lucide-react'
import { CHART_SETTINGS, MARKER_COLORS } from '../../config/chartSettings'
import { renderChart } from './utils/canvasRenderer'
import { computeIndicator } from './utils/indicatorCalc'
import { computeChartMappings } from './utils/chartMappings'
import { detectPatterns } from './utils/patternDetect'
import DrawingCanvas from './DrawingCanvas'
import Toolbar from './Toolbar'
import IndicatorPanel from './IndicatorPanel'
import PatternPanel from './PatternPanel'

export default function FullscreenChart({
  event, data, chartType, showVolume,
  activeIndicators: parentIndicators, setActiveIndicators: setParentIndicators,
  drawings, onAddDrawing, onRemoveDrawing,
  activeTool: parentTool, drawingColor: parentColor, drawingWidth: parentWidth,
  drawingsVisible, hoveredPattern, onClose,
}) {
  const wrapperRef = useRef(null)
  const canvasRef = useRef(null)
  const containerRef = useRef(null)
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 })
  const [crosshairPos, setCrosshairPos] = useState(null)
  const [viewRange, setViewRange] = useState(null)
  const [isPanning, setIsPanning] = useState(false)
  const [panStart, setPanStart] = useState(0)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [candleInfo, setCandleInfo] = useState(null)

  // Own tool state (inherits from parent but can override in fullscreen)
  const [activeTool, setActiveTool] = useState(parentTool)
  const [drawingColor, setDrawingColor] = useState(parentColor)
  const [drawingWidth, setDrawingWidth] = useState(parentWidth)
  const [showIndicatorPanel, setShowIndicatorPanel] = useState(false)
  const [showPatternPanel, setShowPatternPanel] = useState(false)

  const totalCandles = data?.candles?.length || 0
  const visibleCandles = viewRange && data?.candles
    ? data.candles.slice(viewRange.start, viewRange.end) : data?.candles || []

  // Indicators — auf rawCandles (1m) berechnen, auf sichtbare Candles mappen
  const indicatorData = parentIndicators.map((ind, i) => {
    const rawData = data?.rawCandles
    if (!rawData) throw new Error("rawCandles missing in chartData")
    const result = computeIndicator(ind, rawData)
    if (!result) return null
    const rawLen = rawData?.length || 0
    const ratio = rawLen / (data?.candles?.length || rawLen)
    const mapVis = (vals) => {
      if (!vals) return null
      const mapped = vals.map(v => ({ ...v, idx: Math.round(v.idx / ratio) }))
      if (viewRange) {
        return mapped.filter(v => v.idx >= viewRange.start && v.idx < viewRange.end)
          .map(v => ({ ...v, idx: v.idx - viewRange.start }))
      }
      return mapped.filter(v => v.idx >= 0 && v.idx < (data?.candles?.length || 0))
    }
    return {
      ...ind,
      color: CHART_SETTINGS.indicators.defaultColors[i % CHART_SETTINGS.indicators.defaultColors.length],
      data: result.upper ? { upper: mapVis(result.upper), middle: mapVis(result.middle), lower: mapVis(result.lower) }
        : { values: mapVis(result.values || result) },
    }
  }).filter(Boolean)

  const patternHighlights = hoveredPattern && data?.candles
    ? (detectPatterns(data.candles, [hoveredPattern])[hoveredPattern] || []).map(h => h.index) : []

  // Real browser fullscreen
  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      wrapperRef.current?.requestFullscreen?.()
    } else {
      document.exitFullscreen?.()
    }
  }
  useEffect(() => {
    const handler = () => setIsFullscreen(!!document.fullscreenElement)
    document.addEventListener('fullscreenchange', handler)
    // Auto-enter fullscreen
    wrapperRef.current?.requestFullscreen?.()
    return () => document.removeEventListener('fullscreenchange', handler)
  }, [])

  // Resize
  useEffect(() => {
    if (!containerRef.current) return
    const obs = new ResizeObserver(entries => {
      for (const e of entries) setDimensions({ width: e.contentRect.width, height: e.contentRect.height })
    })
    obs.observe(containerRef.current)
    setDimensions({ width: containerRef.current.clientWidth, height: containerRef.current.clientHeight })
    return () => obs.disconnect()
  }, [])

  // Render
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !visibleCandles.length || dimensions.width === 0) return
    renderChart(canvas.getContext('2d'), visibleCandles, {
      width: dimensions.width, height: dimensions.height,
      chartType, showVolume, eventColor: CHART_SETTINGS.eventColors[0],
      indicators: indicatorData, crosshairPos, patternHighlights,
    })
  }, [visibleCandles, dimensions, chartType, showVolume, indicatorData, crosshairPos, patternHighlights])

  // Candle info at cursor
  const updateCandleInfo = useCallback((mouseX) => {
    if (!visibleCandles.length || !dimensions.width) { setCandleInfo(null); return }
    const step = (dimensions.width - 70) / visibleCandles.length
    const idx = Math.round((mouseX - step / 2) / step)
    if (idx < 0 || idx >= visibleCandles.length) { setCandleInfo(null); return }
    const c = visibleCandles[idx]
    const prev1h = visibleCandles[Math.max(0, idx - 60)] // ~1h ago at 1m
    const prev4h = visibleCandles[Math.max(0, idx - 240)]
    const prev24h = visibleCandles[0]
    const pctFrom = (ref) => ref ? ((c.close - ref.close) / ref.close * 100).toFixed(2) : '?'
    setCandleInfo({
      open: c.open, high: c.high, low: c.low, close: c.close,
      volume: c.volume, trades: c.trades || 0,
      change1h: pctFrom(prev1h), change4h: pctFrom(prev4h), changeTotal: pctFrom(prev24h),
      relTime: Math.round(c.relativeTime / 60),
      isGreen: c.close >= c.open,
    })
  }, [visibleCandles, dimensions])

  // Mappings direkt berechnen
  const separateIndCount = indicatorData.filter(i => i.config?.separate && i.visible !== false).length
  const mappings = computeChartMappings(visibleCandles, dimensions.width, dimensions.height, showVolume, separateIndCount)

  // Zoom
  const handleWheel = useCallback((e) => {
    e.preventDefault()
    if (totalCandles < 10) return
    const cur = viewRange || { start: 0, end: totalCandles }
    const vis = cur.end - cur.start
    let nc = Math.round(vis * (e.deltaY > 0 ? 1.15 : 0.87))
    nc = Math.max(10, Math.min(totalCandles, nc))
    if (nc === totalCandles) { setViewRange(null); return }
    const rect = canvasRef.current?.getBoundingClientRect()
    const ratio = rect ? (e.clientX - rect.left) / rect.width : 0.5
    const center = cur.start + Math.round(vis * ratio)
    let s = Math.round(center - nc * ratio), end = s + nc
    if (s < 0) { s = 0; end = nc }
    if (end > totalCandles) { end = totalCandles; s = totalCandles - nc }
    setViewRange({ start: Math.max(0, s), end })
  }, [viewRange, totalCandles])

  // Pan
  const handlePanStart = useCallback((e) => {
    if (e.button === 1 || (e.button === 0 && e.shiftKey)) { setIsPanning(true); setPanStart(e.clientX); e.preventDefault() }
  }, [])
  const handlePanMove = useCallback((e) => {
    if (!isPanning || !viewRange) return
    const rect = canvasRef.current?.getBoundingClientRect()
    if (!rect) return
    const cpp = (viewRange.end - viewRange.start) / rect.width
    const shift = Math.round(-(e.clientX - panStart) * cpp)
    if (!shift) return
    let s = viewRange.start + shift, end = viewRange.end + shift
    if (s < 0) { end -= s; s = 0 }
    if (end > totalCandles) { s -= end - totalCandles; end = totalCandles }
    setViewRange({ start: Math.max(0, s), end }); setPanStart(e.clientX)
  }, [isPanning, panStart, viewRange, totalCandles])

  // Escape
  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') { document.exitFullscreen?.().catch(() => {}); onClose() } }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  return (
    <div ref={wrapperRef} className="fixed inset-0 z-50 bg-gray-900 flex flex-col">
      {/* Toolbar */}
      <Toolbar
        activeTool={activeTool} setActiveTool={setActiveTool}
        drawingColor={drawingColor} setDrawingColor={setDrawingColor}
        drawingWidth={drawingWidth} setDrawingWidth={setDrawingWidth}
        onClearAll={() => {}}
        onToggleIndicators={() => setShowIndicatorPanel(!showIndicatorPanel)}
        onTogglePatterns={() => setShowPatternPanel(!showPatternPanel)}
        showIndicatorPanel={showIndicatorPanel} showPatternPanel={showPatternPanel}
        drawingsVisible={drawingsVisible} onToggleDrawingsVisible={() => {}}
      />

      {/* Header with info */}
      <div className="flex items-center gap-3 px-3 py-1.5 border-b border-gray-700 bg-gray-800/80 text-xs">
        <span className="font-mono font-bold text-white text-sm">{event.symbol}</span>
        <span className={`font-mono ${event.change_percent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
          {event.change_percent >= 0 ? '+' : ''}{event.change_percent?.toFixed(2)}%
        </span>

        {/* OHLCV Info Bar */}
        {candleInfo && (
          <>
            <div className="w-px h-4 bg-gray-600" />
            <span className="text-gray-400">O:</span>
            <span className={candleInfo.isGreen ? 'text-green-400' : 'text-red-400'}>{candleInfo.open.toFixed(4)}</span>
            <span className="text-gray-400">H:</span>
            <span className="text-green-400">{candleInfo.high.toFixed(4)}</span>
            <span className="text-gray-400">L:</span>
            <span className="text-red-400">{candleInfo.low.toFixed(4)}</span>
            <span className="text-gray-400">C:</span>
            <span className={candleInfo.isGreen ? 'text-green-400' : 'text-red-400'}>{candleInfo.close.toFixed(4)}</span>
            <span className="text-gray-400">Vol:</span>
            <span className="text-blue-400">{Math.round(candleInfo.volume).toLocaleString()}</span>
            <div className="w-px h-4 bg-gray-600" />
            <span className="text-gray-500">t={candleInfo.relTime}m</span>
            <span className="text-gray-400">1h:</span>
            <span className={parseFloat(candleInfo.change1h) >= 0 ? 'text-green-400' : 'text-red-400'}>{candleInfo.change1h}%</span>
            <span className="text-gray-400">4h:</span>
            <span className={parseFloat(candleInfo.change4h) >= 0 ? 'text-green-400' : 'text-red-400'}>{candleInfo.change4h}%</span>
            <span className="text-gray-400">Ges:</span>
            <span className={parseFloat(candleInfo.changeTotal) >= 0 ? 'text-green-400' : 'text-red-400'}>{candleInfo.changeTotal}%</span>
          </>
        )}

        <div className="flex-1" />

        {viewRange && (
          <button onClick={() => setViewRange(null)}
            className="flex items-center gap-1 px-2 py-0.5 bg-gray-700 hover:bg-gray-600 rounded text-gray-300">
            <RotateCcw size={11} /> Reset
          </button>
        )}
        <span className="text-gray-600">Scroll=Zoom | Shift+Drag=Pan | Esc=Schliessen</span>
        <button onClick={toggleFullscreen} className="p-1 bg-gray-700 hover:bg-gray-600 rounded">
          {isFullscreen ? <Minimize size={14} /> : <Maximize size={14} />}
        </button>
        <button onClick={() => { document.exitFullscreen?.().catch(() => {}); onClose() }}
          className="p-1 bg-gray-700 hover:bg-red-600 rounded">
          <X size={14} />
        </button>
      </div>

      {/* Main area */}
      <div className="flex-1 flex overflow-hidden">
        {showIndicatorPanel && (
          <IndicatorPanel activeIndicators={parentIndicators} setActiveIndicators={setParentIndicators}
            onClose={() => setShowIndicatorPanel(false)} />
        )}
        {showPatternPanel && (
          <PatternPanel chartData={{ [event.id || 'fs']: data }} candleTimeframe="5m"
            onClose={() => setShowPatternPanel(false)}
            hoveredPattern={hoveredPattern} setHoveredPattern={() => {}} />
        )}

        <div ref={containerRef} className="flex-1 relative"
          onWheel={handleWheel}
          onMouseDown={handlePanStart}
          onMouseMove={(e) => {
            if (isPanning) handlePanMove(e)
            else {
              const rect = canvasRef.current?.getBoundingClientRect()
              if (rect) {
                const pos = { x: e.clientX - rect.left, y: e.clientY - rect.top }
                setCrosshairPos(pos)
                updateCandleInfo(pos.x)
              }
            }
          }}
          onMouseUp={() => setIsPanning(false)}
          onMouseLeave={() => { setCrosshairPos(null); setIsPanning(false); setCandleInfo(null) }}>
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
      </div>
    </div>
  )
}
