import { useEffect, useState, useMemo } from 'react'
import { useSearchStore } from '../../stores/searchStore'
import { useDrawingsStore } from '../../stores/drawingsStore'
import { CHART_SETTINGS } from '../../config/chartSettings'
import api from '../../utils/api'
import Toolbar from './Toolbar'
import ChartGrid from './ChartGrid'
import ChartControls from './ChartControls'
import IndicatorPanel from './IndicatorPanel'
import PatternPanel from './PatternPanel'
import FullscreenChart from './FullscreenChart'
import CounterSearchPanel from './CounterSearchPanel'
import SetManagerPanel from './SetManagerPanel'
import AnomaliesPanel from './AnomaliesPanel'
import { itemsToDrawings } from './utils/itemsToDrawings'
import { aggregateCandles, calculateDerivedValues } from './utils/candleUtils'

export default function ChartViewModule() {
  const { selectedEvents, prehistoryMinutes, setPrehistoryMinutes, searchParams, results, setSearchParams, search: performSearch } = useSearchStore()
  const [mode, setMode] = useState('grid')
  const [chartType, setChartType] = useState('candle')
  const [candleTimeframe, setCandleTimeframe] = useState('5m')
  const [loading, setLoading] = useState(false)
  const [chartData, setChartData] = useState({})
  const [showVolume, setShowVolume] = useState(true)
  const [error, setError] = useState(null)

  // Drawing
  const [activeTool, setActiveTool] = useState('cursor')
  const [drawingColor, setDrawingColor] = useState(CHART_SETTINGS.drawing.defaultColor)
  const [drawingWidth, setDrawingWidth] = useState(CHART_SETTINGS.drawing.defaultLineWidth)
  const drawings = useDrawingsStore(s => s.drawings)
  const setDrawings = (updater) => {
    const current = useDrawingsStore.getState().drawings
    const next = typeof updater === 'function' ? updater(current) : updater
    useDrawingsStore.getState().setAll(next)
  }
  const [drawingsVisible, setDrawingsVisible] = useState(true)

  // Indicators
  const [activeIndicators, setActiveIndicators] = useState([])
  const [showIndicatorPanel, setShowIndicatorPanel] = useState(false)
  const [showPatternPanel, setShowPatternPanel] = useState(false)

  // Pattern hover highlight
  const [hoveredPattern, setHoveredPattern] = useState(null)
  const [showCounterSearch, setShowCounterSearch] = useState(false)
  const [showSetManager, setShowSetManager] = useState(false)
  const [showAnomalies, setShowAnomalies] = useState(false)
  const [syncedHoverTime, setSyncedHoverTime] = useState(null)  // relativeTime in Sek, null=kein Hover
  const [counterResults, setCounterResults] = useState(null)

  // Active chart + Fullscreen
  const [activeChartId, setActiveChartId] = useState(null)
  const [fullscreenEventId, setFullscreenEventId] = useState(null)

  const displayEvents = useMemo(
    () => selectedEvents.slice(0, CHART_SETTINGS.display.maxCharts),
    [selectedEvents]
  )

  useEffect(() => {
    if (displayEvents.length === 0) { setChartData({}); return }
    loadChartData()
  }, [displayEvents, prehistoryMinutes, candleTimeframe])

  const loadChartData = async () => {
    setLoading(true)
    setError(null)
    const newData = {}
    for (const event of displayEvents) {
      try {
        const eventStart = new Date(event.event_start.replace(' ', 'T'))
        const start = new Date(eventStart.getTime() - prehistoryMinutes * 60 * 1000)
        const end = new Date(eventStart.getTime() + event.duration_minutes * 60 * 1000)
        const response = await api.get('/api/v1/search/candles', {
          params: { symbol: event.symbol, start: start.toISOString(), end: end.toISOString(), interval: '1m' }
        })
        const candles = response.data.candles
        if (!candles) throw new Error(`API response missing candles for ${event.symbol}`)
        if (candles.length === 0) continue
        const eventStartTs = eventStart.getTime() / 1000
        const withRelativeTime = candles.map(c => ({ ...c, relativeTime: c.time - eventStartTs }))
        const tfMinutes = CHART_SETTINGS.timeframes.find(t => t.key === candleTimeframe)?.minutes
        if (!tfMinutes) throw new Error(`unknown timeframe: ${candleTimeframe}`)
        const aggregated = aggregateCandles(withRelativeTime, tfMinutes)
        const withDerived = calculateDerivedValues(aggregated)

        // Tagesanfangswert (00:00 Berlin) des Event-Tages laden
        const dateStr = eventStart.toLocaleDateString('sv-SE', { timeZone: 'Europe/Berlin' }) // YYYY-MM-DD
        let dayOpen = null
        try {
          const doRes = await api.get('/api/v1/search/day-open', { params: { symbol: event.symbol, date: dateStr } })
          dayOpen = doRes.data.open
        } catch (err) {
          console.warn(`day-open fehlt fuer ${event.symbol} ${dateStr}`)
          // Kein Fallback — wenn dayOpen fehlt, ist das Event nicht verwendbar
          continue
        }

        newData[event.id] = {
          event, candles: withDerived, rawCandles: withRelativeTime,
          eventStartTime: eventStartTs, dayOpen,
        }
      } catch (err) {
        console.error(`Failed to load candles for ${event.symbol}:`, err)
      }
    }
    setChartData(newData)
    // Drawings fuer neue Events initialisieren
    setDrawings(prev => {
      const next = { ...prev }
      for (const id of Object.keys(newData)) {
        if (!next[id]) next[id] = []
      }
      return next
    })
    setLoading(false)
  }


  const handleLoadSet = async (loadedSet) => {
    if (!loadedSet) return
    // 1) Mainsearch-Params im searchStore wiederherstellen
    if (loadedSet.mainsearch_params && setSearchParams) {
      setSearchParams({
        direction: loadedSet.mainsearch_params.direction,
        targetPercent: loadedSet.mainsearch_params.target_percent,
        maxPercent: loadedSet.mainsearch_params.max_percent,
        durationMinutes: loadedSet.mainsearch_params.duration_minutes,
        startDate: loadedSet.mainsearch_params.start_date?.slice(0,10),
        endDate: loadedSet.mainsearch_params.end_date?.slice(0,10),
        groupIds: loadedSet.mainsearch_params.coin_group_id ? [loadedSet.mainsearch_params.coin_group_id] : [],
      })
    }
    // 2) Primary-Suche triggern (wird im SearchStore gemacht)
    if (performSearch) { try { await performSearch() } catch {} }
    // 3) Drawings auf aktuell aktivem Event projizieren
    if (activeChartId && chartData[activeChartId]) {
      try {
        const data = chartData[activeChartId]
        const drawingsRecon = itemsToDrawings(loadedSet.drawings, data.candles, data.dayOpen)
        setDrawings(prev => ({ ...prev, [activeChartId]: drawingsRecon }))
      } catch (e) {
        console.error('itemsToDrawings failed:', e)
      }
    }
    setShowSetManager(false)
  }

  const addDrawing = (eventId, drawing) => {
    setDrawings(prev => ({ ...prev, [eventId]: [...(prev[eventId] || []), drawing] }))
  }
  const removeDrawing = (eventId, drawingIndex) => {
    setDrawings(prev => ({ ...prev, [eventId]: (prev[eventId] || []).filter((_, i) => i !== drawingIndex) }))
  }
  const clearDrawings = (eventId) => {
    if (eventId) setDrawings(prev => ({ ...prev, [eventId]: [] }))
    else setDrawings({})
  }

  // Pattern data: only pass selected events' chart data
  const patternChartData = useMemo(() => {
    if (activeChartId && chartData[activeChartId]) {
      return { [activeChartId]: chartData[activeChartId] }
    }
    // All displayed (=selected) events
    const result = {}
    for (const ev of displayEvents) {
      if (chartData[ev.id]) result[ev.id] = chartData[ev.id]
    }
    return result
  }, [activeChartId, chartData, displayEvents])

  if (displayEvents.length === 0) {
    return <div className="text-gray-400 text-center py-8">Events aus den Suchergebnissen auswaehlen.</div>
  }

  return (
    <div className="h-full flex flex-col">
      <Toolbar
        activeTool={activeTool} setActiveTool={setActiveTool}
        drawingColor={drawingColor} setDrawingColor={setDrawingColor}
        drawingWidth={drawingWidth} setDrawingWidth={setDrawingWidth}
        onClearAll={() => clearDrawings()}
        onToggleIndicators={() => setShowIndicatorPanel(!showIndicatorPanel)}
        onTogglePatterns={() => setShowPatternPanel(!showPatternPanel)}
        showIndicatorPanel={showIndicatorPanel} showPatternPanel={showPatternPanel}
        drawingsVisible={drawingsVisible}
        onToggleCounterSearch={() => setShowCounterSearch(!showCounterSearch)}
        showCounterSearch={showCounterSearch}
        onToggleSetManager={() => setShowSetManager(!showSetManager)}
        showSetManager={showSetManager}
        onToggleAnomalies={() => setShowAnomalies(!showAnomalies)}
        showAnomalies={showAnomalies}
        onToggleDrawingsVisible={() => setDrawingsVisible(!drawingsVisible)}
      />
      <ChartControls
        mode={mode} setMode={setMode}
        chartType={chartType} setChartType={setChartType}
        candleTimeframe={candleTimeframe} setCandleTimeframe={setCandleTimeframe}
        prehistoryMinutes={prehistoryMinutes} setPrehistoryMinutes={setPrehistoryMinutes}
        showVolume={showVolume} setShowVolume={setShowVolume}
        loading={loading} onRefresh={loadChartData}
        eventCount={displayEvents.length}
      />
      <div className="flex-1 flex overflow-hidden">
        {showIndicatorPanel && (
          <IndicatorPanel activeIndicators={activeIndicators}
          setActiveIndicators={setActiveIndicators}
            onClose={() => setShowIndicatorPanel(false)} />
        )}
        {showPatternPanel && (
          <PatternPanel chartData={patternChartData} candleTimeframe={candleTimeframe}
            onClose={() => setShowPatternPanel(false)}
            hoveredPattern={hoveredPattern} setHoveredPattern={setHoveredPattern} />
        )}
        {showSetManager && (
          <SetManagerPanel
            onClose={() => setShowSetManager(false)}
            onLoadSet={handleLoadSet}
          />
        )}
                {showAnomalies && (
          <AnomaliesPanel
            results={results}
            displayEvents={displayEvents}
            prehistoryMinutes={prehistoryMinutes}
            candleTimeframeMinutes={CHART_SETTINGS.timeframes.find(t => t.key === candleTimeframe)?.minutes}
            primaryContext={searchParams ? {
              search_date_from: searchParams.startDate,
              search_date_to: searchParams.endDate,
              search_percent_min: searchParams.targetPercent,
              search_percent_max: searchParams.maxPercent,
              search_direction: searchParams.direction,
              search_duration_minutes: searchParams.durationMinutes,
              events_at_creation: (results || []).length,
              coin_group_id: searchParams.coinGroupId,
            } : null}
            onClose={() => setShowAnomalies(false)}
          />
        )}
        {showCounterSearch && (
          <CounterSearchPanel
            durationMinutes={displayEvents[0]?.duration_minutes}
            candleTimeframeMinutes={CHART_SETTINGS.timeframes.find(t => t.key === candleTimeframe)?.minutes}
            prehistoryMinutes={prehistoryMinutes}
            direction={searchParams?.direction}
            primaryContext={searchParams ? {
              search_date_from: searchParams.startDate,
              search_date_to: searchParams.endDate,
              search_percent_min: searchParams.targetPercent,
              search_percent_max: searchParams.maxPercent,
              search_direction: searchParams.direction,
              search_duration_minutes: searchParams.durationMinutes,
              events_at_creation: (results || []).length,
              coin_group_id: (searchParams.groupIds && searchParams.groupIds.length > 0) ? searchParams.groupIds[0] : null,
            } : null}
            drawings={drawings} chartData={chartData} activeChartId={activeChartId}
            onClose={() => setShowCounterSearch(false)}
            onResultsFound={setCounterResults}
          />
        )}
        <div className="flex-1 overflow-auto p-2">
          {error && <div className="p-2 text-red-400 text-sm bg-red-900/30 mb-2">{error}</div>}
          <ChartGrid
            displayEvents={displayEvents} chartData={chartData}
            mode={mode} chartType={chartType} showVolume={showVolume}
            activeTool={activeTool} drawingColor={drawingColor} drawingWidth={drawingWidth}
            drawings={drawings} onAddDrawing={addDrawing} onRemoveDrawing={removeDrawing}
            activeIndicators={activeIndicators}
          setActiveIndicators={setActiveIndicators}
            activeChartId={activeChartId} setActiveChartId={setActiveChartId}
            drawingsVisible={drawingsVisible} loading={loading}
            hoveredPattern={hoveredPattern}
            onDoubleClick={(eventId) => setFullscreenEventId(eventId)}
            syncedHoverTime={syncedHoverTime}
            onSyncHover={setSyncedHoverTime}
          />
        </div>
      </div>

      {/* Fullscreen overlay */}
      {fullscreenEventId && chartData[fullscreenEventId] && (
        <FullscreenChart
          event={chartData[fullscreenEventId].event}
          data={chartData[fullscreenEventId]}
          chartType={chartType} showVolume={showVolume}
          activeIndicators={activeIndicators}
          setActiveIndicators={setActiveIndicators}
          drawings={drawings[fullscreenEventId] || []}
          onAddDrawing={(d) => addDrawing(fullscreenEventId, d)}
          onRemoveDrawing={(i) => removeDrawing(fullscreenEventId, i)}
          activeTool={activeTool} drawingColor={drawingColor} drawingWidth={drawingWidth}
          drawingsVisible={drawingsVisible}
          hoveredPattern={hoveredPattern}
          onClose={() => setFullscreenEventId(null)}
        />
      )}
    </div>
  )
}
