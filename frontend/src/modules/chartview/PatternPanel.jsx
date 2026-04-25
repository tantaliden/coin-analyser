import { useState, useMemo } from 'react'
import { X, Search, CandlestickChart } from 'lucide-react'
import { CHART_SETTINGS } from '../../config/chartSettings'
import { detectPatterns } from './utils/patternDetect'

export default function PatternPanel({
  chartData, candleTimeframe, onClose,
  hoveredPattern, setHoveredPattern,
}) {
  const [selectedPattern, setSelectedPattern] = useState('')
  const [detectedResults, setDetectedResults] = useState(null)
  const availablePatterns = CHART_SETTINGS.candlePatterns.available
  const eventCount = Object.keys(chartData).length

  // Detect single pattern across selected events
  const runDetection = () => {
    if (!selectedPattern) return
    const results = {}
    let totalFound = 0
    for (const [eventId, data] of Object.entries(chartData)) {
      if (!data?.candles?.length) continue
      const detected = detectPatterns(data.candles, [selectedPattern])
      const hits = detected[selectedPattern] || []
      if (hits.length > 0) {
        results[eventId] = { symbol: data.event.symbol, hits, changePercent: data.event.change_percent }
        totalFound += hits.length
      }
    }
    setDetectedResults({ pattern: selectedPattern, results, totalFound, totalEvents: eventCount })
  }

  // Full scan all patterns
  const runFullScan = () => {
    const summary = {}
    for (const pattern of availablePatterns) {
      let count = 0
      let eventsWithPattern = 0
      for (const [, data] of Object.entries(chartData)) {
        if (!data?.candles?.length) continue
        const detected = detectPatterns(data.candles, [pattern.id])
        const hits = (detected[pattern.id] || []).length
        if (hits > 0) { count += hits; eventsWithPattern++ }
      }
      if (count > 0) summary[pattern.id] = { count, events: eventsWithPattern }
    }
    setDetectedResults({ pattern: '_all', summary, totalEvents: eventCount })
  }

  return (
    <div className="w-64 border-r border-gray-700 bg-gray-800/80 flex flex-col overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-700">
        <span className="text-xs font-semibold text-gray-300">Kerzen-Muster</span>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-gray-500">{eventCount} Events</span>
          <button onClick={onClose} className="text-gray-500 hover:text-white"><X size={14} /></button>
        </div>
      </div>

      {/* Pattern Selector */}
      <div className="p-2 border-b border-gray-700 space-y-2">
        <select value={selectedPattern} onChange={e => setSelectedPattern(e.target.value)}
          className="input text-xs py-1 w-full">
          <option value="">Muster auswaehlen...</option>
          {availablePatterns.map(p => <option key={p.id} value={p.id}>{p.label}</option>)}
        </select>
        <div className="flex gap-1">
          <button onClick={runDetection} disabled={!selectedPattern}
            className="flex-1 flex items-center justify-center gap-1 px-2 py-1 bg-amber-600 hover:bg-amber-500 disabled:bg-gray-700 rounded text-xs">
            <Search size={10} /> Suchen
          </button>
          <button onClick={runFullScan}
            className="flex items-center justify-center gap-1 px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs"
            title="Alle Muster scannen">
            <CandlestickChart size={10} /> Alle
          </button>
        </div>
      </div>

      {selectedPattern && (
        <div className="px-3 py-1.5 bg-gray-900/50 text-[10px] text-gray-400 border-b border-gray-700">
          {availablePatterns.find(p => p.id === selectedPattern)?.description}
        </div>
      )}

      {/* Results */}
      <div className="flex-1 overflow-auto p-2">
        {!detectedResults && (
          <div className="text-gray-500 text-xs text-center py-4">Muster auswaehlen und suchen</div>
        )}

        {/* Full scan results */}
        {detectedResults?.pattern === '_all' && (
          <div className="space-y-1">
            <div className="text-xs text-gray-400 mb-2">
              Scan ueber {detectedResults.totalEvents} ausgewaehlte Events:
            </div>
            {Object.entries(detectedResults.summary)
              .sort(([, a], [, b]) => b.count - a.count)
              .map(([pid, info]) => {
                const meta = availablePatterns.find(p => p.id === pid)
                const isHovered = hoveredPattern === pid
                return (
                  <div key={pid}
                    className={`flex items-center justify-between text-xs py-1.5 px-2 rounded cursor-pointer transition-colors ${
                      isHovered ? 'bg-amber-900/40 border border-amber-600/50' : 'bg-gray-900/30 border border-transparent hover:bg-gray-800'
                    }`}
                    onMouseEnter={() => setHoveredPattern(pid)}
                    onMouseLeave={() => setHoveredPattern(null)}
                  >
                    <span className="text-gray-300">{meta?.label || pid}</span>
                    <div className="flex items-center gap-2">
                      <span className="text-gray-500 text-[10px]">{info.events}/{detectedResults.totalEvents}</span>
                      <span className="text-amber-400 font-mono">{info.count}x</span>
                    </div>
                  </div>
                )
              })}
            {Object.keys(detectedResults.summary).length === 0 && (
              <div className="text-gray-500 text-xs text-center py-2">Keine Muster gefunden</div>
            )}
          </div>
        )}

        {/* Single pattern results */}
        {detectedResults?.pattern !== '_all' && detectedResults?.results && (
          <div className="space-y-1">
            <div className="text-xs text-gray-400 mb-2">
              {detectedResults.totalFound}x in {Object.keys(detectedResults.results).length}/{detectedResults.totalEvents} Events
            </div>
            {Object.entries(detectedResults.results).map(([eventId, info]) => (
              <div key={eventId}
                className="text-xs py-1 px-2 bg-gray-900/30 rounded flex items-center justify-between hover:bg-gray-800 cursor-pointer"
                onMouseEnter={() => setHoveredPattern(detectedResults.pattern)}
                onMouseLeave={() => setHoveredPattern(null)}
              >
                <span className="font-mono text-gray-200">{info.symbol}</span>
                <div className="flex items-center gap-2">
                  <span className={info.changePercent >= 0 ? 'text-green-400' : 'text-red-400'}>
                    {info.changePercent >= 0 ? '+' : ''}{info.changePercent?.toFixed(1)}%
                  </span>
                  <span className="text-amber-400">{info.hits.length}x</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
