import { useState, useMemo, useCallback, useEffect } from 'react'
import { ChevronUp, ChevronDown, Columns, Shuffle, CheckSquare, Square, CandlestickChart, X } from 'lucide-react'
import { useSearchStore } from '../stores/searchStore'
import { formatDateTime, formatPercent, formatNumber } from '../utils/format'
import api from '../utils/api'

const ALL_COLUMNS = [
  { key: 'symbol', label: 'Symbol', default: true },
  { key: 'event_start', label: 'Start', default: true },
  { key: 'event_end', label: 'Ende', default: false },
  { key: 'start_price', label: 'Preis', default: false },
  { key: 'change_percent', label: '%', default: true },
  { key: 'duration_minutes', label: 'Dauer', default: false },
  { key: 'direction', label: 'Dir', default: false },
  { key: 'volume', label: 'Volume', default: false },
  { key: 'trades_count', label: 'Trades', default: false },
  { key: 'volatility_pct', label: 'Volatilität', default: false },
  { key: 'max_drawdown_pct', label: 'Drawdown', default: false },
]

const TIMEFRAME_OPTIONS = [
  { value: '5m', label: '5m' },
  { value: '10m', label: '10m' },
  { value: '15m', label: '15m' },
  { value: '30m', label: '30m' },
  { value: '1h', label: '1h' },
  { value: '2h', label: '2h' },
  { value: '4h', label: '4h' },
]

const CANDLE_COUNT_OPTIONS = [5, 10, 15, 20, 25, 30]

export default function SearchResultsModule() {
  const {
    results, cascadeResults, indicatorChain, loading,
    selectedEvents, toggleEvent, selectAll, deselectAll, isSelected, maxSelection,
    selectedSetId, prehistoryMinutes, setIndicatorChain,
    useMainSearch, setUseMainSearch, applyCascadeFilter
  } = useSearchStore()
  
  const activeResults = indicatorChain?.length > 0 && cascadeResults?.length > 0 ? cascadeResults : results
  
  const [visibleColumns, setVisibleColumns] = useState(['symbol', 'event_start', 'change_percent'])
  const [sortConfig, setSortConfig] = useState({ key: 'event_start', direction: 'desc' })
  const [showColumnPicker, setShowColumnPicker] = useState(false)
  const [showFilter, setShowFilter] = useState('all')
  
  // Pre-Candles State
  const [showPreCandles, setShowPreCandles] = useState(true)
  const [preCandleTimeframe, setPreCandleTimeframe] = useState('15m')
  const [preCandleCount, setPreCandleCount] = useState(10)
  const [preCandles, setPreCandles] = useState({})
  const [preCandlesLoading, setPreCandlesLoading] = useState(false)
  const [candleFilters, setCandleFilters] = useState({})
  
  // Pattern-Indikator aus Chain finden
  const patternIndicator = useMemo(() => {
    return indicatorChain?.find(ind => ind.indicator_type === 'candle_pattern' && ind.is_active !== false)
  }, [indicatorChain])
  
  // candleFilters aus Pattern-Indikator laden
  useEffect(() => {
    if (patternIndicator?.pattern_data) {
      setCandleFilters(patternIndicator.pattern_data)
    }
  }, [patternIndicator])
  
  const handleSort = (key) => {
    setSortConfig(prev => ({ key, direction: prev.key === key && prev.direction === 'asc' ? 'desc' : 'asc' }))
  }
  
  const toggleColumn = (key) => {
    setVisibleColumns(prev => prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key])
  }
  
  const loadPreCandles = useCallback(async () => {
    if (activeResults.length === 0) return
    setPreCandlesLoading(true)
    try {
      const events = activeResults.map(e => ({
        id: e.id,
        symbol: e.symbol,
        event_start: e.event_start
      }))
      
      let requestParams = {
        events,
        timeframe: preCandleTimeframe,
        candle_count: preCandleCount
      }
      
      if (patternIndicator && prehistoryMinutes) {
        const timeStart = patternIndicator.time_start_minutes || 0
        const timeEnd = patternIndicator.time_end_minutes || 0
        const aggregator = patternIndicator.aggregator || '1m'
        const aggMinutes = aggregator.endsWith('m') ? parseInt(aggregator) :
                          aggregator.endsWith('h') ? parseInt(aggregator) * 60 :
                          aggregator.endsWith('d') ? parseInt(aggregator) * 1440 : 1
        const candleCount = Math.ceil((timeEnd - timeStart) / aggMinutes)
        
        requestParams = {
          events,
          timeframe: aggregator,
          candle_count: candleCount,
          prehistory_minutes: prehistoryMinutes,
          time_end_minutes: timeEnd
        }
      }
      
      const response = await api.post('/api/v1/search/pre-candles', requestParams)
      setPreCandles(response.data.candles || {})
    } catch (err) {
      console.error('Failed to load pre-candles:', err)
    } finally {
      setPreCandlesLoading(false)
    }
  }, [activeResults, preCandleTimeframe, preCandleCount, patternIndicator, prehistoryMinutes])
  
  // Candles automatisch laden wenn Ergebnisse da
  useEffect(() => {
    if (activeResults.length > 0 && showPreCandles && Object.keys(preCandles).length === 0) {
      loadPreCandles()
    }
  }, [activeResults.length, showPreCandles])

  // preCandles neu laden wenn Pattern-Indikator sich ändert
  useEffect(() => {
    setPreCandles({})
  }, [patternIndicator?.item_id, patternIndicator?.is_active])
  
  // Candle-Filter togglen UND im Pattern-Indikator speichern
  const toggleCandleFilter = async (position, color) => {
    const newFilters = { ...candleFilters }
    if (newFilters[position] === color) {
      delete newFilters[position]
    } else {
      newFilters[position] = color
    }
    
    setCandleFilters(newFilters)
    
    if (patternIndicator?.item_id) {
      try {
        await api.put(`/api/v1/indicators/items/${patternIndicator.item_id}`, {
          pattern_data: newFilters
        })
        const updatedChain = indicatorChain.map(ind => 
          ind.item_id === patternIndicator.item_id 
            ? { ...ind, pattern_data: newFilters }
            : ind
        )
        setIndicatorChain(updatedChain)
        applyCascadeFilter(updatedChain.filter(i => i.is_active !== false), prehistoryMinutes)
      } catch (err) {
        console.error('Failed to save pattern:', err)
      }
    }
  }
  
  const clearCandleFilters = async () => {
    setCandleFilters({})
    if (patternIndicator?.item_id) {
      try {
        await api.put(`/api/v1/indicators/items/${patternIndicator.item_id}`, {
          pattern_data: {}
        })
        const updatedChain = indicatorChain.map(ind => 
          ind.item_id === patternIndicator.item_id 
            ? { ...ind, pattern_data: {} }
            : ind
        )
        setIndicatorChain(updatedChain)
        applyCascadeFilter(updatedChain.filter(i => i.is_active !== false), prehistoryMinutes)
      } catch (err) {
        console.error('Failed to clear pattern:', err)
      }
    }
  }
  
  // Neuen Pattern-Indikator erstellen
  const createPatternIndicator = async () => {
    if (Object.keys(candleFilters).length === 0) return
    if (!selectedSetId || !prehistoryMinutes) return
    
    const tfMinutes = parseInt(preCandleTimeframe) || 15
    const patternMinutes = preCandleCount * tfMinutes
    const timeEnd = prehistoryMinutes - 15
    const timeStart = Math.max(0, timeEnd - patternMinutes)
    
    try {
      const res = await api.post(`/api/v1/indicators/sets/${selectedSetId}/items`, {
        field: 'candle_pattern',
        operation: 'match',
        value: null,
        time_start_minutes: timeStart,
        time_end_minutes: timeEnd,
        aggregator: preCandleTimeframe,
        color: '#F59E0B',
        pattern_type: 'exact',
        pattern_data: candleFilters,
        pattern_count: Object.keys(candleFilters).length,
        pattern_consecutive: true
      })
      
      setIndicatorChain([...indicatorChain, {
        item_id: res.data.item_id,
        indicator_type: 'candle_pattern',
        condition_operator: 'match',
        time_start_minutes: timeStart,
        time_end_minutes: timeEnd,
        aggregator: preCandleTimeframe,
        pattern_data: candleFilters,
        color: '#F59E0B',
        is_active: true
      }])
    } catch (err) {
      alert('Fehler: ' + (err.response?.data?.detail || err.message))
    }
  }
  
  // Sortierte + gefilterte Ergebnisse
  const sortedResults = useMemo(() => {
    let filtered = [...activeResults]
    
    if (showFilter === 'selected') {
      filtered = filtered.filter(e => isSelected(e.id))
    } else if (showFilter === 'unselected') {
      filtered = filtered.filter(e => !isSelected(e.id))
    }
    
    // Client-seitiges Candle-Pattern-Filter (nur ohne Indikator-Chain)
    if (showPreCandles && Object.keys(candleFilters).length > 0 && (!indicatorChain || indicatorChain.length === 0)) {
      filtered = filtered.filter(e => {
        const candles = preCandles[e.id] || []
        for (const [pos, color] of Object.entries(candleFilters)) {
          if (candles[parseInt(pos)] !== color) return false
        }
        return true
      })
    }
    
    filtered.sort((a, b) => {
      if (sortConfig.key === '_selected') {
        const aS = isSelected(a.id) ? 1 : 0
        const bS = isSelected(b.id) ? 1 : 0
        if (aS !== bS) return sortConfig.direction === 'asc' ? aS - bS : bS - aS
        return new Date(b.event_start).getTime() - new Date(a.event_start).getTime()
      }
      
      let aVal = a[sortConfig.key]
      let bVal = b[sortConfig.key]
      if (['change_percent', 'duration_minutes', 'volume', 'start_price', 'trades_count', 'volatility_pct', 'max_drawdown_pct'].includes(sortConfig.key)) {
        aVal = parseFloat(aVal) || 0
        bVal = parseFloat(bVal) || 0
      }
      if (['event_start', 'event_end'].includes(sortConfig.key)) {
        aVal = new Date(aVal).getTime()
        bVal = new Date(bVal).getTime()
      }
      if (typeof aVal === 'string' && typeof bVal === 'string') {
        const cmp = aVal.localeCompare(bVal)
        return sortConfig.direction === 'asc' ? cmp : -cmp
      }
      if (aVal < bVal) return sortConfig.direction === 'asc' ? -1 : 1
      if (aVal > bVal) return sortConfig.direction === 'asc' ? 1 : -1
      return 0
    })
    return filtered
  }, [activeResults, sortConfig, showFilter, selectedEvents, showPreCandles, candleFilters, preCandles, indicatorChain])
  
  const formatValue = (key, value) => {
    if (value === null || value === undefined) return '-'
    switch (key) {
      case 'event_start':
      case 'event_end':
        return formatDateTime(value)
      case 'change_percent':
      case 'volatility_pct':
      case 'max_drawdown_pct':
        return formatPercent(value)
      case 'start_price':
        return formatNumber(value, 4)
      case 'volume':
      case 'avg_volume_per_candle':
        return formatNumber(value, 0)
      case 'direction':
        return value === 'up' ? '↑' : '↓'
      default:
        return value
    }
  }
  
  const candleColor = (color) => {
    if (color === 'green') return '#22c55e'
    if (color === 'red') return '#ef4444'
    return '#6b7280'
  }
  
  // Random aus gefilterten Ergebnissen
  const selectRandomFromFiltered = (count) => {
    const shuffled = [...sortedResults].sort(() => Math.random() - 0.5)
    const toSelect = shuffled.slice(0, Math.min(count, maxSelection))
    useSearchStore.setState({ selectedEvents: toSelect })
  }

  return (
    <div className="h-full flex flex-col gap-1 text-xs">
      {/* Header */}
      <div className="flex items-center gap-1.5 flex-wrap px-1 pt-1">
        <span className="text-gray-500">
          {sortedResults.length} Events | {selectedEvents.length}/{maxSelection}
        </span>
        
        {cascadeResults?.length > 0 && (
          <span className="text-blue-400">
            ({results.length} → {cascadeResults.length})
          </span>
        )}
        
        <div className="flex-1" />
        
        <select value={showFilter} onChange={e => setShowFilter(e.target.value)}
          className="bg-zinc-800 text-gray-300 border border-zinc-700 rounded px-1 py-0.5 text-xs">
          <option value="all">Alle</option>
          <option value="selected">Gewählte</option>
          <option value="unselected">Nicht gewählte</option>
        </select>
        
        {/* Random View Buttons */}
        <div className="flex gap-0.5">
          {[12, 16, 24, 32].map(n => (
            <button key={n} onClick={() => selectRandomFromFiltered(n)}
              className="flex items-center gap-0.5 px-1.5 py-0.5 bg-zinc-800 hover:bg-zinc-700 text-gray-400 hover:text-gray-200 rounded border border-zinc-700 text-[10px]">
              <Shuffle size={10} />{n}
            </button>
          ))}
        </div>
        
        {/* Column Picker */}
        <div className="relative">
          <button onClick={() => setShowColumnPicker(!showColumnPicker)}
            className="p-1 bg-zinc-800 hover:bg-zinc-700 rounded border border-zinc-700">
            <Columns size={12} className="text-gray-400" />
          </button>
          {showColumnPicker && (
            <div className="absolute right-0 top-full mt-1 bg-zinc-900 border border-zinc-700 rounded-lg p-2 z-50 min-w-[140px]">
              {ALL_COLUMNS.map(col => (
                <label key={col.key} className="flex items-center gap-2 py-0.5 cursor-pointer text-gray-300 hover:text-white">
                  <input type="checkbox" checked={visibleColumns.includes(col.key)} onChange={() => toggleColumn(col.key)} className="rounded" />
                  {col.label}
                </label>
              ))}
            </div>
          )}
        </div>
      </div>
      
      {/* Candle Controls */}
      <div className="flex items-center gap-1.5 px-1 py-1 bg-zinc-900/50 rounded flex-wrap">
        <CandlestickChart size={12} className="text-gray-500" />
        <select value={preCandleTimeframe} onChange={e => { setPreCandleTimeframe(e.target.value); setCandleFilters({}); setTimeout(loadPreCandles, 100) }}
          className="bg-zinc-800 text-gray-300 border border-zinc-700 rounded px-1 py-0.5 text-xs">
          {TIMEFRAME_OPTIONS.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
        </select>
        <select value={preCandleCount} onChange={e => { setPreCandleCount(parseInt(e.target.value)); setCandleFilters({}); setTimeout(loadPreCandles, 100) }}
          className="bg-zinc-800 text-gray-300 border border-zinc-700 rounded px-1 py-0.5 text-xs">
          {CANDLE_COUNT_OPTIONS.map(n => <option key={n} value={n}>{n}</option>)}
        </select>
        <button onClick={loadPreCandles} className="px-1.5 py-0.5 bg-zinc-800 hover:bg-zinc-700 rounded border border-zinc-700 text-gray-400 hover:text-white text-[10px]">
          Laden
        </button>
        {preCandlesLoading && <span className="text-gray-500">...</span>}
        
        {/* Pattern-Anzeige */}
        {Object.keys(candleFilters).length > 0 && (
          <>
            <div className="h-3 w-px bg-zinc-700" />
            <span className="text-gray-500">Pattern:</span>
            <div className="flex gap-px">
              {Array.from({ length: preCandleCount }).map((_, i) => {
                const fc = candleFilters[i]
                return (
                  <div key={i}
                    onClick={() => { if (fc) toggleCandleFilter(i, fc) }}
                    className="rounded-sm"
                    style={{ 
                      width: 8, height: 12, 
                      backgroundColor: fc ? candleColor(fc) : '#374151',
                      opacity: fc ? 1 : 0.2,
                      cursor: fc ? 'pointer' : 'default',
                      border: fc ? '1px solid white' : 'none'
                    }} 
                  />
                )
              })}
            </div>
            <span className="text-blue-400">({Object.keys(candleFilters).length})</span>
            <button onClick={clearCandleFilters} className="p-0.5 hover:bg-zinc-700 rounded">
              <X size={10} className="text-gray-400" />
            </button>
            {!patternIndicator && selectedSetId && (
              <button onClick={createPatternIndicator}
                className="px-2 py-0.5 bg-blue-600 hover:bg-blue-500 rounded text-white text-[10px]">
                + Indikator
              </button>
            )}
          </>
        )}
      </div>
      
      {/* Results Table */}
      <div className="flex-1 overflow-auto">
        {sortedResults.length === 0 ? (
          <div className="h-full flex items-center justify-center text-gray-500">
            {loading ? 'Suche läuft...' : 'Keine Events'}
          </div>
        ) : (
          <table className="w-full">
            <thead className="bg-zinc-900 sticky top-0 z-10">
              <tr className="text-gray-500 text-left">
                <th className="px-1 py-1 font-normal w-6 cursor-pointer" onClick={() => handleSort('_selected')}>
                  <Square size={12} className={sortConfig.key === '_selected' ? 'text-blue-400' : 'text-gray-600'} />
                </th>
                {ALL_COLUMNS.filter(c => visibleColumns.includes(c.key)).map(col => (
                  <th key={col.key} onClick={() => handleSort(col.key)}
                    className="px-1.5 py-1 font-normal cursor-pointer hover:text-gray-300 select-none whitespace-nowrap">
                    {col.label}
                    {sortConfig.key === col.key && (
                      sortConfig.direction === 'asc' 
                        ? <ChevronUp size={10} className="inline ml-0.5 text-blue-400" />
                        : <ChevronDown size={10} className="inline ml-0.5 text-blue-400" />
                    )}
                  </th>
                ))}
                {/* Candle Header */}
                <th className="px-1 py-1 font-normal">
                  <div className="flex gap-px">
                    {Array.from({ length: preCandleCount }).map((_, i) => (
                      <span key={i} style={{ 
                        width: 12, textAlign: 'center', fontSize: '7px',
                        color: candleFilters[i] ? candleColor(candleFilters[i]) : '#6b7280'
                      }}>{i+1}</span>
                    ))}
                  </div>
                </th>
              </tr>
            </thead>
            <tbody>
              {sortedResults.map((event) => {
                const selected = isSelected(event.id)
                const eventCandles = preCandles[event.id] || []
                return (
                  <tr key={event.id} onClick={() => toggleEvent(event)}
                    className={`border-t border-zinc-800/50 cursor-pointer hover:bg-zinc-800/50 ${selected ? 'bg-blue-900/20' : ''}`}>
                    <td className="px-1 py-0.5">
                      {selected 
                        ? <CheckSquare size={12} className="text-blue-400" /> 
                        : <Square size={12} className="text-gray-600" />}
                    </td>
                    {ALL_COLUMNS.filter(c => visibleColumns.includes(c.key)).map(col => (
                      <td key={col.key} className={`px-1.5 py-0.5 whitespace-nowrap font-mono ${
                        col.key === 'change_percent' ? (event.change_percent >= 0 ? 'text-green-400' : 'text-red-400') :
                        col.key === 'direction' ? (event.direction === 'up' ? 'text-green-400' : 'text-red-400') :
                        col.key === 'symbol' ? 'font-medium text-gray-200' : 'text-gray-400'
                      }`}>
                        {formatValue(col.key, event[col.key])}
                      </td>
                    ))}
                    {/* Candle Blocks */}
                    <td className="px-1 py-0.5">
                      <div className="flex gap-px">
                        {Array.from({ length: preCandleCount }).map((_, i) => {
                          const color = eventCandles[i]
                          const isFiltered = candleFilters[i] === color
                          const hasFilter = candleFilters[i] !== undefined
                          return (
                            <div key={i}
                              onClick={(e) => { e.stopPropagation(); if (color) toggleCandleFilter(i, color) }}
                              className="rounded-sm"
                              style={{ 
                                width: 12, height: 16,
                                backgroundColor: color ? candleColor(color) : '#374151',
                                cursor: color ? 'pointer' : 'default',
                                opacity: hasFilter ? (isFiltered ? 1 : 0.3) : 0.7,
                                border: isFiltered ? '2px solid white' : (hasFilter ? '2px solid #ef4444' : 'none')
                              }} 
                            />
                          )
                        })}
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
