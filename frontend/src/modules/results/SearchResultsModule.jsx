import { useState, useMemo, useCallback, useEffect } from 'react'
import { useSearchStore } from '../../stores/searchStore'
import ResultsHeader from './ResultsHeader'
import ResultsTable from './ResultsTable'
import api from '../../utils/api'

const TIMEFRAME_OPTIONS = [
  { value: '5m', label: '5m' }, { value: '10m', label: '10m' },
  { value: '15m', label: '15m' }, { value: '30m', label: '30m' },
  { value: '1h', label: '1h' }, { value: '2h', label: '2h' },
  { value: '4h', label: '4h' },
]
const CANDLE_COUNT_OPTIONS = [5, 10, 15, 20, 25, 30]

export default function SearchResultsModule() {
  const {
    results, cascadeResults, indicatorChain, loading,
    selectedEvents, toggleEvent, isSelected, maxSelection,
    selectedSetId, prehistoryMinutes, setIndicatorChain,
    applyCascadeFilter
  } = useSearchStore()

  const activeResults = indicatorChain?.length > 0 && cascadeResults?.length > 0 ? cascadeResults : results

  const [sortConfig, setSortConfig] = useState({ key: 'event_start', direction: 'desc' })
  const [showFilter, setShowFilter] = useState('all')
  const [visibleColumns, setVisibleColumns] = useState(['symbol', 'event_start', 'change_percent'])

  // Pre-Candles
  const [showPreCandles, setShowPreCandles] = useState(true)
  const [preCandleTimeframe, setPreCandleTimeframe] = useState('15m')
  const [preCandleCount, setPreCandleCount] = useState(10)
  const [preCandles, setPreCandles] = useState({})
  const [preCandlesLoading, setPreCandlesLoading] = useState(false)
  const [candleFilters, setCandleFilters] = useState({})

  const patternIndicator = useMemo(() => {
    return indicatorChain?.find(ind => ind.indicator_type === 'candle_pattern' && ind.is_active !== false)
  }, [indicatorChain])

  useEffect(() => {
    if (patternIndicator?.pattern_data) setCandleFilters(patternIndicator.pattern_data)
  }, [patternIndicator])

  const loadPreCandles = useCallback(async () => {
    if (activeResults.length === 0) return
    setPreCandlesLoading(true)
    try {
      const events = activeResults.map(e => ({ id: e.id, symbol: e.symbol, event_start: e.event_start }))
      let requestParams = { events, timeframe: preCandleTimeframe, candle_count: preCandleCount }

      if (patternIndicator && prehistoryMinutes) {
        if (patternIndicator.time_start_minutes == null || patternIndicator.time_end_minutes == null || !patternIndicator.aggregator) {
          throw new Error('patternIndicator has missing fields')
        }
        const timeStart = patternIndicator.time_start_minutes
        const timeEnd = patternIndicator.time_end_minutes
        const aggregator = patternIndicator.aggregator
        const aggMinutes = aggregator.endsWith('m') ? parseInt(aggregator) :
          aggregator.endsWith('h') ? parseInt(aggregator) * 60 : 1
        const candleCount = Math.ceil((timeEnd - timeStart) / aggMinutes)
        requestParams = { events, timeframe: aggregator, candle_count: candleCount, prehistory_minutes: prehistoryMinutes, time_end_minutes: timeEnd }
      }

      const response = await api.post('/api/v1/search/pre-candles', requestParams)
      if (!response.data.candles) throw new Error('API response missing candles field')
      setPreCandles(response.data.candles)
    } catch (err) {
      console.error('Failed to load pre-candles:', err)
    } finally {
      setPreCandlesLoading(false)
    }
  }, [activeResults, preCandleTimeframe, preCandleCount, patternIndicator, prehistoryMinutes])

  useEffect(() => {
    if (activeResults.length > 0 && showPreCandles && Object.keys(preCandles).length === 0) loadPreCandles()
  }, [activeResults.length, showPreCandles])

  useEffect(() => { setPreCandles({}) }, [patternIndicator?.item_id, patternIndicator?.is_active])

  const toggleCandleFilter = async (position, color) => {
    const newFilters = { ...candleFilters }
    if (newFilters[position] === color) delete newFilters[position]
    else newFilters[position] = color
    setCandleFilters(newFilters)

    if (patternIndicator?.item_id) {
      try {
        await api.put(`/api/v1/indicators/items/${patternIndicator.item_id}`, { pattern_data: newFilters })
        const updatedChain = indicatorChain.map(ind =>
          ind.item_id === patternIndicator.item_id ? { ...ind, pattern_data: newFilters } : ind
        )
        setIndicatorChain(updatedChain)
        applyCascadeFilter(updatedChain.filter(i => i.is_active !== false), prehistoryMinutes)
      } catch (err) { console.error('Failed to save pattern:', err) }
    }
  }

  const clearCandleFilters = async () => {
    setCandleFilters({})
    if (patternIndicator?.item_id) {
      try {
        await api.put(`/api/v1/indicators/items/${patternIndicator.item_id}`, { pattern_data: {} })
        const updatedChain = indicatorChain.map(ind =>
          ind.item_id === patternIndicator.item_id ? { ...ind, pattern_data: {} } : ind
        )
        setIndicatorChain(updatedChain)
        applyCascadeFilter(updatedChain.filter(i => i.is_active !== false), prehistoryMinutes)
      } catch (err) { console.error('Failed to clear pattern:', err) }
    }
  }

  const createPatternIndicator = async () => {
    if (Object.keys(candleFilters).length === 0 || !selectedSetId || !prehistoryMinutes) return
    const tfMinutes = parseInt(preCandleTimeframe)
    if (isNaN(tfMinutes)) throw new Error('preCandleTimeframe invalid: ' + preCandleTimeframe)
    const patternMinutes = preCandleCount * tfMinutes
    const timeEnd = prehistoryMinutes - 15
    const timeStart = Math.max(0, timeEnd - patternMinutes)

    try {
      const res = await api.post(`/api/v1/indicators/sets/${selectedSetId}/items`, {
        field: 'candle_pattern', operation: 'match', value: null,
        time_start_minutes: timeStart, time_end_minutes: timeEnd,
        aggregator: preCandleTimeframe, color: '#F59E0B',
        pattern_type: 'exact', pattern_data: candleFilters,
        pattern_count: Object.keys(candleFilters).length, pattern_consecutive: true,
      })
      setIndicatorChain([...indicatorChain, {
        item_id: res.data.item_id, indicator_type: 'candle_pattern',
        condition_operator: 'match', time_start_minutes: timeStart, time_end_minutes: timeEnd,
        aggregator: preCandleTimeframe, pattern_data: candleFilters, color: '#F59E0B', is_active: true,
      }])
    } catch (err) { alert('Fehler: ' + (err.response?.data?.detail || err.message)) }
  }

  // Sorted + filtered
  const sortedResults = useMemo(() => {
    let filtered = [...activeResults]
    if (showFilter === 'selected') filtered = filtered.filter(e => isSelected(e.id))
    else if (showFilter === 'unselected') filtered = filtered.filter(e => !isSelected(e.id))

    if (showPreCandles && Object.keys(candleFilters).length > 0 && (!indicatorChain || indicatorChain.length === 0)) {
      filtered = filtered.filter(e => {
        const candles = preCandles[e.id]
        if (!candles) return false
        for (const [pos, color] of Object.entries(candleFilters)) {
          if (candles[parseInt(pos)] !== color) return false
        }
        return true
      })
    }

    filtered.sort((a, b) => {
      if (sortConfig.key === '_selected') {
        const aS = isSelected(a.id) ? 1 : 0, bS = isSelected(b.id) ? 1 : 0
        if (aS !== bS) return sortConfig.direction === 'asc' ? aS - bS : bS - aS
        return new Date(b.event_start).getTime() - new Date(a.event_start).getTime()
      }
      let aVal = a[sortConfig.key], bVal = b[sortConfig.key]
      if (['change_percent', 'duration_minutes', 'volume', 'start_price', 'trades_count', 'volatility_pct', 'max_drawdown_pct'].includes(sortConfig.key)) {
        aVal = parseFloat(aVal); bVal = parseFloat(bVal)
      }
      if (['event_start', 'event_end'].includes(sortConfig.key)) {
        aVal = new Date(aVal).getTime(); bVal = new Date(bVal).getTime()
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

  const selectRandomFromFiltered = (count) => {
    const shuffled = [...sortedResults].sort(() => Math.random() - 0.5)
    useSearchStore.setState({ selectedEvents: shuffled.slice(0, Math.min(count, maxSelection)) })
  }

  return (
    <div className="h-full flex flex-col gap-1 text-xs">
      <ResultsHeader
        totalCount={sortedResults.length}
        cascadeCount={cascadeResults?.length}
        mainCount={results.length}
        hasCascade={cascadeResults?.length > 0}
        showFilter={showFilter}
        setShowFilter={setShowFilter}
        onSelectRandom={selectRandomFromFiltered}
        visibleColumns={visibleColumns}
        setVisibleColumns={setVisibleColumns}
        preCandleTimeframe={preCandleTimeframe}
        setPrecandleTimeframe={setPreCandleTimeframe}
        preCandleCount={preCandleCount}
        setPreCandleCount={setPreCandleCount}
        candleFilters={candleFilters}
        onClearFilters={clearCandleFilters}
        onLoadCandles={loadPreCandles}
        preCandlesLoading={preCandlesLoading}
        timeframeOptions={TIMEFRAME_OPTIONS}
        candleCountOptions={CANDLE_COUNT_OPTIONS}
        patternIndicator={patternIndicator}
        selectedSetId={selectedSetId}
        onCreatePattern={createPatternIndicator}
      />
      <ResultsTable
        results={sortedResults}
        loading={loading}
        visibleColumns={visibleColumns}
        sortConfig={sortConfig}
        onSort={(key) => setSortConfig(prev => ({
          key, direction: prev.key === key && prev.direction === 'asc' ? 'desc' : 'asc'
        }))}
        isSelected={isSelected}
        onToggleEvent={toggleEvent}
        preCandles={preCandles}
        preCandleCount={preCandleCount}
        candleFilters={candleFilters}
        onToggleCandleFilter={toggleCandleFilter}
      />
    </div>
  )
}
