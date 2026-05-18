import { CANDLE_COLOR_MAP, MARKER_COLORS } from '../../config/chartSettings'
import { ChevronUp, ChevronDown, CheckSquare, Square } from 'lucide-react'
import { formatDateTime, formatPercent, formatNumber } from '../../utils/format'

const ALL_COLUMNS = [
  { key: 'symbol', label: 'Symbol' }, { key: 'event_start', label: 'Start' },
  { key: 'event_end', label: 'Ende' }, { key: 'start_price', label: 'Preis' },
  { key: 'change_percent', label: '%' }, { key: 'duration_minutes', label: 'Dauer' },
  { key: 'direction', label: 'Dir' }, { key: 'volume', label: 'Volume' },
  { key: 'trades_count', label: 'Trades' }, { key: 'volatility_pct', label: 'Volatilitaet' },
  { key: 'max_drawdown_pct', label: 'Drawdown' },
]

const formatValue = (key, value) => {
  if (value === null || value === undefined) return '-'
  switch (key) {
    case 'event_start': case 'event_end': return formatDateTime(value)
    case 'change_percent': case 'volatility_pct': case 'max_drawdown_pct': return formatPercent(value)
    case 'start_price': return formatNumber(value, 4)
    case 'volume': return formatNumber(value, 0)
    case 'direction': return value === 'up' ? '\u2191' : '\u2193'
    default: return value
  }
}

const candleColor = (color) => {
  if (color === 'green') return CANDLE_COLOR_MAP.green
  if (color === 'red') return CANDLE_COLOR_MAP.red
  return '#6b7280'
}

export default function ResultsTable({
  results, loading, visibleColumns, sortConfig, onSort,
  isSelected, onToggleEvent,
  preCandles, preCandleCount, candleFilters, onToggleCandleFilter,
}) {
  if (results.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-500">
        {loading ? 'Suche laeuft...' : 'Keine Events'}
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-auto">
      <table className="w-full">
        <thead className="bg-zinc-900 sticky top-0 z-10">
          <tr className="text-gray-500 text-left">
            <th className="px-1 py-1 font-normal w-6 cursor-pointer" onClick={() => onSort('_selected')}>
              <Square size={12} className={sortConfig.key === '_selected' ? 'text-blue-400' : 'text-gray-600'} />
            </th>
            {ALL_COLUMNS.filter(c => visibleColumns.includes(c.key)).map(col => (
              <th key={col.key} onClick={() => onSort(col.key)}
                className="px-1.5 py-1 font-normal cursor-pointer hover:text-gray-300 select-none whitespace-nowrap">
                {col.label}
                {sortConfig.key === col.key && (
                  sortConfig.direction === 'asc'
                    ? <ChevronUp size={10} className="inline ml-0.5 text-blue-400" />
                    : <ChevronDown size={10} className="inline ml-0.5 text-blue-400" />
                )}
              </th>
            ))}
            <th className="px-1 py-1 font-normal">
              <div className="flex gap-px">
                {Array.from({ length: preCandleCount }).map((_, i) => (
                  <span key={i} style={{
                    width: 12, textAlign: 'center', fontSize: '7px',
                    color: candleFilters[i] ? candleColor(candleFilters[i]) : '#6b7280',
                  }}>{i + 1}</span>
                ))}
              </div>
            </th>
          </tr>
        </thead>
        <tbody>
          {results.map(event => {
            const selected = isSelected(event.id)
            const eventCandles = preCandles[event.id]
            if (!eventCandles) return null
            return (
              <tr key={event.id} onClick={() => onToggleEvent(event)}
                className={`border-t border-zinc-800/50 cursor-pointer hover:bg-zinc-800/50 ${selected ? 'bg-blue-900/20' : ''}`}>
                <td className="px-1 py-0.5">
                  {selected ? <CheckSquare size={12} className="text-blue-400" /> : <Square size={12} className="text-gray-600" />}
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
                <td className="px-1 py-0.5">
                  <div className="flex gap-px">
                    {Array.from({ length: preCandleCount }).map((_, i) => {
                      const color = eventCandles[i]
                      const isFiltered = candleFilters[i] === color
                      const hasFilter = candleFilters[i] !== undefined
                      return (
                        <div key={i}
                          onClick={e => { e.stopPropagation(); if (color) onToggleCandleFilter(i, color) }}
                          className="rounded-sm"
                          style={{
                            width: 12, height: 16,
                            backgroundColor: color ? candleColor(color) : '#374151',
                            cursor: color ? 'pointer' : 'default',
                            opacity: hasFilter ? (isFiltered ? 1 : 0.3) : 0.7,
                            border: isFiltered ? '2px solid white' : hasFilter ? `2px solid ${MARKER_COLORS.filterBorderMiss}` : 'none',
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
    </div>
  )
}
