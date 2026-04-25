import { Columns, Shuffle, X, CandlestickChart } from 'lucide-react'
import { useState } from 'react'
import { CANDLE_COLOR_MAP } from '../../config/chartSettings'

const ALL_COLUMNS = [
  { key: 'symbol', label: 'Symbol' }, { key: 'event_start', label: 'Start' },
  { key: 'event_end', label: 'Ende' }, { key: 'start_price', label: 'Preis' },
  { key: 'change_percent', label: '%' }, { key: 'duration_minutes', label: 'Dauer' },
  { key: 'direction', label: 'Dir' }, { key: 'volume', label: 'Volume' },
  { key: 'trades_count', label: 'Trades' }, { key: 'volatility_pct', label: 'Volatilitaet' },
  { key: 'max_drawdown_pct', label: 'Drawdown' },
]

const VIEW_COUNTS = [12, 16, 24, 32]
const DEFAULT_VIEW_COUNT = 32

const candleColor = (color) => {
  if (color === 'green') return CANDLE_COLOR_MAP.green
  if (color === 'red') return CANDLE_COLOR_MAP.red
  return '#6b7280'
}

export default function ResultsHeader({
  totalCount,
  cascadeCount, mainCount, hasCascade,
  showFilter, setShowFilter,
  onSelectRandom,
  visibleColumns, setVisibleColumns,
  preCandleTimeframe, setPrecandleTimeframe,
  preCandleCount, setPreCandleCount,
  candleFilters, onClearFilters,
  onLoadCandles, preCandlesLoading,
  timeframeOptions, candleCountOptions,
  patternIndicator, selectedSetId, onCreatePattern,
}) {
  const [showColumnPicker, setShowColumnPicker] = useState(false)
  const [viewCount, setViewCount] = useState(DEFAULT_VIEW_COUNT)

  const selCls = "bg-zinc-800 text-gray-300 border border-zinc-700 rounded px-1 py-0.5 text-xs"
  const btnCls = "p-1 bg-zinc-800 hover:bg-zinc-700 text-gray-400 hover:text-gray-200 rounded border border-zinc-700"

  return (
    <div className="flex items-center gap-1 px-1 pt-1 text-xs flex-nowrap">
      <span className="text-gray-500">{totalCount}</span>
      {hasCascade && <span className="text-blue-400">({mainCount}&rarr;{cascadeCount})</span>}

      <select value={viewCount} onChange={e => setViewCount(parseInt(e.target.value))} className={selCls}>
        {VIEW_COUNTS.map(n => <option key={n} value={n}>{n}</option>)}
      </select>

      <button onClick={() => onSelectRandom(viewCount)} className={btnCls} title="Zufallsauswahl">
        <Shuffle size={11} />
      </button>

      <select value={showFilter} onChange={e => setShowFilter(e.target.value)} className={selCls}>
        <option value="all">Alle</option>
        <option value="selected">Sel</option>
        <option value="unselected">Un</option>
      </select>

      <CandlestickChart size={12} className="text-gray-500" />

      <select value={preCandleTimeframe}
        onChange={e => { setPrecandleTimeframe(e.target.value); setTimeout(onLoadCandles, 50) }}
        className={selCls}>
        {timeframeOptions.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
      </select>

      <select value={preCandleCount}
        onChange={e => { setPreCandleCount(parseInt(e.target.value)); setTimeout(onLoadCandles, 50) }}
        className={selCls}>
        {candleCountOptions.map(n => <option key={n} value={n}>{n}</option>)}
      </select>

      {preCandlesLoading && <span className="text-gray-500">...</span>}

      {Object.keys(candleFilters).length > 0 && (
        <>
          <span className="text-blue-400">{Object.keys(candleFilters).length}</span>
          <button onClick={onClearFilters} className="p-0.5 hover:bg-zinc-700 rounded" title="Pattern loeschen">
            <X size={10} className="text-gray-400" />
          </button>
          {!patternIndicator && selectedSetId && (
            <button onClick={onCreatePattern}
              className="px-2 py-0.5 bg-blue-600 hover:bg-blue-500 rounded text-white text-[10px]">+I</button>
          )}
        </>
      )}

      <div className="relative ml-auto">
        <button onClick={() => setShowColumnPicker(!showColumnPicker)} className={btnCls}>
          <Columns size={11} className="text-gray-400" />
        </button>
        {showColumnPicker && (
          <div className="absolute right-0 top-full mt-1 bg-zinc-900 border border-zinc-700 rounded p-2 z-50 min-w-[140px]">
            {ALL_COLUMNS.map(col => (
              <label key={col.key} className="flex items-center gap-2 py-0.5 cursor-pointer text-gray-300 hover:text-white text-xs">
                <input type="checkbox" checked={visibleColumns.includes(col.key)}
                  onChange={() => setVisibleColumns(prev =>
                    prev.includes(col.key) ? prev.filter(k => k !== col.key) : [...prev, col.key]
                  )} className="rounded" />
                {col.label}
              </label>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
