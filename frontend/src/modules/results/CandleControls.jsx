import { CANDLE_COLOR_MAP } from '../../config/chartSettings'
import { CandlestickChart, X } from 'lucide-react'

const candleColor = (color) => {
  if (color === 'green') return CANDLE_COLOR_MAP.green
  if (color === 'red') return CANDLE_COLOR_MAP.red
  return '#6b7280'
}

export default function CandleControls({
  preCandleTimeframe, setPrecandleTimeframe,
  preCandleCount, setPreCandleCount,
  candleFilters, onClearFilters,
  onLoadCandles, preCandlesLoading,
  timeframeOptions, candleCountOptions,
  patternIndicator, selectedSetId,
  onCreatePattern,
}) {
  return (
    <div className="flex items-center gap-1.5 px-1 py-1 bg-zinc-900/50 rounded flex-wrap">
      <CandlestickChart size={12} className="text-gray-500" />

      <select value={preCandleTimeframe}
        onChange={e => { setPrecandleTimeframe(e.target.value); setTimeout(onLoadCandles, 100) }}
        className="bg-zinc-800 text-gray-300 border border-zinc-700 rounded px-1 py-0.5 text-xs">
        {timeframeOptions.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
      </select>

      <select value={preCandleCount}
        onChange={e => { setPreCandleCount(parseInt(e.target.value)); setTimeout(onLoadCandles, 100) }}
        className="bg-zinc-800 text-gray-300 border border-zinc-700 rounded px-1 py-0.5 text-xs">
        {candleCountOptions.map(n => <option key={n} value={n}>{n}</option>)}
      </select>

      <button onClick={onLoadCandles}
        className="px-1.5 py-0.5 bg-zinc-800 hover:bg-zinc-700 rounded border border-zinc-700 text-gray-400 hover:text-white text-[10px]">
        Laden
      </button>

      {preCandlesLoading && <span className="text-gray-500">...</span>}

      {Object.keys(candleFilters).length > 0 && (
        <>
          <div className="h-3 w-px bg-zinc-700" />
          <span className="text-gray-500">Pattern:</span>
          <div className="flex gap-px">
            {Array.from({ length: preCandleCount }).map((_, i) => {
              const fc = candleFilters[i]
              return (
                <div key={i} className="rounded-sm" style={{
                  width: 8, height: 12,
                  backgroundColor: fc ? candleColor(fc) : '#374151',
                  opacity: fc ? 1 : 0.2,
                  border: fc ? '1px solid white' : 'none',
                }} />
              )
            })}
          </div>
          <span className="text-blue-400">({Object.keys(candleFilters).length})</span>
          <button onClick={onClearFilters} className="p-0.5 hover:bg-zinc-700 rounded">
            <X size={10} className="text-gray-400" />
          </button>
          {!patternIndicator && selectedSetId && (
            <button onClick={onCreatePattern}
              className="px-2 py-0.5 bg-blue-600 hover:bg-blue-500 rounded text-white text-[10px]">
              + Indikator
            </button>
          )}
        </>
      )}
    </div>
  )
}
