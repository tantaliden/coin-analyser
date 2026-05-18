import { Grid, TrendingUp, RefreshCw } from 'lucide-react'
import { CHART_SETTINGS } from '../../config/chartSettings'

export default function ChartControls({
  mode, setMode,
  chartType, setChartType,
  candleTimeframe, setCandleTimeframe,
  prehistoryMinutes, setPrehistoryMinutes,
  showVolume, setShowVolume,
  loading, onRefresh,
  eventCount,
}) {
  return (
    <div className="flex items-center gap-3 px-2 py-1.5 border-b border-gray-700 flex-wrap text-xs">
      {/* Grid/Overlay Toggle */}
      <div className="flex gap-1">
        <button
          onClick={() => setMode('grid')}
          className={`p-1.5 rounded ${mode === 'grid' ? 'bg-blue-600' : 'bg-gray-700'}`}
          title="Grid-Ansicht"
        >
          <Grid size={14} />
        </button>
        <button
          onClick={() => setMode('overlay')}
          className={`p-1.5 rounded ${mode === 'overlay' ? 'bg-blue-600' : 'bg-gray-700'}`}
          title="Overlay-Ansicht"
        >
          <TrendingUp size={14} />
        </button>
      </div>

      {/* Chart Type */}
      <select
        value={chartType}
        onChange={(e) => setChartType(e.target.value)}
        className="input text-xs py-1"
      >
        <option value="candle">Kerzen</option>
        <option value="line">Linie</option>
      </select>

      {/* Timeframe */}
      <select
        value={candleTimeframe}
        onChange={(e) => setCandleTimeframe(e.target.value)}
        className="input text-xs py-1"
      >
        {CHART_SETTINGS.timeframes.map(tf => (
          <option key={tf.key} value={tf.key}>{tf.label}</option>
        ))}
      </select>

      {/* Prehistory */}
      <div className="flex items-center gap-1">
        <span className="text-gray-400">Vorlauf:</span>
        <input
          type="number"
          value={prehistoryMinutes}
          onChange={(e) => setPrehistoryMinutes(parseInt(e.target.value) || 0)}
          className="input text-xs py-1 w-20"
          min="0"
          step="60"
        />
        <span className="text-gray-400">min</span>
      </div>

      {/* Volume */}
      <label className="flex items-center gap-1">
        <input
          type="checkbox"
          checked={showVolume}
          onChange={(e) => setShowVolume(e.target.checked)}
        />
        Volume
      </label>

      {/* Refresh */}
      <button
        onClick={onRefresh}
        disabled={loading}
        className="p-1.5 bg-gray-700 hover:bg-gray-600 rounded"
        title="Neu laden"
      >
        <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
      </button>

      <span className="text-gray-500 ml-auto">
        {eventCount} Charts
      </span>
    </div>
  )
}
