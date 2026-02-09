import { useEffect, useState, useMemo } from 'react'
import { Grid, RefreshCw, BarChart3, TrendingUp } from 'lucide-react'
import { useSearchStore } from '../stores/searchStore'
import { useConfigStore } from '../stores/configStore'
import api from '../utils/api'
import ChartCanvas from './chart/ChartCanvas'
import { CANDLE_TIMEFRAMES, aggregateCandles, calculateDerivedValues, getEventColor } from './chart/chartUtils'

export default function ChartModule() {
  const { selectedEvents, prehistoryMinutes, setPrehistoryMinutes } = useSearchStore()
  const { getEventColors } = useConfigStore()

  const [mode, setMode] = useState('grid') // grid | overlay
  const [chartType, setChartType] = useState('line') // line | candle
  const [candleTimeframe, setCandleTimeframe] = useState('1m')
  const [loading, setLoading] = useState(false)
  const [chartData, setChartData] = useState({})
  const [showVolume, setShowVolume] = useState(false)

  // Events die angezeigt werden (max 32)
  const displayEvents = useMemo(() => selectedEvents.slice(0, 32), [selectedEvents])

  // Daten laden wenn Events sich ändern
  useEffect(() => {
    if (displayEvents.length === 0) {
      setChartData({})
      return
    }
    loadChartData()
  }, [displayEvents, prehistoryMinutes])

  const loadChartData = async () => {
    setLoading(true)
    const newData = {}

    for (const event of displayEvents) {
      try {
        const response = await api.get('/api/v1/search/candles', {
          params: {
            symbol: event.symbol,
            start_time: event.start_time,
            prehistory_minutes: prehistoryMinutes,
            duration_minutes: event.duration_minutes,
          }
        })

        const candles = response.data.candles || []
        const tfMinutes = CANDLE_TIMEFRAMES.find(t => t.key === candleTimeframe)?.minutes || 1
        const aggregated = aggregateCandles(candles, tfMinutes)
        const withDerived = calculateDerivedValues(aggregated)

        newData[event.id] = {
          event,
          candles: withDerived,
          eventStartTime: event.start_time,
        }
      } catch (error) {
        console.error(`Failed to load candles for ${event.symbol}:`, error)
      }
    }

    setChartData(newData)
    setLoading(false)
  }

  if (displayEvents.length === 0) {
    return (
      <div className="text-gray-400 text-center py-8">
        Wähle Events aus den Suchergebnissen aus.
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col">
      {/* Controls */}
      <div className="flex items-center gap-3 p-2 border-b border-gray-700 flex-wrap">
        {/* Mode Toggle */}
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
          <option value="line">Linie</option>
          <option value="candle">Kerzen</option>
        </select>

        {/* Timeframe */}
        <select
          value={candleTimeframe}
          onChange={(e) => setCandleTimeframe(e.target.value)}
          className="input text-xs py-1"
        >
          {CANDLE_TIMEFRAMES.map(tf => (
            <option key={tf.key} value={tf.key}>{tf.label}</option>
          ))}
        </select>

        {/* Prehistory */}
        <div className="flex items-center gap-1">
          <span className="text-xs text-gray-400">Vorlauf:</span>
          <input
            type="number"
            value={prehistoryMinutes}
            onChange={(e) => setPrehistoryMinutes(parseInt(e.target.value) || 0)}
            className="input text-xs py-1 w-20"
            min="0"
            step="60"
          />
          <span className="text-xs text-gray-400">min</span>
        </div>

        {/* Volume Toggle */}
        <label className="flex items-center gap-1 text-xs">
          <input
            type="checkbox"
            checked={showVolume}
            onChange={(e) => setShowVolume(e.target.checked)}
          />
          Volume
        </label>

        {/* Refresh */}
        <button
          onClick={loadChartData}
          disabled={loading}
          className="p-1.5 bg-gray-700 hover:bg-gray-600 rounded"
          title="Neu laden"
        >
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
        </button>

        <span className="text-xs text-gray-500 ml-auto">
          {displayEvents.length} Charts
        </span>
      </div>

      {/* Chart Grid */}
      <div className="flex-1 overflow-auto p-2">
        {loading ? (
          <div className="text-gray-400 text-center py-8">Laden...</div>
        ) : (
          <div className={mode === 'grid' 
            ? 'grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-2' 
            : 'space-y-2'
          }>
            {displayEvents.map((event, idx) => {
              const data = chartData[event.id]
              if (!data) return null

              return (
                <div 
                  key={event.id} 
                  className="bg-gray-800 rounded border border-gray-700"
                >
                  <div 
                    className="px-2 py-1 border-b border-gray-700 text-xs flex items-center gap-2"
                    style={{ borderLeftColor: getEventColor(idx), borderLeftWidth: 3 }}
                  >
                    <span className="font-mono font-semibold">{event.symbol}</span>
                    <span className={event.percent_change >= 0 ? 'text-green-400' : 'text-red-400'}>
                      {event.percent_change >= 0 ? '+' : ''}{event.percent_change?.toFixed(2)}%
                    </span>
                  </div>
                  <ChartCanvas
                    data={data}
                    eventIndex={idx}
                    chartType={chartType}
                    showVolume={showVolume}
                    height={mode === 'grid' ? 150 : 250}
                  />
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
