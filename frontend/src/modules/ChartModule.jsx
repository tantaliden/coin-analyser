import { useEffect, useState, useMemo } from 'react'
import { Grid, RefreshCw, TrendingUp } from 'lucide-react'
import { useSearchStore } from '../stores/searchStore'
import { useConfigStore } from '../stores/configStore'
import api from '../utils/api'
import ChartCanvas from './chart/ChartCanvas'
import { CANDLE_TIMEFRAMES, aggregateCandles, calculateDerivedValues, getEventColor } from './chart/chartUtils'

export default function ChartModule() {
  const { selectedEvents, prehistoryMinutes, setPrehistoryMinutes } = useSearchStore()
  const { getTimeframeOptions } = useConfigStore()

  const [mode, setMode] = useState('grid')
  const [chartType, setChartType] = useState('line')
  const [candleTimeframe, setCandleTimeframe] = useState('1m')
  const [loading, setLoading] = useState(false)
  const [chartData, setChartData] = useState({})
  const [showVolume, setShowVolume] = useState(false)
  const [error, setError] = useState(null)

  const displayEvents = useMemo(() => selectedEvents.slice(0, 32), [selectedEvents])

  useEffect(() => {
    if (displayEvents.length === 0) {
      setChartData({})
      return
    }
    loadChartData()
  }, [displayEvents, prehistoryMinutes, candleTimeframe])

  const loadChartData = async () => {
    setLoading(true)
    setError(null)
    const newData = {}

    for (const event of displayEvents) {
      try {
        // Event-Start parsen und Zeitraum berechnen
        const eventStart = new Date(event.event_start.replace(' ', 'T'))
        const start = new Date(eventStart.getTime() - prehistoryMinutes * 60 * 1000)
        const end = new Date(eventStart.getTime() + event.duration_minutes * 60 * 1000)

        const response = await api.get('/api/v1/search/candles', {
          params: {
            symbol: event.symbol,
            start: start.toISOString(),
            end: end.toISOString(),
            interval: candleTimeframe
          }
        })

        const candles = response.data.candles || []
        if (candles.length === 0) continue

        // Relative Zeit berechnen (0 = Event-Start)
        const eventStartTs = eventStart.getTime() / 1000
        const withRelativeTime = candles.map(c => ({
          ...c,
          relativeTime: c.time - eventStartTs
        }))

        const withDerived = calculateDerivedValues(withRelativeTime)

        newData[event.id] = {
          event,
          candles: withDerived,
          eventStartTime: eventStartTs
        }
      } catch (err) {
        console.error(`Failed to load candles for ${event.symbol}:`, err)
        setError(`Fehler beim Laden von ${event.symbol}`)
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

        <select
          value={chartType}
          onChange={(e) => setChartType(e.target.value)}
          className="input text-xs py-1"
        >
          <option value="line">Linie</option>
          <option value="candle">Kerzen</option>
        </select>

        <select
          value={candleTimeframe}
          onChange={(e) => setCandleTimeframe(e.target.value)}
          className="input text-xs py-1"
        >
          {CANDLE_TIMEFRAMES.map(tf => (
            <option key={tf.key} value={tf.key}>{tf.label}</option>
          ))}
        </select>

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

        <label className="flex items-center gap-1 text-xs">
          <input
            type="checkbox"
            checked={showVolume}
            onChange={(e) => setShowVolume(e.target.checked)}
          />
          Volume
        </label>

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

      {/* Error */}
      {error && (
        <div className="p-2 text-red-400 text-sm bg-red-900/30">
          {error}
        </div>
      )}

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
                    <span className={event.change_percent >= 0 ? 'text-green-400' : 'text-red-400'}>
                      {event.change_percent >= 0 ? '+' : ''}{event.change_percent?.toFixed(2)}%
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
