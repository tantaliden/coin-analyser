import { useState, useMemo } from 'react'
import { X } from 'lucide-react'
import { ANOMALY_BUCKET_OPTIONS } from './anomalyLabels'
import AnomaliesCascadeView from './AnomaliesCascadeView'
import AnomaliesItemsetView from './AnomaliesItemsetView'

// Wrapper: Cascade-Modus (Einzel-Anomalien) + Itemset-Modus (Kombinationen).
export default function AnomaliesPanel({
  results, displayEvents, prehistoryMinutes, candleTimeframeMinutes,
  primaryContext, onClose,
}) {
  const [mode, setMode] = useState('cascade')  // 'cascade' | 'itemset'
  const [bucketMinutes, setBucketMinutes] = useState(5)
  const [minSupportPct, setMinSupportPct] = useState(50)
  const [minSetSize, setMinSetSize] = useState(2)
  const [maxSetSize, setMaxSetSize] = useState(3)

  const sourceEvents = useMemo(() => {
    if (results && results.length > 0) return results
    if (displayEvents && displayEvents.length > 0) return displayEvents
    return []
  }, [results, displayEvents])

  return (
    <div className="w-96 border-r border-gray-700 bg-gray-800/80 flex flex-col overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-700">
        <div className="flex items-center gap-1">
          <button onClick={() => setMode('cascade')}
            className={`text-xs px-2 py-0.5 rounded ${mode === 'cascade' ? 'bg-indigo-600 text-white' : 'bg-gray-700 text-gray-400'}`}>
            Cascade
          </button>
          <button onClick={() => setMode('itemset')}
            className={`text-xs px-2 py-0.5 rounded ${mode === 'itemset' ? 'bg-indigo-600 text-white' : 'bg-gray-700 text-gray-400'}`}>
            Kombinationen
          </button>
        </div>
        <button onClick={onClose} className="text-gray-500 hover:text-white"><X size={14} /></button>
      </div>

      <div className="px-2 py-1.5 border-b border-gray-700 flex items-center gap-2 text-[10px]">
        <span className="text-gray-500 shrink-0">Events: {sourceEvents.length} | {prehistoryMinutes}m Prehistory</span>
      </div>

      <div className="p-2 border-b border-gray-700 flex items-center gap-2 text-xs">
        <label className="text-[10px] text-gray-500 shrink-0">Bucket:</label>
        <select value={bucketMinutes} onChange={e => setBucketMinutes(e.target.value)}
          className="input text-xs py-1 flex-1">
          {ANOMALY_BUCKET_OPTIONS.map(m => <option key={m} value={m}>{m} min</option>)}
        </select>
      </div>

      {mode === 'cascade' ? (
        <AnomaliesCascadeView
          sourceEvents={sourceEvents}
          prehistoryMinutes={prehistoryMinutes}
          candleTimeframeMinutes={candleTimeframeMinutes}
          bucketMinutes={bucketMinutes}
        />
      ) : (
        <AnomaliesItemsetView
          sourceEvents={sourceEvents}
          prehistoryMinutes={prehistoryMinutes}
          candleTimeframeMinutes={candleTimeframeMinutes}
          bucketMinutes={bucketMinutes}
          minSupportPct={minSupportPct} setMinSupportPct={setMinSupportPct}
          minSetSize={minSetSize} setMinSetSize={setMinSetSize}
          maxSetSize={maxSetSize} setMaxSetSize={setMaxSetSize}
          primaryContext={primaryContext}
        />
      )}
    </div>
  )
}
