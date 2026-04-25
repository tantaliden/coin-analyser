import { useState } from 'react'
import { RefreshCw, Layers, Save } from 'lucide-react'
import api from '../../utils/api'
import { labelFor } from './anomalyLabels'
import SaveAnomalySetDialog from './SaveAnomalySetDialog'

// Kombinations-Modus: Apriori-Itemsets die bei vielen Events gemeinsam auftreten.
// Pro Set: Speichern als Indikator-Set moeglich.
export default function AnomaliesItemsetView({
  sourceEvents, prehistoryMinutes, candleTimeframeMinutes, bucketMinutes,
  minSupportPct, setMinSupportPct,
  minSetSize, setMinSetSize,
  maxSetSize, setMaxSetSize,
  primaryContext,
}) {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [saveDialogSet, setSaveDialogSet] = useState(null)

  const runScan = async () => {
    if (sourceEvents.length === 0) { setError('Keine Ergebnisse'); return }
    setLoading(true); setError(null); setResult(null)
    try {
      const res = await api.post('/api/v1/search/anomalies/itemsets', {
        events: sourceEvents.map(e => ({ symbol: e.symbol, event_start: e.event_start })),
        prehistory_minutes: prehistoryMinutes,
        candle_timeframe: candleTimeframeMinutes,
        bucket_minutes: parseInt(bucketMinutes),
        min_support_pct: parseFloat(minSupportPct),
        min_set_size: parseInt(minSetSize),
        max_set_size: parseInt(maxSetSize),
      })
      setResult(res.data)
      if (res.data.itemsets.length === 0) setError('Keine Kombinationen gefunden — min_support verringern?')
    } catch (err) { setError(err.response?.data?.detail || err.message) }
    finally { setLoading(false) }
  }

  return (
    <>
      <div className="p-2 border-b border-gray-700 space-y-2">
        <div className="grid grid-cols-2 gap-2 text-[10px]">
          <label className="flex flex-col gap-0.5">
            <span className="text-gray-500">Min Support %</span>
            <input type="number" min="1" max="100" value={minSupportPct}
              onChange={e => setMinSupportPct(e.target.value)} className="input text-xs py-1" />
          </label>
          <label className="flex flex-col gap-0.5">
            <span className="text-gray-500">Min Set-Groesse</span>
            <input type="number" min="1" max="10" value={minSetSize}
              onChange={e => setMinSetSize(e.target.value)} className="input text-xs py-1" />
          </label>
          <label className="flex flex-col gap-0.5">
            <span className="text-gray-500">Max Set-Groesse</span>
            <input type="number" min="1" max="10" value={maxSetSize}
              onChange={e => setMaxSetSize(e.target.value)} className="input text-xs py-1" />
          </label>
        </div>
        <button onClick={runScan} disabled={loading || sourceEvents.length === 0}
          className="btn btn-primary w-full text-xs flex items-center justify-center gap-1 disabled:opacity-50">
          {loading ? <RefreshCw size={12} className="animate-spin" /> : <Layers size={12} />}
          Kombinationen finden
        </button>
      </div>

      <div className="flex-1 overflow-auto p-2 text-xs">
        {error && <div className="text-amber-400 bg-amber-900/20 p-2 rounded border border-amber-700/30 mb-2">{error}</div>}
        {result && (
          <div className="mb-2 text-[10px] text-gray-400 bg-gray-900/40 p-2 rounded space-y-0.5">
            <div>Gescannt: {result.scanned_events}/{result.requested_events} | Kombinationen: {result.itemsets.length}</div>
            <div>Events mit Anomalien: {result.events_with_anomalies} | Unique Items: {result.unique_items}</div>
            <div>Total Hits: {result.total_anomaly_hits} | Items &ge; Support: {result.items_meeting_support}</div>
            <div>Min-Count (@{result.min_support_pct}%): {result.min_support_count}</div>
            <div>Hoechster Item-Support: {result.top_item_support_pct}%</div>
          </div>
        )}
        {result && result.items_meeting_support === 0 && result.top_item_support_pct > 0 && (
          <div className="mb-2 text-[10px] text-amber-400 bg-amber-900/20 p-2 rounded border border-amber-700/30">
            Kein Item erreicht {result.min_support_pct}% Support.
            Setz min_support auf {Math.floor(result.top_item_support_pct)}% oder weniger.
          </div>
        )}
        {result && result.itemsets.map((s, i) => (
          <div key={i} className="mb-1 p-2 bg-gray-900/50 rounded border border-gray-700">
            <div className="flex items-center justify-between mb-1">
              <span className="text-[10px] text-gray-500">Groesse {s.size} | {s.view_count} Views ({s.frequency_pct}%)</span>
              <button onClick={() => setSaveDialogSet(s)}
                className="p-1 hover:bg-emerald-600/30 rounded" title="Als Indikator-Set speichern">
                <Save size={12} />
              </button>
            </div>
            <div className="space-y-0.5">
              {s.items.map((it, j) => (
                <div key={j} className="text-[10px] flex items-center justify-between">
                  <span className="text-gray-300">{labelFor(it.metric)}</span>
                  <span className="text-gray-500 font-mono">-{it.bucket_end_min}..-{it.bucket_start_min}m</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {saveDialogSet && (
        <SaveAnomalySetDialog
          itemset={saveDialogSet}
          prehistoryMinutes={prehistoryMinutes}
          candleTimeframeMinutes={candleTimeframeMinutes}
          primaryContext={primaryContext}
          onClose={() => setSaveDialogSet(null)}
        />
      )}
    </>
  )
}
