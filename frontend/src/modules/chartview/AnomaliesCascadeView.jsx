import { useState, useMemo } from 'react'
import { RefreshCw, Sparkles, ChevronRight, Check, RotateCcw } from 'lucide-react'
import api from '../../utils/api'
import { useSearchStore } from '../../stores/searchStore'
import { ANOMALY_BUCKET_OPTIONS, labelFor } from './anomalyLabels'

// Cascade-Modus: Einzel-Anomalien, Gruppe anklicken filtert Events-Subset, re-scan.
export default function AnomaliesCascadeView({ sourceEvents, prehistoryMinutes, candleTimeframeMinutes, bucketMinutes }) {
  const [loading, setLoading] = useState(false)
  const [stack, setStack] = useState([])
  const [error, setError] = useState(null)

  const currentStep = stack.length > 0 ? stack[stack.length - 1] : null
  const currentEvents = currentStep ? currentStep.events : sourceEvents
  const currentReport = currentStep ? currentStep.report : null

  const runScan = async (eventsToScan) => {
    const payload = {
      events: eventsToScan.map(e => ({ symbol: e.symbol, event_start: e.event_start })),
      prehistory_minutes: prehistoryMinutes,
      candle_timeframe: candleTimeframeMinutes,
      bucket_minutes: parseInt(bucketMinutes),
    }
    const res = await api.post('/api/v1/search/anomalies/batch', payload)
    return res.data
  }

  const startScan = async () => {
    if (sourceEvents.length === 0) { setError('Keine Ergebnisse'); return }
    setLoading(true); setError(null)
    try {
      const report = await runScan(sourceEvents)
      setStack([{ label: `Alle (${sourceEvents.length})`, events: sourceEvents, report }])
      if (report.groups.length === 0) setError(`Keine Anomalien in ${report.scanned_events} Events`)
    } catch (err) { setError(err.response?.data?.detail || err.message) }
    finally { setLoading(false) }
  }

  const pickGroup = async (group) => {
    const refSet = new Set(group.event_refs.map(r => `${r.symbol}|${r.event_start}`))
    const matched = currentEvents.filter(e => refSet.has(`${e.symbol}|${e.event_start}`))
    if (matched.length === 0) { setError('Kein Event matcht'); return }
    const label = `${group.metric} -${group.offset_bucket_end_min}..-${group.offset_bucket_start_min}m (${matched.length})`
    setLoading(true); setError(null)
    try {
      const report = await runScan(matched)
      setStack([...stack, { label, events: matched, report, pickedKey: `${group.metric}|${group.offset_bucket_start_min}` }])
    } catch (err) { setError(err.response?.data?.detail || err.message) }
    finally { setLoading(false) }
  }

  const popStep = () => setStack(stack.length <= 1 ? [] : stack.slice(0, -1))
  const resetAll = () => { setStack([]); setError(null) }
  const applyToSelection = () => {
    useSearchStore.setState({ cascadeResults: currentEvents, selectedEvents: currentEvents.slice(0, 32) })
  }

  return (
    <>
      <div className="p-2 border-b border-gray-700 space-y-2">
        <button onClick={startScan} disabled={loading || sourceEvents.length === 0}
          className="btn btn-primary w-full text-xs flex items-center justify-center gap-1 disabled:opacity-50">
          {loading ? <RefreshCw size={12} className="animate-spin" /> : <Sparkles size={12} />}
          {stack.length === 0 ? 'Scan starten' : 'Neu starten'}
        </button>
        {stack.length > 0 && (
          <div className="flex items-center gap-2">
            <button onClick={applyToSelection} className="btn btn-secondary flex-1 text-xs flex items-center justify-center gap-1">
              <Check size={12} /> In Second Search
            </button>
            <button onClick={resetAll} className="btn btn-secondary text-xs px-2"><RotateCcw size={12} /></button>
          </div>
        )}
      </div>

      {stack.length > 0 && (
        <div className="px-2 py-1.5 border-b border-gray-700 bg-gray-900/40">
          <div className="text-[10px] text-gray-500 mb-1">Kette:</div>
          <div className="flex flex-wrap items-center gap-1 text-[10px]">
            {stack.map((step, i) => (
              <div key={i} className="flex items-center gap-0.5">
                <span className={i === stack.length - 1 ? 'text-emerald-400 font-semibold' : 'text-gray-400'}>{step.label}</span>
                {i < stack.length - 1 && <ChevronRight size={10} className="text-gray-600" />}
              </div>
            ))}
          </div>
          {stack.length > 1 && (
            <button onClick={popStep} className="text-[10px] text-blue-400 hover:text-blue-300 mt-1">← einen Schritt zurueck</button>
          )}
        </div>
      )}

      <div className="flex-1 overflow-auto p-2 text-xs">
        {error && <div className="text-amber-400 bg-amber-900/20 p-2 rounded border border-amber-700/30 mb-2">{error}</div>}
        {currentReport && (
          <div className="mb-2 text-[10px] text-gray-400 bg-gray-900/40 p-2 rounded">
            Gescannt: {currentReport.scanned_events}/{currentReport.requested_events}
            &nbsp;|&nbsp; Mit Anomalien: {currentReport.events_with_anomalies}
            &nbsp;|&nbsp; Gruppen: {currentReport.groups.length}
          </div>
        )}
        {currentReport && currentReport.groups
          .filter(g => !stack.some(step => step.pickedKey === `${g.metric}|${g.offset_bucket_start_min}`))
          .map((g, i) => {
          const offsetLabel = `-${g.offset_bucket_end_min}m..-${g.offset_bucket_start_min}m`
          return (
            <button key={i} onClick={() => pickGroup(g)} disabled={loading}
              className="w-full mb-1 p-2 bg-gray-900/50 hover:bg-gray-900/80 rounded border border-gray-700 hover:border-emerald-700 text-left disabled:opacity-50">
              <div className="flex items-center justify-between">
                <div className="flex flex-col">
                  <span className="text-gray-200 font-semibold">{labelFor(g.metric)}</span>
                  <span className="text-[10px] text-gray-500">{offsetLabel} vor Event</span>
                </div>
                <div className="flex flex-col items-end">
                  <span className="text-emerald-400 font-mono text-sm">{g.view_count}</span>
                  <span className="text-[10px] text-gray-500">{g.frequency_pct}% | z̄ {g.avg_abs_z_score}</span>
                </div>
              </div>
            </button>
          )
        })}
      </div>
    </>
  )
}
