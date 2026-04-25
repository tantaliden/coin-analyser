import { useState } from 'react'
import { X, Save } from 'lucide-react'
import api from '../../utils/api'
import { labelFor } from './anomalyLabels'

// Dialog zum Speichern eines Anomalie-Itemsets als Indicator-Set.
export default function SaveAnomalySetDialog({ itemset, prehistoryMinutes, candleTimeframeMinutes, primaryContext, onClose }) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState(null)
  const [saved, setSaved] = useState(null)

  const save = async () => {
    if (!name.trim()) { setError('Name erforderlich'); return }
    if (!primaryContext) { setError('primary_context fehlt'); return }
    setSaving(true); setError(null)
    try {
      const res = await api.post('/api/v1/indicators/sets/from-anomaly-itemset', {
        name: name.trim(),
        description: description.trim(),
        duration_minutes: primaryContext.search_duration_minutes,
        direction: primaryContext.search_direction,
        target_percent: primaryContext.search_percent_min,
        prehistory_minutes: prehistoryMinutes,
        candle_timeframe: candleTimeframeMinutes,
        items: itemset.items.map(it => ({
          metric: it.metric,
          bucket_start_min: it.bucket_start_min,
          bucket_end_min: it.bucket_end_min,
        })),
        primary_context: primaryContext,
        view_count: itemset.view_count,
        frequency_pct: itemset.frequency_pct,
      })
      setSaved(res.data)
    } catch (err) { setError(err.response?.data?.detail || err.message) }
    finally { setSaving(false) }
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-gray-800 border border-gray-700 rounded-lg w-96 max-w-[95vw]">
        <div className="flex items-center justify-between px-3 py-2 border-b border-gray-700">
          <span className="text-sm font-semibold text-gray-200">Anomalie-Set speichern</span>
          <button onClick={onClose} className="text-gray-500 hover:text-white"><X size={16} /></button>
        </div>
        <div className="p-3 space-y-3 text-xs">
          <div className="p-2 bg-gray-900/50 rounded">
            <div className="text-[10px] text-gray-500 mb-1">
              {itemset.view_count} Views ({itemset.frequency_pct}%) | {itemset.size} Anomalien
            </div>
            {itemset.items.map((it, j) => (
              <div key={j} className="flex justify-between text-[10px]">
                <span className="text-gray-300">{labelFor(it.metric)}</span>
                <span className="text-gray-500 font-mono">-{it.bucket_end_min}..-{it.bucket_start_min}m</span>
              </div>
            ))}
          </div>

          <label className="block">
            <span className="text-gray-400 text-[10px] block mb-0.5">Name</span>
            <input type="text" value={name} onChange={e => setName(e.target.value)}
              placeholder="z.B. Volume+RSI-Pre-Pump" disabled={saving || saved}
              className="input text-xs py-1 w-full" />
          </label>
          <label className="block">
            <span className="text-gray-400 text-[10px] block mb-0.5">Beschreibung</span>
            <textarea value={description} onChange={e => setDescription(e.target.value)}
              rows="2" disabled={saving || saved}
              className="input text-xs py-1 w-full resize-none" />
          </label>

          {error && <div className="text-red-400 bg-red-900/20 p-2 rounded text-[10px]">{error}</div>}
          {saved && (
            <div className="text-emerald-400 bg-emerald-900/20 p-2 rounded text-[10px]">
              Gespeichert — Set-ID {saved.set_id} ({saved.item_count} Items)
            </div>
          )}

          <div className="flex gap-2">
            <button onClick={onClose} className="btn btn-secondary flex-1 text-xs">
              {saved ? 'Schliessen' : 'Abbrechen'}
            </button>
            {!saved && (
              <button onClick={save} disabled={saving || !name.trim()}
                className="btn btn-primary flex-1 text-xs flex items-center justify-center gap-1 disabled:opacity-50">
                <Save size={12} /> {saving ? 'Speichert...' : 'Speichern'}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
