import { useState } from 'react'
import { X } from 'lucide-react'
import api from '../../utils/api'

// Dialog zum Speichern eines Sets aus ChartView-Drawings.
// Primary-Context (Zeitraum, Filter, Events) wird mitgeschickt — Pflicht, kein Fallback.
export default function SaveSetDialog({
  criteria, initialPoints, initialPointConfig, globalFuzzy,
  durationMinutes, candleTimeframeMinutes, prehistoryMinutes, direction,
  primaryContext,  // Pflicht — enthaelt searchParams der Primary-Suche
  onClose, onSaved,
}) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [targetPercent, setTargetPercent] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState(null)

  const handleSave = async () => {
    if (!name.trim()) { setError('Name fehlt'); return }
    if (!primaryContext) { setError('Primary-Suche-Kontext fehlt — speichern nicht moeglich'); return }
    if (!globalFuzzy) { setError('Fuzzy-Defaults fehlen'); return }
    if (!initialPointConfig) { setError('Initial-Point-Config fehlt'); return }
    if (durationMinutes == null) { setError('duration_minutes fehlt'); return }
    if (candleTimeframeMinutes == null) { setError('candle_timeframe fehlt'); return }
    if (prehistoryMinutes == null) { setError('prehistory_minutes fehlt'); return }
    if (!direction) { setError('direction fehlt'); return }
    if (!targetPercent) { setError('Ziel-Prozent fehlt'); return }

    setSaving(true); setError(null)
    try {
      const payload = {
        name: name.trim(),
        description: description.trim(),
        duration_minutes: durationMinutes,
        direction: direction,
        target_percent: parseFloat(targetPercent),
        prehistory_minutes: prehistoryMinutes,
        candle_timeframe: candleTimeframeMinutes,
        criteria: criteria || [],
        initial_points: initialPoints || [],
        initial_point_config: initialPointConfig,
        global_fuzzy: globalFuzzy,
        primary_context: primaryContext,
      }
      const res = await api.post('/api/v1/indicators/sets/from-drawings', payload)
      onSaved?.(res.data)
      onClose?.()
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Speichern fehlgeschlagen')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center">
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-4 w-96 space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-200">Als Indikator-Set speichern</h3>
          <button onClick={onClose} className="text-gray-500 hover:text-white"><X size={16} /></button>
        </div>
        <div>
          <label className="text-xs text-gray-400 block mb-1">Name *</label>
          <input value={name} onChange={e => setName(e.target.value)}
            className="input w-full text-sm" placeholder="z.B. Pump-Pattern Alpha" autoFocus />
        </div>
        <div>
          <label className="text-xs text-gray-400 block mb-1">Beschreibung</label>
          <textarea value={description} onChange={e => setDescription(e.target.value)}
            className="input w-full text-xs" rows={2} />
        </div>
        <div>
          <label className="text-xs text-gray-400 block mb-1">Ziel-Prozent * (fuer Kategorisierung beim Backtest)</label>
          <input type="number" step="0.5" value={targetPercent}
            onChange={e => setTargetPercent(e.target.value)}
            className="input w-full text-sm" placeholder="5.0" />
        </div>
        <div className="text-[10px] text-gray-500 space-y-0.5">
          <div>{(criteria?.length || 0)} Kriterien, {(initialPoints?.length || 0)} Initialpunkte</div>
          <div>Primary-Suche: {primaryContext?.events_at_creation} Events | {primaryContext?.search_date_from?.slice(0,10)} – {primaryContext?.search_date_to?.slice(0,10)}</div>
        </div>
        {error && <div className="text-red-400 text-xs bg-red-900/30 p-2 rounded">{error}</div>}
        <button onClick={handleSave} disabled={saving}
          className="btn btn-primary w-full text-sm disabled:opacity-50">
          {saving ? 'Speichere...' : 'Speichern'}
        </button>
      </div>
    </div>
  )
}
