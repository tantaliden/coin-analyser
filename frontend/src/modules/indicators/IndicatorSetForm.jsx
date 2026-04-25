import { useState, useEffect } from 'react'
import { X } from 'lucide-react'
import api from '../../utils/api'
import { useConfigStore } from '../../stores/configStore'

export default function IndicatorSetForm({ onCreated, onCancel }) {
  const { getKlineMetricsDurations } = useConfigStore()
  const [groups, setGroups] = useState([])
  const [form, setForm] = useState({
    name: '',
    description: '',
    coin_group_id: null,
    search_date_from: '',
    search_date_to: '',
    search_percent_min: 5,
    search_percent_max: 100,
    search_direction: 'up',
    search_duration_minutes: 120,
    prehistory_minutes: 720
  })
  const [error, setError] = useState(null)
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    api.get('/api/v1/groups').then(r => setGroups(r.data.groups || [])).catch(() => {})
  }, [])

  const durations = getKlineMetricsDurations()

  const handleSubmit = async () => {
    if (!form.name.trim()) { setError('Name erforderlich'); return }
    setSubmitting(true)
    setError(null)
    try {
      const res = await api.post('/api/v1/indicators/sets', form)
      onCreated(res.data.set_id)
    } catch (e) {
      setError(e.response?.data?.detail || 'Erstellen fehlgeschlagen')
    }
    setSubmitting(false)
  }

  const update = (key, val) => setForm(prev => ({ ...prev, [key]: val }))

  return (
    <div className="p-3 border-b border-gray-700 bg-gray-800/50">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-semibold">Neues Indikator-Set</span>
        <button onClick={onCancel} className="p-1 hover:bg-gray-700 rounded"><X size={14} /></button>
      </div>

      {error && <div className="text-red-400 text-xs mb-2">{error}</div>}

      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="col-span-2">
          <label className="text-gray-400">Name *</label>
          <input value={form.name} onChange={e => update('name', e.target.value)} className="input w-full text-xs py-1" />
        </div>
        <div>
          <label className="text-gray-400">Richtung</label>
          <select value={form.search_direction} onChange={e => update('search_direction', e.target.value)} className="input w-full text-xs py-1">
            <option value="up">Long</option>
            <option value="down">Short</option>
          </select>
        </div>
        <div>
          <label className="text-gray-400">Dauer (Min)</label>
          <select value={form.search_duration_minutes} onChange={e => update('search_duration_minutes', parseInt(e.target.value))} className="input w-full text-xs py-1">
            {durations.map(d => <option key={d} value={d}>{d}</option>)}
          </select>
        </div>
        <div>
          <label className="text-gray-400">Min %</label>
          <input type="number" value={form.search_percent_min} onChange={e => update('search_percent_min', parseFloat(e.target.value))} className="input w-full text-xs py-1" step="0.5" />
        </div>
        <div>
          <label className="text-gray-400">Vorlauf (Min)</label>
          <input type="number" value={form.prehistory_minutes} onChange={e => update('prehistory_minutes', parseInt(e.target.value))} className="input w-full text-xs py-1" step="60" />
        </div>
        <div>
          <label className="text-gray-400">Von</label>
          <input type="date" value={form.search_date_from} onChange={e => update('search_date_from', e.target.value)} className="input w-full text-xs py-1" />
        </div>
        <div>
          <label className="text-gray-400">Bis</label>
          <input type="date" value={form.search_date_to} onChange={e => update('search_date_to', e.target.value)} className="input w-full text-xs py-1" />
        </div>
        <div className="col-span-2">
          <label className="text-gray-400">Coin-Gruppe</label>
          <select value={form.coin_group_id || ''} onChange={e => update('coin_group_id', e.target.value ? parseInt(e.target.value) : null)} className="input w-full text-xs py-1">
            <option value="">Alle Coins</option>
            {groups.map(g => <option key={g.id} value={g.id}>{g.name} ({(g.coins || []).length})</option>)}
          </select>
        </div>
      </div>

      <button onClick={handleSubmit} disabled={submitting}
        className="mt-2 px-3 py-1.5 bg-blue-600 hover:bg-blue-500 rounded text-xs w-full disabled:opacity-50">
        {submitting ? 'Erstelle...' : 'Set erstellen'}
      </button>
    </div>
  )
}
