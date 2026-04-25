import { useState, useEffect } from 'react'
import { Power, RefreshCw, AlertTriangle } from 'lucide-react'
import api from '../../utils/api'

export default function BotModule() {
  const [config, setConfig] = useState(null)
  const [upcoming, setUpcoming] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [saving, setSaving] = useState(false)

  const loadConfig = async () => {
    try {
      const [cfgRes, upRes] = await Promise.all([
        api.get('/api/v1/bot/config'),
        api.get('/api/v1/bot/upcoming').catch(() => ({ data: { events: [] } }))
      ])
      if (cfgRes.data.error) setError(cfgRes.data.error)
      else setConfig(cfgRes.data)
      setUpcoming(upRes.data.events || [])
    } catch (e) {
      setError(e.response?.data?.detail || 'Bot-Config laden fehlgeschlagen')
    }
    setLoading(false)
  }

  useEffect(() => { loadConfig() }, [])

  const updateConfig = async (updates) => {
    setSaving(true)
    setError(null)
    try {
      const res = await api.put('/api/v1/bot/config', updates)
      if (res.data.error) { setError(res.data.error); setSaving(false); return }
      loadConfig()
    } catch (e) {
      setError(e.response?.data?.detail || 'Speichern fehlgeschlagen')
    }
    setSaving(false)
  }

  const executeTrade = async (eventId) => {
    if (!confirm('Trade wirklich ausführen?')) return
    try {
      const res = await api.post(`/api/v1/bot/execute/${eventId}`)
      if (res.data.error) setError(res.data.error)
      else loadConfig()
    } catch (e) {
      setError(e.response?.data?.detail || 'Trade fehlgeschlagen')
    }
  }

  if (loading) return <div className="text-gray-400 p-4">Laden...</div>

  return (
    <div className="h-full flex flex-col">
      {error && (
        <div className="p-2 text-red-400 text-sm bg-red-900/30 flex items-center gap-2">
          <AlertTriangle size={14} /> {error}
        </div>
      )}

      {/* Bot Status + Toggle */}
      <div className="p-3 border-b border-gray-700">
        <div className="flex items-center gap-3 mb-3">
          <button
            onClick={() => updateConfig({ is_active: !config?.is_active })}
            disabled={saving}
            className={`flex items-center gap-2 px-4 py-2 rounded font-semibold text-sm ${
              config?.is_active ? 'bg-green-600 hover:bg-green-500' : 'bg-gray-700 hover:bg-gray-600'
            }`}>
            <Power size={16} />
            {config?.is_active ? 'Bot AKTIV' : 'Bot AUS'}
          </button>
          <button onClick={loadConfig} className="p-2 bg-gray-700 hover:bg-gray-600 rounded" title="Aktualisieren">
            <RefreshCw size={14} />
          </button>
        </div>

        {/* Config */}
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div>
            <label className="text-gray-400">Betrag pro Trade (USDC)</label>
            <input type="number" value={config?.amount_per_trade || 50} min="10" step="10"
              onChange={e => updateConfig({ amount_per_trade: parseFloat(e.target.value) })}
              className="input w-full text-sm py-1.5 mt-1" />
          </div>
          <div className="bg-gray-800 rounded p-2">
            <div className="text-gray-400 mb-1">Heute</div>
            <div className="text-sm">{config?.today_trades || 0} Trades</div>
            <div className={`text-sm font-mono ${(config?.today_profit_loss || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {(config?.today_profit_loss || 0) >= 0 ? '+' : ''}{(config?.today_profit_loss || 0).toFixed(2)} USDC
            </div>
          </div>
        </div>
      </div>

      {/* Upcoming Events */}
      <div className="flex-1 overflow-auto">
        <div className="p-2 text-xs font-semibold text-gray-400 border-b border-gray-700">
          Anstehende Events ({upcoming.length})
        </div>
        {upcoming.length === 0 ? (
          <div className="p-4 text-gray-500 text-sm text-center">Keine anstehenden Events</div>
        ) : (
          <div className="divide-y divide-gray-700/50">
            {upcoming.map(e => (
              <div key={e.event_id} className="p-2 flex items-center gap-2 text-xs">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-semibold">{e.symbol}</span>
                    <span className="text-green-400">+{e.expected_target_pct}%</span>
                    <span className="text-gray-500">{e.expected_duration_min}min</span>
                  </div>
                  <div className="text-gray-500 mt-0.5">
                    {e.set_name} · Start: {e.expected_start ? new Date(e.expected_start).toLocaleString('de-DE', { hour: '2-digit', minute: '2-digit' }) : '?'}
                    {e.take_profit_pct && <span className="ml-1">TP {e.take_profit_pct}%</span>}
                    {e.stop_loss_pct && <span className="ml-1">SL {e.stop_loss_pct}%</span>}
                  </div>
                </div>
                <span className={`px-2 py-0.5 rounded ${e.status === 'active' ? 'bg-green-600/30 text-green-400' : 'bg-yellow-600/30 text-yellow-400'}`}>
                  {e.status}
                </span>
                {config?.is_active && e.status === 'waiting' && (
                  <button onClick={() => executeTrade(e.event_id)}
                    className="px-2 py-1 bg-blue-600 hover:bg-blue-500 rounded">
                    Trade
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
