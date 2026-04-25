import { useEffect, useState } from 'react'
import { X, RefreshCw, Trash2, Play, Download } from 'lucide-react'
import api from '../../utils/api'
import { SCAN_UI_SETTINGS } from '../../config/chartSettings'
import ScanResultsView from './ScanResultsView'

// Einblendbares Panel im ChartView: Sets laden, loeschen, scannen.
export default function SetManagerPanel({ onClose, onLoadSet }) {
  const [sets, setSets] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [scanningSetId, setScanningSetId] = useState(null)
  const [scanResult, setScanResult] = useState(null)
  const [scanPeriodDays, setScanPeriodDays] = useState(SCAN_UI_SETTINGS.defaultPeriodDays)

  const load = async () => {
    setLoading(true); setError(null)
    try {
      const res = await api.get('/api/v1/indicators/sets?only_mine=true')
      // Backend liefert { sets: [...] } — nicht direkt das Array
      setSets(res.data.sets)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Laden fehlgeschlagen')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const handleDelete = async (setId) => {
    if (!confirm('Set wirklich loeschen?')) return
    try {
      await api.delete(`/api/v1/indicators/sets/${setId}`)
      if (scanResult?.set_id === setId) setScanResult(null)
      await load()
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    }
  }

  const handleLoad = async (setId) => {
    setError(null)
    try {
      const res = await api.get(`/api/v1/indicators/sets/${setId}/load`)
      onLoadSet?.(res.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Laden fehlgeschlagen')
    }
  }

  const handleScan = async (setId) => {
    setScanningSetId(setId)
    setError(null)
    setScanResult(null)
    try {
      const res = await api.post(`/api/v1/indicators/sets/${setId}/scan`, {
        period_days: parseInt(scanPeriodDays),
      })
      setScanResult(res.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Scan fehlgeschlagen')
    } finally {
      setScanningSetId(null)
    }
  }

  return (
    <div className="w-96 border-r border-gray-700 bg-gray-800/80 flex flex-col overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-700">
        <span className="text-xs font-semibold text-gray-300">Gespeicherte Sets</span>
        <div className="flex gap-2">
          <button onClick={load} className="text-gray-400 hover:text-white"><RefreshCw size={12} /></button>
          <button onClick={onClose} className="text-gray-500 hover:text-white"><X size={14} /></button>
        </div>
      </div>

      <div className="p-2 border-b border-gray-700 flex items-center gap-2 text-xs">
        <label className="text-gray-400">Scan-Zeitraum:</label>
        <select value={scanPeriodDays} onChange={e => setScanPeriodDays(e.target.value)}
          className="input text-xs py-1 flex-1">
          {SCAN_UI_SETTINGS.periodOptions.map(d => (
            <option key={d} value={d}>{d} Tage</option>
          ))}
        </select>
      </div>

      <div className="flex-1 overflow-auto p-2">
        {loading && <div className="text-xs text-gray-500">Laedt...</div>}
        {error && <div className="text-xs text-red-400 bg-red-900/30 p-2 rounded mb-2">{error}</div>}
        {sets && sets.length === 0 && <div className="text-xs text-gray-500">Noch keine Sets gespeichert</div>}
        {sets && sets.map(s => (
          <div key={s.set_id} className="mb-2 p-2 bg-gray-900/50 rounded border border-gray-700 text-xs">
            <div className="flex items-center justify-between gap-2">
              <span className="font-semibold text-gray-200 truncate">{s.name}</span>
              <div className="flex gap-1 shrink-0">
                <button onClick={() => handleLoad(s.set_id)}
                  className="p-1 hover:bg-blue-700/30 rounded" title="In ChartView laden (Mainsearch + Drawings)">
                  <Download size={12} />
                </button>
                <button onClick={() => handleScan(s.set_id)} disabled={scanningSetId === s.set_id}
                  className="p-1 hover:bg-emerald-700/30 rounded disabled:opacity-50" title="Backsearch starten">
                  {scanningSetId === s.set_id ? <RefreshCw size={12} className="animate-spin" /> : <Play size={12} />}
                </button>
                <button onClick={() => handleDelete(s.set_id)}
                  className="p-1 hover:bg-red-600/30 rounded"><Trash2 size={12} /></button>
              </div>
            </div>
            <div className="text-[10px] text-gray-500 mt-1">
              {s.item_count} Items | {s.duration_minutes}m {s.direction} | Ziel {s.target_percent}%
            </div>
            <div className="text-[10px] text-gray-600 mt-0.5">
              {s.search_date_from?.slice(0,10) || '-'} bis {s.search_date_to?.slice(0,10) || '-'}
              {s.events_at_creation != null && ` | ${s.events_at_creation} Events`}
            </div>
          </div>
        ))}
      </div>

      {scanResult && (
        <div className="border-t border-gray-700 p-2 bg-gray-900/50 overflow-auto max-h-[60%]">
          <ScanResultsView scanResult={scanResult} />
        </div>
      )}
    </div>
  )
}
