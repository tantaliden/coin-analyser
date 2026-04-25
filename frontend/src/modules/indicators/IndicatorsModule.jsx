import { useState, useEffect } from 'react'
import { Plus, Trash2, ChevronRight, ToggleLeft, ToggleRight, Play, Settings } from 'lucide-react'
import api from '../../utils/api'
import { useConfigStore } from '../../stores/configStore'
import IndicatorItemRow from './IndicatorItemRow'
import IndicatorSetForm from './IndicatorSetForm'

export default function IndicatorsModule() {
  const { getIndicatorFields, getLabel } = useConfigStore()
  const [sets, setSets] = useState([])
  const [selectedSet, setSelectedSet] = useState(null)
  const [setDetail, setSetDetail] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showCreate, setShowCreate] = useState(false)
  const [showBacksearch, setShowBacksearch] = useState(false)
  const [backsearchResult, setBacksearchResult] = useState(null)
  const [backsearchLoading, setBacksearchLoading] = useState(false)

  const loadSets = async () => {
    try {
      const res = await api.get('/api/v1/indicators/sets')
      setSets(res.data.sets || [])
      setError(null)
    } catch (e) {
      setError(e.response?.data?.detail || 'Sets laden fehlgeschlagen')
    }
    setLoading(false)
  }

  const loadSetDetail = async (setId) => {
    try {
      const res = await api.get(`/api/v1/indicators/sets/${setId}`)
      setSetDetail(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Set-Details laden fehlgeschlagen')
    }
  }

  useEffect(() => { loadSets() }, [])

  useEffect(() => {
    if (selectedSet) loadSetDetail(selectedSet)
  }, [selectedSet])

  const deleteSet = async (setId) => {
    if (!confirm('Set wirklich löschen?')) return
    try {
      await api.delete(`/api/v1/indicators/sets/${setId}`)
      if (selectedSet === setId) { setSelectedSet(null); setSetDetail(null) }
      loadSets()
    } catch (e) {
      setError(e.response?.data?.detail || 'Löschen fehlgeschlagen')
    }
  }

  const toggleItem = async (setId, itemId) => {
    try {
      await api.put(`/api/v1/indicators/sets/${setId}/items/${itemId}/toggle`)
      loadSetDetail(setId)
    } catch {}
  }

  const deleteItem = async (setId, itemId) => {
    try {
      await api.delete(`/api/v1/indicators/sets/${setId}/items/${itemId}`)
      loadSetDetail(setId)
    } catch {}
  }

  const runBacksearch = async () => {
    if (!setDetail?.set) return
    setBacksearchLoading(true)
    setBacksearchResult(null)
    try {
      const s = setDetail.set
      const res = await api.post(`/api/v1/indicators/sets/${s.set_id}/backsearch`, {
        start_date: s.search_date_from ? new Date(s.search_date_from).toISOString().split('T')[0] : '2024-01-01',
        end_date: s.search_date_to ? new Date(s.search_date_to).toISOString().split('T')[0] : new Date().toISOString().split('T')[0],
        scan_interval_minutes: 60
      })
      setBacksearchResult(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Backsearch fehlgeschlagen')
    }
    setBacksearchLoading(false)
  }

  if (loading) return <div className="text-gray-400 p-4">Laden...</div>

  return (
    <div className="h-full flex flex-col">
      {error && <div className="p-2 text-red-400 text-sm bg-red-900/30">{error}</div>}

      <div className="flex items-center gap-2 p-2 border-b border-gray-700">
        <button onClick={() => setShowCreate(!showCreate)} className="p-1.5 bg-blue-600 hover:bg-blue-500 rounded" title="Neues Set">
          <Plus size={14} />
        </button>
        <span className="text-xs text-gray-400">{sets.length} Sets</span>
      </div>

      {showCreate && (
        <IndicatorSetForm onCreated={(id) => { setShowCreate(false); loadSets(); setSelectedSet(id) }}
          onCancel={() => setShowCreate(false)} />
      )}

      <div className="flex-1 overflow-auto flex">
        {/* Sets Liste */}
        <div className="w-1/3 border-r border-gray-700 overflow-auto">
          {sets.map(s => (
            <div key={s.set_id}
              onClick={() => setSelectedSet(s.set_id)}
              className={`p-2 cursor-pointer hover:bg-gray-700 border-b border-gray-700/50 ${selectedSet === s.set_id ? 'bg-gray-700' : ''}`}>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full" style={{ background: s.color || '#3b82f6' }} />
                <span className="text-sm truncate flex-1">{s.name}</span>
                {s.current_accuracy !== null && (
                  <span className={`text-xs font-mono ${parseFloat(s.current_accuracy) >= 70 ? 'text-green-400' : parseFloat(s.current_accuracy) >= 50 ? 'text-yellow-400' : 'text-red-400'}`}>
                    {parseFloat(s.current_accuracy).toFixed(0)}%
                  </span>
                )}
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {s.item_count || 0} Indikatoren · {s.direction || 'up'}
              </div>
            </div>
          ))}
        </div>

        {/* Set Detail */}
        <div className="flex-1 overflow-auto">
          {setDetail?.set ? (
            <div className="p-2">
              <div className="flex items-center gap-2 mb-3">
                <h4 className="text-sm font-semibold flex-1">{setDetail.set.name}</h4>
                <button onClick={runBacksearch} disabled={backsearchLoading}
                  className="px-2 py-1 bg-green-600 hover:bg-green-500 rounded text-xs flex items-center gap-1 disabled:opacity-50">
                  <Play size={12} /> {backsearchLoading ? 'Läuft...' : 'Gegensuche'}
                </button>
                <button onClick={() => deleteSet(setDetail.set.set_id)}
                  className="p-1 hover:bg-red-600/30 rounded">
                  <Trash2 size={14} />
                </button>
              </div>

              {/* Set Info */}
              <div className="grid grid-cols-3 gap-2 text-xs mb-3 bg-gray-800 rounded p-2">
                <div><span className="text-gray-500">Richtung:</span> {setDetail.set.search_direction || setDetail.set.direction}</div>
                <div><span className="text-gray-500">Ziel:</span> {setDetail.set.search_percent_min || setDetail.set.target_percent}%</div>
                <div><span className="text-gray-500">Dauer:</span> {setDetail.set.search_duration_minutes || setDetail.set.duration_minutes}min</div>
                <div><span className="text-gray-500">Vorlauf:</span> {setDetail.set.prehistory_minutes}min</div>
                <div><span className="text-gray-500">TP:</span> {setDetail.set.take_profit_pct}%</div>
                <div><span className="text-gray-500">SL:</span> {setDetail.set.stop_loss_pct}%</div>
              </div>

              {/* Backsearch Result */}
              {backsearchResult && (
                <div className="mb-3 bg-gray-800 rounded p-2 text-xs">
                  <div className="font-semibold mb-1">Gegensuche Ergebnis</div>
                  <div className="grid grid-cols-3 gap-2">
                    <div className="text-green-400">✓ {backsearchResult.statistics?.green_count || 0} ({backsearchResult.statistics?.green_percent || 0}%)</div>
                    <div className="text-gray-400">◌ {backsearchResult.statistics?.grey_count || 0} ({backsearchResult.statistics?.grey_percent || 0}%)</div>
                    <div className="text-red-400">✗ {backsearchResult.statistics?.red_count || 0} ({backsearchResult.statistics?.red_percent || 0}%)</div>
                  </div>
                  <div className="mt-1 font-semibold">
                    Trefferquote: <span className={backsearchResult.statistics?.probability >= 70 ? 'text-green-400' : 'text-yellow-400'}>
                      {backsearchResult.statistics?.probability || 0}%
                    </span>
                  </div>
                </div>
              )}

              {/* Indicator Items */}
              <div className="text-xs font-semibold text-gray-400 mb-1">Indikatoren</div>
              {(setDetail.items || []).map(item => (
                <IndicatorItemRow key={item.item_id} item={item} setId={setDetail.set.set_id}
                  onToggle={() => toggleItem(setDetail.set.set_id, item.item_id)}
                  onDelete={() => deleteItem(setDetail.set.set_id, item.item_id)} />
              ))}

              {(setDetail.items || []).length === 0 && (
                <div className="text-gray-500 text-xs py-2">Keine Indikatoren. Füge welche im Chart hinzu.</div>
              )}
            </div>
          ) : (
            <div className="p-4 text-gray-500 text-sm text-center">Wähle ein Set aus</div>
          )}
        </div>
      </div>
    </div>
  )
}
