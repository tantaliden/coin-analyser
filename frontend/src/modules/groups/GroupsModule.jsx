import { useState, useEffect } from 'react'
import { Plus, Trash2, X, Search } from 'lucide-react'
import { useConfigStore } from '../../stores/configStore'
import api from '../../utils/api'

export default function GroupsModule() {
  const { getLabel } = useConfigStore()
  const [groups, setGroups] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showCreate, setShowCreate] = useState(false)
  const [newName, setNewName] = useState('')
  const [newColor, setNewColor] = useState('#3B82F6')
  const [selectedGroup, setSelectedGroup] = useState(null)
  const [coinSearch, setCoinSearch] = useState('')
  const [allCoins, setAllCoins] = useState([])
  const [showCoinPicker, setShowCoinPicker] = useState(false)

  const loadGroups = async () => {
    try {
      const res = await api.get('/api/v1/groups')
      setGroups(res.data.groups || [])
      setError(null)
    } catch (e) {
      setError(e.response?.data?.detail || 'Gruppen laden fehlgeschlagen')
    }
    setLoading(false)
  }

  const loadCoins = async () => {
    try {
      const res = await api.get('/api/v1/meta/symbols')
      setAllCoins(res.data.symbols || [])
    } catch {}
  }

  useEffect(() => { loadGroups(); loadCoins() }, [])

  const createGroup = async () => {
    if (!newName.trim()) return
    try {
      await api.post('/api/v1/groups', { name: newName, color: newColor })
      setNewName('')
      setShowCreate(false)
      loadGroups()
    } catch (e) {
      setError(e.response?.data?.detail || 'Erstellen fehlgeschlagen')
    }
  }

  const deleteGroup = async (id) => {
    if (!confirm('Gruppe wirklich löschen?')) return
    try {
      await api.delete(`/api/v1/groups/${id}`)
      if (selectedGroup?.id === id) setSelectedGroup(null)
      loadGroups()
    } catch (e) {
      setError(e.response?.data?.detail || 'Löschen fehlgeschlagen')
    }
  }

  const addCoin = async (symbol) => {
    if (!selectedGroup) return
    try {
      await api.post(`/api/v1/groups/${selectedGroup.id}/coins`, [symbol])
      loadGroups()
    } catch {}
  }

  const removeCoin = async (symbol) => {
    if (!selectedGroup) return
    try {
      await api.delete(`/api/v1/groups/${selectedGroup.id}/coins/${symbol}`)
      loadGroups()
    } catch {}
  }

  const filteredCoins = allCoins.filter(s =>
    s.toLowerCase().includes(coinSearch.toLowerCase()) &&
    !(selectedGroup?.coins || []).includes(s)
  ).slice(0, 50)

  // Sync selectedGroup with updated groups
  useEffect(() => {
    if (selectedGroup) {
      const updated = groups.find(g => g.id === selectedGroup.id)
      if (updated) setSelectedGroup(updated)
    }
  }, [groups])

  if (loading) return <div className="text-gray-400 p-4">Laden...</div>

  return (
    <div className="h-full flex flex-col">
      {error && <div className="p-2 text-red-400 text-sm bg-red-900/30">{error}</div>}

      <div className="flex items-center gap-2 p-2 border-b border-gray-700">
        <button onClick={() => setShowCreate(!showCreate)} className="p-1.5 bg-blue-600 hover:bg-blue-500 rounded" title="Neue Gruppe">
          <Plus size={14} />
        </button>
        <span className="text-xs text-gray-400">{groups.length} Gruppen</span>
      </div>

      {showCreate && (
        <div className="p-2 border-b border-gray-700 flex gap-2">
          <input value={newName} onChange={e => setNewName(e.target.value)} placeholder="Gruppenname"
            className="input flex-1 text-sm" onKeyDown={e => e.key === 'Enter' && createGroup()} />
          <input type="color" value={newColor} onChange={e => setNewColor(e.target.value)} className="w-8 h-8 rounded cursor-pointer" />
          <button onClick={createGroup} className="px-2 py-1 bg-green-600 hover:bg-green-500 rounded text-sm">OK</button>
          <button onClick={() => setShowCreate(false)} className="p-1 hover:bg-gray-700 rounded"><X size={14} /></button>
        </div>
      )}

      <div className="flex-1 overflow-auto flex">
        {/* Gruppen-Liste */}
        <div className="w-1/3 border-r border-gray-700 overflow-auto">
          {groups.map(g => (
            <div key={g.id}
              onClick={() => { setSelectedGroup(g); setShowCoinPicker(false) }}
              className={`p-2 cursor-pointer hover:bg-gray-700 border-b border-gray-700/50 flex items-center gap-2 ${selectedGroup?.id === g.id ? 'bg-gray-700' : ''}`}>
              <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ background: g.color }} />
              <div className="flex-1 min-w-0">
                <div className="text-sm truncate">{g.name}</div>
                <div className="text-xs text-gray-500">{(g.coins || []).length} Coins</div>
              </div>
              <button onClick={(e) => { e.stopPropagation(); deleteGroup(g.id) }}
                className="p-1 hover:bg-red-600/30 rounded opacity-50 hover:opacity-100">
                <Trash2 size={12} />
              </button>
            </div>
          ))}
        </div>

        {/* Coins der Gruppe */}
        <div className="flex-1 overflow-auto">
          {selectedGroup ? (
            <div className="p-2">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-sm font-semibold">{selectedGroup.name}</span>
                <button onClick={() => setShowCoinPicker(!showCoinPicker)}
                  className="px-2 py-1 bg-blue-600 hover:bg-blue-500 rounded text-xs flex items-center gap-1">
                  <Plus size={12} /> Coin
                </button>
              </div>

              {showCoinPicker && (
                <div className="mb-2 bg-gray-800 border border-gray-700 rounded p-2">
                  <div className="flex items-center gap-1 mb-1">
                    <Search size={12} className="text-gray-400" />
                    <input value={coinSearch} onChange={e => setCoinSearch(e.target.value)}
                      placeholder="Symbol suchen..." className="input text-xs py-1 flex-1" autoFocus />
                  </div>
                  <div className="max-h-32 overflow-auto flex flex-wrap gap-1">
                    {filteredCoins.map(s => (
                      <button key={s} onClick={() => addCoin(s)}
                        className="px-2 py-0.5 bg-gray-700 hover:bg-blue-600 rounded text-xs font-mono">
                        {s}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              <div className="flex flex-wrap gap-1">
                {(selectedGroup.coins || []).map(s => (
                  <span key={s} className="inline-flex items-center gap-1 px-2 py-0.5 bg-gray-700 rounded text-xs font-mono">
                    {s}
                    <button onClick={() => removeCoin(s)} className="hover:text-red-400"><X size={10} /></button>
                  </span>
                ))}
                {(selectedGroup.coins || []).length === 0 && (
                  <span className="text-gray-500 text-xs">Keine Coins. Klicke "+ Coin" zum Hinzufügen.</span>
                )}
              </div>
            </div>
          ) : (
            <div className="p-4 text-gray-500 text-sm text-center">Wähle eine Gruppe aus</div>
          )}
        </div>
      </div>
    </div>
  )
}
