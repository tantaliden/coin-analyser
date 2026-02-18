import { useState, useEffect, useMemo } from 'react'
import { Plus, Trash2, X, Search, ChevronUp, ChevronDown } from 'lucide-react'
import api from '../../utils/api'

export default function GroupsModule() {
  const [groups, setGroups] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showCreate, setShowCreate] = useState(false)
  const [newName, setNewName] = useState('')
  const [newColor, setNewColor] = useState('#3B82F6')
  const [selectedGroup, setSelectedGroup] = useState(null)
  const [showCoinPicker, setShowCoinPicker] = useState(false)

  // Coin picker state
  const [coins, setCoins] = useState([])
  const [coinSearch, setCoinSearch] = useState('')
  const [networkFilter, setNetworkFilter] = useState('')
  const [categoryFilter, setCategoryFilter] = useState('')
  const [networks, setNetworks] = useState([])
  const [categories, setCategories] = useState([])
  const [coinsLoading, setCoinsLoading] = useState(false)

  // Sort state
  const [sortCol, setSortCol] = useState('symbol')
  const [sortDir, setSortDir] = useState('asc')

  const loadGroups = async () => {
    try {
      const res = await api.get('/api/v1/groups')
      setGroups(res.data.groups || [])
      setError(null)
    } catch (e) { setError('Gruppen laden fehlgeschlagen') }
    setLoading(false)
  }

  const loadFilters = async () => {
    try {
      const [nRes, cRes] = await Promise.all([
        api.get('/api/v1/coins/networks'), api.get('/api/v1/coins/categories')
      ])
      setNetworks(nRes.data.networks || [])
      setCategories(cRes.data.categories || [])
    } catch {}
  }

  const loadCoins = async () => {
    setCoinsLoading(true)
    try {
      const params = new URLSearchParams()
      if (coinSearch) params.set('search', coinSearch)
      if (networkFilter) params.set('network', networkFilter)
      if (categoryFilter) params.set('category', categoryFilter)
      const res = await api.get(`/api/v1/coins?${params}`)
      setCoins(res.data.coins || [])
    } catch {}
    setCoinsLoading(false)
  }

  useEffect(() => { loadGroups(); loadFilters() }, [])
  useEffect(() => { if (showCoinPicker) loadCoins() }, [showCoinPicker, coinSearch, networkFilter, categoryFilter])

  // Debounce search
  const [searchTimeout, setSearchTimeout] = useState(null)
  const handleSearch = (val) => {
    setCoinSearch(val)
    if (searchTimeout) clearTimeout(searchTimeout)
    setSearchTimeout(setTimeout(() => loadCoins(), 300))
  }

  const createGroup = async () => {
    if (!newName.trim()) return
    try {
      await api.post('/api/v1/groups', { name: newName, color: newColor })
      setNewName(''); setShowCreate(false); loadGroups()
    } catch (e) { setError('Erstellen fehlgeschlagen') }
  }

  const deleteGroup = async (id) => {
    if (!confirm('Gruppe löschen?')) return
    try {
      await api.delete(`/api/v1/groups/${id}`)
      if (selectedGroup?.id === id) setSelectedGroup(null)
      loadGroups()
    } catch {}
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

  useEffect(() => {
    if (selectedGroup) {
      const updated = groups.find(g => g.id === selectedGroup.id)
      if (updated) setSelectedGroup(updated)
    }
  }, [groups])

  const groupCoins = selectedGroup?.coins || []

  // Sort + filter coins
  const toggleSort = (col) => {
    if (sortCol === col) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    } else {
      setSortCol(col)
      setSortDir('asc')
    }
  }

  const SortIcon = ({ col }) => {
    if (sortCol !== col) return <ChevronDown size={10} className="text-gray-600 ml-0.5 inline" />
    return sortDir === 'asc'
      ? <ChevronUp size={10} className="text-blue-400 ml-0.5 inline" />
      : <ChevronDown size={10} className="text-blue-400 ml-0.5 inline" />
  }

  const sortedCoins = useMemo(() => {
    const filtered = coins.filter(c => !groupCoins.includes(c.symbol))
    return filtered.sort((a, b) => {
      let va = '', vb = ''
      if (sortCol === 'symbol') { va = a.symbol; vb = b.symbol }
      else if (sortCol === 'name') { va = a.name || ''; vb = b.name || '' }
      else if (sortCol === 'network') { va = a.network || 'zzz'; vb = b.network || 'zzz' }
      const cmp = va.localeCompare(vb, undefined, { sensitivity: 'base' })
      return sortDir === 'asc' ? cmp : -cmp
    })
  }, [coins, groupCoins, sortCol, sortDir])

  if (loading) return <div className="text-gray-400 p-2 text-xs">Laden...</div>

  return (
    <div className="h-full flex flex-col text-xs overflow-hidden">
      {error && <div className="px-2 py-1 text-red-400 bg-red-900/30">{error}</div>}

      {/* Header */}
      <div className="flex items-center gap-2 px-2 py-1 border-b border-gray-700 flex-shrink-0">
        <button onClick={() => setShowCreate(!showCreate)} className="p-1 bg-blue-600 hover:bg-blue-500 rounded"><Plus size={12} /></button>
        <span className="text-gray-400">{groups.length} Gruppen</span>
      </div>

      {showCreate && (
        <div className="px-2 py-1 border-b border-gray-700 flex gap-1 flex-shrink-0">
          <input value={newName} onChange={e => setNewName(e.target.value)} placeholder="Name"
            className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-2 py-0.5" onKeyDown={e => e.key === 'Enter' && createGroup()} />
          <input type="color" value={newColor} onChange={e => setNewColor(e.target.value)} className="w-6 h-6 rounded cursor-pointer" />
          <button onClick={createGroup} className="px-2 py-0.5 bg-green-600 rounded">OK</button>
          <button onClick={() => setShowCreate(false)} className="p-0.5 hover:bg-gray-700 rounded"><X size={12} /></button>
        </div>
      )}

      <div className="flex-1 flex overflow-hidden">
        {/* Gruppen-Liste links */}
        <div className="w-40 border-r border-gray-700 overflow-auto flex-shrink-0">
          {groups.map(g => (
            <div key={g.id} onClick={() => { setSelectedGroup(g); setShowCoinPicker(false) }}
              className={`px-2 py-1.5 cursor-pointer hover:bg-gray-700 border-b border-gray-700/50 flex items-center gap-1.5 ${selectedGroup?.id === g.id ? 'bg-gray-700' : ''}`}>
              <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: g.color }} />
              <div className="flex-1 min-w-0">
                <div className="truncate">{g.name}</div>
                <div className="text-gray-500" style={{fontSize:'10px'}}>{(g.coins || []).length} Coins</div>
              </div>
              <button onClick={e => { e.stopPropagation(); deleteGroup(g.id) }}
                className="p-0.5 hover:bg-red-600/30 rounded opacity-30 hover:opacity-100"><Trash2 size={10} /></button>
            </div>
          ))}
        </div>

        {/* Rechts: Coins der Gruppe + Picker */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {selectedGroup ? (<>
            {/* Gruppe Header + Coins */}
            <div className="px-2 py-1 border-b border-gray-700 flex items-center gap-2 flex-shrink-0">
              <span className="font-semibold">{selectedGroup.name}</span>
              <span className="text-gray-500">{groupCoins.length} Coins</span>
              <button onClick={() => setShowCoinPicker(!showCoinPicker)}
                className="ml-auto px-2 py-0.5 bg-blue-600 hover:bg-blue-500 rounded flex items-center gap-1">
                <Plus size={10} /> Coin
              </button>
            </div>

            {/* Coins der Gruppe */}
            <div className="px-2 py-1 flex flex-wrap gap-1 border-b border-gray-700/50 max-h-20 overflow-auto flex-shrink-0">
              {groupCoins.length === 0 && <span className="text-gray-500">Keine Coins</span>}
              {groupCoins.map(s => (
                <span key={s} className="inline-flex items-center gap-0.5 px-1.5 py-0.5 bg-gray-700 rounded font-mono">
                  {s.replace('USDC','')}
                  <button onClick={() => removeCoin(s)} className="hover:text-red-400"><X size={8} /></button>
                </span>
              ))}
            </div>

            {/* Coin Picker Tabelle */}
            {showCoinPicker && (
              <div className="flex-1 flex flex-col overflow-hidden">
                {/* Filter-Zeile */}
                <div className="px-2 py-1 flex items-center gap-1 border-b border-gray-700 flex-shrink-0">
                  <Search size={10} className="text-gray-400" />
                  <input value={coinSearch} onChange={e => handleSearch(e.target.value)} placeholder="Suchen..."
                    className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-1.5 py-0.5" autoFocus />
                  <select value={networkFilter} onChange={e => setNetworkFilter(e.target.value)}
                    className="bg-zinc-800 border border-zinc-700 rounded px-1 py-0.5 max-w-32">
                    <option value="">Alle Chains</option>
                    {networks.map(n => <option key={n} value={n}>{n}</option>)}
                  </select>
                  <select value={categoryFilter} onChange={e => setCategoryFilter(e.target.value)}
                    className="bg-zinc-800 border border-zinc-700 rounded px-1 py-0.5 max-w-40">
                    <option value="">Alle Kategorien</option>
                    {categories.map(c => <option key={c} value={c}>{c}</option>)}
                  </select>
                  <span className="text-gray-500 ml-1">{sortedCoins.length}</span>
                </div>

                {/* Tabelle */}
                <div className="flex-1 overflow-auto">
                  <table className="w-full">
                    <thead className="sticky top-0 bg-zinc-900 z-10">
                      <tr className="text-gray-500 text-left">
                        <th className="px-2 py-1 font-normal w-8"></th>
                        <th className="px-2 py-1 font-normal cursor-pointer hover:text-gray-300 select-none" onClick={() => toggleSort('symbol')}>
                          Symbol<SortIcon col="symbol" />
                        </th>
                        <th className="px-2 py-1 font-normal cursor-pointer hover:text-gray-300 select-none" onClick={() => toggleSort('name')}>
                          Name<SortIcon col="name" />
                        </th>
                        <th className="px-2 py-1 font-normal cursor-pointer hover:text-gray-300 select-none" onClick={() => toggleSort('network')}>
                          Chain<SortIcon col="network" />
                        </th>
                        <th className="px-2 py-1 font-normal">Kategorien</th>
                      </tr>
                    </thead>
                    <tbody>
                      {coinsLoading ? (
                        <tr><td colSpan={5} className="text-center py-4 text-gray-500">Laden...</td></tr>
                      ) : sortedCoins.length === 0 ? (
                        <tr><td colSpan={5} className="text-center py-4 text-gray-500">Keine Ergebnisse</td></tr>
                      ) : sortedCoins.map(c => (
                        <tr key={c.symbol} className="border-t border-zinc-800 hover:bg-zinc-800/50">
                          <td className="px-2 py-0.5">
                            <button onClick={() => addCoin(c.symbol)}
                              className="p-0.5 bg-blue-600 hover:bg-blue-500 rounded"><Plus size={10} /></button>
                          </td>
                          <td className="px-2 py-0.5 font-mono font-medium">{c.symbol.replace('USDC','')}</td>
                          <td className="px-2 py-0.5 text-gray-300">{c.name || '-'}</td>
                          <td className="px-2 py-0.5">
                            {c.network ? (
                              <span className="px-1.5 py-0.5 bg-zinc-700 rounded" style={{fontSize:'10px'}}>{c.network}</span>
                            ) : <span className="text-gray-600">-</span>}
                          </td>
                          <td className="px-2 py-0.5">
                            <div className="flex flex-wrap gap-0.5 max-w-xs">
                              {(c.categories || []).slice(0, 3).map((cat, i) => (
                                <span key={i} className="px-1 py-0 bg-zinc-700/50 rounded" style={{fontSize:'9px'}}>{cat}</span>
                              ))}
                              {(c.categories || []).length > 3 && (
                                <span className="text-gray-500" style={{fontSize:'9px'}}>+{c.categories.length - 3}</span>
                              )}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {!showCoinPicker && (
              <div className="flex-1 flex items-center justify-center text-gray-600">
                Klicke "+ Coin" um Coins hinzuzufügen
              </div>
            )}
          </>) : (
            <div className="flex-1 flex items-center justify-center text-gray-500">Gruppe auswählen</div>
          )}
        </div>
      </div>
    </div>
  )
}
