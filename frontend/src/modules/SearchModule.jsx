import { useState, useEffect } from 'react'
import { Search, TrendingUp, TrendingDown } from 'lucide-react'
import { useSearchStore } from '../stores/searchStore'
import { useConfigStore } from '../stores/configStore'
import api from '../utils/api'

export default function SearchModule() {
  const { searchParams, setSearchParams, search, isSearching, searchError } = useSearchStore()
  const { getKlineMetricsDurations } = useConfigStore()
  const [groups, setGroups] = useState([])

  const durations = getKlineMetricsDurations()

  useEffect(() => {
    api.get('/api/v1/groups').then(r => setGroups(r.data.groups || [])).catch(() => {})
  }, [])

  const handleSearch = async (e) => {
    e.preventDefault()
    try { await search() } catch {}
  }

  const toggleGroup = (groupId) => {
    const current = searchParams.groupIds || []
    if (current.includes(groupId)) {
      setSearchParams({ groupIds: current.filter(id => id !== groupId) })
    } else {
      setSearchParams({ groupIds: [...current, groupId] })
    }
  }

  return (
    <form onSubmit={handleSearch} className="space-y-3">
      {/* Richtung */}
      <div>
        <label className="block text-gray-400 text-xs mb-1">Richtung</label>
        <div className="flex gap-2">
          <button type="button" onClick={() => setSearchParams({ direction: 'up' })}
            className={`flex-1 flex items-center justify-center gap-1 py-2 rounded text-sm ${searchParams.direction === 'up' ? 'bg-green-600 text-white' : 'bg-gray-700 text-gray-300'}`}>
            <TrendingUp size={14} /> Long
          </button>
          <button type="button" onClick={() => setSearchParams({ direction: 'down' })}
            className={`flex-1 flex items-center justify-center gap-1 py-2 rounded text-sm ${searchParams.direction === 'down' ? 'bg-red-600 text-white' : 'bg-gray-700 text-gray-300'}`}>
            <TrendingDown size={14} /> Short
          </button>
        </div>
      </div>

      {/* Ziel-Prozent */}
      <div className="flex gap-2">
        <div className="flex-1">
          <label className="block text-gray-400 text-xs mb-1">Min %</label>
          <input type="number" value={searchParams.targetPercent}
            onChange={e => setSearchParams({ targetPercent: parseFloat(e.target.value) || 0 })}
            className="input w-full text-sm" step="0.5" min="0.1" max="100" />
        </div>
        <div className="flex-1">
          <label className="block text-gray-400 text-xs mb-1">Max %</label>
          <input type="number" value={searchParams.maxPercent || 100}
            onChange={e => setSearchParams({ maxPercent: parseFloat(e.target.value) || 100 })}
            className="input w-full text-sm" step="0.5" min="0.1" max="1000" />
        </div>
      </div>

      {/* Duration */}
      <div>
        <label className="block text-gray-400 text-xs mb-1">Zeitfenster</label>
        <select value={searchParams.durationMinutes}
          onChange={e => setSearchParams({ durationMinutes: parseInt(e.target.value) })}
          className="input w-full text-sm">
          {durations.map(d => <option key={d} value={d}>{d} Min</option>)}
        </select>
      </div>

      {/* Datum */}
      <div className="flex gap-2">
        <div className="flex-1">
          <label className="block text-gray-400 text-xs mb-1">Von *</label>
          <input type="date" value={searchParams.startDate}
            onChange={e => setSearchParams({ startDate: e.target.value })}
            className="input w-full text-sm" required />
        </div>
        <div className="flex-1">
          <label className="block text-gray-400 text-xs mb-1">Bis *</label>
          <input type="date" value={searchParams.endDate}
            onChange={e => setSearchParams({ endDate: e.target.value })}
            className="input w-full text-sm" required />
        </div>
      </div>

      {/* Coin-Gruppen */}
      {groups.length > 0 && (
        <div>
          <label className="block text-gray-400 text-xs mb-1">Coin-Gruppen</label>
          <div className="flex flex-wrap gap-1">
            {groups.map(g => (
              <button key={g.id} type="button" onClick={() => toggleGroup(g.id)}
                className={`px-2 py-1 rounded text-xs flex items-center gap-1 ${(searchParams.groupIds || []).includes(g.id) ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'}`}>
                <div className="w-2 h-2 rounded-full" style={{ background: g.color }} />
                {g.name}
              </button>
            ))}
          </div>
        </div>
      )}

      {searchError && <div className="text-red-400 text-sm bg-red-900/30 p-2 rounded">{searchError}</div>}

      <button type="submit"
        disabled={isSearching || !searchParams.startDate || !searchParams.endDate}
        className="btn btn-primary w-full flex items-center justify-center gap-2 disabled:opacity-50 text-sm">
        <Search size={14} />
        {isSearching ? 'Suche läuft...' : 'Suchen'}
      </button>
    </form>
  )
}
