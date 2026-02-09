import { useState, useEffect } from 'react'
import { Search, TrendingUp, TrendingDown } from 'lucide-react'
import { useSearchStore } from '../stores/searchStore'
import { useConfigStore } from '../stores/configStore'
import api from '../utils/api'

export default function SearchModule() {
  const { searchParams, setSearchParams, search, isSearching } = useSearchStore()
  const { getKlineMetricsDurations } = useConfigStore()
  
  const [groups, setGroups] = useState([])
  const durations = getKlineMetricsDurations()

  useEffect(() => {
    loadGroups()
  }, [])

  const loadGroups = async () => {
    try {
      const response = await api.get('/api/v1/groups')
      setGroups(response.data || [])
    } catch (error) {
      console.error('Failed to load groups:', error)
    }
  }

  const handleSearch = async (e) => {
    e.preventDefault()
    try {
      await search()
    } catch (error) {
      // Error handling in store
    }
  }

  return (
    <form onSubmit={handleSearch} className="space-y-4">
      {/* Richtung */}
      <div>
        <label className="block text-gray-400 text-xs mb-1">Richtung</label>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => setSearchParams({ direction: 'up' })}
            className={`flex-1 flex items-center justify-center gap-1 py-2 rounded ${
              searchParams.direction === 'up' 
                ? 'bg-green-600 text-white' 
                : 'bg-gray-700 text-gray-300'
            }`}
          >
            <TrendingUp size={16} /> Long
          </button>
          <button
            type="button"
            onClick={() => setSearchParams({ direction: 'down' })}
            className={`flex-1 flex items-center justify-center gap-1 py-2 rounded ${
              searchParams.direction === 'down' 
                ? 'bg-red-600 text-white' 
                : 'bg-gray-700 text-gray-300'
            }`}
          >
            <TrendingDown size={16} /> Short
          </button>
        </div>
      </div>

      {/* Ziel-Prozent */}
      <div>
        <label className="block text-gray-400 text-xs mb-1">Ziel %</label>
        <input
          type="number"
          value={searchParams.targetPercent}
          onChange={(e) => setSearchParams({ targetPercent: parseFloat(e.target.value) || 0 })}
          className="input w-full"
          step="0.5"
          min="0"
        />
      </div>

      {/* Duration */}
      <div>
        <label className="block text-gray-400 text-xs mb-1">Zeitfenster (Min)</label>
        <select
          value={searchParams.durationMinutes}
          onChange={(e) => setSearchParams({ durationMinutes: parseInt(e.target.value) })}
          className="input w-full"
        >
          {durations.map(d => (
            <option key={d} value={d}>{d} Min</option>
          ))}
        </select>
      </div>

      {/* Datum Von */}
      <div>
        <label className="block text-gray-400 text-xs mb-1">Von</label>
        <input
          type="date"
          value={searchParams.dateFrom || ''}
          onChange={(e) => setSearchParams({ dateFrom: e.target.value || null })}
          className="input w-full"
        />
      </div>

      {/* Datum Bis */}
      <div>
        <label className="block text-gray-400 text-xs mb-1">Bis</label>
        <input
          type="date"
          value={searchParams.dateTo || ''}
          onChange={(e) => setSearchParams({ dateTo: e.target.value || null })}
          className="input w-full"
        />
      </div>

      {/* Coingruppe */}
      <div>
        <label className="block text-gray-400 text-xs mb-1">Coingruppe</label>
        <select
          value={searchParams.groupId || ''}
          onChange={(e) => setSearchParams({ groupId: e.target.value ? parseInt(e.target.value) : null })}
          className="input w-full"
        >
          <option value="">Alle Coins</option>
          {groups.map(g => (
            <option key={g.id} value={g.id}>{g.name}</option>
          ))}
        </select>
      </div>

      {/* Submit */}
      <button
        type="submit"
        disabled={isSearching}
        className="btn btn-primary w-full flex items-center justify-center gap-2"
      >
        <Search size={16} />
        {isSearching ? 'Suche läuft...' : 'Suchen'}
      </button>
    </form>
  )
}
