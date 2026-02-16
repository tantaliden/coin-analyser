import { useState, useEffect } from 'react'
import { Search, TrendingUp, TrendingDown, ArrowLeftRight, ChevronDown, Clock } from 'lucide-react'
import { useSearchStore } from '../stores/searchStore'
import { useConfigStore } from '../stores/configStore'
import DateRangePicker from '../components/DateRangePicker'
import api from '../utils/api'

const WEEKDAY_LABELS = ['Mo','Di','Mi','Do','Fr','Sa','So']

export default function SearchModule() {
  const { searchParams, setSearchParams, search, isSearching, searchError } = useSearchStore()
  const { getKlineMetricsDurations } = useConfigStore()
  const [groups, setGroups] = useState([])
  const [showAdvanced, setShowAdvanced] = useState(false)

  const durations = getKlineMetricsDurations()

  useEffect(() => {
    api.get('/api/v1/groups').then(r => setGroups(r.data.groups || [])).catch(() => {})
  }, [])

  // Auto-open advanced if filters are active
  useEffect(() => {
    if (searchParams.weekdays?.length > 0 || searchParams.hourStart >= 0) setShowAdvanced(true)
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

  const toggleWeekday = (day) => {
    const current = searchParams.weekdays || []
    if (current.includes(day)) {
      setSearchParams({ weekdays: current.filter(d => d !== day) })
    } else {
      setSearchParams({ weekdays: [...current, day].sort() })
    }
  }

  const setWeekdayPreset = (preset) => {
    if (preset === 'all') setSearchParams({ weekdays: [] })
    else if (preset === 'workdays') setSearchParams({ weekdays: [0,1,2,3,4] })
    else if (preset === 'weekend') setSearchParams({ weekdays: [5,6] })
  }

  const hourActive = searchParams.hourStart >= 0 && searchParams.hourEnd >= 0
  const toggleHourFilter = () => {
    if (hourActive) {
      setSearchParams({ hourStart: -1, hourEnd: -1 })
    } else {
      setSearchParams({ hourStart: 8, hourEnd: 16 })
    }
  }

  const advancedActive = (searchParams.weekdays?.length > 0) || hourActive

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
          <button type="button" onClick={() => setSearchParams({ direction: 'both' })}
            className={`flex-1 flex items-center justify-center gap-1 py-2 rounded text-sm ${searchParams.direction === 'both' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'}`}>
            <ArrowLeftRight size={14} /> Both
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
            className="input w-full text-sm" step="any" />
        </div>
        <div className="flex-1">
          <label className="block text-gray-400 text-xs mb-1">Max %</label>
          <input type="number" value={searchParams.maxPercent || ''}
            onChange={e => setSearchParams({ maxPercent: parseFloat(e.target.value) || '' })}
            className="input w-full text-sm" step="any" placeholder="∞" />
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

      {/* Zeitraum */}
      <DateRangePicker
        startDate={searchParams.startDate}
        endDate={searchParams.endDate}
        onChange={(start, end) => setSearchParams({ startDate: start, endDate: end })}
      />

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

      {/* Advanced Toggle */}
      <button type="button" onClick={() => setShowAdvanced(!showAdvanced)}
        className="w-full flex items-center justify-between px-2 py-1.5 bg-gray-800 hover:bg-gray-700 rounded text-xs text-gray-400">
        <span className="flex items-center gap-1">
          <Clock size={12} />
          Erweitert
          {advancedActive && <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />}
        </span>
        <ChevronDown size={12} className={`transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
      </button>

      {showAdvanced && (
        <div className="space-y-3 p-2 bg-gray-800/50 rounded border border-gray-700">
          {/* Wochentage */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-gray-400 text-xs">Wochentage</label>
              <div className="flex gap-1">
                <button type="button" onClick={() => setWeekdayPreset('all')}
                  className={`px-1.5 py-0.5 rounded text-xs ${!searchParams.weekdays?.length ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'}`}>
                  Alle
                </button>
                <button type="button" onClick={() => setWeekdayPreset('workdays')}
                  className={`px-1.5 py-0.5 rounded text-xs ${JSON.stringify(searchParams.weekdays)==='[0,1,2,3,4]' ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'}`}>
                  Mo–Fr
                </button>
                <button type="button" onClick={() => setWeekdayPreset('weekend')}
                  className={`px-1.5 py-0.5 rounded text-xs ${JSON.stringify(searchParams.weekdays)==='[5,6]' ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'}`}>
                  Sa–So
                </button>
              </div>
            </div>
            <div className="flex gap-1">
              {WEEKDAY_LABELS.map((label, idx) => (
                <button key={idx} type="button" onClick={() => toggleWeekday(idx)}
                  className={`flex-1 py-1.5 rounded text-xs font-medium ${
                    !searchParams.weekdays?.length || searchParams.weekdays.includes(idx)
                      ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-500'
                  }`}>
                  {label}
                </button>
              ))}
            </div>
          </div>

          {/* Uhrzeit */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-gray-400 text-xs">Uhrzeit (Berlin)</label>
              <button type="button" onClick={toggleHourFilter}
                className={`px-1.5 py-0.5 rounded text-xs ${hourActive ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'}`}>
                {hourActive ? 'Aktiv' : 'Aus'}
              </button>
            </div>
            {hourActive && (
              <div className="flex items-center gap-2">
                <div className="flex-1">
                  <select value={searchParams.hourStart}
                    onChange={e => setSearchParams({ hourStart: parseInt(e.target.value) })}
                    className="input w-full text-sm">
                    {Array.from({length:24}, (_,i) => (
                      <option key={i} value={i}>{String(i).padStart(2,'0')}:00</option>
                    ))}
                  </select>
                </div>
                <span className="text-gray-500">–</span>
                <div className="flex-1">
                  <select value={searchParams.hourEnd}
                    onChange={e => setSearchParams({ hourEnd: parseInt(e.target.value) })}
                    className="input w-full text-sm">
                    {Array.from({length:24}, (_,i) => (
                      <option key={i} value={i}>{String(i).padStart(2,'0')}:59</option>
                    ))}
                  </select>
                </div>
              </div>
            )}
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
