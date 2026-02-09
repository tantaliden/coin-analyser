import { useState, useMemo } from 'react'
import { ChevronUp, ChevronDown, Check, Square, CheckSquare } from 'lucide-react'
import { useSearchStore } from '../stores/searchStore'
import { formatPercent, formatDateTime, formatVolume } from '../utils/format'

export default function SearchResultsModule() {
  const { results, selectedEvents, toggleEvent, selectEvents } = useSearchStore()
  const [sortField, setSortField] = useState('change_percent')
  const [sortDir, setSortDir] = useState('desc')

  const sortedResults = useMemo(() => {
    return [...results].sort((a, b) => {
      const aVal = a[sortField] ?? 0
      const bVal = b[sortField] ?? 0
      return sortDir === 'asc' ? aVal - bVal : bVal - aVal
    })
  }, [results, sortField, sortDir])

  const handleSort = (field) => {
    if (sortField === field) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortDir('desc')
    }
  }

  const allSelected = results.length > 0 && results.every(r => 
    selectedEvents.some(e => e.id === r.id)
  )

  const toggleAll = () => {
    if (allSelected) {
      selectEvents([])
    } else {
      selectEvents(results.slice(0, 32))
    }
  }

  const isSelected = (event) => selectedEvents.some(e => e.id === event.id)

  const SortIcon = ({ field }) => {
    if (sortField !== field) return null
    return sortDir === 'asc' ? <ChevronUp size={12} /> : <ChevronDown size={12} />
  }

  if (results.length === 0) {
    return (
      <div className="text-gray-400 text-center py-8">
        Keine Ergebnisse. Starte eine Suche.
      </div>
    )
  }

  return (
    <div className="overflow-auto h-full">
      <table className="w-full text-sm">
        <thead className="bg-gray-800 sticky top-0">
          <tr>
            <th className="p-2 text-left">
              <button onClick={toggleAll} className="hover:text-blue-400">
                {allSelected ? <CheckSquare size={14} /> : <Square size={14} />}
              </button>
            </th>
            <th 
              className="p-2 text-left cursor-pointer hover:text-blue-400"
              onClick={() => handleSort('symbol')}
            >
              Symbol <SortIcon field="symbol" />
            </th>
            <th 
              className="p-2 text-right cursor-pointer hover:text-blue-400"
              onClick={() => handleSort('change_percent')}
            >
              % <SortIcon field="change_percent" />
            </th>
            <th 
              className="p-2 text-right cursor-pointer hover:text-blue-400"
              onClick={() => handleSort('duration_minutes')}
            >
              Dauer <SortIcon field="duration_minutes" />
            </th>
            <th 
              className="p-2 text-right cursor-pointer hover:text-blue-400"
              onClick={() => handleSort('volume')}
            >
              Vol <SortIcon field="volume" />
            </th>
            <th className="p-2 text-left">Start</th>
          </tr>
        </thead>
        <tbody>
          {sortedResults.map((event) => (
            <tr 
              key={event.id}
              onClick={() => toggleEvent(event)}
              className={`border-b border-gray-700 cursor-pointer hover:bg-gray-700 ${
                isSelected(event) ? 'bg-blue-900/30' : ''
              }`}
            >
              <td className="p-2">
                {isSelected(event) 
                  ? <Check size={14} className="text-blue-400" /> 
                  : <Square size={14} className="text-gray-500" />
                }
              </td>
              <td className="p-2 font-mono">{event.symbol}</td>
              <td className={`p-2 text-right font-mono ${
                event.change_percent >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {formatPercent(event.change_percent)}
              </td>
              <td className="p-2 text-right">{event.duration_minutes}m</td>
              <td className="p-2 text-right font-mono">{formatVolume(event.volume)}</td>
              <td className="p-2 text-gray-400 text-xs">{event.event_start}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="p-2 text-xs text-gray-500 border-t border-gray-700">
        {results.length} Ergebnisse | {selectedEvents.length} ausgewählt (max 32)
      </div>
    </div>
  )
}
