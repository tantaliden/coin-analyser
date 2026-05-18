import { useState, useEffect } from 'react'
import { ChevronDown, Clock } from 'lucide-react'

const WEEKDAY_LABELS = ['Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'So']

export default function AdvancedFilters({ weekdays, hourStart, hourEnd, onWeekdaysChange, onHoursChange }) {
  const [showAdvanced, setShowAdvanced] = useState(false)
  const hourActive = hourStart >= 0 && hourEnd >= 0
  const advancedActive = weekdays.length > 0 || hourActive

  useEffect(() => {
    if (advancedActive) setShowAdvanced(true)
  }, [])

  const toggleWeekday = (day) => {
    if (weekdays.includes(day)) {
      onWeekdaysChange(weekdays.filter(d => d !== day))
    } else {
      onWeekdaysChange([...weekdays, day].sort())
    }
  }

  const setPreset = (preset) => {
    if (preset === 'all') onWeekdaysChange([])
    else if (preset === 'workdays') onWeekdaysChange([0, 1, 2, 3, 4])
    else if (preset === 'weekend') onWeekdaysChange([5, 6])
  }

  const toggleHours = () => {
    if (hourActive) onHoursChange(-1, -1)
    else onHoursChange(8, 16)
  }

  return (
    <>
      <button
        type="button"
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="w-full flex items-center justify-between px-2 py-1.5 bg-gray-800 hover:bg-gray-700 rounded text-xs text-gray-400"
      >
        <span className="flex items-center gap-1">
          <Clock size={12} /> Erweitert
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
                {[
                  { key: 'all', label: 'Alle', check: !weekdays.length },
                  { key: 'workdays', label: 'Mo-Fr', check: JSON.stringify(weekdays) === '[0,1,2,3,4]' },
                  { key: 'weekend', label: 'Sa-So', check: JSON.stringify(weekdays) === '[5,6]' },
                ].map(p => (
                  <button key={p.key} type="button" onClick={() => setPreset(p.key)}
                    className={`px-1.5 py-0.5 rounded text-xs ${p.check ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'}`}>
                    {p.label}
                  </button>
                ))}
              </div>
            </div>
            <div className="flex gap-1">
              {WEEKDAY_LABELS.map((label, idx) => (
                <button key={idx} type="button" onClick={() => toggleWeekday(idx)}
                  className={`flex-1 py-1.5 rounded text-xs font-medium ${
                    !weekdays.length || weekdays.includes(idx)
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
              <button type="button" onClick={toggleHours}
                className={`px-1.5 py-0.5 rounded text-xs ${hourActive ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'}`}>
                {hourActive ? 'Aktiv' : 'Aus'}
              </button>
            </div>
            {hourActive && (
              <div className="flex items-center gap-2">
                <select value={hourStart} onChange={e => onHoursChange(parseInt(e.target.value), hourEnd)}
                  className="input flex-1 text-sm">
                  {Array.from({ length: 24 }, (_, i) => (
                    <option key={i} value={i}>{String(i).padStart(2, '0')}:00</option>
                  ))}
                </select>
                <span className="text-gray-500">-</span>
                <select value={hourEnd} onChange={e => onHoursChange(hourStart, parseInt(e.target.value))}
                  className="input flex-1 text-sm">
                  {Array.from({ length: 24 }, (_, i) => (
                    <option key={i} value={i}>{String(i).padStart(2, '0')}:59</option>
                  ))}
                </select>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  )
}
