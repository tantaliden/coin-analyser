import { useState, useRef, useEffect } from 'react'
import { ChevronLeft, ChevronRight, Calendar } from 'lucide-react'

const MONTHS = ['Jan','Feb','Mär','Apr','Mai','Jun','Jul','Aug','Sep','Okt','Nov','Dez']
const DAYS = ['Mo','Di','Mi','Do','Fr','Sa','So']

function daysInMonth(y, m) { return new Date(y, m + 1, 0).getDate() }
function firstDow(y, m) { return (new Date(y, m, 1).getDay() + 6) % 7 } // Mo=0
function fmt(d) { return d ? `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}` : '' }
function fmtShort(d) { return d ? `${String(d.getDate()).padStart(2,'0')}.${String(d.getMonth()+1).padStart(2,'0')}.${d.getFullYear()}` : '' }
function sameDay(a, b) { return a && b && a.getFullYear()===b.getFullYear() && a.getMonth()===b.getMonth() && a.getDate()===b.getDate() }
function inRange(d, s, e) { return s && e && d >= s && d <= e }

export default function DateRangePicker({ startDate, endDate, onChange }) {
  const [open, setOpen] = useState(false)
  const [pickState, setPickState] = useState(0) // 0=pick start, 1=pick end
  const [viewYear, setViewYear] = useState(() => {
    if (endDate) return new Date(endDate).getFullYear()
    return new Date().getFullYear()
  })
  const [viewMonth, setViewMonth] = useState(() => {
    if (endDate) return new Date(endDate).getMonth()
    return new Date().getMonth()
  })
  const [tempStart, setTempStart] = useState(null)
  const [hoverDate, setHoverDate] = useState(null)
  const ref = useRef(null)

  // Parse string dates
  const startD = startDate ? new Date(startDate + 'T00:00:00') : null
  const endD = endDate ? new Date(endDate + 'T00:00:00') : null

  useEffect(() => {
    const handler = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false) }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const openPicker = () => {
    setOpen(true)
    setPickState(0)
    setTempStart(null)
    setHoverDate(null)
    // Navigate to current end date or now
    if (endD) { setViewYear(endD.getFullYear()); setViewMonth(endD.getMonth()) }
    else { const n = new Date(); setViewYear(n.getFullYear()); setViewMonth(n.getMonth()) }
  }

  const prevMonth = () => {
    if (viewMonth === 0) { setViewMonth(11); setViewYear(viewYear - 1) }
    else setViewMonth(viewMonth - 1)
  }
  const nextMonth = () => {
    if (viewMonth === 11) { setViewMonth(0); setViewYear(viewYear + 1) }
    else setViewMonth(viewMonth + 1)
  }

  const handleDayClick = (day) => {
    const clicked = new Date(viewYear, viewMonth, day)
    if (pickState === 0) {
      setTempStart(clicked)
      setPickState(1)
    } else {
      let s = tempStart, e = clicked
      if (e < s) { [s, e] = [e, s] }
      onChange(fmt(s), fmt(e))
      setOpen(false)
      setPickState(0)
      setTempStart(null)
    }
  }

  // Quick presets
  const applyPreset = (daysBack) => {
    const e = new Date()
    const s = new Date(); s.setDate(s.getDate() - daysBack)
    onChange(fmt(s), fmt(e))
    setOpen(false)
  }

  const days = daysInMonth(viewYear, viewMonth)
  const offset = firstDow(viewYear, viewMonth)
  const cells = []
  for (let i = 0; i < offset; i++) cells.push(null)
  for (let d = 1; d <= days; d++) cells.push(d)

  // Determine visual range for highlighting
  const rangeStart = pickState === 1 ? tempStart : startD
  const rangeEnd = pickState === 1 ? (hoverDate || tempStart) : endD

  const displayText = (startD && endD)
    ? `${fmtShort(startD)} – ${fmtShort(endD)}`
    : (startD ? `${fmtShort(startD)} – ...` : 'Zeitraum wählen')

  return (
    <div className="relative" ref={ref}>
      <label className="block text-gray-400 text-xs mb-1">Zeitraum *</label>
      <button type="button" onClick={openPicker}
        className="input w-full text-sm text-left flex items-center gap-2">
        <Calendar size={12} className="text-gray-400 flex-shrink-0" />
        <span className={startD ? 'text-white' : 'text-gray-500'}>{displayText}</span>
      </button>

      {open && (
        <div className="absolute z-50 top-full mt-1 left-0 bg-zinc-900 border border-zinc-700 rounded-lg shadow-xl p-2 w-64">
          {/* Hint */}
          <div className="text-center text-xs mb-1" style={{color: pickState === 0 ? '#9ca3af' : '#60a5fa'}}>
            {pickState === 0 ? 'Startdatum wählen' : 'Enddatum wählen'}
          </div>

          {/* Month Nav */}
          <div className="flex items-center justify-between mb-1">
            <button type="button" onClick={prevMonth} className="p-1 hover:bg-zinc-700 rounded"><ChevronLeft size={14} /></button>
            <span className="text-sm font-medium">{MONTHS[viewMonth]} {viewYear}</span>
            <button type="button" onClick={nextMonth} className="p-1 hover:bg-zinc-700 rounded"><ChevronRight size={14} /></button>
          </div>

          {/* Day Headers */}
          <div className="grid grid-cols-7 gap-0 text-center">
            {DAYS.map(d => <div key={d} className="text-gray-500 py-0.5" style={{fontSize:'10px'}}>{d}</div>)}
          </div>

          {/* Days Grid */}
          <div className="grid grid-cols-7 gap-0">
            {cells.map((day, i) => {
              if (!day) return <div key={`e${i}`} />
              const dt = new Date(viewYear, viewMonth, day)
              const isStart = sameDay(dt, rangeStart)
              const isEnd = sameDay(dt, rangeEnd)
              const isIn = rangeStart && rangeEnd && inRange(dt, 
                rangeStart < rangeEnd ? rangeStart : rangeEnd,
                rangeStart < rangeEnd ? rangeEnd : rangeStart
              )
              const isToday = sameDay(dt, new Date())

              let cls = 'py-1 text-center text-xs rounded cursor-pointer '
              if (isStart || isEnd) cls += 'bg-blue-600 text-white font-medium '
              else if (isIn) cls += 'bg-blue-600/20 text-blue-300 '
              else if (isToday) cls += 'text-yellow-400 hover:bg-zinc-700 '
              else cls += 'text-gray-300 hover:bg-zinc-700 '

              return (
                <div key={day} className={cls}
                  onClick={() => handleDayClick(day)}
                  onMouseEnter={() => pickState === 1 && setHoverDate(dt)}>
                  {day}
                </div>
              )
            })}
          </div>

          {/* Quick Presets */}
          <div className="flex gap-1 mt-2 pt-2 border-t border-zinc-700">
            {[{l:'7T',d:7},{l:'30T',d:30},{l:'90T',d:90},{l:'1J',d:365},{l:'2J',d:730}].map(p => (
              <button key={p.d} type="button" onClick={() => applyPreset(p.d)}
                className="flex-1 text-center py-1 bg-zinc-800 hover:bg-zinc-700 rounded text-xs text-gray-400">
                {p.l}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
