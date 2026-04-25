import { useState } from 'react'
import { Settings2, X, ChevronDown, ChevronUp, ArrowUp, ArrowDown } from 'lucide-react'
import FuzzyPanel from './FuzzyPanel'

/**
 * Zeigt extrahierte Suchkriterien mit Pro-Kriterium Unschaerfe-Override.
 * Default: Kriterium nutzt globalFuzzy. Click auf Settings → eigene Toleranzen.
 */
export default function CriteriaList({ criteria, setCriteria, globalFuzzy }) {
  const [expanded, setExpanded] = useState({})

  if (!criteria.length) {
    return <div className="text-gray-500 text-xs text-center py-3">Keine Kriterien. Zeichne Marker/Linien im Chart.</div>
  }

  const toggleExpand = (idx) => setExpanded(p => ({ ...p, [idx]: !p[idx] }))

  const updateFuzzy = (idx, newFuzzy) => {
    const next = [...criteria]
    next[idx] = { ...next[idx], fuzzy: newFuzzy }
    setCriteria(next)
  }

  const resetFuzzy = (idx) => {
    const next = [...criteria]
    delete next[idx].fuzzy
    setCriteria(next)
  }

  const removeCriterion = (idx) => {
    setCriteria(criteria.filter((_, i) => i !== idx))
  }

  const moveCriterion = (idx, dir) => {
    const ni = idx + dir
    if (ni < 0 || ni >= criteria.length) return
    const next = [...criteria]
    ;[next[idx], next[ni]] = [next[ni], next[idx]]
    setCriteria(next)
  }

  const kindColors = {
    value: 'text-blue-400',
    range: 'text-purple-400',
    slope: 'text-amber-400',
    ratio: 'text-teal-400',
    pattern: 'text-pink-400',
  }

  return (
    <div className="space-y-1">
      {criteria.map((c, idx) => {
        const hasOwnFuzzy = !!c.fuzzy
        const isExpanded = expanded[idx]
        return (
          <div key={idx} className="bg-gray-900/50 rounded border border-gray-700/50">
            <div className="flex items-center gap-1 p-1.5">
              <span className="text-[9px] text-gray-500 w-3">{idx + 1}</span>
              <span className={`text-[10px] uppercase font-semibold w-10 ${kindColors[c.kind] || 'text-gray-400'}`}>
                {c.kind}
              </span>
              <span className="text-xs text-gray-200 flex-1 truncate">{c.meta?.label || c.field}</span>
              <button onClick={() => moveCriterion(idx, -1)} disabled={idx === 0}
                className="p-0.5 text-gray-500 hover:text-gray-300 disabled:opacity-30" title="Nach oben">
                <ArrowUp size={10} />
              </button>
              <button onClick={() => moveCriterion(idx, 1)} disabled={idx === criteria.length - 1}
                className="p-0.5 text-gray-500 hover:text-gray-300 disabled:opacity-30" title="Nach unten">
                <ArrowDown size={10} />
              </button>
              <button onClick={() => toggleExpand(idx)}
                className={`p-0.5 rounded ${hasOwnFuzzy ? 'text-blue-400' : 'text-gray-500 hover:text-gray-300'}`}
                title={hasOwnFuzzy ? 'Individuelle Unschaerfe' : 'Global-Unschaerfe'}>
                <Settings2 size={11} />
              </button>
              <button onClick={() => toggleExpand(idx)} className="text-gray-500 hover:text-gray-300">
                {isExpanded ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
              </button>
              <button onClick={() => removeCriterion(idx)} className="text-gray-500 hover:text-red-400">
                <X size={11} />
              </button>
            </div>

            {isExpanded && (
              <div className="border-t border-gray-700/50 p-2 bg-gray-900/30 space-y-2">
                {/* Detail-Info */}
                <div className="text-[10px] text-gray-500 space-y-0.5">
                  <div>Feld: <span className="text-gray-300">{c.field}</span></div>
                  {c.value != null && <div>Ziel: <span className="text-gray-300">{c.value.toFixed?.(4) || c.value}</span></div>}
                  {c.time_offset != null && <div>Zeitpunkt: <span className="text-gray-300">{c.time_offset}m</span></div>}
                  {c.time_offset2 != null && <div>Ende: <span className="text-gray-300">{c.time_offset2}m</span></div>}
                </div>

                {/* Fuzzy Override Toggle */}
                <div className="flex items-center gap-2">
                  <label className="flex items-center gap-1 cursor-pointer text-[10px]">
                    <input type="checkbox" checked={hasOwnFuzzy}
                      onChange={e => {
                        if (e.target.checked) updateFuzzy(idx, { ...globalFuzzy })
                        else resetFuzzy(idx)
                      }} />
                    <span className="text-gray-400">Individuelle Unschaerfe</span>
                  </label>
                </div>

                {/* Fuzzy Panel */}
                {hasOwnFuzzy && (
                  <div className="pt-1 border-t border-gray-700/50">
                    <FuzzyPanel values={c.fuzzy} onChange={f => updateFuzzy(idx, f)} title="" />
                  </div>
                )}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
