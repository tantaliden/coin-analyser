import { useState } from 'react'
import { ToggleLeft, ToggleRight, Trash2, ChevronDown, ChevronRight, Sliders } from 'lucide-react'
import FuzzyEditor from './FuzzyEditor'

export default function IndicatorItemRow({ item, setId, onToggle, onDelete, onFuzzyUpdated }) {
  const [expanded, setExpanded] = useState(false)

  const opLabel = { '>': '>', '<': '<', '>=': '≥', '<=': '≤', 'between': '↔' }[item.condition_operator] || item.condition_operator
  const typeLabel = item.indicator_type === 'candle_pattern' ? 'Candle-Pattern' : item.indicator_type
  const valueStr = item.condition_operator === 'between'
    ? `${item.condition_value} – ${item.condition_value2}`
    : `${opLabel} ${item.condition_value ?? ''}`

  const fuzzy = item.fuzzy_config || {}

  return (
    <div className={`mb-1 rounded ${item.is_active ? 'bg-gray-800' : 'bg-gray-800/40 opacity-60'}`}>
      <div className="flex items-center gap-2 p-2 text-xs">
        <button onClick={() => setExpanded(x => !x)} className="p-0.5 hover:bg-gray-700 rounded text-gray-400">
          {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        </button>
        <div className="w-2 h-4 rounded" style={{ background: item.color || '#3b82f6' }} />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            {item.is_initial_point && <span className="text-[9px] bg-yellow-900/50 text-yellow-300 px-1 rounded">INIT</span>}
            <span className="font-semibold">{typeLabel}</span>
            <span className="text-gray-400">{valueStr}{item.indicator_type !== 'candle_pattern' ? '%' : ''}</span>
            <span className="text-gray-500">{item.aggregator}</span>
          </div>
          <div className="text-gray-500 mt-0.5">
            {item.time_start_minutes}min – {item.time_end_minutes}min
            {item.pattern_data && <span className="ml-1 text-yellow-400">🕯</span>}
            <span className="ml-2 text-[10px]">
              <Sliders size={10} className="inline" /> v±{fuzzy.valueTolerance ?? 0} t±{fuzzy.timeTolerance ?? 0}
            </span>
          </div>
        </div>
        <button onClick={onToggle} className="p-1 hover:bg-gray-700 rounded" title={item.is_active ? 'Deaktivieren' : 'Aktivieren'}>
          {item.is_active ? <ToggleRight size={16} className="text-green-400" /> : <ToggleLeft size={16} className="text-gray-500" />}
        </button>
        <button onClick={onDelete} className="p-1 hover:bg-red-600/30 rounded"><Trash2 size={12} /></button>
      </div>

      {expanded && (
        <div className="px-2 pb-2">
          <FuzzyEditor
            itemId={item.item_id}
            initialFuzzy={fuzzy}
            onSaved={(f) => onFuzzyUpdated?.(item.item_id, f)}
          />
        </div>
      )}
    </div>
  )
}
