import { ToggleLeft, ToggleRight, Trash2 } from 'lucide-react'

export default function IndicatorItemRow({ item, setId, onToggle, onDelete }) {
  const opLabel = { '>': '>', '<': '<', '>=': '≥', '<=': '≤', 'between': '↔' }[item.condition_operator] || item.condition_operator

  const typeLabel = item.indicator_type === 'candle_pattern' ? 'Candle-Pattern' : item.indicator_type

  const valueStr = item.condition_operator === 'between'
    ? `${item.condition_value} – ${item.condition_value2}`
    : `${opLabel} ${item.condition_value ?? ''}`

  return (
    <div className={`flex items-center gap-2 p-2 mb-1 rounded text-xs ${item.is_active ? 'bg-gray-800' : 'bg-gray-800/40 opacity-60'}`}>
      <div className="w-2 h-full rounded" style={{ background: item.color || '#3b82f6' }} />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-semibold">{typeLabel}</span>
          <span className="text-gray-400">{valueStr}%</span>
          <span className="text-gray-500">{item.aggregator}</span>
        </div>
        <div className="text-gray-500 mt-0.5">
          {item.time_start_minutes}min – {item.time_end_minutes}min
          {item.pattern_data && <span className="ml-1 text-yellow-400">🕯</span>}
        </div>
      </div>
      <button onClick={onToggle} className="p-1 hover:bg-gray-700 rounded" title={item.is_active ? 'Deaktivieren' : 'Aktivieren'}>
        {item.is_active ? <ToggleRight size={16} className="text-green-400" /> : <ToggleLeft size={16} className="text-gray-500" />}
      </button>
      <button onClick={onDelete} className="p-1 hover:bg-red-600/30 rounded"><Trash2 size={12} /></button>
    </div>
  )
}
