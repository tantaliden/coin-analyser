import { TrendingUp, TrendingDown, ArrowLeftRight } from 'lucide-react'

const DIRECTIONS = [
  { value: 'up', label: 'Long', icon: TrendingUp, activeClass: 'bg-green-600 text-white' },
  { value: 'both', label: 'Both', icon: ArrowLeftRight, activeClass: 'bg-blue-600 text-white' },
  { value: 'down', label: 'Short', icon: TrendingDown, activeClass: 'bg-red-600 text-white' },
]

export default function DirectionFilter({ direction, onChange }) {
  return (
    <div>
      <label className="block text-gray-400 text-xs mb-1">Richtung</label>
      <div className="flex gap-2">
        {DIRECTIONS.map(d => {
          const Icon = d.icon
          return (
            <button
              key={d.value}
              type="button"
              onClick={() => onChange(d.value)}
              className={`flex-1 flex items-center justify-center gap-1 py-2 rounded text-sm ${
                direction === d.value ? d.activeClass : 'bg-gray-700 text-gray-300'
              }`}
            >
              <Icon size={14} /> {d.label}
            </button>
          )
        })}
      </div>
    </div>
  )
}
