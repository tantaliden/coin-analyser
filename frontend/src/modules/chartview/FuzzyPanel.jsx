import { FUZZY_DEFAULTS } from '../../config/chartSettings'

export default function FuzzyPanel({ values, onChange, title = 'Unschaerfe' }) {
  const ranges = FUZZY_DEFAULTS.ranges
  const v = values || FUZZY_DEFAULTS.global

  const set = (key, val) => onChange({ ...v, [key]: parseFloat(val) || 0 })
  const setBool = (key, val) => onChange({ ...v, [key]: val })
  const setNum = (key, val) => onChange({ ...v, [key]: val === '' ? null : parseFloat(val) })

  return (
    <div className="space-y-2">
      {title && <span className="text-xs font-semibold text-gray-300">{title}</span>}

      {/* Mode Toggle */}
      <div className="flex gap-1">
        <button onClick={() => setBool('useRange', false)}
          className={`flex-1 text-[10px] py-1 rounded ${!v.useRange ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-400'}`}>
          Toleranz
        </button>
        <button onClick={() => setBool('useRange', true)}
          className={`flex-1 text-[10px] py-1 rounded ${v.useRange ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-400'}`}>
          Wertbereich
        </button>
      </div>

      {/* Range Mode: Min/Max absolute */}
      {v.useRange && (
        <div className="flex gap-1">
          <div className="flex-1">
            <label className="text-[10px] text-gray-500 block">Min</label>
            <input type="number" value={v.rangeMin ?? ''} onChange={e => setNum('rangeMin', e.target.value)}
              className="input text-xs py-0.5 w-full" step="any" placeholder="-inf" />
          </div>
          <div className="flex-1">
            <label className="text-[10px] text-gray-500 block">Max</label>
            <input type="number" value={v.rangeMax ?? ''} onChange={e => setNum('rangeMax', e.target.value)}
              className="input text-xs py-0.5 w-full" step="any" placeholder="+inf" />
          </div>
        </div>
      )}

      {/* Tolerance Sliders */}
      {Object.entries(ranges).map(([key, cfg]) => (
        <div key={key} className="flex items-center gap-2">
          <span className="text-[10px] text-gray-400 w-28">{cfg.label}:</span>
          <input type="range" value={v[key] ?? FUZZY_DEFAULTS.global[key]}
            onChange={e => set(key, e.target.value)}
            min={cfg.min} max={cfg.max} step={cfg.step}
            className="flex-1 h-1 accent-blue-500" />
          <span className="text-[10px] text-gray-300 w-10 text-right font-mono">
            {v[key] ?? FUZZY_DEFAULTS.global[key]}{cfg.unit}
          </span>
        </div>
      ))}
    </div>
  )
}
