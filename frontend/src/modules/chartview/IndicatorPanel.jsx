import { useState } from 'react'
import { X, Plus, Trash2, ToggleLeft, ToggleRight } from 'lucide-react'
import { CHART_SETTINGS, DEFAULT_INDICATORS } from '../../config/chartSettings'

export default function IndicatorPanel({ activeIndicators, setActiveIndicators, onClose }) {
  const [selectedType, setSelectedType] = useState('')
  const available = CHART_SETTINGS.indicators.available

  const addIndicator = (type, label, params) => {
    const config = available.find(a => a.id === type)
    if (!config) throw new Error(`indicator type not found: ${type}`)
    if (!label) throw new Error(`label required for indicator ${type}`)
    if (!params) throw new Error(`params required for indicator ${type}`)
    setActiveIndicators(prev => [...prev, {
      id: `${type}_${Date.now()}`,
      type, label, params, config, visible: true,
    }])
  }

  const addFromDropdown = () => {
    if (!selectedType) return
    const config = available.find(a => a.id === selectedType)
    if (!config) throw new Error(`indicator type not found: ${selectedType}`)
    const defaultParams = {}
    config.params.forEach(p => { defaultParams[p.key] = p.default })
    addIndicator(selectedType, config.label, defaultParams)
    setSelectedType('')
  }

  const removeIndicator = (id) => setActiveIndicators(prev => prev.filter(i => i.id !== id))
  const toggleIndicator = (id) => setActiveIndicators(prev => prev.map(i => i.id === id ? { ...i, visible: !i.visible } : i))
  const updateParam = (id, key, value) => setActiveIndicators(prev => prev.map(i =>
    i.id === id ? { ...i, params: { ...i.params, [key]: parseFloat(value) } } : i
  ))

  const isActive = (type, params) => activeIndicators.some(i =>
    i.type === type && JSON.stringify(i.params) === JSON.stringify(params)
  )

  return (
    <div className="w-56 border-r border-gray-700 bg-gray-800/80 flex flex-col overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-700">
        <span className="text-xs font-semibold text-gray-300">Indikatoren</span>
        <button onClick={onClose} className="text-gray-500 hover:text-white"><X size={14} /></button>
      </div>

      {/* Schnellzugriff: Standard-Indikatoren */}
      <div className="p-2 border-b border-gray-700">
        <span className="text-[10px] text-gray-500 block mb-1">Schnellzugriff</span>
        <div className="flex flex-wrap gap-1">
          {DEFAULT_INDICATORS.map((di, idx) => {
            const active = isActive(di.type, di.params)
            return (
              <button key={idx}
                onClick={() => {
                  if (active) {
                    setActiveIndicators(prev => prev.filter(i => !(i.type === di.type && JSON.stringify(i.params) === JSON.stringify(di.params))))
                  } else {
                    addIndicator(di.type, di.label, di.params)
                  }
                }}
                className={`px-1.5 py-0.5 rounded text-[10px] ${active ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-400 hover:bg-gray-600'}`}
              >
                {di.label}
              </button>
            )
          })}
        </div>
      </div>

      {/* Custom hinzufuegen */}
      <div className="flex items-center gap-1 p-2 border-b border-gray-700">
        <select value={selectedType} onChange={e => setSelectedType(e.target.value)} className="input text-xs py-1 flex-1">
          <option value="">Hinzufuegen...</option>
          {available.map(a => <option key={a.id} value={a.id}>{a.label}</option>)}
        </select>
        <button onClick={addFromDropdown} disabled={!selectedType}
          className="p-1 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 rounded">
          <Plus size={12} />
        </button>
      </div>

      {/* Aktive Indikatoren */}
      <div className="flex-1 overflow-auto p-2 space-y-2">
        {activeIndicators.length === 0 && (
          <div className="text-gray-500 text-xs text-center py-4">Keine Indikatoren aktiv</div>
        )}
        {activeIndicators.map((ind, idx) => {
          const color = CHART_SETTINGS.indicators.defaultColors[idx % CHART_SETTINGS.indicators.defaultColors.length]
          return (
            <div key={ind.id} className={`bg-gray-900/50 rounded p-2 ${!ind.visible ? 'opacity-50' : ''}`}>
              <div className="flex items-center gap-2 mb-1">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
                <span className="text-xs font-medium text-gray-200 flex-1">{ind.label}</span>
                <button onClick={() => toggleIndicator(ind.id)} className="text-gray-400 hover:text-blue-400">
                  {ind.visible ? <ToggleRight size={14} className="text-blue-400" /> : <ToggleLeft size={14} />}
                </button>
                <button onClick={() => removeIndicator(ind.id)} className="text-gray-500 hover:text-red-400">
                  <Trash2 size={10} />
                </button>
              </div>
              {ind.config?.params?.map(p => (
                <div key={p.key} className="flex items-center gap-1 mt-1">
                  <span className="text-[10px] text-gray-500 w-12">{p.label}:</span>
                  <input type="number" value={ind.params[p.key]}
                    onChange={e => updateParam(ind.id, p.key, e.target.value)}
                    className="input text-[10px] py-0.5 w-16"
                    min={p.min} max={p.max} step={p.step || 1} />
                </div>
              ))}
            </div>
          )
        })}
      </div>
    </div>
  )
}
