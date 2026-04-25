import { useEffect, useState } from 'react'
import { Plus, X } from 'lucide-react'
import api from '../../utils/api'

// Panel zum manuellen Hinzufuegen von Kriterien (nicht gezeichnet, sondern Form-basiert).
// Sinnvoll fuer nicht-preisbasierte Felder (funding, OI, spread, BBO etc).
export default function ManualCriterionPanel({ onAdd, onClose }) {
  const [fields, setFields] = useState(null)
  const [groups, setGroups] = useState(null)
  const [error, setError] = useState(null)

  const [field, setField] = useState('')
  const [kind, setKind] = useState('value')       // 'value' | 'range' | 'slope'
  const [operator, setOperator] = useState('>')
  const [value, setValue] = useState('')
  const [value2, setValue2] = useState('')
  const [timeFrom, setTimeFrom] = useState(-30)   // Minuten vor Event
  const [timeTo, setTimeTo] = useState(0)

  useEffect(() => {
    api.get('/api/v1/indicators/fields').then(res => {
      setFields(res.data.fields)
      setGroups(res.data.groups)
      setField(res.data.fields[0]?.name || '')
    }).catch(err => setError(err.response?.data?.detail || err.message))
  }, [])

  const submit = () => {
    if (!field) { setError('Feld waehlen'); return }
    const v = parseFloat(value)
    if (isNaN(v)) { setError('value muss Zahl sein'); return }
    const crit = {
      kind,
      field,
      value: v,
      value2: kind === 'range' ? parseFloat(value2) : null,
      time_offset_from: parseInt(timeFrom),
      time_offset_to: parseInt(timeTo),
      fuzzy: {
        valueTolerance: 0.1, timeTolerance: 2.0,
        slopeTolerance: 0.1, ratioTolerance: 0.1, useRange: false,
      },
    }
    // Operator fuer value: '>', '<', '='
    // Die Backend-Logik mappt kind -> operator; hier geben wir operator durch den kind-Parameter
    // Fuer value mit operator '>' setzen wir kind='slope' mit positivem value (>value)? Nein
    // Einfacher: Backend akzeptiert value-kind mit condition_operator explizit. Wir geben operator mit.
    crit.condition_operator = operator
    onAdd(crit)
    setError(null)
    // Werte fuer nextes Hinzufuegen stehen lassen, nur value clearen
    setValue(''); setValue2('')
  }

  if (error && !fields) return <div className="p-3 text-xs text-amber-400">{error}</div>
  if (!fields) return <div className="p-3 text-xs text-gray-500">Laedt Feldliste...</div>

  return (
    <div className="w-96 border-r border-gray-700 bg-gray-800/80 flex flex-col overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-700">
        <span className="text-xs font-semibold text-gray-300">Manuelles Kriterium</span>
        <button onClick={onClose} className="text-gray-500 hover:text-white"><X size={14} /></button>
      </div>

      <div className="flex-1 overflow-auto p-3 space-y-2 text-xs">
        {error && <div className="text-red-400 bg-red-900/20 p-2 rounded">{error}</div>}

        <label className="block">
          <span className="text-gray-500 block mb-0.5">Feld</span>
          <select value={field} onChange={e => setField(e.target.value)} className="input text-xs py-1 w-full">
            {groups.map(g => (
              <optgroup key={g.group} label={g.group}>
                {g.fields.map(fn => {
                  const f = fields.find(x => x.name === fn)
                  return f ? <option key={fn} value={fn}>{f.label}</option> : null
                })}
              </optgroup>
            ))}
          </select>
        </label>

        <label className="block">
          <span className="text-gray-500 block mb-0.5">Bedingungsart</span>
          <select value={kind} onChange={e => setKind(e.target.value)} className="input text-xs py-1 w-full">
            <option value="value">Einzelwert (value)</option>
            <option value="range">Bereich (value..value2)</option>
            <option value="slope">Steigung (slope)</option>
            <option value="ratio">Verhaeltnis (ratio)</option>
          </select>
        </label>

        {kind === 'value' && (
          <label className="block">
            <span className="text-gray-500 block mb-0.5">Operator</span>
            <select value={operator} onChange={e => setOperator(e.target.value)} className="input text-xs py-1 w-full">
              <option value=">">&gt;</option>
              <option value="<">&lt;</option>
              <option value=">=">&gt;=</option>
              <option value="<=">&lt;=</option>
              <option value="=">=</option>
              <option value="!=">!=</option>
            </select>
          </label>
        )}

        <label className="block">
          <span className="text-gray-500 block mb-0.5">{kind === 'range' ? 'Minimum' : 'Wert'}</span>
          <input type="number" step="any" value={value} onChange={e => setValue(e.target.value)}
                 className="input text-xs py-1 w-full" placeholder="z.B. 0.0002 fuer funding" />
        </label>

        {kind === 'range' && (
          <label className="block">
            <span className="text-gray-500 block mb-0.5">Maximum</span>
            <input type="number" step="any" value={value2} onChange={e => setValue2(e.target.value)}
                   className="input text-xs py-1 w-full" />
          </label>
        )}

        <div className="grid grid-cols-2 gap-2">
          <label className="block">
            <span className="text-gray-500 block mb-0.5">Zeit von (min v. Event)</span>
            <input type="number" step="1" value={timeFrom} onChange={e => setTimeFrom(e.target.value)}
                   className="input text-xs py-1 w-full" />
          </label>
          <label className="block">
            <span className="text-gray-500 block mb-0.5">Zeit bis (min v. Event)</span>
            <input type="number" step="1" value={timeTo} onChange={e => setTimeTo(e.target.value)}
                   className="input text-xs py-1 w-full" />
          </label>
        </div>

        <button onClick={submit}
                className="btn btn-primary w-full text-xs flex items-center justify-center gap-1">
          <Plus size={12} /> Kriterium hinzufuegen
        </button>
      </div>
    </div>
  )
}
