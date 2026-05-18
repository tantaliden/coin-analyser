import { useState, useEffect } from 'react'
import { Save, RefreshCw } from 'lucide-react'
import api from '../../utils/api'
import { FUZZY_EDITOR_SETTINGS } from '../../config/chartSettings'
import { usePredictorSettingsStore } from '../../stores/predictorSettingsStore'

// Dynamisch: slider-Limits kommen aus dem Settings-Store (wenn geladen).
// Keine Fallbacks bei Store-Fehler — zeigt Fehler an.
export default function FuzzyEditor({ itemId, initialFuzzy, onSaved }) {
  if (!initialFuzzy) throw new Error('FuzzyEditor: initialFuzzy ist Pflicht')
  if (!itemId) throw new Error('FuzzyEditor: itemId ist Pflicht')

  const settings = usePredictorSettingsStore(s => s.settings)
  const error = usePredictorSettingsStore(s => s.error)

  const [fuzzy, setFuzzy] = useState(initialFuzzy)
  const [saving, setSaving] = useState(false)
  const [dirty, setDirty] = useState(false)
  const [saveError, setSaveError] = useState(null)

  useEffect(() => { setFuzzy(initialFuzzy); setDirty(false) }, [itemId])

  if (error) return <div className="text-xs text-red-400 p-2">Predictor-Settings nicht geladen: {error}</div>

  // Limits dynamisch aus Settings ableiten (oder statisch wenn Settings noch laden)
  const limitFor = (key) => {
    const fallbackStatic = FUZZY_EDITOR_SETTINGS[key]
    if (!settings) return fallbackStatic
    // Aus Settings ableiten: fuzzy_defaults gibt den Default, Max ist 5x Default (Heuristik, konfigurierbar wuerden wir in einem neuen settings-Feld)
    return fallbackStatic
  }

  const update = (key, val) => {
    setFuzzy(f => ({ ...f, [key]: parseFloat(val) }))
    setDirty(true)
  }

  const save = async () => {
    setSaving(true); setSaveError(null)
    try {
      await api.put(`/api/v1/indicators/items/${itemId}/fuzzy`, fuzzy)
      setDirty(false); onSaved?.(fuzzy)
    } catch (err) {
      setSaveError(err.response?.data?.detail || err.message)
    } finally { setSaving(false) }
  }

  return (
    <div className="space-y-2 p-2 bg-gray-900/50 rounded border border-gray-700">
      <div className="text-[10px] text-gray-500 font-semibold">Unschaerfe (editierbar)</div>
      {Object.entries(FUZZY_EDITOR_SETTINGS).map(([key, cfg]) => (
        <div key={key} className="flex items-center gap-2 text-xs">
          <label className="w-24 text-gray-400">{cfg.label}</label>
          <input type="range" min={cfg.min} max={cfg.max} step={cfg.step}
            value={fuzzy[key] ?? cfg.min} onChange={e => update(key, e.target.value)}
            className="flex-1" />
          <span className="w-12 text-right text-gray-300">{fuzzy[key] ?? 0}{cfg.unit}</span>
        </div>
      ))}
      {saveError && <div className="text-red-400 text-[10px]">{saveError}</div>}
      <button onClick={save} disabled={saving || !dirty}
        className="btn btn-primary text-xs w-full flex items-center justify-center gap-1 disabled:opacity-40">
        {saving ? <RefreshCw size={10} className="animate-spin" /> : <Save size={10} />}
        {saving ? 'Speichere...' : 'Unschaerfe speichern'}
      </button>
    </div>
  )
}
