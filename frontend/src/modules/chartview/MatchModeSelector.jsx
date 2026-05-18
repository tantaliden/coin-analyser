import { MATCH_MODES } from '../../config/chartSettings'

/**
 * Kompakter Selector fuer Gegensuche-Match-Modus.
 * Bei 'atleast' erscheint zusaetzlich ein Zahl-Eingabefeld.
 */
export default function MatchModeSelector({
  mode, setMode, threshold, setThreshold, totalCriteria,
}) {
  const current = MATCH_MODES.find(m => m.id === mode)

  return (
    <div className="space-y-1">
      <label className="text-[10px] text-gray-500 block">Match-Modus</label>
      <select value={mode} onChange={e => setMode(e.target.value)}
        className="input text-xs py-1 w-full">
        {MATCH_MODES.map(m => <option key={m.id} value={m.id}>{m.label}</option>)}
      </select>
      <div className="text-[9px] text-gray-500 italic">{current?.description}</div>

      {mode === 'atleast' && (
        <div className="flex items-center gap-2 pt-1">
          <span className="text-[10px] text-gray-400">Mindestens:</span>
          <input type="number" min={1} max={Math.max(1, totalCriteria)}
            value={threshold}
            onChange={e => setThreshold(parseInt(e.target.value))}
            className="input text-xs py-0.5 w-14" />
          <span className="text-[10px] text-gray-500">von {totalCriteria}</span>
        </div>
      )}
    </div>
  )
}
