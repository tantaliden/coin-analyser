import { useState, useEffect, useMemo, useRef } from 'react'
import { Search, X, ChevronDown, ChevronUp, RefreshCw, Save } from 'lucide-react'
import { COUNTER_SEARCH_SETTINGS, FUZZY_DEFAULTS, MATCH_MODES, MATCH_MODE_DEFAULT, LIVE_FEEDBACK } from '../../config/chartSettings'
import { useSearchStore } from '../../stores/searchStore'
import FuzzyPanel from './FuzzyPanel'
import CriteriaList from './CriteriaList'
import ManualCriterionPanel from './ManualCriterionPanel'
import SaveSetDialog from './SaveSetDialog'
import MatchModeSelector from './MatchModeSelector'
import { collectCriteriaFromAllCharts, splitInitialAndCriteria } from './utils/drawingsToCriteria'
import api from '../../utils/api'

export default function CounterSearchPanel({
  drawings, chartData, activeChartId, durationMinutes, candleTimeframeMinutes, prehistoryMinutes, direction, primaryContext,
  onClose, onResultsFound,
}) {
  const primaryResults = useSearchStore(s => s.results)

  const [periodDays, setPeriodDays] = useState(COUNTER_SEARCH_SETTINGS.defaultPeriodDays)
  const [globalFuzzy, setGlobalFuzzy] = useState({ ...FUZZY_DEFAULTS.global })
  const [showGlobalFuzzy, setShowGlobalFuzzy] = useState(false)
  const [criteria, setCriteria] = useState([])
  const [showManualCrit, setShowManualCrit] = useState(false)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  const [matchMode, setMatchMode] = useState(MATCH_MODE_DEFAULT)
  const [matchThreshold, setMatchThreshold] = useState(1)
  const [liveEnabled, setLiveEnabled] = useState(LIVE_FEEDBACK.enabled)
  const [showSaveDialog, setShowSaveDialog] = useState(false)
  const debounceRef = useRef(null)

  const extractedCriteria = useMemo(
    () => collectCriteriaFromAllCharts(drawings, chartData, activeChartId),
    [drawings, chartData, activeChartId]
  )

  useEffect(() => {
    setCriteria(prev => {
      return extractedCriteria.map((c, i) => {
        const existing = prev[i]
        if (existing && existing.kind === c.kind && existing.field === c.field) {
          return { ...c, fuzzy: existing.fuzzy }
        }
        return c
      })
    })
  }, [extractedCriteria])

  // Threshold auf atleast passend anpassen
  useEffect(() => {
    if (matchMode === 'atleast') {
      if (matchThreshold > criteria.length) setMatchThreshold(Math.max(1, criteria.length))
      if (matchThreshold < 1 && criteria.length > 0) setMatchThreshold(1)
    }
  }, [criteria.length, matchMode])

  const runSearch = async () => {
    if (criteria.length === 0) {
      setError('Keine Kriterien. Zeichne Marker oder Linien im Chart.')
      return
    }
    setLoading(true); setError(null)
    try {
      const payload = {
        criteria: criteria.map(c => ({
          kind: c.kind, field: c.field, field2: c.field2,
          value: c.value, value2: c.value2,
          time_offset: c.time_offset || 0,
          time_offset2: c.time_offset2,
          pattern_id: c.pattern_id,
          fuzzy: c.fuzzy,
        })),
        global_fuzzy: globalFuzzy,
        period_days: periodDays,
        hl_only: true,
        direction: 'both',
        match_mode: matchMode,
        match_threshold: matchThreshold,
        duration_minutes: durationMinutes || 120,
        candle_timeframe: candleTimeframeMinutes || 1,
      }
      const res = await api.post('/api/v1/search/counter/find', payload)
      setResults(res.data)
      onResultsFound?.(res.data.matches || [])
    } catch (err) {
      setError(err.response?.data?.detail || 'Gegensuche fehlgeschlagen')
    } finally {
      setLoading(false)
    }
  }

  // Live-Feedback: Nach jeder Aenderung an Kriterien/Fuzzy/Mode automatisch suchen
  useEffect(() => {
    if (!liveEnabled) return
    if (criteria.length === 0) return
    if (criteria.length > LIVE_FEEDBACK.maxCriteria) return
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => {
      runSearch()
    }, LIVE_FEEDBACK.debounceMs)
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current) }
  }, [criteria, globalFuzzy, periodDays, matchMode, matchThreshold, liveEnabled])

  // Cross-Reference: wie viele der Treffer sind auch in der Primary-Search?
  const crossRef = useMemo(() => {
    if (!results?.matches || !primaryResults?.length) return null
    const primaryKeys = new Set(primaryResults.map(e => `${e.symbol}|${e.event_start}`))
    let inPrimary = 0
    for (const m of results.matches) {
      const key = `${m.symbol}|${m.event_start}`
      if (primaryKeys.has(key)) inPrimary++
    }
    return { inPrimary, total: results.matches.length, primaryTotal: primaryResults.length }
  }, [results, primaryResults])

  return (
    <div className="w-80 border-r border-gray-700 bg-gray-800/80 flex flex-col overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-700">
        <span className="text-xs font-semibold text-gray-300">Gegensuche</span>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1 text-[10px] text-gray-400 cursor-pointer">
            <input type="checkbox" checked={liveEnabled} onChange={e => setLiveEnabled(e.target.checked)} />
            Live
          </label>
          <button onClick={onClose} className="text-gray-500 hover:text-white"><X size={14} /></button>
        </div>
      </div>

      <div className="p-3 space-y-3 border-b border-gray-700">
        <div>
          <label className="text-[10px] text-gray-500 block mb-0.5">Zeitraum</label>
          <select value={periodDays} onChange={e => setPeriodDays(parseInt(e.target.value))}
            className="input text-xs py-1 w-full">
            {COUNTER_SEARCH_SETTINGS.periodOptions.map(d => (
              <option key={d} value={d}>{d} Tage</option>
            ))}
          </select>
        </div>

        <MatchModeSelector
          mode={matchMode} setMode={setMatchMode}
          threshold={matchThreshold} setThreshold={setMatchThreshold}
          totalCriteria={criteria.length}
        />

        <button onClick={() => setShowGlobalFuzzy(!showGlobalFuzzy)}
          className="w-full flex items-center justify-between text-xs text-gray-400 hover:text-gray-200">
          <span>Globale Unschaerfe</span>
          {showGlobalFuzzy ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        </button>
        {showGlobalFuzzy && (
          <div className="p-2 bg-gray-900/50 rounded border border-gray-700">
            <FuzzyPanel values={globalFuzzy} onChange={setGlobalFuzzy} title="" />
          </div>
        )}

        <div className="text-[10px] text-gray-500">
          {criteria.length} Kriterien | {activeChartId ? '1 Chart aktiv' : 'Alle Charts'}
        </div>

        <button onClick={runSearch} disabled={loading || criteria.length === 0}
          className="btn btn-primary w-full flex items-center justify-center gap-2 text-xs disabled:opacity-50">
          {loading ? <RefreshCw size={12} className="animate-spin" /> : <Search size={12} />}
          {loading ? 'Suche laeuft...' : 'Gegensuche starten'}
        </button>

        <button onClick={() => setShowSaveDialog(true)} disabled={criteria.length === 0}
          className="btn w-full flex items-center justify-center gap-2 text-xs bg-emerald-700 hover:bg-emerald-600 disabled:opacity-50">
          <Save size={12} /> Als Indikator-Set speichern
        </button>

        {error && <div className="text-red-400 text-[10px] bg-red-900/30 p-2 rounded">{error}</div>}
      </div>

      <div className="border-b border-gray-700 p-2 max-h-64 overflow-auto">
        <div className="text-[10px] text-gray-500 mb-1">Kriterien:</div>
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs text-gray-400">Kriterien ({criteria.length})</span>
          <button onClick={() => setShowManualCrit(!showManualCrit)}
                  className="text-xs text-blue-400 hover:text-blue-300">
            {showManualCrit ? '− Feld-Kriterium' : '+ Feld-Kriterium'}
          </button>
        </div>
        {showManualCrit && (
          <ManualCriterionPanel
            onAdd={(c) => setCriteria(prev => [...prev, c])}
            onClose={() => setShowManualCrit(false)}
          />
        )}
        <CriteriaList criteria={criteria} setCriteria={setCriteria} globalFuzzy={globalFuzzy} />
      </div>

      {results && (
        <div className="flex-1 overflow-auto p-2">
          <div className="text-xs text-gray-400 mb-2 space-y-0.5">
            <div>{results.total_found} Treffer in {results.period_days}T</div>
            {crossRef && (
              <div className="text-blue-400 text-[10px]">
                {crossRef.inPrimary}/{crossRef.total} auch in Hauptsuche ({crossRef.primaryTotal})
              </div>
            )}
          </div>
          <div className="space-y-1">
            {(results.matches || []).slice(0, 100).map((m, i) => {
              const key = `${m.symbol}|${m.event_start}`
              const inPrimary = primaryResults?.some(e => `${e.symbol}|${e.event_start}` === key)
              return (
                <div key={i} className={`flex items-center justify-between text-xs py-1 px-2 rounded hover:bg-gray-800 ${
                  inPrimary ? 'bg-blue-900/30 border border-blue-700/30' : 'bg-gray-900/30'
                }`}>
                  <span className="font-mono text-gray-200">{m.symbol}</span>
                  <div className="flex items-center gap-2">
                    {m.matched_count != null && (
                      <span className="text-gray-500 text-[10px]">{m.matched_count}/{m.total_criteria}</span>
                    )}
                    <span className={m.change_percent >= 0 ? 'text-green-400' : 'text-red-400'}>
                      {m.change_percent >= 0 ? '+' : ''}{m.change_percent?.toFixed(1)}%
                    </span>
                    <span className="text-blue-400 font-mono">{(m.match_score * 100).toFixed(0)}%</span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {showSaveDialog && (() => {
        const split = splitInitialAndCriteria(criteria)
        return (
          <SaveSetDialog
            criteria={split.criteria}
            initialPoints={split.initialPoints}
            initialPointConfig={{
              match_mode: matchMode,
              match_threshold: matchThreshold,
              enforce_sequence: matchMode === 'sequence',
              window_minutes: 30,
            }}
            globalFuzzy={globalFuzzy}
            durationMinutes={durationMinutes}
            candleTimeframeMinutes={candleTimeframeMinutes}
            prehistoryMinutes={prehistoryMinutes}
            direction={direction}
            primaryContext={primaryContext}
            onClose={() => setShowSaveDialog(false)}
            onSaved={(data) => { alert(`Set gespeichert: ${data.name} (ID ${data.set_id})`) }}
          />
        )
      })()}
    </div>
  )
}
