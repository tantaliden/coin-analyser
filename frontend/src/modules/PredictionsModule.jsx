import { useEffect, useState, useRef } from 'react'
import { Power, RefreshCw, AlertTriangle, TrendingUp, TrendingDown, X, Send, Filter, Settings, Save, Bot } from 'lucide-react'
import api from '../utils/api'

const SCOPES = [
  { id: 'visible', label: 'Aktiv (<1h)' },
  { id: 'open', label: 'Alle offen' },
  { id: 'closed', label: 'Abgeschlossen' },
  { id: 'all', label: 'Alle' },
  { id: 'errors', label: 'Errors' },
]

function pct(entry, target, side) {
  if (!entry || !target) return 0
  const raw = ((target - entry) / entry) * 100
  return side === 'long' ? raw : -raw
}

function ageMin(created) {
  if (!created) return 0
  return Math.floor((Date.now() - new Date(created).getTime()) / 60000)
}

function fmt(n, d = 4) {
  if (n == null || isNaN(n)) return '-'
  return Number(n).toLocaleString('de-DE', { maximumFractionDigits: d })
}

// Datum kompakt: "29.04 13:45"
function fmtDt(iso) {
  if (!iso) return '-'
  const d = new Date(iso)
  const dd = String(d.getDate()).padStart(2, '0')
  const mm = String(d.getMonth() + 1).padStart(2, '0')
  const hh = String(d.getHours()).padStart(2, '0')
  const mi = String(d.getMinutes()).padStart(2, '0')
  return `${dd}.${mm} ${hh}:${mi}`
}

// Dauer "27m" / "2h14" / "3d4h"
function durStr(fromIso, toIso) {
  if (!fromIso || !toIso) return ''
  const sec = Math.max(0, (new Date(toIso).getTime() - new Date(fromIso).getTime()) / 1000)
  if (sec < 60) return `${Math.round(sec)}s`
  const m = Math.floor(sec / 60)
  if (m < 60) return `${m}m`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h${m % 60}`
  return `${Math.floor(h / 24)}d${h % 24}h`
}

// Fallback-Heuristik wenn coinMeta nicht verfügbar
function trimPxFallback(v) {
  if (v == null || isNaN(v)) return v
  const a = Math.abs(v)
  if (a >= 1000) return Number(v.toFixed(2))
  if (a >= 1)    return Number(v.toFixed(4))
  if (a >= 0.01) return Number(v.toFixed(5))
  if (a >= 0.001) return Number(v.toFixed(6))
  return Number(v.toFixed(8))
}

// HL-konforme Rundung: max (6-szDec) Decimals UND max 5 sig figs.
// Beide Constraints muessen erfuellt sein, sonst lehnt HL ab.
function roundHL(v, szDec, sigFigs = 5) {
  if (v == null || isNaN(v) || v <= 0) return v
  const maxDec = Math.max(0, 6 - (szDec ?? 0))
  // Erst auf 5 sig figs runden
  const dForSig = sigFigs - Math.floor(Math.log10(Math.abs(v))) - 1
  const sigRounded = Number(v.toFixed(Math.max(0, dForSig)))
  // Dann auf max-Decimals begrenzen
  return Number(sigRounded.toFixed(maxDec))
}

// Echte HL-Decimals aus hl_meta (coin-meta API). Fallback bei fehlendem Eintrag.
function trimPxBySymbol(v, symbol, coinMeta) {
  if (v == null || isNaN(v)) return v
  const meta = coinMeta?.[symbol]
  if (!meta) return trimPxFallback(v)
  return roundHL(v, meta.sz_decimals)
}

// Format mit fixen Decimals (für Anzeige als String) — basierend auf HL-konformem Wert
function fmtPx(v, symbol, coinMeta) {
  if (v == null || isNaN(v)) return '-'
  const meta = coinMeta?.[symbol]
  if (!meta) return Number(v).toLocaleString('de-DE', { maximumFractionDigits: 6 })
  const rounded = roundHL(v, meta.sz_decimals)
  // Anzeige-Decimals: aus DB (price_decimals = HL aktueller markPx-Decimals)
  const dec = meta.price_decimals ?? Math.max(0, 6 - (meta.sz_decimals ?? 0))
  return Number(rounded).toLocaleString('de-DE', { maximumFractionDigits: dec })
}

export default function PredictionsModule() {
  const [status, setStatus] = useState(null)
  const [list, setList] = useState([])
  const [scope, setScope] = useState('visible')
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const [orderTarget, setOrderTarget] = useState(null) // prediction für Modal
  const [showSettings, setShowSettings] = useState(false)
  const [showAutoTrade, setShowAutoTrade] = useState(false)
  const [search, setSearch] = useState('')
  const [coinMeta, setCoinMeta] = useState({})  // {symbol: {sz_decimals, price_decimals, max_leverage}}
  const [sortKey, setSortKey] = useState('created_at')
  const [sortAsc, setSortAsc] = useState(false)

  // coinMeta einmal laden (TTL 30s im Backend, hier reicht 1x bei Mount)
  useEffect(() => {
    api.get('/api/v1/predictor/coin-meta').then(r => setCoinMeta(r.data || {})).catch(() => {})
  }, [])

  const scopeRef = useRef(scope)
  scopeRef.current = scope

  const load = async () => {
    const requestedScope = scopeRef.current
    try {
      const [st, pr] = await Promise.all([
        api.get('/api/v1/predictor/status'),
        api.get(`/api/v1/predictor/predictions?scope=${requestedScope}&limit=200`),
      ])
      // Drop result wenn scope wechselte waehrend Fetch lief (Race-Schutz)
      if (requestedScope !== scopeRef.current) return
      setStatus(st.data)
      setList(pr.data.predictions || [])
      setError(null)
    } catch (e) {
      setError(e.response?.data?.detail || 'Laden fehlgeschlagen')
    }
    setLoading(false)
  }

  useEffect(() => { load() }, [scope])
  useEffect(() => {
    if (scope === 'closed') return
    let inFlight = false
    const tick = async () => {
      if (inFlight) return
      inFlight = true
      try { await load() } finally { inFlight = false }
    }
    const t = setInterval(tick, 1000)
    return () => clearInterval(t)
  }, [scope])

  const toggleService = async () => {
    try {
      const action = status?.service_running ? 'stop' : 'start'
      await api.post(`/api/v1/predictor/service/${action}`)
      await load()
    } catch (e) {
      setError(e.response?.data?.detail || 'Service-Steuerung fehlgeschlagen')
    }
  }

  if (loading) return <div className="text-gray-400 p-4">Laden...</div>

  const filteredList = (() => {
    const q = search.trim().toLowerCase()
    const arr = q ? list.filter(p => {
      const hay = [p.symbol, p.side, p.status, p.rule_name, p.source].filter(Boolean).join(' ').toLowerCase()
      return hay.includes(q)
    }) : list
    const cmp = (a, b) => {
      let va = a[sortKey], vb = b[sortKey]
      if (sortKey === 'created_at' || sortKey === 'closed_at') {
        va = va ? new Date(va).getTime() : 0
        vb = vb ? new Date(vb).getTime() : 0
      }
      if (va == null && vb == null) return 0
      if (va == null) return 1
      if (vb == null) return -1
      if (typeof va === 'string') return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va)
      return sortAsc ? va - vb : vb - va
    }
    return [...arr].sort(cmp)
  })()

  const toggleSort = (k) => {
    if (sortKey === k) setSortAsc(!sortAsc)
    else { setSortKey(k); setSortAsc(false) }
  }

  return (
    <div className="h-full flex flex-col text-xs">
      {error && (
        <div className="p-2 text-red-400 text-sm bg-red-900/30 flex items-center gap-2">
          <AlertTriangle size={14} /> {error}
        </div>
      )}

      {/* Header */}
      <div className="p-3 border-b border-gray-700">
        <div className="flex items-center gap-3 mb-3">
          <button
            onClick={toggleService}
            className={`flex items-center gap-2 px-4 py-2 rounded font-semibold text-sm transition-colors ${
              status?.service_running ? 'bg-green-600 hover:bg-green-500' : 'bg-gray-700 hover:bg-gray-600'
            }`}>
            <Power size={16} />
            {status?.service_running ? 'Predictor AKTIV' : 'Predictor AUS'}
          </button>
          <span className="text-gray-500">Predictor Auto-Lerner</span>
          <button onClick={() => setShowAutoTrade(true)}
            className={`ml-auto px-3 py-2 rounded flex items-center gap-2 ${
              status?.auto_trade ? 'bg-purple-700 hover:bg-purple-600' : 'bg-gray-700 hover:bg-gray-600'
            }`}
            title="Auto-Trade Einstellungen">
            <Bot size={14} /> Auto-Trade {status?.auto_trade ? '(an)' : '(aus)'}
          </button>
          <button onClick={() => setShowSettings(true)} className="px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded flex items-center gap-2" title="Einstellungen">
            <Settings size={14} /> Settings
          </button>
          <button onClick={load} className="px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded flex items-center gap-2">
            <RefreshCw size={14} /> Refresh
          </button>
        </div>

        <div className="grid grid-cols-5 gap-2">
          <Stat label="Offen" value={status?.open_count ?? 0} />
          <Stat label="Wins" value={status?.wins ?? 0} color="text-green-400" />
          <Stat label="Losses" value={status?.losses ?? 0} color="text-red-400" />
          <Stat label="WR" value={status?.winrate_pct != null ? status.winrate_pct + '%' : '-'} />
          <Stat label="Threshold" value={(status?.threshold ?? 0).toFixed(3)} />
        </div>
      </div>

      {/* Filter */}
      <div className="p-2 border-b border-gray-700 flex items-center gap-2">
        <Filter size={12} className="text-gray-500" />
        {SCOPES.map(sc => (
          <button key={sc.id} onClick={() => setScope(sc.id)}
            className={`px-2 py-1 rounded text-[10px] ${scope === sc.id ? 'bg-blue-600' : 'bg-gray-800 hover:bg-gray-700'}`}>
            {sc.label}
          </button>
        ))}
        <input
          type="text"
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Suche: symbol, side, status, rule..."
          className="ml-2 px-2 py-1 bg-gray-800 border border-gray-700 rounded text-[10px] w-64 focus:outline-none focus:border-blue-500"
        />
        {search && (
          <button onClick={() => setSearch('')} className="p-1 hover:bg-gray-700 rounded text-gray-400" title="Suche leeren">
            <X size={12} />
          </button>
        )}
        <span className="text-gray-500 text-[10px] ml-2">Sort:</span>
        {[
          ['created_at', 'Alter'],
          ['symbol', 'Symbol'],
          ['side', 'Side'],
          ['effective_leverage', 'Hebel'],
          ['score', 'Score'],
          ['pnl_pct', 'PnL'],
        ].map(([k, l]) => (
          <button key={k} onClick={() => toggleSort(k)}
            className={`px-1.5 py-0.5 rounded text-[10px] ${sortKey === k ? 'bg-blue-600' : 'bg-gray-800 hover:bg-gray-700'}`}>
            {l}{sortKey === k && (sortAsc ? ' ↑' : ' ↓')}
          </button>
        ))}
        <span className="ml-auto text-gray-500 text-[10px]">
          {(() => {
            const f = filteredList.length
            return search ? `${f} / ${list.length}` : `${list.length} Einträge`
          })()}
        </span>
      </div>

      {/* List */}
      <div className="flex-1 overflow-auto">
        {filteredList.length === 0 ? (
          <div className="p-4 text-gray-500 text-center">{search ? 'Keine Treffer' : 'Keine Predictions'}</div>
        ) : (
          <div className="divide-y divide-gray-700/50">
            {filteredList.map(p => (
              <PredictionRow key={p.id} p={p} coinMeta={coinMeta} onOrder={() => setOrderTarget(p)} />
            ))}
          </div>
        )}
      </div>

      {orderTarget && (
        <OrderModal prediction={orderTarget} coinMeta={coinMeta} onClose={() => setOrderTarget(null)} onSuccess={load} />
      )}
      {showSettings && (
        <SettingsModal onClose={() => setShowSettings(false)} onSaved={load} />
      )}
      {showAutoTrade && (
        <AutoTradeModal onClose={() => setShowAutoTrade(false)} onSaved={load} />
      )}
    </div>
  )
}

function SettingsModal({ onClose, onSaved }) {
  const [cfg, setCfg] = useState(null)
  const [err, setErr] = useState(null)
  const [saving, setSaving] = useState(false)
  const [walletCfg, setWalletCfg] = useState(null)

  useEffect(() => {
    api.get('/api/v1/predictor/config').then(r => setCfg(r.data)).catch(e => setErr(e.message))
    api.get('/api/v1/wallet/config').then(r => setWalletCfg(r.data)).catch(e => setErr(e.message))
  }, [])

  if (!cfg) {
    return (
      <div className="absolute inset-0 bg-black/70 flex items-center justify-center z-50">
        <div className="bg-gray-900 border border-gray-700 rounded-lg p-4 w-[600px]">
          {err ? <div className="text-red-400">{err}</div> : <div className="text-gray-400">Lade...</div>}
          <div className="mt-3 flex justify-end"><button onClick={onClose} className="px-3 py-1 bg-gray-700 rounded text-sm">Schließen</button></div>
        </div>
      </div>
    )
  }

  const set = (path, value) => {
    const c = JSON.parse(JSON.stringify(cfg))
    const parts = path.split('.')
    let o = c
    for (let i = 0; i < parts.length - 1; i++) {
      if (!o[parts[i]]) o[parts[i]] = {}
      o = o[parts[i]]
    }
    o[parts[parts.length - 1]] = value
    setCfg(c)
  }

  const num = (path, dflt = 0) => {
    const parts = path.split('.')
    let o = cfg
    for (const p of parts) { if (o == null) return dflt; o = o[p] }
    return o ?? dflt
  }

  const submit = async () => {
    setSaving(true); setErr(null)
    try {
      await api.put('/api/v1/predictor/config', { config: cfg })
      if (walletCfg) {
        await api.put('/api/v1/wallet/config', { config: walletCfg })
      }
      onSaved && onSaved()
      onClose()
    } catch (e) {
      setErr(e.response?.data?.detail || 'Speichern fehlgeschlagen')
    }
    setSaving(false)
  }

  const wset = (key, value) => setWalletCfg(prev => ({ ...(prev || {}), [key]: value }))
  const wnum = (key, dflt = 0) => (walletCfg && walletCfg[key] != null) ? walletCfg[key] : dflt

  const trading = cfg.trading || {}
  const cal = cfg.self_calibration || {}
  const learner = cfg.online_learner || {}
  const bandit = cfg.bandit || {}
  const mh = cfg.multi_head || {}
  const modify = cfg.modify_bandit || {}
  const whale = cfg.whale_tracker || {}
  const stalker = cfg.stalker || {}
  const stalkerRegime = stalker.btc_regime || {}
  const stalkerIdent = stalker.coin_identity || {}
  const stalkerCross = stalker.cross_coin || {}

  const parseList = (str) => {
    return str.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n) && n > 0)
  }
  const parseListSigned = (str) => {
    return str.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n) && n !== 0)
  }
  const formatList = (arr) => Array.isArray(arr) ? arr.join(', ') : ''

  return (
    <div className="absolute inset-0 bg-black/70 flex items-center justify-center z-50">
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-4 w-[640px] max-h-[90vh] overflow-auto">
        <div className="flex items-center justify-between mb-3">
          <div className="font-semibold">Predictor-Einstellungen</div>
          <button onClick={onClose} className="p-1 hover:bg-gray-700 rounded"><X size={16} /></button>
        </div>
        {err && <div className="bg-red-900/40 text-red-300 p-2 rounded mb-2 text-sm">{err}</div>}

        <Section title="Predictor v5 (Multi-Head: DirectionClassifier + 2× MagnitudeRegressor)">
          <Field label="Min. Direction-Confidence (target nach Cold-Start)">
            <NumInput value={num('multi_head.min_direction_confidence', 0.95)} step="0.01" min="0.5" max="0.999"
              onChange={v => set('multi_head.min_direction_confidence', parseFloat(v))} />
          </Field>
          <Field label="Max. gleichzeitige Predictions">
            <NumInput value={num('multi_head.max_open_predictions', 50)} step="1" min="1" max="500"
              onChange={v => set('multi_head.max_open_predictions', parseInt(v))} />
          </Field>
          <Field label="Min. TP-% (Schwelle, kleiner: skip)">
            <NumInput value={num('multi_head.min_tp_pct', 0.5)} step="0.1" min="0.1" max="10"
              onChange={v => set('multi_head.min_tp_pct', parseFloat(v))} />
          </Field>
          <Field label="Max. SL-% (Risiko-Cap, größer: skip)">
            <NumInput value={num('multi_head.max_sl_pct', 2.5)} step="0.1" min="0.5" max="20"
              onChange={v => set('multi_head.max_sl_pct', parseFloat(v))} />
          </Field>
          <Field label="TP Safety-Factor (× Predicted-Peak)">
            <NumInput value={num('multi_head.tp_safety_factor', 0.6)} step="0.05" min="0.1" max="2"
              onChange={v => set('multi_head.tp_safety_factor', parseFloat(v))} />
          </Field>
          <Field label="SL Safety-Factor (× Predicted-Trough)">
            <NumInput value={num('multi_head.sl_safety_factor', 1.3)} step="0.05" min="0.5" max="3"
              onChange={v => set('multi_head.sl_safety_factor', parseFloat(v))} />
          </Field>
          <Field label="Cold-Start: min N für Random→Modell-Switch">
            <NumInput value={num('multi_head.cold_start_min_n', 200)} step="10" min="0" max="10000"
              onChange={v => set('multi_head.cold_start_min_n', parseInt(v))} />
          </Field>
          <Field label="Cold-Start: threshold während Lern-Phase">
            <NumInput value={num('multi_head.cold_start_threshold', 0.5)} step="0.05" min="0" max="1"
              onChange={v => set('multi_head.cold_start_threshold', parseFloat(v))} />
          </Field>
          <Field label="Cold-Start: Max Predictions / Scan">
            <NumInput value={num('multi_head.cold_start_max_per_scan', 10)} step="1" min="1" max="200"
              onChange={v => set('multi_head.cold_start_max_per_scan', parseInt(v))} />
          </Field>
          <Field label="Cold-Start: TP-Magnitude in % (raw, vor safety)">
            <NumInput value={num('multi_head.cold_start_tp_raw_pct', 1.5)} step="0.1" min="0.1" max="10"
              onChange={v => set('multi_head.cold_start_tp_raw_pct', parseFloat(v))} />
          </Field>
          <Field label="Cold-Start: SL-Magnitude in % (raw, vor safety)">
            <NumInput value={num('multi_head.cold_start_sl_raw_pct', 1.0)} step="0.1" min="0.1" max="10"
              onChange={v => set('multi_head.cold_start_sl_raw_pct', parseFloat(v))} />
          </Field>
          <Field label="Timeout (Min)">
            <NumInput value={Math.round(num('timeout_hours', 1) * 60)} step="15" min="15" max="10080"
              onChange={v => set('timeout_hours', parseInt(v) / 60)} />
          </Field>
          <Field label="Cooldown pro Symbol (s)">
            <NumInput value={num('cooldown_seconds_per_symbol', 300)} step="60" min="0" max="86400"
              onChange={v => set('cooldown_seconds_per_symbol', parseInt(v))} />
          </Field>
          <Field label="Universe Top-N (Coins nach Volume)">
            <NumInput value={num('universe_top_n', 200)} step="10" min="10" max="500"
              onChange={v => set('universe_top_n', parseInt(v))} />
          </Field>
          <Field label="Scan-Pass-Intervall (s) — wie oft neue Predictions geprüft">
            <NumInput value={num('scan_interval_seconds', 60)} step="1" min="1" max="3600"
              onChange={v => set('scan_interval_seconds', parseInt(v))} />
          </Field>
          <Field label="Watch-Pass-Intervall (s) — wie oft TP/SL geprüft wird">
            <NumInput value={num('watch_interval_seconds', 1)} step="1" min="1" max="60"
              onChange={v => set('watch_interval_seconds', parseInt(v))} />
          </Field>

          <div style={{gridColumn: '1 / -1', marginTop: 8, fontSize: 12, color: '#9ca3af'}}>
            Lern-Gewichte (Multi-Head Direction-Classifier):
          </div>
          <Field label="Loss-Weight (falsche Richtung: härter)">
            <NumInput value={num('multi_head.loss_weight', 1.0)} step="0.5" min="0" max="10"
              onChange={v => set('multi_head.loss_weight', parseFloat(v))} />
          </Field>
          <Field label="Win-Weight (richtige Richtung)">
            <NumInput value={num('multi_head.win_weight', 1.0)} step="0.5" min="0" max="10"
              onChange={v => set('multi_head.win_weight', parseFloat(v))} />
          </Field>
          <Field label="Timeout-Wrong-Weight (Drift in Gegenrichtung)">
            <NumInput value={num('multi_head.timeout_wrong_weight', 1.0)} step="0.5" min="0" max="10"
              onChange={v => set('multi_head.timeout_wrong_weight', parseFloat(v))} />
          </Field>
          <Field label="Timeout-Correct-Weight (Drift in Predict-Richtung)">
            <NumInput value={num('multi_head.timeout_correct_weight', 0.3)} step="0.1" min="0" max="2"
              onChange={v => set('multi_head.timeout_correct_weight', parseFloat(v))} />
          </Field>
          <Field label="Timeout-Flat-Threshold % (drunter: skip-label)">
            <NumInput value={num('multi_head.timeout_flat_threshold_pct', 0.3)} step="0.05" min="0.05" max="2"
              onChange={v => set('multi_head.timeout_flat_threshold_pct', parseFloat(v))} />
          </Field>
        </Section>

        <Section title="Stalker-Predictor (pro-Coin Baseline + Cross-Coin + Coin-Identity)">
          <Field label="Aktiv">
            <input type="checkbox" checked={!!stalker.enabled}
              onChange={e => set('stalker.enabled', e.target.checked)} />
          </Field>
          <Field label="Baseline-Refresh (h)">
            <NumInput value={num('stalker.baseline_refresh_hours', 24)} step="1" min="1" max="168"
              onChange={v => set('stalker.baseline_refresh_hours', parseInt(v))} />
          </Field>
          <Field label="Baseline-Lookback (Tage)">
            <NumInput value={num('stalker.baseline_lookback_days', 14)} step="1" min="3" max="90"
              onChange={v => set('stalker.baseline_lookback_days', parseInt(v))} />
          </Field>
          <Field label="Min. Samples pro Bucket (sonst skip)">
            <NumInput value={num('stalker.baseline_min_samples_per_bucket', 5)} step="1" min="1" max="500"
              onChange={v => set('stalker.baseline_min_samples_per_bucket', parseInt(v))} />
          </Field>
          <Field label="BTC-Regime Fenster (h)">
            <NumInput value={num('stalker.btc_regime.source_window_hours', 4)} step="1" min="1" max="48"
              onChange={v => set('stalker.btc_regime.source_window_hours', parseInt(v))} />
          </Field>
          <Field label="BTC-Bull Schwelle (% Change)">
            <NumInput value={num('stalker.btc_regime.bull_threshold_pct', 1.0)} step="0.1" min="0" max="20"
              onChange={v => set('stalker.btc_regime.bull_threshold_pct', parseFloat(v))} />
          </Field>
          <Field label="BTC-Bear Schwelle (% Change)">
            <NumInput value={num('stalker.btc_regime.bear_threshold_pct', -1.0)} step="0.1" min="-20" max="0"
              onChange={v => set('stalker.btc_regime.bear_threshold_pct', parseFloat(v))} />
          </Field>
          <Field label="Cross-Coin Reference (Komma-Liste)">
            <input type="text"
              defaultValue={(stalkerCross.reference_coins || []).join(', ')}
              onBlur={e => set('stalker.cross_coin.reference_coins', e.target.value.split(',').map(s => s.trim()).filter(Boolean))}
              className="w-full bg-gray-800 px-2 py-1.5 rounded border border-gray-600 text-xs"/>
          </Field>
          <Field label="Corr-Fenster Minuten (Komma-Liste)">
            <input type="text"
              defaultValue={(stalkerCross.corr_windows_minutes || []).join(', ')}
              onBlur={e => set('stalker.cross_coin.corr_windows_minutes', e.target.value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n) && n > 0))}
              className="w-full bg-gray-800 px-2 py-1.5 rounded border border-gray-600 text-xs"/>
          </Field>
          <Field label="Lead/Lag Fenster (min)">
            <NumInput value={num('stalker.cross_coin.lead_lag_window_minutes', 5)} step="1" min="1" max="60"
              onChange={v => set('stalker.cross_coin.lead_lag_window_minutes', parseInt(v))} />
          </Field>
          <Field label="Lead/Lag Schwelle (% Reference-Move)">
            <NumInput value={num('stalker.cross_coin.lead_lag_threshold_pct', 0.2)} step="0.05" min="0" max="5"
              onChange={v => set('stalker.cross_coin.lead_lag_threshold_pct', parseFloat(v))} />
          </Field>
          <Field label="Coin-Identity Hash-Slots">
            <NumInput value={num('stalker.coin_identity.hash_slots', 16)} step="1" min="4" max="128"
              onChange={v => set('stalker.coin_identity.hash_slots', parseInt(v))} />
          </Field>
          <Field label="Network Hash-Slots">
            <NumInput value={num('stalker.coin_identity.network_hash_slots', 8)} step="1" min="2" max="64"
              onChange={v => set('stalker.coin_identity.network_hash_slots', parseInt(v))} />
          </Field>
        </Section>

        <Section title="Whale-Tracker (Stufe B: HL Leaderboard + Position-Snapshots)">
          <Field label="Aktiv">
            <input type="checkbox" checked={!!whale.enabled}
              onChange={e => set('whale_tracker.enabled', e.target.checked)} />
          </Field>
          <Field label="Leaderboard-Refresh (h)">
            <NumInput value={num('whale_tracker.leaderboard_refresh_hours', 6)} step="1" min="1" max="48"
              onChange={v => set('whale_tracker.leaderboard_refresh_hours', parseInt(v))} />
          </Field>
          <Field label="Fills-Poll (s)">
            <NumInput value={num('whale_tracker.fills_poll_seconds', 300)} step="30" min="60" max="3600"
              onChange={v => set('whale_tracker.fills_poll_seconds', parseInt(v))} />
          </Field>
          <Field label="Top-N Wallets">
            <NumInput value={num('whale_tracker.top_n_wallets', 50)} step="10" min="10" max="500"
              onChange={v => set('whale_tracker.top_n_wallets', parseInt(v))} />
          </Field>
          <Field label="Min. 7d-PnL für Whale ($)">
            <NumInput value={num('whale_tracker.min_rolling_7d_pnl_usd', 100000)} step="10000" min="1000" max="10000000"
              onChange={v => set('whale_tracker.min_rolling_7d_pnl_usd', parseInt(v))} />
          </Field>
        </Section>

        <Section title="Trader (passt TP/SL laufender Trades alle 30s an)">
          <Field label="Aktiv (nur wenn Auto-Trade an)">
            <input type="checkbox" checked={!!modify.enabled}
              onChange={e => set('modify_bandit.enabled', e.target.checked)} />
          </Field>
          <Field label="Tick-Intervall (s)">
            <NumInput value={num('modify_bandit.interval_seconds', 30)} step="5" min="5" max="300"
              onChange={v => set('modify_bandit.interval_seconds', parseInt(v))} />
          </Field>
          <Field label="Min. Trade-Alter bevor Modify (s)">
            <NumInput value={num('modify_bandit.min_open_age_seconds', 60)} step="10" min="0" max="600"
              onChange={v => set('modify_bandit.min_open_age_seconds', parseInt(v))} />
          </Field>
          <Field label="TP-Delta-Buckets (% in 0.1-Schritten — Komma-Liste, ± erlaubt)">
            <input type="text"
              defaultValue={formatList(modify.tp_delta_buckets)}
              onBlur={e => set('modify_bandit.tp_delta_buckets', parseListSigned(e.target.value))}
              className="w-full bg-gray-800 px-2 py-1.5 rounded border border-gray-600 text-xs"/>
          </Field>
          <Field label="SL-Delta-Buckets (% in 0.1-Schritten — Komma-Liste, ± erlaubt)">
            <input type="text"
              defaultValue={formatList(modify.sl_delta_buckets)}
              onBlur={e => set('modify_bandit.sl_delta_buckets', parseListSigned(e.target.value))}
              className="w-full bg-gray-800 px-2 py-1.5 rounded border border-gray-600 text-xs"/>
          </Field>
          <Field label="Exploration Start">
            <NumInput value={num('modify_bandit.exploration_init', 1.0)} step="0.05" min="0" max="1"
              onChange={v => set('modify_bandit.exploration_init', parseFloat(v))} />
          </Field>
          <Field label="Exploration Decay / Decision">
            <NumInput value={num('modify_bandit.exploration_decay_per_decision', 0.0001)} step="0.00005" min="0" max="0.01"
              onChange={v => set('modify_bandit.exploration_decay_per_decision', parseFloat(v))} />
          </Field>
          <Field label="Exploration Floor">
            <NumInput value={num('modify_bandit.exploration_floor', 0.05)} step="0.01" min="0" max="0.5"
              onChange={v => set('modify_bandit.exploration_floor', parseFloat(v))} />
          </Field>
        </Section>

        <Section title="Lifecycle">
          <Field label="Aus Liste verstecken nach (h)">
            <NumInput value={num('hide_after_hours', 1)} step="0.5" min="0.1" max="48" onChange={v => set('hide_after_hours', parseFloat(v))} />
          </Field>
          <Field label="Cooldown / Symbol (s)">
            <NumInput value={num('cooldown_seconds_per_symbol', 300)} step="30" min="0" max="3600" onChange={v => set('cooldown_seconds_per_symbol', parseInt(v))} />
          </Field>
          <Field label="Universe Top-N">
            <NumInput value={num('universe_top_n', 50)} step="10" min="5" max="230" onChange={v => set('universe_top_n', parseInt(v))} />
          </Field>
        </Section>

        <Section title="Trading-Defaults (vorausgefüllt im Order-Modal)">
          <Field label="Hebel">
            <NumInput value={trading.default_leverage ?? 5} step="1" min="1" max="20" onChange={v => set('trading.default_leverage', parseInt(v))} />
          </Field>
          <Field label="Trade-Size ($)">
            <NumInput value={trading.default_size_usd ?? 20} step="5" min="10" max="10000" onChange={v => set('trading.default_size_usd', parseFloat(v))} />
          </Field>
          <Field label="Trail-Stop (%)">
            <NumInput value={trading.default_trailing_stop_pct ?? 1} step="0.1" min="0.1" max="20" onChange={v => set('trading.default_trailing_stop_pct', parseFloat(v))} />
          </Field>
          <Field label="Trail standardmäßig an">
            <input type="checkbox" checked={!!trading.default_use_trailing}
              onChange={e => set('trading.default_use_trailing', e.target.checked)} />
          </Field>
        </Section>

        <Section title="Lerner / Kalibrierung">
          <Field label="Online-Lerner aktiv">
            <input type="checkbox" checked={!!learner.enabled}
              onChange={e => set('online_learner.enabled', e.target.checked)} />
          </Field>
          <Field label="Min-Closes bevor Modell genutzt">
            <NumInput value={learner.min_closed_to_use ?? 50} step="10" min="0" max="10000" onChange={v => set('online_learner.min_closed_to_use', parseInt(v))} />
          </Field>
          <Field label="Selbstkalibrierung aktiv">
            <input type="checkbox" checked={!!cal.enabled}
              onChange={e => set('self_calibration.enabled', e.target.checked)} />
          </Field>
          <Field label="Ziel-WR">
            <NumInput value={cal.target_winrate ?? 0.55} step="0.01" min="0.1" max="0.95" onChange={v => set('self_calibration.target_winrate', parseFloat(v))} />
          </Field>
          <Field label="Rolling-Window">
            <NumInput value={cal.rolling_window ?? 100} step="10" min="20" max="1000" onChange={v => set('self_calibration.rolling_window', parseInt(v))} />
          </Field>
        </Section>

        <Section title="Wallet (HL REST-Fallback-Caches — live aus settings)">
          <Field label="HL-State-Cache TTL (s) — Balance/Positions/Orders">
            <NumInput value={wnum('hl_cache_ttl_seconds', 1)} step="0.5" min="0.1" max="60"
              onChange={v => wset('hl_cache_ttl_seconds', parseFloat(v))} />
          </Field>
          <Field label="HL-Fills-Cache TTL (s) — Trade-History">
            <NumInput value={wnum('hl_fills_ttl_seconds', 30)} step="1" min="1" max="600"
              onChange={v => wset('hl_fills_ttl_seconds', parseFloat(v))} />
          </Field>
        </Section>

        <div className="mt-4 flex justify-end gap-2">
          <button onClick={onClose} className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-sm">Abbrechen</button>
          <button onClick={submit} disabled={saving}
            className="px-4 py-1.5 bg-blue-600 hover:bg-blue-500 rounded text-sm flex items-center gap-2 disabled:opacity-50">
            <Save size={14} /> {saving ? 'Speichere...' : 'Speichern'}
          </button>
        </div>
      </div>
    </div>
  )
}

function RewardTiersEditor({ bandit, set }) {
  // Defaults: [[50, 0.75], [70, 1.0], [80, 1.2], [90, 1.5], [9999, 2.5]]
  const tiers = Array.isArray(bandit?.reward_mult_tiers) && bandit.reward_mult_tiers.length >= 5
    ? bandit.reward_mult_tiers
    : [[50, 0.75], [70, 1.0], [80, 1.2], [90, 1.5], [9999, 2.5]]

  const updateTier = (idx, field, val) => {
    const next = tiers.map((t, i) => i === idx ? [
      field === 0 ? parseFloat(val) : t[0],
      field === 1 ? parseFloat(val) : t[1]
    ] : [t[0], t[1]])
    set('bandit.reward_mult_tiers', next)
  }

  const labels = ['Stufe 1 (≤)', 'Stufe 2 (≤)', 'Stufe 3 (≤)', 'Stufe 4 (≤)', 'Stufe 5 (darüber)']
  return (
    <div style={{gridColumn: '1 / -1'}}>
      <div className="grid grid-cols-[1fr_90px_90px] gap-2 text-[11px] text-gray-400 mb-1">
        <div>Stufe</div><div>Schwelle %</div><div>Multiplikator</div>
      </div>
      {tiers.map((t, i) => (
        <div key={i} className="grid grid-cols-[1fr_90px_90px] gap-2 mb-1 items-center">
          <div className="text-xs text-gray-300">{labels[i]}</div>
          <input type="number" defaultValue={t[0]} step="1" min="0" max="9999"
            disabled={i === 4}
            onBlur={e => updateTier(i, 0, e.target.value)}
            className="w-full bg-gray-800 px-2 py-1 rounded border border-gray-600 text-xs disabled:opacity-50" />
          <input type="number" defaultValue={t[1]} step="0.05" min="0" max="10"
            onBlur={e => updateTier(i, 1, e.target.value)}
            className="w-full bg-gray-800 px-2 py-1 rounded border border-gray-600 text-xs" />
        </div>
      ))}
    </div>
  )
}

function Section({ title, children }) {
  return (
    <div className="mb-3 border border-gray-700/50 rounded p-2">
      <div className="text-gray-400 text-[11px] font-semibold uppercase mb-2">{title}</div>
      <div className="grid grid-cols-2 gap-3">{children}</div>
    </div>
  )
}

function NumInput({ value, onChange, ...rest }) {
  return (
    <input type="number" value={value ?? 0} {...rest}
      onChange={e => onChange(e.target.value)}
      className="w-full bg-gray-800 px-2 py-1.5 rounded border border-gray-600 text-xs" />
  )
}

function Stat({ label, value, color = 'text-blue-400' }) {
  return (
    <div className="bg-gray-800 rounded p-1.5 text-center">
      <div className="text-gray-500 text-[10px]">{label}</div>
      <div className={`font-mono font-semibold ${color}`}>{value}</div>
    </div>
  )
}

function PredictionRow({ p, coinMeta, onOrder }) {
  const isLong = p.side === 'long'
  const slPct = pct(p.entry_px, p.sl_px, p.side)
  const tpPct = pct(p.entry_px, p.tp_px, p.side)
  const ageM = ageMin(p.created_at)
  const isOpen = p.status === 'open'
  const pnl = p.pnl_pct

  const livePx = p.live_px ?? p.last_px
  const liveMove = (isOpen && livePx && p.entry_px) ? (isLong
    ? (livePx / p.entry_px - 1) * 100
    : (1 - livePx / p.entry_px) * 100) : null
  const peakMove = (p.peak_px && p.entry_px) ? (isLong
    ? (p.peak_px / p.entry_px - 1) * 100
    : (1 - p.trough_px / p.entry_px) * 100) : null
  const troughMove = (p.trough_px && p.entry_px) ? (isLong
    ? (p.trough_px / p.entry_px - 1) * 100
    : (1 - p.peak_px / p.entry_px) * 100) : null

  return (
    <div className="p-2 hover:bg-gray-800/50">
      <div className="flex items-center gap-2">
        <span className={`flex items-center gap-1 font-mono font-semibold w-24 ${isLong ? 'text-green-400' : 'text-red-400'}`} title={p.side}>
          {isLong ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
          <span className="truncate">{p.symbol}</span>
        </span>
        <span className="text-amber-400 font-mono text-[11px] w-10" title="Effektiver Hebel (HL-gekappt)">
          {p.effective_leverage ? `${p.effective_leverage}x` : '—'}
        </span>
        <span className="text-gray-500">@{fmtPx(p.entry_px, p.symbol, coinMeta)}</span>
        {isOpen && liveMove != null && (
          <span className={`font-mono font-semibold ${liveMove >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {liveMove >= 0 ? '+' : ''}{liveMove.toFixed(2)}%
          </span>
        )}
        <span className="text-green-400">TP {fmtPx(p.tp_px, p.symbol, coinMeta)} ({tpPct.toFixed(2)}%)</span>
        <span className="text-red-400">SL {fmtPx(p.sl_px, p.symbol, coinMeta)} ({slPct.toFixed(2)}%)</span>
        <span className="text-gray-500 ml-auto"></span>
        {isOpen ? (
          <span className="text-gray-400 text-[10px]">{ageM}m</span>
        ) : (
          <>
            <span className="text-gray-500 text-[10px]" title={`Geöffnet: ${p.created_at}\nGeschlossen: ${p.closed_at}`}>
              {fmtDt(p.created_at)} → {fmtDt(p.closed_at)} ({durStr(p.created_at, p.closed_at)})
            </span>
            <span className={`px-1.5 py-0.5 rounded text-[10px] ${
              p.status === 'win' ? 'bg-green-900/60 text-green-300' :
              p.status === 'loss' ? 'bg-red-900/60 text-red-300' :
              p.status === 'timeout' ? 'bg-gray-800 text-gray-400' :
              p.status === 'auto_trade_failed' ? 'bg-yellow-900/40 text-yellow-300' :
              p.status === 'auto_close_failsafe' ? 'bg-orange-900/40 text-orange-300' :
              'bg-gray-800 text-gray-400'
            }`} title={
              p.status === 'auto_trade_failed' ? 'HL-Order schlug fehl (Rate-Limit, kein Trade auf HL)' :
              p.status === 'auto_close_failsafe' ? 'HL-Position eröffnet aber TP/SL-Setup gescheitert → Failsafe-Auto-Close' : ''
            }>{p.status} {pnl != null && p.status !== 'auto_trade_failed' && p.status !== 'auto_close_failsafe' ? pnl.toFixed(2) + '%' : ''}</span>
          </>
        )}
        {isOpen && (
          <button onClick={onOrder} className="px-2 py-1 bg-blue-600 hover:bg-blue-500 rounded text-[10px] flex items-center gap-1">
            <Send size={10} /> Order
          </button>
        )}
      </div>
      <div className="text-gray-600 text-[10px] mt-0.5 ml-[68px]">
        {p.rule_name} · {isOpen ? 'live' : 'exit'} {fmtPx(isOpen ? livePx : p.exit_px, p.symbol, coinMeta)}
        {peakMove != null && troughMove != null && (
          <span> · peak <span className="text-green-500">+{peakMove.toFixed(2)}%</span>
            {' '}/ trough <span className="text-red-500">{troughMove.toFixed(2)}%</span></span>
        )}
        {!isOpen && p.predicted_up_pct != null && p.predicted_down_pct != null && (
          <span className="text-zinc-500"> · Predictor-TP {p.predicted_up_pct}% / SL {p.predicted_down_pct}%</span>
        )}
      </div>
    </div>
  )
}

// Long-TP: tp > entry → tp_pct > 0; SL: sl < entry → sl_pct > 0 (Abstand nach unten).
// Short-TP: tp < entry → tp_pct > 0 (Abstand nach unten); SL: sl > entry → sl_pct > 0.
// pct_to_px / px_to_pct sind so gestaltet, dass tp_pct und sl_pct IMMER positiv sind
// und den Abstand zum Entry vor Hebel beschreiben.
function tpPctToPx(entry, side, pct) {
  if (!entry || pct == null || isNaN(pct)) return 0
  return side === 'long' ? entry * (1 + pct / 100) : entry * (1 - pct / 100)
}
function slPctToPx(entry, side, pct) {
  if (!entry || pct == null || isNaN(pct)) return 0
  return side === 'long' ? entry * (1 - pct / 100) : entry * (1 + pct / 100)
}
function tpPxToPct(entry, side, px) {
  if (!entry || !px) return 0
  return side === 'long' ? (px / entry - 1) * 100 : (1 - px / entry) * 100
}
function slPxToPct(entry, side, px) {
  if (!entry || !px) return 0
  return side === 'long' ? (1 - px / entry) * 100 : (px / entry - 1) * 100
}

function OrderModal({ prediction: p, coinMeta, onClose, onSuccess }) {
  const isLong = p.side === 'long'
  const entry = p.entry_px

  const initialTpPct = tpPxToPct(entry, p.side, p.tp_px)
  const initialSlPct = slPxToPct(entry, p.side, p.sl_px)

  const [cfg, setCfg] = useState(null)
  const [form, setForm] = useState({
    leverage: 5, size_usd: 20,
    tp_pct: Number(initialTpPct.toFixed(2)),
    sl_pct: Number(initialSlPct.toFixed(2)),
    tp_px: trimPxBySymbol(p.tp_px, p.symbol, coinMeta),
    sl_px: trimPxBySymbol(p.sl_px, p.symbol, coinMeta),
    use_trailing: false, trailing_stop_pct: 1.0,
  })
  const [submitting, setSubmitting] = useState(false)
  const [err, setErr] = useState(null)

  useEffect(() => {
    api.get('/api/v1/predictor/config').then(r => {
      const t = r.data?.trading || {}
      setForm(f => ({
        ...f,
        leverage: t.default_leverage ?? f.leverage,
        size_usd: t.default_size_usd ?? f.size_usd,
        trailing_stop_pct: t.default_trailing_stop_pct ?? f.trailing_stop_pct,
        use_trailing: t.default_use_trailing ?? f.use_trailing,
      }))
      setCfg(r.data)
    }).catch(() => {})
  }, [])

  // Bidirektionale Sync: tp_pct ↔ tp_px, sl_pct ↔ sl_px
  const setTpPct = (v) => {
    const pct = parseFloat(v)
    setForm(f => ({ ...f, tp_pct: v, tp_px: trimPxBySymbol(tpPctToPx(entry, p.side, pct), p.symbol, coinMeta) }))
  }
  const setTpPx = (v) => {
    const px = parseFloat(v)
    setForm(f => ({ ...f, tp_px: v, tp_pct: Number(tpPxToPct(entry, p.side, px).toFixed(2)) }))
  }
  const setSlPct = (v) => {
    const pct = parseFloat(v)
    setForm(f => ({ ...f, sl_pct: v, sl_px: trimPxBySymbol(slPctToPx(entry, p.side, pct), p.symbol, coinMeta) }))
  }
  const setSlPx = (v) => {
    const px = parseFloat(v)
    setForm(f => ({ ...f, sl_px: v, sl_pct: Number(slPxToPct(entry, p.side, px).toFixed(2)) }))
  }

  const lev = Number(form.leverage) || 1
  const tpPnL = (Number(form.tp_pct) || 0) * lev
  const slPnL = (Number(form.sl_pct) || 0) * lev

  const submit = async () => {
    setSubmitting(true); setErr(null)
    try {
      const payload = {
        leverage: parseInt(form.leverage),
        size_usd: parseFloat(form.size_usd),
        tp_px: parseFloat(form.tp_px),
        sl_px: parseFloat(form.sl_px),
        use_trailing: !!form.use_trailing,
        trailing_stop_pct: parseFloat(form.trailing_stop_pct),
      }
      await api.post(`/api/v1/predictor/predictions/${p.id}/order`, payload)
      onSuccess && onSuccess()
      onClose()
    } catch (e) {
      setErr(e.response?.data?.detail || 'Order fehlgeschlagen')
    }
    setSubmitting(false)
  }

  return (
    <div className="absolute inset-0 bg-black/70 flex items-center justify-center z-50">
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-4 w-[560px] max-h-[90vh] overflow-auto">
        <div className="flex items-center justify-between mb-3">
          <div className="font-semibold flex items-center gap-2">
            <span className={isLong ? 'text-green-400' : 'text-red-400'}>
              {isLong ? <TrendingUp size={14} className="inline" /> : <TrendingDown size={14} className="inline" />}
              {' '}{p.symbol} {p.side}
            </span>
            <span className="text-gray-500 text-xs">@ {fmtPx(p.entry_px, p.symbol, coinMeta)}</span>
          </div>
          <button onClick={onClose} className="p-1 hover:bg-gray-700 rounded"><X size={16} /></button>
        </div>

        <div className="text-xs text-gray-500 mb-3">
          Regel: <span className="text-gray-300">{p.rule_name}</span>
        </div>

        {err && <div className="bg-red-900/40 text-red-300 p-2 rounded mb-2 text-sm">{err}</div>}

        <div className="grid grid-cols-2 gap-3 text-xs">
          <Field label="Hebel">
            <select value={form.leverage} onChange={e => setForm({ ...form, leverage: parseInt(e.target.value) })}
              className="w-full bg-gray-800 px-2 py-1.5 rounded border border-gray-600">
              {[1,2,3,4,5,6,7,8,9,10].map(v => <option key={v} value={v}>{v}x</option>)}
            </select>
          </Field>
          <Field label="Trade-Size ($)">
            <input type="number" min="10" step="5" value={form.size_usd}
              onChange={e => setForm({ ...form, size_usd: parseFloat(e.target.value) })}
              className="w-full bg-gray-800 px-2 py-1.5 rounded border border-gray-600" />
          </Field>

          <Field label="Take-Profit % (vor Hebel)">
            <input type="number" step="0.1" value={form.tp_pct}
              onChange={e => setTpPct(e.target.value)}
              className="w-full bg-gray-800 px-2 py-1.5 rounded border border-gray-600 font-mono" />
            <div className="text-[10px] text-gray-500 mt-0.5">= +{tpPnL.toFixed(1)}% PnL bei {lev}x</div>
          </Field>
          <Field label="Take-Profit ($)">
            <input type="number" step="any" value={form.tp_px}
              onChange={e => setTpPx(e.target.value)}
              className="w-full bg-gray-800 px-2 py-1.5 rounded border border-gray-600 font-mono" />
            <div className="text-[10px] text-gray-500 mt-0.5">&nbsp;</div>
          </Field>

          <Field label="Stop-Loss % (vor Hebel)">
            <input type="number" step="0.1" value={form.sl_pct}
              onChange={e => setSlPct(e.target.value)}
              className="w-full bg-gray-800 px-2 py-1.5 rounded border border-gray-600 font-mono" />
            <div className="text-[10px] text-gray-500 mt-0.5">= -{slPnL.toFixed(1)}% PnL bei {lev}x</div>
          </Field>
          <Field label="Stop-Loss ($)">
            <input type="number" step="any" value={form.sl_px}
              onChange={e => setSlPx(e.target.value)}
              className="w-full bg-gray-800 px-2 py-1.5 rounded border border-gray-600 font-mono" />
            <div className="text-[10px] text-gray-500 mt-0.5">&nbsp;</div>
          </Field>

          <Field label="Trailing-Stop">
            <label className="flex items-center gap-2 mt-1.5">
              <input type="checkbox" checked={form.use_trailing}
                onChange={e => setForm({ ...form, use_trailing: e.target.checked })} />
              <input type="number" min="0.1" max="20" step="0.1" value={form.trailing_stop_pct}
                disabled={!form.use_trailing}
                onChange={e => setForm({ ...form, trailing_stop_pct: parseFloat(e.target.value) })}
                className="w-20 bg-gray-800 px-2 py-1 rounded border border-gray-600 disabled:opacity-40" />
              <span className="text-gray-500">%</span>
            </label>
          </Field>
        </div>

        <div className="mt-4 flex justify-end gap-2">
          <button onClick={onClose} className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-sm">Abbrechen</button>
          <button onClick={submit} disabled={submitting}
            className="px-4 py-1.5 bg-blue-600 hover:bg-blue-500 rounded text-sm flex items-center gap-2 disabled:opacity-50">
            <Send size={14} /> {submitting ? 'Sende...' : 'Order platzieren'}
          </button>
        </div>
      </div>
    </div>
  )
}

function AutoTradeModal({ onClose, onSaved }) {
  const [cfg, setCfg] = useState(null)
  const [err, setErr] = useState(null)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    api.get('/api/v1/predictor/config').then(r => setCfg(r.data)).catch(e => setErr(e.message))
  }, [])

  if (!cfg) {
    return (
      <div className="absolute inset-0 bg-black/70 flex items-center justify-center z-50">
        <div className="bg-gray-900 border border-gray-700 rounded-lg p-4 w-[600px]">
          {err ? <div className="text-red-400">{err}</div> : <div className="text-gray-400">Lade...</div>}
          <div className="mt-3 flex justify-end"><button onClick={onClose} className="px-3 py-1 bg-gray-700 rounded text-sm">Schließen</button></div>
        </div>
      </div>
    )
  }

  const t = cfg.trading || {}

  const set = (key, value) => {
    const c = JSON.parse(JSON.stringify(cfg))
    if (!c.trading) c.trading = {}
    c.trading[key] = value
    setCfg(c)
  }

  const submit = async () => {
    setSaving(true); setErr(null)
    try {
      await api.put('/api/v1/predictor/config', { config: cfg })
      onSaved && onSaved()
      onClose()
    } catch (e) {
      setErr(e.response?.data?.detail || 'Speichern fehlgeschlagen')
    }
    setSaving(false)
  }

  const usePred = t.use_predictor_targets !== false  // default true
  const split = !!t.long_short_split
  const longEnabled = t.auto_long_enabled !== false  // default true
  const shortEnabled = t.auto_short_enabled !== false  // default true

  return (
    <div className="absolute inset-0 bg-black/70 flex items-center justify-center z-50">
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-4 w-[640px] max-h-[90vh] overflow-auto">
        <div className="flex items-center justify-between mb-3">
          <div className="font-semibold flex items-center gap-2"><Bot size={16} /> Auto-Trade</div>
          <button onClick={onClose} className="p-1 hover:bg-gray-700 rounded"><X size={16} /></button>
        </div>
        {err && <div className="bg-red-900/40 text-red-300 p-2 rounded mb-2 text-sm">{err}</div>}

        <Section title="Aktivierung">
          <Field label="Auto-Trade aktiv">
            <input type="checkbox" checked={!!t.auto_trade}
              onChange={e => set('auto_trade', e.target.checked)} />
          </Field>
          <Field label="Hebel">
            <NumInput value={t.default_leverage ?? 5} step="1" min="1" max="20" onChange={v => set('default_leverage', parseInt(v))} />
          </Field>
          <Field label="Trade-Size ($)">
            <NumInput value={t.default_size_usd ?? 20} step="5" min="10" max="10000" onChange={v => set('default_size_usd', parseFloat(v))} />
          </Field>
          <Field label="Long handeln">
            <input type="checkbox" checked={longEnabled}
              onChange={e => set('auto_long_enabled', e.target.checked)} />
          </Field>
          <Field label="Short handeln">
            <input type="checkbox" checked={shortEnabled}
              onChange={e => set('auto_short_enabled', e.target.checked)} />
          </Field>
        </Section>

        <Section title="Take-Profit / Stop-Loss">
          <Field label="Nach Predictor-Vorgaben">
            <input type="checkbox" checked={usePred}
              onChange={e => set('use_predictor_targets', e.target.checked)} />
          </Field>
          <div></div>
          {!usePred && (
            <>
              <Field label="Long/Short getrennt">
                <input type="checkbox" checked={split}
                  onChange={e => set('long_short_split', e.target.checked)} />
              </Field>
              <div></div>
              {!split && (
                <>
                  <Field label="TP % (vor Hebel)">
                    <NumInput value={t.global_tp_pct ?? 3.0} step="0.1" min="0.1" max="50" onChange={v => set('global_tp_pct', parseFloat(v))} />
                  </Field>
                  <Field label="SL % (vor Hebel)">
                    <NumInput value={t.global_sl_pct ?? 1.5} step="0.1" min="0.1" max="50" onChange={v => set('global_sl_pct', parseFloat(v))} />
                  </Field>
                </>
              )}
              {split && (
                <>
                  <Field label="TP % Long">
                    <NumInput value={t.global_tp_pct_long ?? 3.0} step="0.1" min="0.1" max="50" onChange={v => set('global_tp_pct_long', parseFloat(v))} />
                  </Field>
                  <Field label="SL % Long">
                    <NumInput value={t.global_sl_pct_long ?? 1.5} step="0.1" min="0.1" max="50" onChange={v => set('global_sl_pct_long', parseFloat(v))} />
                  </Field>
                  <Field label="TP % Short">
                    <NumInput value={t.global_tp_pct_short ?? 3.0} step="0.1" min="0.1" max="50" onChange={v => set('global_tp_pct_short', parseFloat(v))} />
                  </Field>
                  <Field label="SL % Short">
                    <NumInput value={t.global_sl_pct_short ?? 1.5} step="0.1" min="0.1" max="50" onChange={v => set('global_sl_pct_short', parseFloat(v))} />
                  </Field>
                </>
              )}
            </>
          )}
        </Section>

        <div className="text-[10px] text-gray-500 mb-3 px-1">
          Hinweis: Auto-Trade greift bei jeder neuen Predictor-Prediction. Hebel und Size gelten für alle Auto-Trades.
          {usePred && ' TP/SL kommen aus der jeweiligen Predictor-Action.'}
        </div>

        <div className="mt-4 flex justify-end gap-2">
          <button onClick={onClose} className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-sm">Abbrechen</button>
          <button onClick={submit} disabled={saving}
            className="px-4 py-1.5 bg-blue-600 hover:bg-blue-500 rounded text-sm flex items-center gap-2 disabled:opacity-50">
            <Save size={14} /> {saving ? 'Speichere...' : 'Speichern'}
          </button>
        </div>
      </div>
    </div>
  )
}

function Field({ label, children }) {
  return (
    <div>
      <label className="text-gray-500 text-[10px]">{label}</label>
      {children}
    </div>
  )
}
