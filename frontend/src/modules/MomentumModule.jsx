import React, { useState, useEffect, useCallback } from 'react'
import { Play, Square, TrendingUp, TrendingDown, Settings, RefreshCw, Trash2, AlertCircle, CheckCircle, XCircle, Clock, Loader2, ArrowUp, ArrowDown, Rocket, X, EyeOff, Eye, BarChart2 } from 'lucide-react'
import api from '../utils/api'
import { useSearchStore } from '../stores/searchStore'

const STATUS_ICONS = {
  active: <Clock size={13} style={{ color: '#3b82f6' }} />,
  hit_tp: <CheckCircle size={13} style={{ color: '#22c55e' }} />,
  hit_sl: <XCircle size={13} style={{ color: '#ef4444' }} />,
  invalidated: <AlertCircle size={13} style={{ color: '#f59e0b' }} />,
  expired: <Clock size={13} style={{ color: '#6b7280' }} />
}

const COLUMNS = [
  { key: 'status',          label: 'Status',  width: 50,  align: 'center' },
  { key: 'direction',       label: 'Dir',     width: 32,  align: 'center' },
  { key: 'symbol',          label: 'Symbol',  width: 80,  align: 'left' },
  { key: 'entry_price',     label: 'Entry',   width: 76,  align: 'right' },
  { key: 'current_price',   label: 'Kurs',    width: 76,  align: 'right' },
  { key: 'current_pct',     label: 'P/L',     width: 54,  align: 'right' },
  { key: 'take_profit_pct', label: 'TP%',     width: 44,  align: 'right' },
  { key: 'stop_loss_pct',   label: 'SL%',     width: 44,  align: 'right' },
  { key: 'confidence',      label: 'Conf',    width: 38,  align: 'right' },
  { key: 'max_favorable',   label: 'Max',     width: 48,  align: 'right' },
  { key: 'detected_at',     label: 'Erkannt', width: 72,  align: 'right' },
  { key: 'actions',         label: '',        width: 48,  align: 'center', sortable: false },
]

export default function MomentumModule() {
  const { selectEvents, setPrehistoryMinutes } = useSearchStore()
  const [tab, setTab] = useState('predictions')
  const [config, setConfig] = useState(null)
  const [predictions, setPredictions] = useState([])
  const [totalPredictions, setTotalPredictions] = useState(0)
  const [stats, setStats] = useState({})
  const [activePredictions, setActivePredictions] = useState(0)
  const [loading, setLoading] = useState(false)
  const [statusFilter, setStatusFilter] = useState('active')
  const [hideShort, setHideShort] = useState(false)
  const [hideTradedFilter, setHideTradedFilter] = useState(true)
  const [showSettings, setShowSettings] = useState(false)
  const [editConfig, setEditConfig] = useState({})
  const [groups, setGroups] = useState([])
  const [sortCol, setSortCol] = useState('detected_at')
  const [sortDir, setSortDir] = useState('desc')
  const [lastUpdate, setLastUpdate] = useState(null)
  const [showOptLog, setShowOptLog] = useState(false)
  const [optLog, setOptLog] = useState([])
  const [tradeDialog, setTradeDialog] = useState(null)
  const [tradeLoading, setTradeLoading] = useState(false)
  const [tradeResult, setTradeResult] = useState(null)
  const [statsDirection, setStatsDirection] = useState('')
  const [expandedId, setExpandedId] = useState(null)
  const [statsDrill, setStatsDrill] = useState(null) // z.B. {period:'24h',dir:'long',type:'tp'}
  const [resolvedPreds, setResolvedPreds] = useState([])
  const [tradeStats, setTradeStats] = useState({})
  // 2h Scanner State
  const [config2h, setConfig2h] = useState(null)
  const [editConfig2h, setEditConfig2h] = useState({})
  const [predictions2h, setPredictions2h] = useState([])
  const [totalPredictions2h, setTotalPredictions2h] = useState(0)
  const [stats2h, setStats2h] = useState({})
  const [activePredictions2h, setActivePredictions2h] = useState(0)
  const [tradeStats2h, setTradeStats2h] = useState({})
  const [showSettings2h, setShowSettings2h] = useState(false)

  const loadConfig = useCallback(async () => {
    try {
      const res = await api.get('/api/v1/momentum/config')
      setConfig(res.data)
      setEditConfig(res.data)
    } catch (err) { console.error('Config load failed:', err) }
  }, [])

  const loadConfig2h = useCallback(async () => {
    try {
      const res = await api.get('/api/v1/momentum/config/2h')
      setConfig2h(res.data)
      setEditConfig2h(res.data)
    } catch (err) { console.error('Config 2h load failed:', err) }
  }, [])

  const loadPredictions = useCallback(async () => {
    try {
      const params = { limit: 200 }
      if (statusFilter) params.status = statusFilter
      if (hideShort) params.direction = 'long'
      if (hideTradedFilter) params.hide_traded = true
      const res = await api.get('/api/v1/momentum/predictions', { params })
      setPredictions(res.data.predictions || [])
      setTotalPredictions(res.data.total || 0)
      setLastUpdate(new Date())
    } catch (err) { console.error('Predictions load failed:', err) }
  }, [statusFilter, hideShort, hideTradedFilter])

  const loadPredictions2h = useCallback(async () => {
    try {
      const params = { limit: 200, scanner_type: 'cnn_2h' }
      if (statusFilter) params.status = statusFilter
      if (hideShort) params.direction = 'long'
      if (hideTradedFilter) params.hide_traded = true
      const res = await api.get('/api/v1/momentum/predictions', { params })
      setPredictions2h(res.data.predictions || [])
      setTotalPredictions2h(res.data.total || 0)
    } catch (err) { console.error('Predictions 2h load failed:', err) }
  }, [statusFilter, hideShort, hideTradedFilter])

  const loadStats = useCallback(async () => {
    try {
      const res = await api.get('/api/v1/momentum/stats')
      setStats(res.data.stats || {})
      setActivePredictions(res.data.active_predictions || 0)
      setTradeStats(res.data.trade_stats || {})
    } catch (err) { console.error('Stats load failed:', err) }
  }, [])

  const loadStats2h = useCallback(async () => {
    try {
      const res = await api.get('/api/v1/momentum/stats', { params: { scanner_type: 'cnn_2h' } })
      setStats2h(res.data.stats || {})
      setActivePredictions2h(res.data.active_predictions || 0)
      setTradeStats2h(res.data.trade_stats || {})
    } catch (err) { console.error('Stats 2h load failed:', err) }
  }, [])

  const loadGroups = useCallback(async () => {
    try {
      const res = await api.get('/api/v1/groups')
      setGroups(res.data.groups || res.data || [])
    } catch (err) { console.error('Groups load failed:', err) }
  }, [])

  const loadResolvedPreds = useCallback(async () => {
    try {
      const res = await api.get('/api/v1/momentum/predictions', { params: { limit: 500 } })
      setResolvedPreds((res.data.predictions || []).filter(p => p.status !== 'active'))
    } catch (err) { console.error('Resolved load failed:', err) }
  }, [])

  const loadOptLog = useCallback(async () => {
    try {
      const res = await api.get('/api/v1/momentum/optimizations')
      setOptLog(res.data || [])
    } catch (err) { console.error('OptLog load failed:', err) }
  }, [])

  useEffect(() => {
    loadConfig(); loadConfig2h(); loadPredictions(); loadPredictions2h(); loadStats(); loadStats2h(); loadGroups()
    const interval = setInterval(() => { loadPredictions(); loadPredictions2h(); loadStats(); loadStats2h() }, 15000)
    return () => clearInterval(interval)
  }, [loadConfig, loadConfig2h, loadPredictions, loadPredictions2h, loadStats, loadStats2h, loadGroups])

  useEffect(() => { loadPredictions(); loadPredictions2h() }, [statusFilter, hideShort, hideTradedFilter])

  useEffect(() => { if (tab === 'stats') loadResolvedPreds() }, [tab, loadResolvedPreds])

  const toggleScanner = async () => {
    if (!config) return
    setLoading(true)
    try {
      const res = await api.put('/api/v1/momentum/config', { is_active: !config.is_active })
      setConfig(res.data); setEditConfig(res.data)
    } catch (err) { console.error('Toggle failed:', err) }
    setLoading(false)
  }

  const toggleScanner2h = async () => {
    if (!config2h) return
    setLoading(true)
    try {
      const res = await api.put('/api/v1/momentum/config/2h', { is_active: !config2h.is_active })
      setConfig2h(res.data); setEditConfig2h(res.data)
    } catch (err) { console.error('Toggle 2h failed:', err) }
    setLoading(false)
  }

  const saveConfig = async () => {
    setLoading(true)
    try {
      const updates = {}
      for (const k of ['idle_seconds','min_target_pct','stop_loss_pct','min_confidence','scan_all_symbols','coin_group_id','tp_sl_mode','fixed_tp_pct','fixed_sl_pct','long_fixed_tp_pct','long_fixed_sl_pct','short_fixed_tp_pct','short_fixed_sl_pct','range_tp_min','range_tp_max','range_sl_min','range_sl_max']) {
        if (editConfig[k] !== config[k]) updates[k] = editConfig[k]
      }
      if (Object.keys(updates).length) {
        const res = await api.put('/api/v1/momentum/config', updates)
        setConfig(res.data); setEditConfig(res.data)
      }
      setShowSettings(false)
    } catch (err) { console.error('Save failed:', err) }
    setLoading(false)
  }

  const saveConfig2h = async () => {
    setLoading(true)
    try {
      const updates = {}
      for (const k of ['idle_seconds','min_confidence','scan_all_symbols','coin_group_id','tp_sl_mode','long_fixed_tp_pct','long_fixed_sl_pct','short_fixed_tp_pct','short_fixed_sl_pct']) {
        if (editConfig2h[k] !== config2h[k]) updates[k] = editConfig2h[k]
      }
      if (Object.keys(updates).length) {
        const res = await api.put('/api/v1/momentum/config/2h', updates)
        setConfig2h(res.data); setEditConfig2h(res.data)
      }
      setShowSettings2h(false)
    } catch (err) { console.error('Save 2h failed:', err) }
    setLoading(false)
  }

  const showInChart = (p) => {
    const detected = new Date(p.detected_at)
    const resolved = p.resolved_at ? new Date(p.resolved_at) : new Date()
    const durationMin = Math.max(Math.round((resolved - detected) / 60000), 10)
    const eventStart = detected.toISOString().replace('T', ' ').replace(/\.\d+Z$/, '')
    selectEvents([{
      id: `pred_${p.prediction_id}`,
      symbol: p.symbol,
      event_start: eventStart,
      duration_minutes: durationMin,
      change_percent: p.actual_result_pct || p.current_pct || 0,
    }])
    setPrehistoryMinutes(Math.max(durationMin, 60))
  }

  const cancelPrediction = async (id) => {
    try { await api.delete(`/api/v1/momentum/predictions/${id}`); loadPredictions(); loadStats() }
    catch (err) { console.error('Cancel failed:', err) }
  }

  const openTradeDialog = (prediction) => {
    setTradeDialog({
      prediction_id: prediction.prediction_id,
      symbol: prediction.symbol,
      direction: prediction.direction,
      entry_price: prediction.entry_price,
      current_price: prediction.current_price,
      take_profit_pct: Number(prediction.take_profit_pct),
      stop_loss_pct: Number(prediction.stop_loss_pct),
      confidence: prediction.confidence,
    })
    setTradeResult(null)
  }

  const executeTrade = async () => {
    if (!tradeDialog) return
    setTradeLoading(true)
    setTradeResult(null)
    try {
      const res = await api.post(`/api/v1/momentum/trade/${tradeDialog.prediction_id}`, {
        take_profit_pct: tradeDialog.take_profit_pct,
        stop_loss_pct: tradeDialog.stop_loss_pct,
      })
      setTradeResult({ success: true, data: res.data })
      loadPredictions()
    } catch (err) {
      const msg = err.response?.data?.detail || err.response?.data?.error || err.message
      setTradeResult({ success: false, error: msg })
    }
    setTradeLoading(false)
  }

  const handleSort = (key) => {
    if (key === 'actions') return
    if (sortCol === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortCol(key); setSortDir('asc') }
  }

  const currentPredictions = tab === 'predictions_2h' ? predictions2h : predictions
  const sortedPredictions = [...currentPredictions].sort((a, b) => {
    let va = a[sortCol], vb = b[sortCol]
    if (sortCol === 'symbol') { va = va || ''; vb = vb || ''; return sortDir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va) }
    if (sortCol === 'detected_at') { va = new Date(va || 0).getTime(); vb = new Date(vb || 0).getTime() }
    if (sortCol === 'status') { const order = { active: 0, hit_tp: 1, hit_sl: 2, invalidated: 3, expired: 4 }; va = order[va] ?? 5; vb = order[vb] ?? 5 }
    if (va == null) va = -Infinity; if (vb == null) vb = -Infinity
    return sortDir === 'asc' ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1)
  })

  const formatPrice = (p) => {
    if (p == null) return '-'
    const n = Number(p)
    if (n < 0.01) return n.toFixed(8)
    if (n < 1) return n.toFixed(6)
    if (n < 100) return n.toFixed(4)
    return n.toFixed(2)
  }
  const formatPct = (p) => p != null ? `${p >= 0 ? '+' : ''}${Number(p).toFixed(1)}%` : '-'
  const formatTime = (t) => t ? new Date(t).toLocaleString('de-DE', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' }) : '-'

  const s = {
    label: { fontSize: '0.6875rem', color: 'var(--color-muted)' },
    value: { fontSize: '0.8125rem', fontWeight: 600 },
    input: { width: '100%', padding: '4px 8px', fontSize: '0.75rem', background: 'var(--color-bg)', border: '1px solid var(--color-border)', borderRadius: 4, color: 'var(--color-text)' },
    select: { width: '100%', padding: '4px 8px', fontSize: '0.75rem', background: 'var(--color-bg)', border: '1px solid var(--color-border)', borderRadius: 4, color: 'var(--color-text)' },
    btn: { padding: '4px 10px', fontSize: '0.75rem', borderRadius: 4, border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4 },
    tab: (active) => ({ padding: '4px 10px', fontSize: '0.75rem', cursor: 'pointer', border: 'none', borderBottom: active ? '2px solid #3b82f6' : '2px solid transparent', background: 'none', color: active ? 'var(--color-text)' : 'var(--color-muted)', fontWeight: active ? 600 : 400 }),
    th: { padding: '3px 4px', fontSize: '0.5625rem', fontWeight: 600, color: 'var(--color-muted)', textTransform: 'uppercase', letterSpacing: '0.03em', cursor: 'pointer', userSelect: 'none', whiteSpace: 'nowrap', borderBottom: '1px solid var(--color-border)' },
    td: { padding: '3px 4px', fontSize: '0.6875rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' },
  }

  const SortIcon = ({ col }) => {
    if (sortCol !== col) return null
    return sortDir === 'asc' ? <ArrowUp size={9} style={{ marginLeft: 1 }} /> : <ArrowDown size={9} style={{ marginLeft: 1 }} />
  }

  const renderCell = (p, col) => {
    switch (col.key) {
      case 'status': return <span style={{ display: 'flex', justifyContent: 'center' }}>{STATUS_ICONS[p.status]}{p.traded && <span title="Gehandelt" style={{ marginLeft: 2, fontSize: '0.5rem', color: '#f59e0b' }}>$</span>}</span>
      case 'direction': return p.direction === 'long'
        ? <TrendingUp size={13} style={{ color: '#22c55e' }} />
        : <TrendingDown size={13} style={{ color: '#ef4444' }} />
      case 'symbol': return <span style={{ fontWeight: 600 }}>{p.symbol.replace('USDC', '')}</span>
      case 'entry_price': return <span style={{ color: 'var(--color-muted)' }}>{formatPrice(p.entry_price)}</span>
      case 'current_price': return <span style={{ fontWeight: 500 }}>{formatPrice(p.current_price)}</span>
      case 'current_pct': {
        const isResolved = p.status && p.status !== 'active'
        const v = isResolved ? (p.actual_result_pct ?? p.current_pct) : p.current_pct
        if (v == null) return '-'
        const color = v > 0 ? '#22c55e' : v < 0 ? '#ef4444' : 'var(--color-muted)'
        return <span style={{ color, fontWeight: 600 }}>{formatPct(v)}</span>
      }
      case 'take_profit_pct': return <span style={{ color: '#22c55e' }}>{Number(p.take_profit_pct).toFixed(1)}</span>
      case 'stop_loss_pct': return <span style={{ color: '#ef4444' }}>{Number(p.stop_loss_pct).toFixed(1)}</span>
      case 'confidence': return <span>{p.confidence}</span>
      case 'max_favorable': {
        const mf = p.max_favorable_pct || p.peak_pct
        if (!mf || p.status === 'active') return <span style={{ color: 'var(--color-muted)' }}>-</span>
        const dd = p.max_adverse_pct
        const sl = p.correction_data?.optimal_sl
        const slGain = p.correction_data?.optimal_gain
        const tip = [
          `Peak: ${Number(mf).toFixed(1)}%`,
          dd ? `Drawdown: ${Number(dd).toFixed(1)}%` : null,
          sl ? `Optimal: SL ${sl}% → ${slGain}%` : null
        ].filter(Boolean).join('\n')
        return <span title={tip} style={{ cursor: 'help', fontWeight: 500 }}>
          <span style={{ color: '#3b82f6' }}>{Number(mf).toFixed(1)}%</span>
          {dd ? <span style={{ color: '#ef4444', fontSize: '0.5rem', marginLeft: 2 }}>{Number(dd).toFixed(0)}</span> : null}
        </span>
      }
      case 'detected_at': return <span style={{ color: 'var(--color-muted)', fontSize: '0.5625rem' }}>{formatTime(p.detected_at)}</span>
      case 'actions': return (
        <span style={{ display: 'flex', gap: 4, justifyContent: 'center' }}>
          <button onClick={(e) => { e.stopPropagation(); showInChart(p) }} title="Im Chart anzeigen"
            style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}>
            <BarChart2 size={11} style={{ color: '#8b5cf6' }} />
          </button>
          {p.status === 'active' && p.direction === 'long' && !p.traded && (
            <button onClick={() => openTradeDialog(p)} title="An Tradebot übergeben"
              style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}>
              <Rocket size={11} style={{ color: '#3b82f6' }} />
            </button>
          )}
          {p.status === 'active' && (
            <button onClick={() => cancelPrediction(p.prediction_id)} title="Löschen"
              style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}>
              <Trash2 size={11} style={{ color: '#ef4444' }} />
            </button>
          )}
        </span>
      )
      default: return '-'
    }
  }

  // Stats nach direction filtern
  const getStatsKey = (period) => statsDirection ? `${statsDirection}_${period}` : period
  const currentStats = tab === 'stats' ? stats : stats2h
  const currentTradeStats = tab === 'stats' ? tradeStats : tradeStats2h
  const currentActive = tab === 'predictions_2h' ? activePredictions2h : activePredictions
  const currentConfig = tab === 'predictions_2h' ? config2h : config

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: 5, fontSize: '0.75rem' }}>
      
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {tab !== 'predictions_2h' ? (
            <>
              <button onClick={toggleScanner} disabled={loading}
                style={{ ...s.btn, background: config?.is_active ? '#ef4444' : '#22c55e', color: '#fff' }}>
                {loading ? <Loader2 size={14} className="spin" /> : config?.is_active ? <><Square size={14} /> Stop</> : <><Play size={14} /> Start</>}
              </button>
              <span style={{ fontSize: '0.625rem', color: config?.is_active ? '#22c55e' : 'var(--color-muted)' }}>
                {config?.is_active ? `Scannt (${activePredictions} aktiv)` : 'Gestoppt'}
              </span>
            </>
          ) : (
            <>
              <button onClick={toggleScanner2h} disabled={loading}
                style={{ ...s.btn, background: config2h?.is_active ? '#ef4444' : '#22c55e', color: '#fff' }}>
                {loading ? <Loader2 size={14} className="spin" /> : config2h?.is_active ? <><Square size={14} /> Stop</> : <><Play size={14} /> Start</>}
              </button>
              <span style={{ fontSize: '0.625rem', color: config2h?.is_active ? '#22c55e' : 'var(--color-muted)' }}>
                {config2h?.is_active ? `2h-Scan (${activePredictions2h} aktiv)` : '2h Gestoppt'}
              </span>
            </>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          {lastUpdate && <span style={{ fontSize: '0.5625rem', color: 'var(--color-muted)' }}>{lastUpdate.toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>}
          <button onClick={() => setHideShort(!hideShort)} title={hideShort ? 'Short anzeigen' : 'Short ausblenden'}
            style={{ ...s.btn, background: hideShort ? '#f59e0b' : 'var(--color-bg)', color: hideShort ? '#fff' : 'var(--color-text)' }}>
            {hideShort ? <Eye size={12} /> : <EyeOff size={12} />}
            <span style={{ fontSize: '0.5625rem' }}>S</span>
          </button>
          <button onClick={() => { loadPredictions(); loadPredictions2h(); loadStats(); loadStats2h() }} style={{ ...s.btn, background: 'var(--color-bg)' }}><RefreshCw size={12} /></button>
          {tab === 'predictions_2h'
            ? <button onClick={() => setShowSettings2h(!showSettings2h)} style={{ ...s.btn, background: showSettings2h ? '#3b82f6' : 'var(--color-bg)', color: showSettings2h ? '#fff' : 'var(--color-text)' }}><Settings size={12} /></button>
            : <button onClick={() => setShowSettings(!showSettings)} style={{ ...s.btn, background: showSettings ? '#3b82f6' : 'var(--color-bg)', color: showSettings ? '#fff' : 'var(--color-text)' }}><Settings size={12} /></button>
          }
        </div>
      </div>

      {/* Settings */}
      {showSettings && (
        <div style={{ padding: 8, background: 'var(--color-bg)', borderRadius: 6, border: '1px solid var(--color-border)', display: 'flex', flexDirection: 'column', gap: 6 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
            <div><div style={s.label}>Min. Confidence</div><input type="number" value={editConfig.min_confidence || 60} onChange={e => setEditConfig({...editConfig, min_confidence: parseInt(e.target.value)})} style={s.input} /></div>
            <div><div style={s.label}>Idle (Sek.)</div><input type="number" value={editConfig.idle_seconds || 60} onChange={e => setEditConfig({...editConfig, idle_seconds: parseInt(e.target.value)})} style={s.input} /></div>
            <div style={{ gridColumn: '1/-1' }}>
              <div style={s.label}>Coin-Gruppe</div>
              <select value={editConfig.scan_all_symbols ? 'all' : (editConfig.coin_group_id || 'all')}
                onChange={e => { if (e.target.value === 'all') setEditConfig({...editConfig, scan_all_symbols: true, coin_group_id: null}); else setEditConfig({...editConfig, scan_all_symbols: false, coin_group_id: parseInt(e.target.value)}) }} style={s.select}>
                <option value="all">Alle Symbole</option>
                {groups.map(g => <option key={g.id || g.group_id} value={g.id || g.group_id}>{g.name}</option>)}
              </select>
            </div>
            <div style={{ gridColumn: '1/-1', borderTop: '1px solid var(--color-border)', paddingTop: 6, marginTop: 2 }}>
              <div style={s.label}>TP/SL Modus</div>
              <div style={{ display: 'flex', gap: 4, marginTop: 3 }}>
                {['dynamic', 'fixed', 'range'].map(m => (
                  <button key={m} onClick={() => setEditConfig({...editConfig, tp_sl_mode: m})}
                    style={{ ...s.btn, fontSize: '0.5625rem', padding: '2px 8px', background: editConfig.tp_sl_mode === m ? '#3b82f6' : 'var(--color-surface)', color: editConfig.tp_sl_mode === m ? '#fff' : 'var(--color-text)' }}>
                    {m === 'dynamic' ? 'Dynamisch (ATR)' : m === 'fixed' ? 'Fest' : 'Bereich'}
                  </button>
                ))}
              </div>
            </div>
            {editConfig.tp_sl_mode === 'fixed' && <>
              <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:'4px'}}>
                <div><div style={s.label}>Long TP %</div><input type="number" step="0.5" value={editConfig.long_fixed_tp_pct || editConfig.fixed_tp_pct || 2} onChange={e => setEditConfig({...editConfig, long_fixed_tp_pct: parseFloat(e.target.value)})} style={s.input} /></div>
                <div><div style={s.label}>Long SL %</div><input type="number" step="0.5" value={editConfig.long_fixed_sl_pct || editConfig.fixed_sl_pct || 2} onChange={e => setEditConfig({...editConfig, long_fixed_sl_pct: parseFloat(e.target.value)})} style={s.input} /></div>
                <div><div style={s.label}>Short TP %</div><input type="number" step="0.5" value={editConfig.short_fixed_tp_pct || editConfig.fixed_tp_pct || 2} onChange={e => setEditConfig({...editConfig, short_fixed_tp_pct: parseFloat(e.target.value)})} style={s.input} /></div>
                <div><div style={s.label}>Short SL %</div><input type="number" step="0.5" value={editConfig.short_fixed_sl_pct || editConfig.fixed_sl_pct || 2} onChange={e => setEditConfig({...editConfig, short_fixed_sl_pct: parseFloat(e.target.value)})} style={s.input} /></div>
              </div>
            </>}
            {editConfig.tp_sl_mode === 'range' && <>
              <div><div style={s.label}>TP Min %</div><input type="number" step="0.5" value={editConfig.range_tp_min || 5} onChange={e => setEditConfig({...editConfig, range_tp_min: parseFloat(e.target.value)})} style={s.input} /></div>
              <div><div style={s.label}>TP Max %</div><input type="number" step="0.5" value={editConfig.range_tp_max || 15} onChange={e => setEditConfig({...editConfig, range_tp_max: parseFloat(e.target.value)})} style={s.input} /></div>
              <div><div style={s.label}>SL Min %</div><input type="number" step="0.5" value={editConfig.range_sl_min || 2} onChange={e => setEditConfig({...editConfig, range_sl_min: parseFloat(e.target.value)})} style={s.input} /></div>
              <div><div style={s.label}>SL Max %</div><input type="number" step="0.5" value={editConfig.range_sl_max || 6} onChange={e => setEditConfig({...editConfig, range_sl_max: parseFloat(e.target.value)})} style={s.input} /></div>
            </>}
            {editConfig.tp_sl_mode === 'dynamic' && <>
              <div><div style={s.label}>Min. Target %</div><input type="number" step="0.5" value={editConfig.min_target_pct || 5} onChange={e => setEditConfig({...editConfig, min_target_pct: parseFloat(e.target.value)})} style={s.input} /></div>
              <div><div style={s.label}>Max SL %</div><input type="number" step="0.5" value={editConfig.stop_loss_pct || 2} onChange={e => setEditConfig({...editConfig, stop_loss_pct: parseFloat(e.target.value)})} style={s.input} /></div>
            </>}
          </div>
          <button onClick={saveConfig} disabled={loading} style={{ ...s.btn, background: '#3b82f6', color: '#fff', justifyContent: 'center' }}>Speichern</button>
          <div style={{ borderTop: '1px solid var(--color-border)', paddingTop: 6, marginTop: 2 }}>
            <button onClick={() => { setShowOptLog(!showOptLog); if (!showOptLog) loadOptLog() }}
              style={{ ...s.btn, fontSize: '0.5625rem', background: 'var(--color-surface)', color: 'var(--color-muted)', width: '100%', justifyContent: 'center' }}>
              {showOptLog ? 'Optimierungen ausblenden' : 'Optimierungen anzeigen'}
            </button>
            {showOptLog && (
              <div style={{ marginTop: 6, display: 'flex', flexDirection: 'column', gap: 4, maxHeight: 250, overflow: 'auto' }}>
                {optLog.length === 0 ? (
                  <div style={{ fontSize: '0.625rem', color: 'var(--color-muted)', textAlign: 'center', padding: 8 }}>
                    Noch keine Optimierungen. Läuft täglich um 08:00 und 20:00.
                  </div>
                ) : optLog.map(o => (
                  <div key={o.optimization_id} style={{ padding: 6, background: 'var(--color-surface)', borderRadius: 4, border: '1px solid var(--color-border)', fontSize: '0.625rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                      <span style={{ fontWeight: 600 }}>{new Date(o.run_at).toLocaleString('de-DE', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })}</span>
                      <span style={{ padding: '0 4px', borderRadius: 3, fontSize: '0.5625rem', fontWeight: 600,
                        background: o.recommendation === 'apply' ? 'rgba(34,197,94,0.15)' : 'rgba(107,114,128,0.15)',
                        color: o.recommendation === 'apply' ? '#22c55e' : 'var(--color-muted)' }}>
                        {o.applied ? 'ANGEWENDET' : o.recommendation === 'apply' ? 'EMPFOHLEN' : 'KEINE ÄNDERUNG'}
                      </span>
                    </div>
                    <div style={{ color: 'var(--color-muted)', lineHeight: 1.4 }}>
                      <div>{o.total_predictions} Predictions: {o.total_tp} TP, {o.total_sl} SL, {o.total_expired} Exp → <span style={{ fontWeight: 600 }}>{o.current_hit_rate}%</span></div>
                      {o.best_variant && (
                        <div style={{ marginTop: 2 }}>Beste: <span style={{ color: '#3b82f6' }}>{o.best_variant.label || JSON.stringify(o.best_variant)}</span> → <span style={{ fontWeight: 600, color: '#22c55e' }}>{o.best_sim_hit_rate}%</span></div>
                      )}
                      {o.changes_applied && Object.keys(o.changes_applied).length > 0 && (
                        <div style={{ marginTop: 2, color: '#f59e0b' }}>Änderungen: {Object.entries(o.changes_applied).map(([k, v]) => `${k}: ${v.old} → ${v.new}`).join(', ')}</div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* 2h Settings */}
      {showSettings2h && tab === 'predictions_2h' && (
        <div style={{ padding: 8, background: 'var(--color-bg)', borderRadius: 6, border: '1px solid var(--color-border)', display: 'flex', flexDirection: 'column', gap: 6 }}>
          <div style={{ fontWeight: 600, fontSize: '0.6875rem', color: '#f59e0b' }}>2h-Scanner Einstellungen</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
            <div><div style={s.label}>Min. Confidence</div><input type="number" value={editConfig2h.min_confidence || 60} onChange={e => setEditConfig2h({...editConfig2h, min_confidence: parseInt(e.target.value)})} style={s.input} /></div>
            <div><div style={s.label}>Idle (Sek.)</div><input type="number" value={editConfig2h.idle_seconds || 60} onChange={e => setEditConfig2h({...editConfig2h, idle_seconds: parseInt(e.target.value)})} style={s.input} /></div>
            <div><div style={s.label}>Long TP %</div><input type="number" step="0.5" value={editConfig2h.long_fixed_tp_pct || 2} onChange={e => setEditConfig2h({...editConfig2h, long_fixed_tp_pct: parseFloat(e.target.value)})} style={s.input} /></div>
            <div><div style={s.label}>Long SL %</div><input type="number" step="0.5" value={editConfig2h.long_fixed_sl_pct || 2} onChange={e => setEditConfig2h({...editConfig2h, long_fixed_sl_pct: parseFloat(e.target.value)})} style={s.input} /></div>
            <div><div style={s.label}>Short TP %</div><input type="number" step="0.5" value={editConfig2h.short_fixed_tp_pct || 2} onChange={e => setEditConfig2h({...editConfig2h, short_fixed_tp_pct: parseFloat(e.target.value)})} style={s.input} /></div>
            <div><div style={s.label}>Short SL %</div><input type="number" step="0.5" value={editConfig2h.short_fixed_sl_pct || 2} onChange={e => setEditConfig2h({...editConfig2h, short_fixed_sl_pct: parseFloat(e.target.value)})} style={s.input} /></div>
          </div>
          <button onClick={saveConfig2h} disabled={loading} style={{ ...s.btn, background: '#3b82f6', color: '#fff', justifyContent: 'center' }}>Speichern</button>
        </div>
      )}

      {/* Stats Bar */}
      {stats[getStatsKey('all')] && (
        <div style={{ display: 'flex', gap: 8, padding: '3px 0', flexWrap: 'wrap', alignItems: 'center' }}>
          <div><span style={s.label}>Gesamt: </span><span style={s.value}>{stats[getStatsKey('all')].total_predictions}</span></div>
          <div><span style={s.label}>Treffer: </span><span style={{ ...s.value, color: '#22c55e' }}>{stats[getStatsKey('all')].hit_rate_pct}%</span></div>
          <div><span style={s.label}>Ø Result: </span><span style={{ ...s.value, color: (stats[getStatsKey('all')].avg_result_pct || 0) >= 0 ? '#22c55e' : '#ef4444' }}>{formatPct(stats[getStatsKey('all')].avg_result_pct)}</span></div>
          <div><span style={s.label}>Best: </span><span style={{ ...s.value, color: '#22c55e' }}>{formatPct(stats[getStatsKey('all')].best_result_pct)}</span></div>
        </div>
      )}

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 0, borderBottom: '1px solid var(--color-border)' }}>
        <button onClick={() => setTab('predictions')} style={s.tab(tab === 'predictions')}>Scanner ({totalPredictions})</button>
        <button onClick={() => setTab('predictions_2h')} style={s.tab(tab === 'predictions_2h')}>2h ({totalPredictions2h})</button>
        <button onClick={() => setTab('stats')} style={s.tab(tab === 'stats')}>Statistik</button>
      </div>

      {/* Filter */}
      {(tab === 'predictions' || tab === 'predictions_2h') && (
        <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
          {['', 'active', 'hit_tp', 'hit_sl', 'invalidated', 'expired'].map(st => (
            <button key={st} onClick={() => setStatusFilter(st)}
              style={{ ...s.btn, background: statusFilter === st ? '#3b82f6' : 'var(--color-bg)', color: statusFilter === st ? '#fff' : 'var(--color-text)', fontSize: '0.5625rem', padding: '2px 5px' }}>
              {st === '' ? 'Alle' : st === 'hit_tp' ? 'TP ✓' : st === 'hit_sl' ? 'SL ✗' : st === 'active' ? 'Aktiv' : st === 'invalidated' ? 'Invalid.' : 'Expired'}
            </button>
          ))}
        </div>
      )}



      {/* Content */}
      <div style={{ flex: 1, overflow: 'auto' }}>
        {(tab === 'predictions' || tab === 'predictions_2h') && (
          sortedPredictions.length === 0
            ? <div style={{ textAlign: 'center', color: 'var(--color-muted)', padding: 20 }}>Keine Predictions</div>
            : <table style={{ width: '100%', borderCollapse: 'collapse', tableLayout: 'fixed' }}>
                <thead>
                  <tr>
                    {COLUMNS.map(col => (
                      <th key={col.key} onClick={() => col.sortable !== false && handleSort(col.key)}
                        style={{ ...s.th, width: col.width, textAlign: col.align, cursor: col.sortable === false ? 'default' : 'pointer' }}>
                        <span style={{ display: 'inline-flex', alignItems: 'center' }}>{col.label}<SortIcon col={col.key} /></span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sortedPredictions.map(p => {
                    const isExp = expandedId === p.prediction_id
                    const sl_sims = p.correction_data?.sl_simulations
                    const hasDetail = p.status !== 'active' && (p.max_favorable_pct || p.peak_pct)
                    return (
                      <React.Fragment key={p.prediction_id}>
                        <tr style={{ borderBottom: isExp ? 'none' : '1px solid var(--color-border)', opacity: p.traded ? 0.6 : 1, cursor: hasDetail ? 'pointer' : 'default' }}
                          onClick={() => hasDetail && setExpandedId(isExp ? null : p.prediction_id)}
                          onMouseEnter={e => e.currentTarget.style.background = 'var(--color-bg)'}
                          onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                          {COLUMNS.map(col => (
                            <td key={col.key} style={{ ...s.td, textAlign: col.align, width: col.width }}>{renderCell(p, col)}</td>
                          ))}
                        </tr>
                        {isExp && (
                          <tr style={{ borderBottom: '1px solid var(--color-border)' }}>
                            <td colSpan={COLUMNS.length} style={{ padding: '6px 8px', background: 'var(--color-bg)', fontSize: '0.625rem' }}>
                              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 6, marginBottom: sl_sims ? 6 : 0 }}>
                                <div>
                                  <span style={s.label}>Peak</span><br/>
                                  <span style={{ color: '#3b82f6', fontWeight: 600 }}>{formatPct(p.max_favorable_pct || p.peak_pct)}</span>
                                </div>
                                <div>
                                  <span style={s.label}>Drawdown</span><br/>
                                  <span style={{ color: '#ef4444', fontWeight: 600 }}>{formatPct(p.max_adverse_pct || p.trough_pct)}</span>
                                </div>
                                <div>
                                  <span style={s.label}>Result</span><br/>
                                  <span style={{ color: (p.actual_result_pct||0) >= 0 ? '#22c55e' : '#ef4444', fontWeight: 600 }}>{formatPct(p.actual_result_pct)}</span>
                                  <span style={{ marginLeft: 4, color: 'var(--color-muted)' }}>{p.status === 'hit_tp' ? 'TP' : p.status === 'hit_sl' ? 'SL' : p.status}</span>
                                </div>
                                <div>
                                  <span style={s.label}>Optimal SL</span><br/>
                                  {p.correction_data?.optimal_sl 
                                    ? <span style={{ color: '#f59e0b', fontWeight: 600 }}>{p.correction_data.optimal_sl}% → {formatPct(p.correction_data.optimal_gain)}</span>
                                    : <span style={{ color: 'var(--color-muted)' }}>-</span>}
                                </div>
                              </div>
                              {sl_sims && (
                                <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
                                  {Object.entries(sl_sims).sort((a,b) => parseFloat(a[0]) - parseFloat(b[0])).map(([sl, sim]) => {
                                    const isOpt = parseFloat(sl) === p.correction_data?.optimal_sl
                                    return (
                                      <span key={sl} style={{ 
                                        padding: '2px 5px', borderRadius: 3, fontSize: '0.5625rem',
                                        background: isOpt ? 'rgba(245,158,11,0.2)' : sim.stopped ? 'rgba(239,68,68,0.1)' : 'rgba(34,197,94,0.1)',
                                        border: `1px solid ${isOpt ? '#f59e0b' : sim.stopped ? '#ef444433' : '#22c55e33'}`,
                                        color: sim.stopped && !isOpt ? '#ef4444' : 'var(--color-text)'
                                      }}>
                                        SL {sl}%: {sim.stopped ? <span style={{ color: '#ef4444' }}>gestoppt</span> : <span style={{ color: '#22c55e' }}>+{sim.max_gain}%</span>}
                                      </span>
                                    )
                                  })}
                                </div>
                              )}
                              {p.traded && <div style={{ marginTop: 4, color: '#f59e0b', fontSize: '0.5625rem' }}>💰 Gehandelt</div>}
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    )
                  })}
                </tbody>
              </table>
        )}

        {tab === 'stats' && (() => {
          // Filter resolved predictions fuer drill-down
          const drillPreds = statsDrill ? resolvedPreds.filter(p => {
            if (statsDrill.dir && p.direction !== statsDrill.dir) return false
            const now = new Date()
            if (statsDrill.period === '24h' && (now - new Date(p.detected_at)) > 86400000) return false
            if (statsDrill.period === '7d' && (now - new Date(p.detected_at)) > 7*86400000) return false
            if (statsDrill.period === '30d' && (now - new Date(p.detected_at)) > 30*86400000) return false
            if (statsDrill.type === 'tp') return p.status === 'hit_tp'
            if (statsDrill.type === 'sl') return p.status === 'hit_sl'
            if (statsDrill.type === 'all') return ['hit_tp','hit_sl','expired','invalidated'].includes(p.status)
            return true
          }) : []

          const Clk = ({children, drill, style: st2}) => (
            <span onClick={(e) => { e.stopPropagation(); setStatsDrill(prev => prev && prev.dir === drill.dir && prev.period === drill.period && prev.type === drill.type ? null : drill) }}
              style={{ cursor: 'pointer', textDecoration: 'underline', textDecorationStyle: 'dotted', textUnderlineOffset: 2, ...st2 }}>
              {children}
            </span>
          )

          const Row = ({data, label, color, icon, dir, period}) => {
            if (!data || !data.total_predictions) return null
            const rPct = data.avg_result_pct || 0
            const miss = data.total_predictions - (data.correct_predictions || 0)
            return (
              <div style={{ display: 'grid', gridTemplateColumns: '48px 1fr 1fr 1fr 1fr', gap: 4, alignItems: 'center', padding: '3px 0', fontSize: '0.625rem' }}>
                <Clk drill={{period, dir, type:'all'}} style={{ fontWeight: 600, color }}>{icon} {label}</Clk>
                <span><Clk drill={{period, dir, type:'all'}} style={{}}>{data.total_predictions}</Clk> <span style={s.label}>pred</span></span>
                <span>
                  <Clk drill={{period, dir, type:'tp'}} style={{ color: '#22c55e', fontWeight: 600 }}>{data.correct_predictions || 0}</Clk>
                  <span style={s.label}> TP </span>
                  <Clk drill={{period, dir, type:'sl'}} style={{ color: '#ef4444', fontWeight: 600 }}>{miss}</Clk>
                  <span style={s.label}> SL</span>
                </span>
                <span style={{ color: rPct >= 0 ? '#22c55e' : '#ef4444' }}>{formatPct(rPct)} <span style={s.label}>avg</span></span>
                <span><span style={{ color: '#22c55e' }}>{formatPct(data.best_result_pct)}</span> / <span style={{ color: '#ef4444' }}>{formatPct(data.worst_result_pct)}</span></span>
              </div>
            )
          }

          const isDrillFor = (period, dir) => statsDrill && statsDrill.period === period && (!dir || statsDrill.dir === dir)

          const PredCard = ({p}) => {
            const isTP = p.status === 'hit_tp'
            const borderColor = isTP ? 'rgba(34,197,94,0.3)' : 'rgba(239,68,68,0.3)'
            const bgColor = isTP ? 'rgba(34,197,94,0.05)' : 'rgba(239,68,68,0.05)'
            return (
              <div onClick={() => showInChart(p)} style={{ padding: '4px 6px', marginBottom: 3, background: bgColor, borderRadius: 4, border: `1px solid ${borderColor}`, fontSize: '0.5625rem', cursor: 'pointer' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '70px 1fr 1fr 1fr 1fr 52px', gap: 4, alignItems: 'center' }}>
                  <span style={{ fontWeight: 700 }}>
                    <span style={{ color: isTP ? '#22c55e' : '#ef4444', marginRight: 3 }}>{isTP ? 'TP' : 'SL'}</span>
                    {p.symbol.replace('USDC','')}{p.traded ? ' $' : ''}
                  </span>
                  <span>
                    <span style={s.label}>Result </span>
                    <span style={{ color: (p.actual_result_pct||0) >= 0 ? '#22c55e' : '#ef4444', fontWeight: 600 }}>{formatPct(p.actual_result_pct)}</span>
                  </span>
                  <span>
                    <span style={s.label}>Peak </span>
                    <span style={{ color: '#3b82f6', fontWeight: 600 }}>{formatPct(p.max_favorable_pct || p.peak_pct)}</span>
                  </span>
                  <span>
                    <span style={s.label}>DD </span>
                    <span style={{ color: '#ef4444' }}>{formatPct(p.max_adverse_pct || p.trough_pct)}</span>
                  </span>
                  <span>
                    {p.correction_data?.optimal_sl
                      ? <><span style={s.label}>Opt </span><span style={{ color: '#f59e0b' }}>SL {p.correction_data.optimal_sl}%→{formatPct(p.correction_data.optimal_gain)}</span></>
                      : <span style={{ color: 'var(--color-muted)' }}>-</span>}
                  </span>
                  <span style={{ color: 'var(--color-muted)', textAlign: 'right' }}>{formatTime(p.resolved_at || p.detected_at)}</span>
                </div>
              </div>
            )
          }

          return (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, padding: 4 }}>
              {/* Trade P/L */}
              {Object.keys(tradeStats).length > 0 && (
                <div style={{ padding: '6px 8px', background: 'var(--color-bg)', borderRadius: 6, border: '1px solid var(--color-border)' }}>
                  <div style={{ fontWeight: 700, fontSize: '0.75rem', marginBottom: 4 }}>Trades (echtes Geld)</div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6, fontSize: '0.625rem' }}>
                    {['7d', '30d', 'all'].map(p => {
                      const ts = tradeStats[p]
                      if (!ts || !ts.trades) return null
                      const pnlColor = ts.realized_pnl >= 0 ? '#22c55e' : '#ef4444'
                      return (
                        <div key={p} style={{ textAlign: 'center' }}>
                          <div style={{ fontWeight: 600, marginBottom: 2 }}>{p === 'all' ? 'Gesamt' : p}</div>
                          <div><span style={s.label}>{ts.trades} Trades</span></div>
                          <div style={{ color: pnlColor, fontWeight: 700, fontSize: '0.75rem' }}>{ts.realized_pnl >= 0 ? '+' : ''}{ts.realized_pnl.toFixed(2)} $</div>
                          <div style={{ fontSize: '0.5625rem', color: 'var(--color-muted)' }}>Buy {ts.total_buy.toFixed(0)} / Sell {ts.total_sell.toFixed(0)}</div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}
              {['24h', '7d', '30d', 'all'].map(period => {
                const st = stats[getStatsKey(period)]
                const stL = stats[`long_${period}`]
                const stS = stats[`short_${period}`]
                if (!st) return <div key={period} style={{ color: 'var(--color-muted)', fontSize: '0.6875rem' }}>{period}: Keine Daten</div>
                return (
                  <div key={period} style={{ padding: '6px 8px', background: 'var(--color-bg)', borderRadius: 6, border: '1px solid var(--color-border)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2, cursor: 'pointer' }}
                      onClick={() => setStatsDrill(prev => prev && prev.period === period && !prev.dir ? null : {period, dir: null, type:'all'})}>
                      <span style={{ fontWeight: 700, fontSize: '0.75rem' }}>{period === 'all' ? 'Gesamt' : period}</span>
                      <span style={{ fontSize: '0.5625rem', color: 'var(--color-muted)' }}>{st.total_predictions} Pred | <span style={{ color: '#22c55e' }}>{st.hit_rate_pct}%</span> | Ø {formatPct(st.avg_result_pct)}</span>
                    </div>
                    <Row data={stL} label="Long" color="#22c55e" icon="▲" dir="long" period={period} />
                    <Row data={stS} label="Short" color="#ef4444" icon="▼" dir="short" period={period} />
                    {isDrillFor(period) && drillPreds.length > 0 && (
                      <div style={{ marginTop: 4, borderTop: '1px solid var(--color-border)', paddingTop: 4 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                          <span style={{ fontSize: '0.5625rem', fontWeight: 600 }}>
                            {statsDrill.dir ? (statsDrill.dir === 'long' ? '▲ Long' : '▼ Short') : 'Alle'}
                            {statsDrill.type === 'tp' ? ' Treffer' : statsDrill.type === 'sl' ? ' Verfehlt' : ''} ({drillPreds.length})
                          </span>
                          <span onClick={() => setStatsDrill(null)} style={{ cursor: 'pointer', fontSize: '0.5625rem', color: 'var(--color-muted)' }}>✕</span>
                        </div>
                        {drillPreds.sort((a,b) => new Date(b.detected_at) - new Date(a.detected_at)).map(p => (
                          <PredCard key={p.prediction_id} p={p} />
                        ))}
                      </div>
                    )}
                    {isDrillFor(period) && drillPreds.length === 0 && (
                      <div style={{ marginTop: 4, fontSize: '0.5625rem', color: 'var(--color-muted)' }}>Keine Predictions für diesen Filter</div>
                    )}
                  </div>
                )
              })}
            </div>
          )
        })()}
      </div>

      {/* Trade Dialog */}
      {tradeDialog && (
        <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.6)', zIndex: 9999, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
          onClick={(e) => { if (e.target === e.currentTarget && !tradeLoading) setTradeDialog(null) }}>
          <div style={{ background: 'var(--color-surface)', border: '1px solid var(--color-border)', borderRadius: 8, padding: 16, width: 320, maxWidth: '90vw' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <span style={{ fontWeight: 700, fontSize: '0.875rem' }}>
                <Rocket size={14} style={{ marginRight: 6, color: '#3b82f6' }} />
                Trade: {tradeDialog.symbol.replace('USDC', '')}
              </span>
              <button onClick={() => !tradeLoading && setTradeDialog(null)} style={{ background: 'none', border: 'none', cursor: 'pointer' }}><X size={16} /></button>
            </div>

            {tradeResult?.success ? (
              <div style={{ padding: 12, background: 'rgba(34,197,94,0.1)', borderRadius: 6, border: '1px solid rgba(34,197,94,0.3)' }}>
                <div style={{ color: '#22c55e', fontWeight: 700, marginBottom: 8 }}>Trade ausgeführt!</div>
                <div style={{ fontSize: '0.75rem', display: 'flex', flexDirection: 'column', gap: 3 }}>
                  <div>Gekauft: <b>{tradeResult.data.sell_qty}</b> @ <b>{tradeResult.data.buy_price}</b></div>
                  <div>Amount: <b>{tradeResult.data.amount_usdt} USDC</b></div>
                  <div style={{ color: '#22c55e' }}>TP: {tradeResult.data.tp_price} ({tradeResult.data.tp_pct}%)</div>
                  <div style={{ color: '#ef4444' }}>SL: {tradeResult.data.sl_price} ({tradeResult.data.sl_pct}%)</div>
                </div>
                <button onClick={() => setTradeDialog(null)} style={{ ...s.btn, background: 'var(--color-bg)', justifyContent: 'center', width: '100%', marginTop: 10 }}>Schließen</button>
              </div>
            ) : (
              <>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 12 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
                    <span style={s.label}>Entry (aktuell)</span>
                    <span style={{ fontWeight: 600 }}>{formatPrice(tradeDialog.current_price || tradeDialog.entry_price)}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
                    <span style={s.label}>Confidence</span>
                    <span style={{ fontWeight: 600 }}>{tradeDialog.confidence}</span>
                  </div>
                  <div>
                    <div style={s.label}>Take Profit %</div>
                    <input type="number" step="0.5" min="0.5" value={tradeDialog.take_profit_pct}
                      onChange={e => setTradeDialog({...tradeDialog, take_profit_pct: parseFloat(e.target.value) || 0})} style={s.input} />
                  </div>
                  <div>
                    <div style={s.label}>Stop Loss %</div>
                    <input type="number" step="0.5" min="0.5" value={tradeDialog.stop_loss_pct}
                      onChange={e => setTradeDialog({...tradeDialog, stop_loss_pct: parseFloat(e.target.value) || 0})} style={s.input} />
                  </div>
                </div>

                {tradeResult?.error && (
                  <div style={{ padding: 8, background: 'rgba(239,68,68,0.1)', borderRadius: 4, border: '1px solid rgba(239,68,68,0.3)', fontSize: '0.6875rem', color: '#ef4444', marginBottom: 8 }}>
                    {tradeResult.error}
                  </div>
                )}

                <div style={{ display: 'flex', gap: 6 }}>
                  <button onClick={() => setTradeDialog(null)} disabled={tradeLoading}
                    style={{ ...s.btn, flex: 1, justifyContent: 'center', background: 'var(--color-bg)' }}>Abbrechen</button>
                  <button onClick={executeTrade} disabled={tradeLoading}
                    style={{ ...s.btn, flex: 1, justifyContent: 'center', background: '#22c55e', color: '#fff', fontWeight: 700 }}>
                    {tradeLoading ? <Loader2 size={14} className="spin" /> : <><Rocket size={14} /> Kaufen</>}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
