import { useState, useEffect, useCallback } from 'react'
import { Wallet, RefreshCw, TrendingUp, TrendingDown, AlertCircle, X, Edit2, Trash2, Bot, DollarSign, Check, Tag, Filter, ArrowRightLeft, Loader2, BarChart2 } from 'lucide-react'
import api from '../../utils/api'
import { useSearchStore } from '../../stores/searchStore'

export default function WalletModule() {
  const { selectEvents, setPrehistoryMinutes } = useSearchStore()
  const [status, setStatus] = useState({ configured: false })
  const [hlStatus, setHlStatus] = useState({ configured: false })
  const [balance, setBalance] = useState(null)
  const [hlBalance, setHlBalance] = useState(null)
  const [positions, setPositions] = useState([])
  const [hlPositions, setHlPositions] = useState([])
  const [timeoutHours, setTimeoutHours] = useState(2)
  const [orders, setOrders] = useState([])
  const [hlOrders, setHlOrders] = useState([])
  const [paperWallet, setPaperWallet] = useState(null)
  const [paperPositions, setPaperPositions] = useState([])
  const [history, setHistory] = useState([])
  const [realizedPnl, setRealizedPnl] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('positions')
  const [posSource, setPosSource] = useState('hyperliquid')
  const [ordSource, setOrdSource] = useState('hyperliquid')
  const [editingOrder, setEditingOrder] = useState(null)
  const [creatingOrder, setCreatingOrder] = useState(null)
  const [hideWithOrders, setHideWithOrders] = useState(false)
  const [convertDialog, setConvertDialog] = useState(false)
  const [convertAmount, setConvertAmount] = useState('')
  const [convertLoading, setConvertLoading] = useState(false)
  const [convertResult, setConvertResult] = useState(null)
  const [selectedCoins, setSelectedCoins] = useState(new Set())
  const [closing, setClosing] = useState(false)
  const [histSort, setHistSort] = useState({ col: 'executed_at', asc: false })
  const [posSort, setPosSort] = useState({ col: '', asc: false })
  const [quickTPs, setQuickTPs] = useState([0.5, 1, 2, 3])
  const [editingQuickTPs, setEditingQuickTPs] = useState(false)
  const [newQuickTP, setNewQuickTP] = useState('')
  const [tpUpdating, setTpUpdating] = useState(false)
  const [tpMsg, setTpMsg] = useState(null)

  const loadData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [statusRes, hlStatusRes] = await Promise.all([
        api.get('/api/v1/wallet/status'),
        api.get('/api/v1/wallet/hl/status')
      ])
      setStatus(statusRes.data)
      setHlStatus(hlStatusRes.data)

      const promises = []
      if (statusRes.data.configured) {
        promises.push(
          api.get('/api/v1/wallet/balance'), api.get('/api/v1/wallet/positions'),
          api.get('/api/v1/wallet/orders')
        )
      } else {
        promises.push(null, null, null)
      }
      // History + PnL immer laden (enthält RL-Trades, nicht nur Binance)
      promises.push(
        api.get('/api/v1/wallet/history?days=30&limit=500'),
        api.get('/api/v1/wallet/realized-pnl?days=7')
      )
      if (hlStatusRes.data.configured) {
        promises.push(
          api.get('/api/v1/wallet/hl/balance'), api.get('/api/v1/wallet/hl/positions'),
          api.get('/api/v1/wallet/hl/orders')
        )
      } else {
        promises.push(null, null, null)
      }

      const results = await Promise.all(promises.map(p => p || Promise.resolve(null)))
      const [balRes, posRes, ordRes, histRes, pnlRes, hlBalRes, hlPosRes, hlOrdRes] = results

      if (balRes) {
        if (balRes.data.error) setError(balRes.data.error)
        else setBalance(balRes.data)
      }
      setPositions(posRes?.data?.positions || [])
      setOrders(ordRes?.data?.orders || [])
      setHistory(histRes?.data?.trades || [])
      setRealizedPnl(pnlRes?.data || null)
      if (hlBalRes) setHlBalance(hlBalRes.data?.error ? null : hlBalRes.data)
      setHlPositions(hlPosRes?.data?.positions || [])
      setTimeoutHours(hlPosRes?.data?.timeout_hours ?? 2)
      setHlOrders(hlOrdRes?.data?.orders || [])
      // Paper-Wallet (virtuelle Engine) — immer laden, leicht
      try {
        const [pw, pp] = await Promise.all([
          api.get('/api/v1/predictor/paper/wallet'),
          api.get('/api/v1/predictor/paper/positions?scope=open'),
        ])
        setPaperWallet(pw.data || null)
        setPaperPositions(pp.data?.positions || [])
      } catch { /* paper optional */ }
    } catch (err) { setError('Fehler beim Laden') }
    finally { setLoading(false) }
  }, [])

  useEffect(() => {
    loadData()
    let inFlight = false
    const tick = async () => {
      if (inFlight) return
      inFlight = true
      try { await loadData() } finally { inFlight = false }
    }
    const i = setInterval(tick, 1000)
    return () => clearInterval(i)
  }, [loadData])

  useEffect(() => {
    api.get('/api/v1/wallet/hl/quick-tp-percentages')
      .then(r => { if (Array.isArray(r.data?.values)) setQuickTPs(r.data.values) })
      .catch(() => {})
  }, [])

  const cancelOrder = async (symbol, orderId) => {
    if (!confirm(`Order ${orderId} stornieren?`)) return
    try { const r = await api.delete(`/api/v1/wallet/orders/${symbol}/${orderId}`); r.data.error ? alert(r.data.error) : loadData() }
    catch { alert('Fehler') }
  }
  const updateOrder = async () => {
    if (!editingOrder) return
    try {
      await api.delete(`/api/v1/wallet/orders/${editingOrder.symbol}/${editingOrder.orderId}`)
      const r = await api.post('/api/v1/wallet/orders', { symbol: editingOrder.symbol, side: 'SELL', type: 'LIMIT', price: parseFloat(editingOrder.newPrice), quantity: editingOrder.quantity })
      r.data.error ? alert(r.data.error) : (setEditingOrder(null), loadData())
    } catch { alert('Fehler') }
  }
  const createSellOrder = async () => {
    if (!creatingOrder) return
    try {
      const r = await api.post('/api/v1/wallet/orders', { symbol: creatingOrder.symbol, side: 'SELL', type: 'LIMIT', price: parseFloat(creatingOrder.price), quantity: parseFloat(creatingOrder.quantity) })
      r.data.error ? alert(r.data.error) : (setCreatingOrder(null), loadData())
    } catch { alert('Fehler') }
  }
  const getOrderForPosition = (posSymbol) => {
    const base = posSymbol.replace(/USD[TC]$/, '')
    return orders.find(o => o.symbol.replace(/USD[TC]$/, '') === base && o.side === 'SELL')
  }

  const toggleCoin = (coin) => {
    setSelectedCoins(prev => {
      const next = new Set(prev)
      next.has(coin) ? next.delete(coin) : next.add(coin)
      return next
    })
  }

  const closeSelected = async () => {
    if (selectedCoins.size === 0) return
    if (!confirm(`${selectedCoins.size} Position(en) schließen?`)) return
    setClosing(true)
    try {
      const r = await api.post('/api/v1/wallet/hl/close', { coins: [...selectedCoins] })
      const failed = (r.data.results || []).filter(x => !x.success)
      if (failed.length > 0) alert('Fehler: ' + failed.map(f => f.coin + ': ' + (f.error || '?')).join(', '))
      setSelectedCoins(new Set())
    } catch (e) { alert('Close fehlgeschlagen') }
    setClosing(false)
  }

  const toggleTimeout = async (coin, side, currentEnabled) => {
    try {
      await api.post(`/api/v1/wallet/hl/positions/${coin}/${side}/timeout`,
                     { enabled: !currentEnabled })
      setHlPositions(prev => prev.map(p =>
        (p.coin === coin && p.direction === side) ? { ...p, timeout_enabled: !currentEnabled } : p))
    } catch (e) { alert('Timeout-Toggle fehlgeschlagen') }
  }

  const fmtHM = (iso) => {
    if (!iso) return ''
    try { const d = new Date(iso); return d.toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit' }) }
    catch { return '' }
  }

  const toggleHistSort = (col) => {
    setHistSort(prev => prev.col === col ? { col, asc: !prev.asc } : { col, asc: false })
  }
  const togglePosSort = (col) => {
    setPosSort(prev => prev.col === col ? { col, asc: !prev.asc } : { col, asc: false })
  }
  const posSortIcon = (col) => posSort.col === col ? (posSort.asc ? ' ▲' : ' ▼') : ''
  const sortedHlPositions = posSort.col ? [...hlPositions].sort((a, b) => {
    const { col, asc } = posSort
    const get = (p) => col === 'pnl_pct'
      ? (p.roe_percent ?? (p.margin_used > 0 ? p.unrealized_pnl / p.margin_used * 100 : 0))
      : p[col]
    let va = get(a), vb = get(b)
    if (va == null) va = ''
    if (vb == null) vb = ''
    if (typeof va === 'string') return asc ? va.localeCompare(vb) : vb.localeCompare(va)
    return asc ? va - vb : vb - va
  }) : hlPositions
  const allHlSelected = sortedHlPositions.length > 0 && sortedHlPositions.every(p => selectedCoins.has(p.coin))
  const someHlSelected = sortedHlPositions.some(p => selectedCoins.has(p.coin))
  const toggleAllHl = () => {
    if (allHlSelected) setSelectedCoins(new Set())
    else setSelectedCoins(new Set(sortedHlPositions.map(p => p.coin)))
  }
  const applyTP = async (pct) => {
    if (selectedCoins.size === 0) return
    if (!confirm(`TP auf ${pct}% (vor Hebel) für ${selectedCoins.size} Position(en) setzen?`)) return
    setTpUpdating(true); setTpMsg(null)
    try {
      const r = await api.post('/api/v1/wallet/hl/update-tp', { coins: [...selectedCoins], tp_pct: pct })
      const ok = (r.data.results || []).filter(x => x.success).length
      const fail = (r.data.results || []).filter(x => !x.success)
      setTpMsg(`${ok} OK${fail.length ? `, ${fail.length} Fehler` : ''}`)
      if (fail.length) console.warn('TP-Update Fehler:', fail)
      loadData()
    } catch (e) { setTpMsg('Fehler: ' + (e.response?.data?.detail || e.message)) }
    setTpUpdating(false)
    setTimeout(() => setTpMsg(null), 4000)
  }
  const saveQuickTPs = async (next) => {
    try {
      const r = await api.put('/api/v1/wallet/hl/quick-tp-percentages', { values: next })
      if (r.data?.values) setQuickTPs(r.data.values)
    } catch {}
  }
  const addQuickTP = () => {
    const v = parseFloat(newQuickTP)
    if (!v || v <= 0) return
    const next = [...new Set([...quickTPs, Math.round(v * 100) / 100])].sort((a, b) => a - b)
    setNewQuickTP('')
    saveQuickTPs(next)
  }
  const removeQuickTP = (val) => saveQuickTPs(quickTPs.filter(v => v !== val))
  const sortedHistory = [...history].sort((a, b) => {
    const { col, asc } = histSort
    let va = a[col], vb = b[col]
    if (va == null) va = ''
    if (vb == null) vb = ''
    if (typeof va === 'string') return asc ? va.localeCompare(vb) : vb.localeCompare(va)
    return asc ? va - vb : vb - va
  })
  const sortIcon = (col) => histSort.col === col ? (histSort.asc ? ' \u25B2' : ' \u25BC') : ''

  const showPositionChart = (pos) => {
    const now = new Date()
    const hoursBack = 24
    const start = new Date(now.getTime() - hoursBack * 3600000)
    const eventStart = start.toISOString().replace('T', ' ').replace(/\.\d+Z$/, '')
    selectEvents([{
      id: `wallet_${pos.symbol}`,
      symbol: pos.symbol,
      event_start: eventStart,
      duration_minutes: hoursBack * 60,
      change_percent: pos.pnl_percent || 0,
    }])
    setPrehistoryMinutes(60)
  }

  const doConvert = async (amount) => {
    setConvertLoading(true)
    setConvertResult(null)
    try {
      const res = await api.post('/api/v1/wallet/convert-usdc', { amount: amount || null })
      if (res.data.error) setConvertResult({ error: res.data.error })
      else { setConvertResult(res.data); loadData() }
    } catch (err) { setConvertResult({ error: err.response?.data?.detail || err.message }) }
    setConvertLoading(false)
  }

  const fp = (v, d = 2) => { if (v == null) return '-'; let dec = d; if (d > 2 && Math.abs(v) > 0 && Math.abs(v) < 1) { const zeros = Math.max(0, Math.floor(-Math.log10(Math.abs(v)))); dec = Math.max(d, zeros + 3); } return new Intl.NumberFormat('de-DE', { minimumFractionDigits: dec, maximumFractionDigits: dec }).format(v) }
  const fPnl = (v) => v == null ? '-' : `${v >= 0 ? '+' : ''}$${fp(v)}`
  const fPct = (v) => v == null ? '-' : `${v >= 0 ? '+' : ''}${fp(v)}%`
  const fDate = (s) => s ? new Date(s).toLocaleDateString('de-DE', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' }) : '-'

  if (!status.configured && !hlStatus.configured && !loading) return (
    <div className="h-full flex flex-col items-center justify-center text-zinc-500 p-4">
      <Wallet className="w-8 h-8 mb-2 opacity-50" />
      <p className="text-xs">Kein API-Key konfiguriert</p>
    </div>
  )
  if (error) return (
    <div className="h-full flex flex-col items-center justify-center text-red-400 p-3">
      <AlertCircle className="w-8 h-8 mb-2" />
      <p className="text-xs">{error}</p>
      <button onClick={loadData} className="mt-2 px-3 py-1 bg-zinc-700 hover:bg-zinc-600 rounded text-xs">Erneut</button>
    </div>
  )

  const totalUPnl = positions.reduce((s, p) => s + (p.unrealized_pnl || 0), 0)
  const hlUPnl = hlPositions.reduce((s, p) => s + (p.unrealized_pnl || 0), 0)
  const binanceTotal = balance?.total_portfolio || 0
  const hlTotal = hlBalance?.equity ?? hlBalance?.account_value ?? 0
  const hlAvail = hlBalance?.available ?? 0
  const grandTotal = binanceTotal + hlTotal

  return (
    <div className="h-full flex flex-col overflow-hidden text-xs">
      {/* Balance Bar */}
      {(balance || hlBalance || paperWallet) && (
        <div className="flex items-center gap-3 px-2 py-1.5 border-b border-zinc-700/50 bg-zinc-800/30 flex-shrink-0 flex-wrap">
          {balance && (
            <>
              <div><span className="text-yellow-500 font-semibold">BIN</span> <span className="font-mono">${fp(binanceTotal)}</span></div>
              {balance.usdc_balance > 0 && (
                <div className="flex items-center gap-1">
                  <span className="text-zinc-500">USDC</span>
                  <span className="font-mono text-blue-300">${fp(balance.usdc_balance)}</span>
                  <button onClick={() => { setConvertDialog(true); setConvertAmount(''); setConvertResult(null) }}
                    className="ml-0.5 p-0.5 rounded hover:bg-zinc-600 text-blue-400" title="USDC → USDT konvertieren">
                    <ArrowRightLeft className="w-3 h-3" />
                  </button>
                </div>
              )}
              <div className={totalUPnl >= 0 ? 'text-green-400' : 'text-red-400'}>{fPnl(totalUPnl)}</div>
            </>
          )}
          {balance && hlBalance && <span className="text-zinc-600">|</span>}
          {hlBalance && (
            <>
              <div><span className="text-emerald-500 font-semibold">HL</span> <span className="font-mono">${fp(hlTotal)}</span></div>
              {hlAvail > 0 && (
                <div className="flex items-center gap-1">
                  <span className="text-zinc-500">Frei</span>
                  <span className="font-mono text-emerald-300">${fp(hlAvail)}</span>
                </div>
              )}
              {hlUPnl !== 0 && <div className={hlUPnl >= 0 ? 'text-green-400' : 'text-red-400'}>{fPnl(hlUPnl)}</div>}
            </>
          )}
          {paperWallet && (
            <>
              <span className="text-zinc-600">|</span>
              <div><span className="text-indigo-400 font-semibold">PAPER{paperWallet.paper_mode ? '' : '*'}</span> <span className="font-mono">${fp(paperWallet.balance)}</span></div>
              <div className={paperWallet.total_return_pct >= 0 ? 'text-green-400' : 'text-red-400'}>
                {paperWallet.total_return_pct >= 0 ? '+' : ''}{paperWallet.total_return_pct}%
              </div>
              <div className="text-zinc-500 text-[10px]">W/L/TO {paperWallet.wins}/{paperWallet.losses}/{paperWallet.timeouts}</div>
            </>
          )}
          {balance && hlBalance && <span className="text-zinc-600">|</span>}
          <div><span className="text-zinc-500">Gesamt</span> <span className="font-mono font-semibold">${fp(grandTotal)}</span></div>
          <div className={(realizedPnl?.realized_pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'}><span className="text-zinc-500">7d</span> {fPnl(realizedPnl?.realized_pnl)}</div>
          <span className="ml-auto flex items-center gap-1 text-[10px] text-zinc-500">
            <span className={`w-1.5 h-1.5 rounded-full ${loading ? 'bg-yellow-400 animate-pulse' : 'bg-green-400'}`}></span>
            live
          </span>
        </div>
      )}

      {/* Tabs */}
      <div className="flex items-center gap-1 px-2 py-1 border-b border-zinc-700/50 flex-shrink-0">
        {[
          { id: 'positions', label: `Pos (${posSource === 'binance' ? (hideWithOrders ? positions.filter(p => !getOrderForPosition(p.symbol)).length : positions.length) : posSource === 'paper' ? paperPositions.length : hlPositions.length})` },
          { id: 'orders', label: `Orders (${ordSource === 'binance' ? orders.length : hlOrders.length})` },
          { id: 'history', label: 'Hist' }
        ].map(t => (
          <button key={t.id} onClick={() => setActiveTab(t.id)}
            className={`px-2 py-0.5 rounded ${activeTab === t.id ? 'bg-blue-600 text-white' : 'text-zinc-400 hover:text-white'}`}>{t.label}</button>
        ))}
        {(activeTab === 'positions' || activeTab === 'orders') && hlStatus.configured && (
          <div className="ml-2 flex items-center gap-0.5 bg-zinc-800 rounded px-0.5">
            <button onClick={() => activeTab === 'positions' ? setPosSource('binance') : setOrdSource('binance')}
              className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${(activeTab === 'positions' ? posSource : ordSource) === 'binance' ? 'bg-yellow-600 text-white' : 'text-zinc-500 hover:text-zinc-300'}`}>BIN</button>
            <button onClick={() => activeTab === 'positions' ? setPosSource('hyperliquid') : setOrdSource('hyperliquid')}
              className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${(activeTab === 'positions' ? posSource : ordSource) === 'hyperliquid' ? 'bg-emerald-600 text-white' : 'text-zinc-500 hover:text-zinc-300'}`}>HL</button>
            {activeTab === 'positions' && (
              <button onClick={() => setPosSource('paper')}
                className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${posSource === 'paper' ? 'bg-indigo-600 text-white' : 'text-zinc-500 hover:text-zinc-300'}`}>PAPER</button>
            )}
          </div>
        )}
        {activeTab === 'positions' && posSource === 'binance' && (
          <button onClick={() => setHideWithOrders(!hideWithOrders)}
            className={`ml-auto p-1 rounded ${hideWithOrders ? 'text-blue-400' : 'text-zinc-500'}`}>
            <Filter className="w-3 h-3" />
          </button>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto">
        {activeTab === 'positions' && posSource === 'binance' && (
          <table className="w-full">
            <thead className="sticky top-0 bg-zinc-900">
              <tr className="text-zinc-500 text-left">
                <th className="px-2 py-1 font-normal">Symbol</th>
                <th className="px-2 py-1 font-normal text-right">Menge</th>
                <th className="px-2 py-1 font-normal text-right">Einstieg</th>
                <th className="px-2 py-1 font-normal text-right">Aktuell</th>
                <th className="px-2 py-1 font-normal text-right">Wert</th>
                <th className="px-2 py-1 font-normal text-right">P/L</th>
                <th className="px-2 py-1 font-normal w-8"></th>
              </tr>
            </thead>
            <tbody>
              {positions.filter(p => !hideWithOrders || !getOrderForPosition(p.symbol)).map((pos, i) => {
                const hasOrder = getOrderForPosition(pos.symbol)
                return (
                  <tr key={i} className="border-t border-zinc-800 hover:bg-zinc-800/50">
                    <td className="px-2 py-1 font-mono font-medium">
                      <button onClick={() => showPositionChart(pos)} className="text-purple-400 hover:text-purple-300 mr-1" title="Chart anzeigen">
                        <BarChart2 className="w-3 h-3 inline" />
                      </button>
                      {pos.symbol.replace('USDC','')}
                      {pos.is_bot_trade && <Bot className="w-3 h-3 text-blue-400 inline ml-1" />}
                      {hasOrder && <Tag className="w-3 h-3 text-yellow-400 inline ml-1" />}
                    </td>
                    <td className="px-2 py-1 text-right font-mono">{fp(pos.quantity, 4)}</td>
                    <td className="px-2 py-1 text-right font-mono">${fp(pos.avg_entry_price, 4)}</td>
                    <td className="px-2 py-1 text-right font-mono">${fp(pos.current_price, 4)}</td>
                    <td className="px-2 py-1 text-right font-mono">${fp(pos.value_usdt)}</td>
                    <td className={`px-2 py-1 text-right font-mono ${pos.pnl_percent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {fPct(pos.pnl_percent)}
                    </td>
                    <td className="px-2 py-1 text-right">
                      {creatingOrder?.symbol === pos.symbol ? (
                        <div className="flex items-center gap-1">
                          <input type="number" value={creatingOrder.quantity} onChange={e => setCreatingOrder({...creatingOrder, quantity: e.target.value})}
                            className="w-16 bg-zinc-700 rounded px-1 py-0.5 text-xs" step="any" />
                          <span className="text-zinc-500">@</span>
                          <input type="number" value={creatingOrder.price} onChange={e => setCreatingOrder({...creatingOrder, price: e.target.value})}
                            className="w-20 bg-zinc-700 rounded px-1 py-0.5 text-xs" step="any" />
                          <button onClick={createSellOrder} className="p-0.5 bg-green-600 rounded"><Check className="w-3 h-3" /></button>
                          <button onClick={() => setCreatingOrder(null)} className="p-0.5 bg-zinc-600 rounded"><X className="w-3 h-3" /></button>
                        </div>
                      ) : (
                        <button onClick={() => setCreatingOrder({ symbol: pos.symbol, quantity: pos.quantity, price: (pos.current_price * 1.05).toFixed(4) })}
                          disabled={!!hasOrder} className={hasOrder ? 'text-zinc-700' : 'text-zinc-500 hover:text-white'}>
                          <DollarSign className="w-3 h-3" />
                        </button>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}

        {activeTab === 'positions' && posSource === 'paper' && (
          <table className="w-full">
            <thead className="sticky top-0 bg-zinc-900">
              <tr className="text-zinc-500 text-left">
                <th className="px-2 py-1 font-normal">Symbol</th>
                <th className="px-2 py-1 font-normal">Side</th>
                <th className="px-2 py-1 font-normal text-right">Hebel</th>
                <th className="px-2 py-1 font-normal text-right">Einstieg</th>
                <th className="px-2 py-1 font-normal text-right">TP</th>
                <th className="px-2 py-1 font-normal text-right">SL</th>
                <th className="px-2 py-1 font-normal text-right">Margin</th>
                <th className="px-2 py-1 font-normal text-right">Alter</th>
              </tr>
            </thead>
            <tbody>
              {paperPositions.length === 0 && (
                <tr><td colSpan={8} className="px-2 py-4 text-center text-zinc-600">Keine offenen Paper-Positionen</td></tr>
              )}
              {paperPositions.map((pos, i) => {
                const ageMin = pos.opened_at ? Math.round((Date.now() - new Date(pos.opened_at).getTime())/60000) : 0
                return (
                  <tr key={i} className="border-t border-zinc-800 hover:bg-zinc-800/50">
                    <td className="px-2 py-1 font-mono font-medium">{pos.symbol}</td>
                    <td className={`px-2 py-1 font-semibold ${pos.side==='long'?'text-emerald-400':'text-red-400'}`}>{pos.side==='long'?'L':'S'}</td>
                    <td className="px-2 py-1 text-right font-mono">{pos.leverage}x</td>
                    <td className="px-2 py-1 text-right font-mono">${fp(pos.entry_px, 6)}</td>
                    <td className="px-2 py-1 text-right font-mono text-emerald-300">${fp(pos.tp_px, 6)}</td>
                    <td className="px-2 py-1 text-right font-mono text-red-300">${fp(pos.sl_px, 6)}</td>
                    <td className="px-2 py-1 text-right font-mono">${fp(pos.margin_usd, 2)}</td>
                    <td className="px-2 py-1 text-right font-mono text-zinc-400">{ageMin}m</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
        {activeTab === 'positions' && posSource === 'hyperliquid' && (<>
          <div className="px-2 py-1.5 border-b border-zinc-800 flex items-center gap-2 flex-wrap bg-zinc-800/40">
            <span className="text-zinc-500">TP setzen (% vor Hebel):</span>
            {quickTPs.map(v => (
              <span key={v} className="flex items-center">
                <button onClick={() => applyTP(v)} disabled={selectedCoins.size === 0 || tpUpdating}
                  className="px-2 py-0.5 rounded bg-blue-700 hover:bg-blue-600 disabled:opacity-30 disabled:cursor-not-allowed text-xs font-mono">
                  {v}%
                </button>
                {editingQuickTPs && (
                  <button onClick={() => removeQuickTP(v)} title="Entfernen"
                    className="ml-0.5 text-red-400 hover:text-red-300"><X className="w-3 h-3" /></button>
                )}
              </span>
            ))}
            {editingQuickTPs && (
              <span className="flex items-center gap-1">
                <input type="number" value={newQuickTP} onChange={e => setNewQuickTP(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') addQuickTP() }}
                  placeholder="z.B. 1.5" step="0.1" min="0.01"
                  className="w-16 bg-zinc-700 rounded px-1 py-0.5 text-xs" />
                <button onClick={addQuickTP} className="px-1.5 py-0.5 bg-green-600 rounded text-xs">+</button>
              </span>
            )}
            <button onClick={() => setEditingQuickTPs(!editingQuickTPs)} title={editingQuickTPs ? 'Fertig' : 'Werte bearbeiten'}
              className={`ml-1 p-0.5 rounded ${editingQuickTPs ? 'text-blue-400' : 'text-zinc-500 hover:text-zinc-300'}`}>
              <Edit2 className="w-3 h-3" />
            </button>
            {selectedCoins.size > 0 && (
              <span className="text-zinc-400 ml-2">{selectedCoins.size} markiert</span>
            )}
            {tpMsg && <span className="text-zinc-300 ml-1">{tpMsg}</span>}
            {selectedCoins.size > 0 && (
              <button onClick={closeSelected} disabled={closing}
                className="ml-auto px-3 py-1 bg-red-600 hover:bg-red-500 rounded text-xs font-semibold disabled:opacity-50">
                {closing ? 'Schließe...' : `Close (${selectedCoins.size})`}
              </button>
            )}
          </div>
          <table className="w-full">
            <thead className="sticky top-0 bg-zinc-900">
              <tr className="text-zinc-500 text-left">
                <th className="px-2 py-1 font-normal cursor-pointer hover:text-zinc-300" onClick={() => togglePosSort('coin')}>Coin{posSortIcon('coin')}</th>
                <th className="px-2 py-1 font-normal text-center cursor-pointer hover:text-zinc-300" onClick={() => togglePosSort('direction')}>D/H{posSortIcon('direction')}</th>
                <th className="px-2 py-1 font-normal text-right cursor-pointer hover:text-zinc-300" onClick={() => togglePosSort('entry_price')}>Einstieg{posSortIcon('entry_price')}</th>
                <th className="px-2 py-1 font-normal text-right cursor-pointer hover:text-zinc-300" onClick={() => togglePosSort('peak_pct')} title="Maximale Profit-Bewegung waehrend Trade (aus Predictor-DB)">Peak{posSortIcon('peak_pct')}</th>
                <th className="px-2 py-1 font-normal text-right cursor-pointer hover:text-zinc-300" onClick={() => togglePosSort('trough_pct')} title="Maximaler Drawdown waehrend Trade (aus Predictor-DB)">Trough{posSortIcon('trough_pct')}</th>
                <th className="px-2 py-1 font-normal text-right cursor-pointer hover:text-zinc-300" onClick={() => togglePosSort('current_price')}>Aktuell{posSortIcon('current_price')}</th>
                <th className="px-2 py-1 font-normal text-right cursor-pointer hover:text-zinc-300" onClick={() => togglePosSort('unrealized_pnl')}>uPnL{posSortIcon('unrealized_pnl')}</th>
                <th className="px-2 py-1 font-normal text-right cursor-pointer hover:text-zinc-300" onClick={() => togglePosSort('pnl_pct')}>%{posSortIcon('pnl_pct')}</th>
                <th className="px-2 py-1 font-normal text-center w-8">
                  <input type="checkbox"
                    ref={el => { if (el) el.indeterminate = someHlSelected && !allHlSelected }}
                    checked={allHlSelected} onChange={toggleAllHl}
                    className="rounded" title="Alle markieren" />
                </th>
                <th className="px-2 py-1 font-normal text-center w-8" title={`Timeout (${timeoutHours}h) ein/aus`}>T</th>
              </tr>
            </thead>
            <tbody>
              {sortedHlPositions.length === 0 ? (
                <tr><td colSpan={10} className="text-center py-4 text-zinc-500">Keine offenen Positionen</td></tr>
              ) : sortedHlPositions.map((pos, i) => (
                <tr key={i} className="border-t border-zinc-800 hover:bg-zinc-800/50">
                  <td className="px-2 py-1 font-mono font-medium">{pos.coin}</td>
                  <td className="px-2 py-1 text-center font-semibold whitespace-nowrap">
                    <span className={pos.direction === 'long' ? 'text-green-400' : 'text-red-400'}>
                      {pos.direction === 'long' ? 'L' : 'S'}
                    </span>
                    <span className="text-amber-400 ml-1">{pos.leverage}x</span>
                    {pos.timeout_enabled === false && (
                      <span className="text-zinc-500 ml-0.5 line-through" title="Timeout deaktiviert">T</span>
                    )}
                  </td>
                  <td className="px-2 py-1 text-right font-mono whitespace-nowrap">
                    ${fp(pos.entry_price, 4)}
                    {pos.opened_at && <span className="text-zinc-600 ml-1 text-[10px]">{fmtHM(pos.opened_at)}</span>}
                  </td>
                  <td className="px-2 py-1 text-right font-mono text-green-500"
                      title="Peak: maximale Profit-Bewegung waehrend des Trades">
                    {pos.peak_pct != null ? `+${pos.peak_pct.toFixed(2)}%` : '-'}
                  </td>
                  <td className="px-2 py-1 text-right font-mono text-red-500"
                      title="Trough: maximaler Drawdown waehrend des Trades">
                    {pos.trough_pct != null ? `${pos.trough_pct.toFixed(2)}%` : '-'}
                  </td>
                  <td className="px-2 py-1 text-right font-mono">${fp(pos.current_price, 4)}</td>
                  <td className={`px-2 py-1 text-right font-mono ${pos.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {fPnl(pos.unrealized_pnl)}
                  </td>
                  <td className={`px-2 py-1 text-right font-mono ${(pos.roe_percent ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}
                      title={`Margin-PnL: PnL ÷ Margin (${pos.margin_used} $) — Coin-Bewegung wäre ${pos.position_value ? (pos.unrealized_pnl / pos.position_value * 100).toFixed(2) : 0}%`}>
                    {fPct(pos.roe_percent ?? (pos.margin_used > 0 ? pos.unrealized_pnl / pos.margin_used * 100 : 0))}
                  </td>
                  <td className="px-2 py-1 text-center">
                    <input type="checkbox" checked={selectedCoins.has(pos.coin)}
                      onChange={() => toggleCoin(pos.coin)} className="rounded" />
                  </td>
                  <td className="px-2 py-1 text-center">
                    <button onClick={() => toggleTimeout(pos.coin, pos.direction, pos.timeout_enabled !== false)}
                      className={`text-[11px] font-semibold ${pos.timeout_enabled === false ? 'text-zinc-500 line-through hover:text-zinc-300' : 'text-amber-400 hover:text-amber-300'}`}
                      title={pos.timeout_enabled === false ? `Timeout aus — klicken um zu aktivieren (${timeoutHours}h)` : `Timeout aktiv (${timeoutHours}h) — klicken um zu deaktivieren`}>
                      T
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {sortedHlPositions.length > 0 && selectedCoins.size > 0 && (
            <div className="px-2 py-2 border-t border-zinc-800 flex justify-end">
              <button onClick={closeSelected} disabled={closing}
                className="px-3 py-1.5 bg-red-600 hover:bg-red-500 rounded text-xs font-semibold disabled:opacity-50">
                {closing ? 'Schließe...' : `Close Selected (${selectedCoins.size})`}
              </button>
            </div>
          )}
        </>)}

        {activeTab === 'orders' && ordSource === 'binance' && (
          <table className="w-full">
            <thead className="sticky top-0 bg-zinc-900">
              <tr className="text-zinc-500 text-left">
                <th className="px-2 py-1 font-normal">Symbol</th>
                <th className="px-2 py-1 font-normal">Side</th>
                <th className="px-2 py-1 font-normal text-right">Preis</th>
                <th className="px-2 py-1 font-normal text-right">Aktuell</th>
                <th className="px-2 py-1 font-normal text-right">Diff</th>
                <th className="px-2 py-1 font-normal text-right">Menge</th>
                <th className="px-2 py-1 font-normal w-12"></th>
              </tr>
            </thead>
            <tbody>
              {orders.map((o, i) => {
                const pos = positions.find(p => p.symbol === o.symbol)
                const cur = pos?.current_price
                const diff = cur && o.price ? ((o.price - cur) / cur * 100) : null
                return (
                  <tr key={i} className="border-t border-zinc-800 hover:bg-zinc-800/50">
                    <td className="px-2 py-1 font-mono font-medium">{o.symbol.replace('USDC','')}</td>
                    <td className={`px-2 py-1 ${o.side === 'BUY' ? 'text-green-400' : 'text-red-400'}`}>{o.side}</td>
                    <td className="px-2 py-1 text-right font-mono">${fp(o.price, 4)}</td>
                    <td className="px-2 py-1 text-right font-mono">{cur ? `$${fp(cur, 4)}` : '-'}</td>
                    <td className={`px-2 py-1 text-right font-mono ${diff !== null ? (diff >= 0 ? 'text-green-400' : 'text-red-400') : ''}`}>
                      {diff !== null ? `${diff >= 0 ? '+' : ''}${diff.toFixed(1)}%` : '-'}
                    </td>
                    <td className="px-2 py-1 text-right font-mono">{fp(o.quantity, 4)}</td>
                    <td className="px-2 py-1 text-right">
                      <div className="flex items-center justify-end gap-1">
                        <button onClick={() => setEditingOrder({ symbol: o.symbol, orderId: o.order_id, currentPrice: cur, newPrice: o.price, quantity: o.quantity })}
                          className="p-0.5 hover:bg-zinc-600 rounded text-blue-400"><Edit2 className="w-3 h-3" /></button>
                        <button onClick={() => cancelOrder(o.symbol, o.order_id)}
                          className="p-0.5 hover:bg-zinc-600 rounded text-red-400"><Trash2 className="w-3 h-3" /></button>
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}

        {activeTab === 'orders' && ordSource === 'hyperliquid' && (
          <table className="w-full">
            <thead className="sticky top-0 bg-zinc-900">
              <tr className="text-zinc-500 text-left">
                <th className="px-2 py-1 font-normal">Coin</th>
                <th className="px-2 py-1 font-normal">Side</th>
                <th className="px-2 py-1 font-normal text-right">Preis</th>
                <th className="px-2 py-1 font-normal text-right">Size</th>
              </tr>
            </thead>
            <tbody>
              {hlOrders.length === 0 ? (
                <tr><td colSpan={4} className="text-center py-4 text-zinc-500">Keine offenen Orders</td></tr>
              ) : hlOrders.map((o, i) => (
                <tr key={i} className="border-t border-zinc-800 hover:bg-zinc-800/50">
                  <td className="px-2 py-1 font-mono font-medium">{o.coin}</td>
                  <td className={`px-2 py-1 ${o.side === 'BUY' ? 'text-green-400' : 'text-red-400'}`}>{o.side}</td>
                  <td className="px-2 py-1 text-right font-mono">${fp(o.price, 4)}</td>
                  <td className="px-2 py-1 text-right font-mono">{fp(o.size, 4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {/* Edit Order Overlay */}
        {editingOrder && (
          <div className="px-2 py-1.5 bg-zinc-800 border-t border-zinc-700 flex items-center gap-2">
            <span className="text-zinc-400">{editingOrder.symbol} neuer Preis:</span>
            <input type="number" value={editingOrder.newPrice} onChange={e => setEditingOrder({...editingOrder, newPrice: e.target.value})}
              className="w-24 bg-zinc-700 rounded px-2 py-0.5" step="any" />
            <button onClick={updateOrder} className="px-2 py-0.5 bg-blue-600 rounded">OK</button>
            <button onClick={() => setEditingOrder(null)} className="px-2 py-0.5 bg-zinc-600 rounded">X</button>
          </div>
        )}

        {activeTab === 'history' && (
          <table className="w-full">
            <thead className="sticky top-0 bg-zinc-900">
              <tr className="text-zinc-500 text-left">
                <th className="px-2 py-1 font-normal cursor-pointer hover:text-zinc-300" onClick={() => toggleHistSort('symbol')}>Symbol{sortIcon('symbol')}</th>
                <th className="px-2 py-1 font-normal cursor-pointer hover:text-zinc-300" onClick={() => toggleHistSort('side')}>Dir{sortIcon('side')}</th>
                <th className="px-2 py-1 font-normal text-right cursor-pointer hover:text-zinc-300" onClick={() => toggleHistSort('leverage')}>Lev{sortIcon('leverage')}</th>
                <th className="px-2 py-1 font-normal text-right cursor-pointer hover:text-zinc-300" onClick={() => toggleHistSort('quote_amount')}>Invest{sortIcon('quote_amount')}</th>
                <th className="px-2 py-1 font-normal text-right cursor-pointer hover:text-zinc-300" onClick={() => toggleHistSort('sold_for')}>Zurück{sortIcon('sold_for')}</th>
                <th className="px-2 py-1 font-normal text-right cursor-pointer hover:text-zinc-300" onClick={() => toggleHistSort('pnl_percent')}>PnL{sortIcon('pnl_percent')}</th>
                <th className="px-2 py-1 font-normal cursor-pointer hover:text-zinc-300" onClick={() => toggleHistSort('exit_reason')}>Exit{sortIcon('exit_reason')}</th>
                <th className="px-2 py-1 font-normal text-right cursor-pointer hover:text-zinc-300" onClick={() => toggleHistSort('executed_at')}>Datum{sortIcon('executed_at')}</th>
              </tr>
            </thead>
            <tbody>
              {sortedHistory.length === 0 ? (
                <tr><td colSpan={8} className="text-center py-4 text-zinc-500">Keine Trades (30d)</td></tr>
              ) : sortedHistory.map((t, i) => (
                <tr key={i} className="border-t border-zinc-800 hover:bg-zinc-800/50">
                  <td className="px-2 py-1 font-mono font-medium">
                    {t.symbol?.replace('USDC','')}
                  </td>
                  <td className={`px-2 py-1 ${t.side === 'long' ? 'text-green-400' : t.side === 'short' ? 'text-red-400' : t.side === 'buy' ? 'text-green-400' : 'text-red-400'}`}>
                    {t.side === 'long' ? 'L' : t.side === 'short' ? 'S' : t.side?.toUpperCase()}
                  </td>
                  <td className="px-2 py-1 text-right font-mono text-zinc-400">{t.leverage ? `${t.leverage}x` : '-'}</td>
                  <td className="px-2 py-1 text-right font-mono"
                      title={t.quote_amount ? `Notional: $${fp(t.quote_amount)}` : ''}>${fp(t.margin_usd ?? t.quote_amount)}</td>
                  <td className={`px-2 py-1 text-right font-mono ${(t.pnl_usd ?? 0) > 0 ? 'text-green-400' : (t.pnl_usd ?? 0) < 0 ? 'text-red-400' : 'text-zinc-400'}`}>
                    {(t.margin_usd ?? t.quote_amount) ? `$${fp((t.margin_usd ?? t.quote_amount) + (t.pnl_usd ?? 0))}` : '-'}
                  </td>
                  <td className={`px-2 py-1 text-right font-mono ${t.pnl_percent > 0 ? 'text-green-400' : t.pnl_percent < 0 ? 'text-red-400' : 'text-zinc-400'}`}>
                    {t.pnl_percent != null ? `${t.pnl_percent > 0 ? '+' : ''}${fp(t.pnl_percent, 2)}%` : '-'}
                  </td>
                  <td className="px-2 py-1 text-zinc-400 text-xs">{t.exit_reason || '-'}</td>
                  <td className="px-2 py-1 text-right text-zinc-400">{fDate(t.executed_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Convert Dialog */}
      {convertDialog && (
        <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center" onClick={e => { if (e.target === e.currentTarget && !convertLoading) setConvertDialog(false) }}>
          <div className="bg-zinc-800 border border-zinc-700 rounded-lg p-4 w-72">
            <div className="flex justify-between items-center mb-3">
              <span className="font-semibold text-sm flex items-center gap-1.5"><ArrowRightLeft className="w-4 h-4 text-blue-400" /> USDC → USDC</span>
              <button onClick={() => !convertLoading && setConvertDialog(false)}><X className="w-4 h-4 text-zinc-400" /></button>
            </div>

            {convertResult?.status === 'success' ? (
              <div className="p-3 bg-green-900/30 border border-green-700/40 rounded">
                <div className="text-green-400 font-semibold mb-1">Konvertiert!</div>
                <div className="text-xs text-zinc-300">{convertResult.usdc_sold} USDC → {convertResult.usdt_received} USDC</div>
                <button onClick={() => setConvertDialog(false)} className="mt-3 w-full py-1 bg-zinc-700 hover:bg-zinc-600 rounded text-xs">Schließen</button>
              </div>
            ) : (
              <>
                <div className="text-xs text-zinc-400 mb-2">Verfügbar: <span className="text-blue-300 font-mono">{fp(balance?.usdc_free || 0)} USDC</span></div>
                <div className="flex gap-1.5 mb-3">
                  <input type="number" placeholder="Betrag (leer = Max)" value={convertAmount}
                    onChange={e => setConvertAmount(e.target.value)} min="5" step="1"
                    className="flex-1 bg-zinc-700 border border-zinc-600 rounded px-2 py-1 text-xs" />
                  <button onClick={() => setConvertAmount(String(Math.floor(balance?.usdc_free || 0)))}
                    className="px-2 py-1 bg-zinc-700 hover:bg-zinc-600 rounded text-xs text-zinc-300">Max</button>
                  <button onClick={() => setConvertAmount(String(Math.floor((balance?.usdc_free || 0) / 2)))}
                    className="px-2 py-1 bg-zinc-700 hover:bg-zinc-600 rounded text-xs text-zinc-300">50%</button>
                </div>

                {convertResult?.error && (
                  <div className="p-2 bg-red-900/30 border border-red-700/40 rounded text-xs text-red-400 mb-2">{convertResult.error}</div>
                )}

                <div className="flex gap-2">
                  <button onClick={() => setConvertDialog(false)} disabled={convertLoading}
                    className="flex-1 py-1 bg-zinc-700 hover:bg-zinc-600 rounded text-xs">Abbrechen</button>
                  <button onClick={() => doConvert(convertAmount ? parseFloat(convertAmount) : null)} disabled={convertLoading}
                    className="flex-1 py-1 bg-blue-600 hover:bg-blue-500 rounded text-xs font-semibold flex items-center justify-center gap-1">
                    {convertLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : <><ArrowRightLeft className="w-3 h-3" /> Convert</>}
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
