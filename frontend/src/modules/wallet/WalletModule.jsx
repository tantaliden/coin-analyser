import { useState, useEffect, useCallback } from 'react'
import { Wallet, RefreshCw, TrendingUp, TrendingDown, AlertCircle, X, Edit2, Trash2, Bot, DollarSign, Check, Tag, Filter, ArrowRightLeft, Loader2 } from 'lucide-react'
import api from '../../utils/api'

export default function WalletModule() {
  const [status, setStatus] = useState({ configured: false })
  const [balance, setBalance] = useState(null)
  const [positions, setPositions] = useState([])
  const [orders, setOrders] = useState([])
  const [history, setHistory] = useState([])
  const [realizedPnl, setRealizedPnl] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('positions')
  const [editingOrder, setEditingOrder] = useState(null)
  const [creatingOrder, setCreatingOrder] = useState(null)
  const [hideWithOrders, setHideWithOrders] = useState(false)
  const [convertDialog, setConvertDialog] = useState(false)
  const [convertAmount, setConvertAmount] = useState('')
  const [convertLoading, setConvertLoading] = useState(false)
  const [convertResult, setConvertResult] = useState(null)

  const loadData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const statusRes = await api.get('/api/v1/wallet/status')
      setStatus(statusRes.data)
      if (!statusRes.data.configured) { setLoading(false); return }
      const [balRes, posRes, ordRes, histRes, pnlRes] = await Promise.all([
        api.get('/api/v1/wallet/balance'), api.get('/api/v1/wallet/positions'),
        api.get('/api/v1/wallet/orders'), api.get('/api/v1/wallet/history?days=7'),
        api.get('/api/v1/wallet/realized-pnl?days=7')
      ])
      if (balRes.data.error) setError(balRes.data.error)
      else setBalance(balRes.data)
      setPositions(posRes.data.positions || [])
      setOrders(ordRes.data.orders || [])
      setHistory(histRes.data.trades || [])
      setRealizedPnl(pnlRes.data)
    } catch (err) { setError('Fehler beim Laden') }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { loadData(); const i = setInterval(loadData, 30000); return () => clearInterval(i) }, [loadData])

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

  if (!status.configured && !loading) return (
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

  return (
    <div className="h-full flex flex-col overflow-hidden text-xs">
      {/* Balance Bar */}
      {balance && (
        <div className="flex items-center gap-3 px-2 py-1.5 border-b border-zinc-700/50 bg-zinc-800/30 flex-shrink-0">
          <div><span className="text-zinc-500">USDC</span> <span className="font-mono">${fp(balance.usdt_balance)}</span></div>
          {balance.usdc_balance > 0 && (
            <div className="flex items-center gap-1">
              <span className="text-zinc-500">USDC</span>
              <span className="font-mono text-blue-300">${fp(balance.usdc_balance)}</span>
              <button onClick={() => { setConvertDialog(true); setConvertAmount(''); setConvertResult(null) }}
                className="ml-0.5 p-0.5 rounded hover:bg-zinc-600 text-blue-400" title="USDC → USDC konvertieren">
                <ArrowRightLeft className="w-3 h-3" />
              </button>
            </div>
          )}
          <div><span className="text-zinc-500">Portfolio</span> <span className="font-mono font-semibold">${fp(balance.total_portfolio)}</span></div>
          <div className={totalUPnl >= 0 ? 'text-green-400' : 'text-red-400'}>{fPnl(totalUPnl)}</div>
          <div className={(realizedPnl?.realized_pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'}><span className="text-zinc-500">7d</span> {fPnl(realizedPnl?.realized_pnl)}</div>
          <button onClick={loadData} disabled={loading} className="ml-auto p-0.5 hover:bg-zinc-700 rounded">
            <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      )}

      {/* Tabs */}
      <div className="flex items-center gap-1 px-2 py-1 border-b border-zinc-700/50 flex-shrink-0">
        {[
          { id: 'positions', label: `Pos (${hideWithOrders ? positions.filter(p => !getOrderForPosition(p.symbol)).length : positions.length})` },
          { id: 'orders', label: `Orders (${orders.length})` },
          { id: 'history', label: 'Hist' }
        ].map(t => (
          <button key={t.id} onClick={() => setActiveTab(t.id)}
            className={`px-2 py-0.5 rounded ${activeTab === t.id ? 'bg-blue-600 text-white' : 'text-zinc-400 hover:text-white'}`}>{t.label}</button>
        ))}
        {activeTab === 'positions' && (
          <button onClick={() => setHideWithOrders(!hideWithOrders)}
            className={`ml-auto p-1 rounded ${hideWithOrders ? 'text-blue-400' : 'text-zinc-500'}`}>
            <Filter className="w-3 h-3" />
          </button>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto">
        {activeTab === 'positions' && (
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

        {activeTab === 'orders' && (
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
                <th className="px-2 py-1 font-normal">Symbol</th>
                <th className="px-2 py-1 font-normal">Side</th>
                <th className="px-2 py-1 font-normal text-right">Preis</th>
                <th className="px-2 py-1 font-normal text-right">Menge</th>
                <th className="px-2 py-1 font-normal text-right">Wert</th>
                <th className="px-2 py-1 font-normal text-right">Datum</th>
              </tr>
            </thead>
            <tbody>
              {history.length === 0 ? (
                <tr><td colSpan={6} className="text-center py-4 text-zinc-500">Keine Trades (7d)</td></tr>
              ) : history.map((t, i) => (
                <tr key={i} className="border-t border-zinc-800 hover:bg-zinc-800/50">
                  <td className="px-2 py-1 font-mono font-medium">
                    {t.symbol?.replace('USDC','')} {t.is_bot_trade && <Bot className="w-3 h-3 text-blue-400 inline ml-1" />}
                  </td>
                  <td className={`px-2 py-1 ${t.side === 'buy' ? 'text-green-400' : 'text-red-400'}`}>{t.side?.toUpperCase()}</td>
                  <td className="px-2 py-1 text-right font-mono">${fp(t.price, 4)}</td>
                  <td className="px-2 py-1 text-right font-mono">{fp(t.quantity, 4)}</td>
                  <td className="px-2 py-1 text-right font-mono">${fp(t.quote_amount)}</td>
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
