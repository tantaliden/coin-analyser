import { useState, useEffect, useCallback } from 'react'
import { Wallet, RefreshCw, TrendingUp, TrendingDown, AlertCircle, X, Edit2, Trash2, Bot, Clock, DollarSign, Check, Tag, Filter } from 'lucide-react'
import api from '../../utils/api'

export default function WalletModule() {
  const [status, setStatus] = useState({ configured: false, message: '' })
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

  const loadData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const statusRes = await api.get('/api/v1/wallet/status')
      setStatus(statusRes.data)
      if (!statusRes.data.configured) {
        setLoading(false)
        return
      }
      const [balRes, posRes, ordRes, histRes, pnlRes] = await Promise.all([
        api.get('/api/v1/wallet/balance'),
        api.get('/api/v1/wallet/positions'),
        api.get('/api/v1/wallet/orders'),
        api.get('/api/v1/wallet/history?days=7'),
        api.get('/api/v1/wallet/realized-pnl?days=7')
      ])
      if (balRes.data.error) setError(balRes.data.error)
      else setBalance(balRes.data)
      setPositions(posRes.data.positions || [])
      setOrders(ordRes.data.orders || [])
      setHistory(histRes.data.trades || [])
      setRealizedPnl(pnlRes.data)
    } catch (err) {
      console.error('Wallet load error:', err)
      setError('Fehler beim Laden der Wallet-Daten')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadData()
    const interval = setInterval(loadData, 30000)
    return () => clearInterval(interval)
  }, [loadData])

  const cancelOrder = async (symbol, orderId) => {
    if (!confirm(`Order ${orderId} wirklich stornieren?`)) return
    try {
      const res = await api.delete(`/api/v1/wallet/orders/${symbol}/${orderId}`)
      if (res.data.error) alert(res.data.error)
      else loadData()
    } catch (err) { alert('Fehler beim Stornieren') }
  }

  const updateOrder = async () => {
    if (!editingOrder) return
    try {
      await api.delete(`/api/v1/wallet/orders/${editingOrder.symbol}/${editingOrder.orderId}`)
      const res = await api.post('/api/v1/wallet/orders', {
        symbol: editingOrder.symbol, side: 'SELL', type: 'LIMIT',
        price: parseFloat(editingOrder.newPrice), quantity: editingOrder.quantity
      })
      if (res.data.error) alert(res.data.error)
      else { setEditingOrder(null); loadData() }
    } catch (err) { alert('Fehler beim Aktualisieren der Order') }
  }

  const createSellOrder = async () => {
    if (!creatingOrder) return
    try {
      const res = await api.post('/api/v1/wallet/orders', {
        symbol: creatingOrder.symbol, side: 'SELL', type: 'LIMIT',
        price: parseFloat(creatingOrder.price), quantity: parseFloat(creatingOrder.quantity)
      })
      if (res.data.error) alert(res.data.error)
      else { setCreatingOrder(null); loadData() }
    } catch (err) { alert('Fehler beim Erstellen der Order') }
  }

  const getOrderForPosition = (posSymbol) => {
    const baseAsset = posSymbol.replace(/USD[TC]$/, '')
    return orders.find(o => {
      const orderBase = o.symbol.replace(/USD[TC]$/, '')
      return orderBase === baseAsset && o.side === 'SELL'
    })
  }

  const formatPrice = (price, decimals = 2) => {
    if (price === null || price === undefined) return '-'
    return new Intl.NumberFormat('de-DE', { minimumFractionDigits: decimals, maximumFractionDigits: decimals }).format(price)
  }
  const formatPnl = (value) => {
    if (value === null || value === undefined) return '-'
    return `${value >= 0 ? '+' : ''}$${formatPrice(value)}`
  }
  const formatPercent = (value) => {
    if (value === null || value === undefined) return '-'
    return `${value >= 0 ? '+' : ''}${formatPrice(value)}%`
  }
  const formatDate = (dateStr) => {
    if (!dateStr) return '-'
    return new Date(dateStr).toLocaleDateString('de-DE', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })
  }

  if (!status.configured && !loading) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-zinc-500 p-4">
        <Wallet className="w-12 h-12 mb-4 opacity-50" />
        <p className="text-sm text-center">Kein API-Key konfiguriert</p>
        <p className="text-xs mt-2 text-center">Binance API-Key im Profil hinterlegen</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-red-400 p-4">
        <AlertCircle className="w-12 h-12 mb-4" />
        <p className="text-sm text-center">{error}</p>
        <button onClick={loadData} className="mt-4 px-4 py-2 bg-zinc-700 hover:bg-zinc-600 rounded text-sm">Erneut versuchen</button>
      </div>
    )
  }

  const totalUnrealizedPnl = positions.reduce((sum, p) => sum + (p.unrealized_pnl || 0), 0)

  return (
    <div className="h-full flex flex-col p-3 gap-3 overflow-hidden">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-white flex items-center gap-2">
          <Wallet className="w-4 h-4" /> Wallet
        </h2>
        <button onClick={loadData} disabled={loading} className="p-1 hover:bg-zinc-700 rounded">
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {balance && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
          <div className="bg-zinc-800/50 rounded p-2">
            <div className="text-xs text-zinc-500">USDT Balance</div>
            <div className="text-sm font-semibold">${formatPrice(balance.usdt_balance)}</div>
          </div>
          <div className="bg-zinc-800/50 rounded p-2">
            <div className="text-xs text-zinc-500">Portfolio</div>
            <div className="text-sm font-semibold">${formatPrice(balance.total_portfolio)}</div>
          </div>
          <div className="bg-zinc-800/50 rounded p-2">
            <div className="text-xs text-zinc-500">Unrealisiert</div>
            <div className={`text-sm font-semibold ${totalUnrealizedPnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {formatPnl(totalUnrealizedPnl)}
            </div>
          </div>
          <div className="bg-zinc-800/50 rounded p-2">
            <div className="text-xs text-zinc-500">Realisiert (7d)</div>
            <div className={`text-sm font-semibold ${(realizedPnl?.realized_pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {formatPnl(realizedPnl?.realized_pnl)}
            </div>
          </div>
        </div>
      )}

      <div className="flex gap-1 border-b border-zinc-700">
        {[
          { id: 'positions', label: `Positionen (${hideWithOrders ? positions.filter(p => !getOrderForPosition(p.symbol)).length : positions.length})` },
          { id: 'orders', label: `Orders (${orders.length})` },
          { id: 'history', label: 'Historie' }
        ].map(tab => (
          <button key={tab.id} onClick={() => setActiveTab(tab.id)}
            className={`px-3 py-1.5 text-xs font-medium border-b-2 -mb-px transition-colors ${
              activeTab === tab.id ? 'border-blue-500 text-blue-400' : 'border-transparent text-zinc-400 hover:text-zinc-200'
            }`}>{tab.label}</button>
        ))}
        {activeTab === 'positions' && (
          <button onClick={() => setHideWithOrders(!hideWithOrders)}
            className={`ml-auto p-1.5 rounded text-xs flex items-center gap-1 ${hideWithOrders ? 'bg-blue-500/20 text-blue-400' : 'text-zinc-400 hover:text-white'}`}
            title={hideWithOrders ? 'Alle anzeigen' : 'Mit Order ausblenden'}>
            <Filter className="w-3 h-3" />
          </button>
        )}
      </div>

      <div className="flex-1 overflow-auto">
        {activeTab === 'positions' && (
          <div className="space-y-2">
            {positions.length === 0 ? (
              <div className="text-center py-8 text-zinc-500 text-sm">Keine offenen Positionen</div>
            ) : (
              positions.filter(p => !hideWithOrders || !getOrderForPosition(p.symbol)).map((pos, idx) => {
                const existingOrder = getOrderForPosition(pos.symbol)
                return (
                  <div key={idx} className="bg-zinc-800/50 rounded p-3">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{pos.symbol}</span>
                        {pos.is_bot_trade && (
                          <span className="flex items-center gap-1 text-xs bg-blue-500/20 text-blue-300 px-1.5 py-0.5 rounded">
                            <Bot className="w-3 h-3" />
                            {pos.indicator_set_accuracy && `${pos.indicator_set_accuracy}%`}
                          </span>
                        )}
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-semibold text-white">${formatPrice(pos.value_usdt)}</div>
                        <div className={`text-xs ${pos.pnl_percent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {formatPercent(pos.pnl_percent)}
                        </div>
                      </div>
                    </div>
                    <div className="grid grid-cols-4 gap-2 text-xs mb-2">
                      <div><span className="text-zinc-500">Menge</span><div>{formatPrice(pos.quantity, 6)}</div></div>
                      <div><span className="text-zinc-500">Einstieg</span><div>${formatPrice(pos.avg_entry_price, 4)}</div></div>
                      <div><span className="text-zinc-500">Aktuell</span><div>${formatPrice(pos.current_price, 4)}</div></div>
                      <div><span className="text-zinc-500">P/L</span>
                        <div className={pos.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}>{formatPnl(pos.unrealized_pnl)}</div>
                      </div>
                    </div>
                    <div className="flex items-center justify-between mt-2 pt-2 border-t border-zinc-700/50">
                      {existingOrder && (
                        <div className="flex items-center gap-2 px-2 py-1 bg-yellow-500/20 rounded text-xs">
                          <Tag className="w-3 h-3 text-yellow-400" /><span className="text-yellow-400">Open Order</span>
                        </div>
                      )}
                      {creatingOrder?.symbol === pos.symbol ? (
                        <div className="flex items-center gap-2 ml-auto">
                          <input type="number" value={creatingOrder.quantity}
                            onChange={(e) => setCreatingOrder({...creatingOrder, quantity: e.target.value})}
                            placeholder="Menge" className="w-20 bg-zinc-600 rounded px-2 py-1 text-xs" step="any" />
                          <span className="text-zinc-500 text-xs">@</span>
                          <input type="number" value={creatingOrder.price}
                            onChange={(e) => setCreatingOrder({...creatingOrder, price: e.target.value})}
                            placeholder="Preis" className="w-24 bg-zinc-600 rounded px-2 py-1 text-xs" step="any" />
                          <button onClick={createSellOrder} className="p-1 bg-green-600 hover:bg-green-700 rounded"><Check className="w-3 h-3" /></button>
                          <button onClick={() => setCreatingOrder(null)} className="p-1 bg-zinc-600 hover:bg-zinc-500 rounded"><X className="w-3 h-3" /></button>
                        </div>
                      ) : (
                        <button onClick={() => setCreatingOrder({
                            symbol: pos.symbol, quantity: pos.quantity,
                            price: (pos.current_price * 1.05).toFixed(4), currentPrice: pos.current_price
                          })} disabled={!!existingOrder}
                          className={`text-xs flex items-center gap-1 ml-auto ${existingOrder ? 'text-zinc-600 cursor-not-allowed' : 'text-zinc-400 hover:text-white'}`}>
                          <DollarSign className="w-3 h-3" /> Sell Order erstellen
                        </button>
                      )}
                    </div>
                  </div>
                )
              })
            )}
          </div>
        )}

        {activeTab === 'orders' && (
          <div className="space-y-2">
            {orders.length === 0 ? (
              <div className="text-center py-8 text-zinc-500 text-sm">Keine offenen Orders</div>
            ) : (
              orders.map((order, idx) => {
                const position = positions.find(p => p.symbol === order.symbol)
                const currentPrice = position?.current_price
                const priceDiff = currentPrice && order.price ? ((order.price - currentPrice) / currentPrice * 100) : null
                return (
                  <div key={idx} className="bg-zinc-800/50 rounded p-3">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{order.symbol}</span>
                        <span className={`text-xs px-1.5 py-0.5 rounded ${order.side === 'BUY' ? 'bg-green-500/20 text-green-300' : 'bg-red-500/20 text-red-300'}`}>{order.side}</span>
                        <span className="text-xs text-zinc-500">{order.type}</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <button onClick={() => setEditingOrder({
                            symbol: order.symbol, orderId: order.order_id, currentPrice,
                            newPrice: order.price, quantity: order.quantity
                          })} className="p-1 hover:bg-zinc-600 rounded text-blue-400" title="Order bearbeiten">
                          <Edit2 className="w-4 h-4" />
                        </button>
                        <button onClick={() => cancelOrder(order.symbol, order.order_id)}
                          className="p-1 hover:bg-zinc-600 rounded text-red-400" title="Order stornieren">
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                    <div className="grid grid-cols-4 gap-2 text-xs">
                      <div><span className="text-zinc-500">Aktuell</span><div>${currentPrice ? formatPrice(currentPrice, 4) : '-'}</div></div>
                      <div><span className="text-zinc-500">Order-Preis</span>
                        <div className="flex items-center gap-1">${formatPrice(order.price, 4)}
                          {priceDiff !== null && (
                            <span className={`text-xs ${priceDiff >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                              ({priceDiff >= 0 ? '+' : ''}{priceDiff.toFixed(1)}%)
                            </span>
                          )}
                        </div>
                      </div>
                      {order.stop_price && <div><span className="text-zinc-500">Stop</span><div>${formatPrice(order.stop_price, 4)}</div></div>}
                      <div><span className="text-zinc-500">Menge</span><div>{formatPrice(order.quantity, 6)}</div></div>
                    </div>
                    {editingOrder?.orderId === order.order_id && (
                      <div className="mt-3 p-2 bg-zinc-700/50 rounded">
                        <div className="text-xs text-zinc-400 mb-2">Neuer Verkaufspreis:</div>
                        <div className="flex items-center gap-2">
                          <input type="number" value={editingOrder.newPrice}
                            onChange={(e) => setEditingOrder({...editingOrder, newPrice: e.target.value})}
                            className="flex-1 bg-zinc-600 rounded px-2 py-1 text-sm" step="any" />
                          <button onClick={updateOrder} className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs font-medium">Aktualisieren</button>
                          <button onClick={() => setEditingOrder(null)} className="px-3 py-1 bg-zinc-600 hover:bg-zinc-500 rounded text-xs">Abbrechen</button>
                        </div>
                        <div className="text-xs text-zinc-500 mt-1">⚠️ Order wird storniert und mit neuem Preis erstellt</div>
                      </div>
                    )}
                  </div>
                )
              })
            )}
          </div>
        )}

        {activeTab === 'history' && (
          <div className="space-y-2">
            {history.length === 0 ? (
              <div className="text-center py-8 text-zinc-500 text-sm">Keine Trades in den letzten 7 Tagen</div>
            ) : (
              history.map((trade, idx) => (
                <div key={idx} className="bg-zinc-800/50 rounded p-2 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className={`w-8 h-8 rounded flex items-center justify-center ${trade.side === 'buy' ? 'bg-green-500/20' : 'bg-red-500/20'}`}>
                      {trade.side === 'buy' ? <TrendingUp className="w-4 h-4 text-green-400" /> : <TrendingDown className="w-4 h-4 text-red-400" />}
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium">{trade.symbol}</span>
                        {trade.is_bot_trade && <Bot className="w-3 h-3 text-blue-400" />}
                      </div>
                      <div className="text-xs text-zinc-500">{formatDate(trade.executed_at)}</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm">${formatPrice(trade.quote_amount)}</div>
                    <div className="text-xs text-zinc-500">{formatPrice(trade.quantity, 6)} @ ${formatPrice(trade.price, 4)}</div>
                  </div>
                </div>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  )
}
