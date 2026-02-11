import { useState, useEffect } from 'react'
import { RefreshCw, Wallet, TrendingUp, AlertCircle } from 'lucide-react'
import api from '../../utils/api'

export default function WalletModule() {
  const [status, setStatus] = useState(null)
  const [balance, setBalance] = useState(null)
  const [positions, setPositions] = useState([])
  const [orders, setOrders] = useState([])
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('balance')

  const loadAll = async () => {
    setLoading(true)
    setError(null)
    try {
      const statusRes = await api.get('/api/v1/wallet/status')
      setStatus(statusRes.data)
      if (!statusRes.data.configured) {
        setError('Kein Binance API Key konfiguriert')
        setLoading(false)
        return
      }
      const [balRes, posRes, ordRes, histRes] = await Promise.all([
        api.get('/api/v1/wallet/balance').catch(() => ({ data: {} })),
        api.get('/api/v1/wallet/positions').catch(() => ({ data: { positions: [] } })),
        api.get('/api/v1/wallet/orders').catch(() => ({ data: { orders: [] } })),
        api.get('/api/v1/wallet/history?days=7').catch(() => ({ data: { trades: [] } }))
      ])
      if (balRes.data.error) setError(balRes.data.error)
      else setBalance(balRes.data)
      setPositions(posRes.data.positions || [])
      setOrders(ordRes.data.orders || [])
      setHistory(histRes.data.trades || [])
    } catch (e) {
      setError(e.response?.data?.detail || 'Wallet laden fehlgeschlagen')
    }
    setLoading(false)
  }

  useEffect(() => { loadAll() }, [])

  const tabs = [
    { id: 'balance', label: 'Übersicht' },
    { id: 'positions', label: `Positionen (${positions.length})` },
    { id: 'orders', label: `Orders (${orders.length})` },
    { id: 'history', label: 'Historie' }
  ]

  if (loading) return <div className="text-gray-400 p-4">Laden...</div>

  return (
    <div className="h-full flex flex-col">
      {error && (
        <div className="p-2 text-yellow-400 text-sm bg-yellow-900/20 flex items-center gap-2">
          <AlertCircle size={14} /> {error}
        </div>
      )}

      <div className="flex items-center gap-1 p-2 border-b border-gray-700">
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            className={`px-2 py-1 rounded text-xs ${tab === t.id ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'}`}>
            {t.label}
          </button>
        ))}
        <button onClick={loadAll} className="ml-auto p-1.5 bg-gray-700 hover:bg-gray-600 rounded" title="Aktualisieren">
          <RefreshCw size={12} />
        </button>
      </div>

      <div className="flex-1 overflow-auto p-2">
        {tab === 'balance' && balance && (
          <div className="space-y-3">
            <div className="bg-gray-800 rounded p-3">
              <div className="text-gray-400 text-xs mb-1">Portfolio</div>
              <div className="text-2xl font-bold">${balance.total_portfolio?.toFixed(2) || '0.00'}</div>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div className="bg-gray-800 rounded p-3">
                <div className="text-gray-400 text-xs">USDT frei</div>
                <div className="text-lg font-mono">${balance.usdt_balance?.toFixed(2) || '0.00'}</div>
              </div>
              <div className="bg-gray-800 rounded p-3">
                <div className="text-gray-400 text-xs">In Positionen</div>
                <div className="text-lg font-mono">${balance.positions_value?.toFixed(2) || '0.00'}</div>
              </div>
            </div>
          </div>
        )}

        {tab === 'positions' && (
          <div className="space-y-1">
            {positions.length === 0 ? <div className="text-gray-500 text-sm">Keine Positionen</div> : null}
            {positions.map((p, i) => (
              <div key={i} className="flex items-center gap-2 p-2 bg-gray-800 rounded text-sm">
                <span className="font-mono font-semibold w-24">{p.asset}</span>
                <span className="text-gray-400 flex-1">{p.quantity?.toFixed(6)}</span>
                <span className="font-mono">${p.value_usdt?.toFixed(2)}</span>
              </div>
            ))}
          </div>
        )}

        {tab === 'orders' && (
          <div className="space-y-1">
            {orders.length === 0 ? <div className="text-gray-500 text-sm">Keine offenen Orders</div> : null}
            {orders.map((o, i) => (
              <div key={i} className="flex items-center gap-2 p-2 bg-gray-800 rounded text-xs">
                <span className="font-mono font-semibold">{o.symbol}</span>
                <span className={o.side === 'BUY' ? 'text-green-400' : 'text-red-400'}>{o.side}</span>
                <span className="text-gray-400 flex-1">{o.quantity}</span>
                {o.price && <span className="font-mono">${o.price}</span>}
              </div>
            ))}
          </div>
        )}

        {tab === 'history' && (
          <div className="space-y-1">
            {history.length === 0 ? <div className="text-gray-500 text-sm">Keine Trades in den letzten 7 Tagen</div> : null}
            {history.map((t, i) => (
              <div key={i} className="flex items-center gap-2 p-2 bg-gray-800 rounded text-xs">
                <span className="font-mono">{t.symbol}</span>
                <span className={t.side === 'buy' ? 'text-green-400' : 'text-red-400'}>{t.side}</span>
                <span className="flex-1 text-gray-400">{t.quantity} @ ${t.price?.toFixed(4)}</span>
                <span className="text-gray-500">{t.executed_at ? new Date(t.executed_at).toLocaleDateString('de-DE') : ''}</span>
              </div>
            ))}
          </div>
        )}

        {!status?.configured && (
          <div className="text-center py-8">
            <Wallet size={32} className="mx-auto text-gray-600 mb-2" />
            <div className="text-gray-400 text-sm">Binance API Key nicht konfiguriert</div>
            <div className="text-gray-500 text-xs mt-1">Hinterlege deinen API Key in den Einstellungen</div>
          </div>
        )}
      </div>
    </div>
  )
}
