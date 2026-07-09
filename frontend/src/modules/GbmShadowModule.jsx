import { useEffect, useState } from 'react'
import api from '../utils/api'

// GBM Cross-Sectional Composite — marktneutraler Paper-Shadow (Top-k long / Bottom-k short,
// alle Horizonte roh+residual gleichgewichtet). Einziger Ansatz der alle ehrlichen Tests ueberlebte.
export default function GbmShadowModule() {
  const [d, setD] = useState(null)
  const [err, setErr] = useState(null)
  const [tab, setTab] = useState('open')

  useEffect(() => {
    let alive = true
    const load = () => api.get('/api/v1/gbm/paper')
      .then(r => { if (alive) { setD(r.data); setErr(null) } })
      .catch(e => { if (alive) setErr(e?.message || 'Fehler') })
    load()
    const t = setInterval(load, 5000)
    return () => { alive = false; clearInterval(t) }
  }, [])

  if (err) return <div style={{ padding: 12, color: '#f66' }}>Fehler: {err}</div>
  if (!d) return <div style={{ padding: 12, color: '#888' }}>lade…</div>

  const col = (v) => (v > 0 ? '#3c3' : v < 0 ? '#f55' : '#aaa')
  const sgn = (v) => (v > 0 ? '+' : '')
  const num = (x, p = 5) => (x == null ? '—' : Number(x).toPrecision(p))

  const Cell = ({ label, value, color }) => (
    <div style={{ background: '#1b1b22', borderRadius: 6, padding: '8px 10px', minWidth: 92 }}>
      <div style={{ fontSize: 11, color: '#888' }}>{label}</div>
      <div style={{ fontSize: 18, fontWeight: 600, color: color || '#ddd' }}>{value}</div>
    </div>
  )

  const curve = d.equity_curve || []
  let spark = null
  if (curve.length > 1) {
    const ys = curve.map(p => p.equity)
    const min = Math.min(...ys, d.start_balance), max = Math.max(...ys, d.start_balance)
    const W = 320, H = 60, span = (max - min) || 1
    const pts = curve.map((p, i) =>
      `${(i / (curve.length - 1)) * W},${H - ((p.equity - min) / span) * H}`).join(' ')
    const y0 = H - ((d.start_balance - min) / span) * H
    spark = (
      <svg width={W} height={H} style={{ background: '#15151b', borderRadius: 6 }}>
        <line x1="0" y1={y0} x2={W} y2={y0} stroke="#444" strokeDasharray="3 3" />
        <polyline points={pts} fill="none" stroke={col(d.realized_usd)} strokeWidth="1.5" />
      </svg>
    )
  }

  const longs = (d.open_positions || []).filter(o => o.side === 'long')
  const shorts = (d.open_positions || []).filter(o => o.side === 'short')

  const posRow = (o) => (
    <tr key={o.id} style={{ borderTop: '1px solid #2a2a33' }}>
      <td style={{ color: o.side === 'long' ? '#3c3' : '#f55', fontWeight: 600 }}>{o.side}</td>
      <td>{o.symbol}</td>
      <td style={{ textAlign: 'right', color: '#aaa' }}>{o.score}</td>
      <td style={{ textAlign: 'right' }}>{num(o.entry_px)}</td>
      <td style={{ textAlign: 'right' }}>{num(o.current_px)}</td>
      <td style={{ textAlign: 'right', color: col(o.live_pnl_usd) }}>
        {o.live_pnl_usd == null ? '—' : `${sgn(o.live_pnl_usd)}${o.live_pnl_usd}$`}
      </td>
      <td style={{ textAlign: 'right', color: col(o.live_pnl_pct) }}>
        {o.live_pnl_pct == null ? '—' : `${sgn(o.live_pnl_pct)}${o.live_pnl_pct}%`}
      </td>
    </tr>
  )

  return (
    <div style={{ padding: 12, color: '#ddd', fontSize: 13, overflow: 'auto', height: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 10 }}>
        <strong>Cross-Sectional · marktneutral · Paper-Shadow</strong>
        <span style={{ fontSize: 11, color: '#888' }}>
          Top-{d.k} long / Bottom-{d.k} short · Horizonte {(d.lookback_hours_list || []).join('/')}h · Rebalance {d.rebalance_hours}h · {d.leverage}x · ${d.trade_size_usd}/Bein · kein Echtgeld
        </span>
      </div>

      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 10 }}>
        <Cell label="Equity" value={`$${num(d.equity, 7)}`} color={col(d.equity - d.start_balance)} />
        <Cell label="Return" value={`${sgn(d.total_return_pct)}${d.total_return_pct}%`} color={col(d.total_return_pct)} />
        <Cell label="Realisiert" value={`${sgn(d.realized_usd)}${d.realized_usd}$`} color={col(d.realized_usd)} />
        <Cell label="Unrealisiert" value={`${sgn(d.unrealized_usd)}${d.unrealized_usd}$`} color={col(d.unrealized_usd)} />
        <Cell label="Offen" value={`${longs.length}L / ${shorts.length}S`} />
        <Cell label="Rebalances" value={d.closed} />
        <Cell label="Bein-WR" value={d.leg_win_rate_pct == null ? '—' : `${d.leg_win_rate_pct}%`} />
      </div>

      {spark && <div style={{ marginBottom: 10 }}>{spark}</div>}

      <div style={{ display: 'flex', gap: 8, marginBottom: 6 }}>
        {['open', 'history'].map(t => (
          <button key={t} onClick={() => setTab(t)}
            style={{ background: tab === t ? '#2a2a40' : '#1b1b22', color: '#ddd', border: '1px solid #333',
                     borderRadius: 5, padding: '4px 10px', cursor: 'pointer', fontSize: 12 }}>
            {t === 'open' ? `Offen (${d.open})` : `Historie (${(d.history || []).length})`}
          </button>
        ))}
      </div>

      {tab === 'open' && (
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead><tr style={{ color: '#888', textAlign: 'left' }}>
            <th>Seite</th><th>Coin</th><th style={{ textAlign: 'right' }}>Score</th>
            <th style={{ textAlign: 'right' }}>Entry</th><th style={{ textAlign: 'right' }}>Jetzt</th>
            <th style={{ textAlign: 'right' }}>PnL$</th><th style={{ textAlign: 'right' }}>PnL%</th>
          </tr></thead>
          <tbody>
            {longs.map(posRow)}{shorts.map(posRow)}
            {(d.open_positions || []).length === 0 &&
              <tr><td colSpan="7" style={{ color: '#888', padding: 8 }}>keine offenen Positionen</td></tr>}
          </tbody>
        </table>
      )}

      {tab === 'history' && (
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead><tr style={{ color: '#888', textAlign: 'left' }}>
            <th>Seite</th><th>Coin</th><th style={{ textAlign: 'right' }}>Entry</th>
            <th style={{ textAlign: 'right' }}>Exit</th><th style={{ textAlign: 'right' }}>PnL$</th>
            <th style={{ textAlign: 'right' }}>PnL%</th><th>zu</th>
          </tr></thead>
          <tbody>
            {(d.history || []).map((h, i) => (
              <tr key={i} style={{ borderTop: '1px solid #2a2a33' }}>
                <td style={{ color: h.side === 'long' ? '#3c3' : '#f55' }}>{h.side}</td>
                <td>{h.symbol}</td>
                <td style={{ textAlign: 'right' }}>{num(h.entry_px)}</td>
                <td style={{ textAlign: 'right' }}>{num(h.exit_px)}</td>
                <td style={{ textAlign: 'right', color: col(h.pnl_usd) }}>{sgn(h.pnl_usd)}{Number(h.pnl_usd).toFixed(2)}$</td>
                <td style={{ textAlign: 'right', color: col(h.pnl_pct) }}>{sgn(h.pnl_pct)}{Number(h.pnl_pct).toFixed(2)}%</td>
                <td style={{ color: '#888', fontSize: 11 }}>{h.closed_at ? new Date(h.closed_at).toLocaleString('de-DE', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' }) : '—'}</td>
              </tr>
            ))}
            {(d.history || []).length === 0 &&
              <tr><td colSpan="7" style={{ color: '#888', padding: 8 }}>noch keine Rebalance-Schließungen</td></tr>}
          </tbody>
        </table>
      )}

      <div style={{ marginTop: 10, fontSize: 11, color: '#666' }}>{d.note}</div>
    </div>
  )
}
