import { useEffect, useState } from 'react'
import api from '../utils/api'

// nnf-Predictor: 2 Varianten parallel als Paper-Shadow. A=OHLCV(57), B=+Funding(63).
// Reach-TP/SL, Observer (schließt bei Richtungswechsel), online-lernend. Kein Echtgeld.
const col = (v) => (v > 0 ? '#3c3' : v < 0 ? '#f55' : '#aaa')
const sgn = (v) => (v > 0 ? '+' : '')
const num = (x, p = 5) => (x == null ? '—' : Number(x).toPrecision(p))
const dtf = (t) => (t ? new Date(t).toLocaleString('de-DE', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' }) : '—')

function Variant({ v, accent }) {
  const [tab, setTab] = useState('open')
  if (!v) return null

  const curve = v.equity_curve || []
  let spark = null
  if (curve.length > 1) {
    const ys = curve.map(p => p.equity)
    const min = Math.min(...ys, v.start_balance), max = Math.max(...ys, v.start_balance)
    const W = 300, H = 44, span = (max - min) || 1
    const pts = curve.map((p, i) => `${(i / (curve.length - 1)) * W},${H - ((p.equity - min) / span) * H}`).join(' ')
    const y0 = H - ((v.start_balance - min) / span) * H
    spark = (
      <svg width={W} height={H} style={{ background: '#15151b', borderRadius: 6, width: '100%' }}>
        <line x1="0" y1={y0} x2={W} y2={y0} stroke="#444" strokeDasharray="3 3" />
        <polyline points={pts} fill="none" stroke={col(v.realized_usd)} strokeWidth="1.5" />
      </svg>
    )
  }

  return (
    <div style={{ flex: 1, minWidth: 300, background: '#16161c', borderRadius: 8, padding: 10, borderTop: `2px solid ${accent}` }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
        <strong style={{ color: accent }}>{v.label}</strong>
        <span style={{ fontSize: 11, color: '#888' }}>WR {v.win_rate_pct != null ? `${v.win_rate_pct}%` : '—'} · gelernt {v.learned}</span>
      </div>
      <div style={{ display: 'flex', gap: 16, marginBottom: 8 }}>
        <div><div style={{ fontSize: 10, color: '#888' }}>Equity</div><div style={{ fontSize: 17, fontWeight: 600, color: col(v.equity - v.start_balance) }}>${v.equity}</div></div>
        <div><div style={{ fontSize: 10, color: '#888' }}>Return</div><div style={{ fontSize: 17, fontWeight: 600, color: col(v.total_return_pct) }}>{sgn(v.total_return_pct)}{v.total_return_pct}%</div></div>
        <div><div style={{ fontSize: 10, color: '#888' }}>real./unr.</div><div style={{ fontSize: 13, color: '#bbb' }}>{sgn(v.realized_usd)}${v.realized_usd} / {sgn(v.unrealized_usd)}${v.unrealized_usd}</div></div>
        <div><div style={{ fontSize: 10, color: '#888' }}>offen/zu</div><div style={{ fontSize: 13, color: '#bbb' }}>{v.open} / {v.closed}</div></div>
      </div>
      {spark}
      <div style={{ display: 'flex', gap: 12, margin: '8px 0 4px', fontSize: 12 }}>
        {['open', 'history'].map(t => (
          <span key={t} onClick={() => setTab(t)} style={{ cursor: 'pointer', color: tab === t ? '#fff' : '#888', borderBottom: tab === t ? `2px solid ${accent}` : 'none', paddingBottom: 2 }}>
            {t === 'open' ? `Offen (${v.open})` : `Historie (${v.closed})`}
          </span>
        ))}
      </div>
      {tab === 'open' ? (
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
          <thead><tr style={{ color: '#888', textAlign: 'left' }}><th>Coin</th><th></th><th>conf</th><th>TP/SL</th><th>Hbl</th><th>PnL%</th><th>gekauft</th></tr></thead>
          <tbody>
            {(v.open_positions || []).map((p, i) => (
              <tr key={i} style={{ borderTop: '1px solid #2a2a33' }}>
                <td>{p.symbol}</td>
                <td style={{ color: p.side === 'long' ? '#3c3' : '#f77', fontWeight: 700 }}>{p.side === 'long' ? '▲' : '▼'}</td>
                <td style={{ color: '#9ac' }}>{p.conf == null ? '—' : Number(p.conf).toFixed(2)}</td>
                <td style={{ color: '#888' }}>{p.tp_pct == null ? '—' : `+${Number(p.tp_pct).toFixed(1)}/-${Number(p.sl_pct).toFixed(1)}`}</td>
                <td style={{ color: '#9ac' }}>{p.leverage == null ? '—' : `${p.leverage}x`}</td>
                <td style={{ color: col(p.live_pnl_pct) }}>{p.live_pnl_pct == null ? '—' : `${sgn(p.live_pnl_pct)}${p.live_pnl_pct}%`}</td>
                <td style={{ color: '#888' }}>{dtf(p.opened_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
          <thead><tr style={{ color: '#888', textAlign: 'left' }}><th>Coin</th><th></th><th>Entry→Exit</th><th>PnL%</th><th>PnL$</th><th>Grund</th><th>gekauft</th><th>verkauft</th></tr></thead>
          <tbody>
            {(v.history || []).map((p, i) => (
              <tr key={i} style={{ borderTop: '1px solid #2a2a33' }}>
                <td>{p.symbol}</td>
                <td style={{ color: p.side === 'long' ? '#3c3' : '#f77', fontWeight: 700 }}>{p.side === 'long' ? '▲' : '▼'}</td>
                <td>{num(p.entry_px)} → {num(p.exit_px)}</td>
                <td style={{ color: col(p.pnl_pct) }}>{sgn(p.pnl_pct)}{Number(p.pnl_pct).toFixed(2)}%</td>
                <td style={{ color: col(p.pnl_usd) }}>{sgn(p.pnl_usd)}${Number(p.pnl_usd).toFixed(2)}</td>
                <td style={{ color: p.exit_reason === 'TP' ? '#3c3' : p.exit_reason === 'SL' ? '#f55' : '#aa8' }}>{p.exit_reason}</td>
                <td style={{ color: '#888' }}>{dtf(p.opened_at)}</td>
                <td style={{ color: '#888' }}>{dtf(p.closed_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

export default function NnfPredictorModule() {
  const [d, setD] = useState(null)
  const [err, setErr] = useState(null)

  useEffect(() => {
    let alive = true
    const load = () => api.get('/api/v1/nnf/paper')
      .then(r => { if (alive) { setD(r.data); setErr(null) } })
      .catch(e => { if (alive) setErr(e?.message || 'Fehler') })
    load(); const t = setInterval(load, 5000)
    return () => { alive = false; clearInterval(t) }
  }, [])

  if (err) return <div style={{ padding: 12, color: '#f66' }}>Fehler: {err}</div>
  if (!d) return <div style={{ padding: 12, color: '#888' }}>lade…</div>

  return (
    <div style={{ padding: 12, color: '#ddd', fontSize: 13, overflow: 'auto', height: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 10 }}>
        <strong>nnf-Predictor · <span style={{ color: '#6a9', fontSize: 12 }}>2 Varianten, Reach-TP/SL, Observer, online-lernend</span></strong>
        <span style={{ fontSize: 11, color: '#888' }}>conf≥{d.min_conf} · ${d.trade_size_usd}×≤{d.leverage_cap} · max {d.max_open} · Timeout {d.timeout_hours}h · kein Echtgeld</span>
      </div>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        <Variant v={d.a} accent="#7aa6ff" />
        <Variant v={d.b} accent="#e0a050" />
        <Variant v={d.c} accent="#5cc98a" />
      </div>
    </div>
  )
}
