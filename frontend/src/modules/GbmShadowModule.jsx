import { useEffect, useState } from 'react'
import api from '../utils/api'

// GBM-Predictor Paper-Shadow — regime-adaptiv, kein Echtgeld. Eigene Welt (gbm_paper_*).
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
  const be = d.backtest_expectation || {}

  const Cell = ({ label, value, color }) => (
    <div style={{ background: '#1b1b22', borderRadius: 6, padding: '8px 10px', minWidth: 92 }}>
      <div style={{ fontSize: 11, color: '#888' }}>{label}</div>
      <div style={{ fontSize: 18, fontWeight: 600, color: color || '#ddd' }}>{value}</div>
    </div>
  )

  // Equity-Kurve
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

  // Regime: +1 = Modell direkt, -1 = invertiert
  const regDirect = d.regime_sign === 1
  const regBox = (
    <div style={{
      background: '#15151b', borderRadius: 6, padding: '8px 10px', minWidth: 150,
      border: `1px solid ${regDirect ? '#2a5' : '#a52'}`
    }}>
      <div style={{ fontSize: 11, color: '#888' }}>Regime-Vorzeichen</div>
      <div style={{ fontSize: 16, fontWeight: 600, color: regDirect ? '#5d5' : '#f95' }}>
        {d.regime_sign == null ? '—' : regDirect ? 'DIREKT ▲' : 'INVERTIERT ⟳'}
      </div>
      <div style={{ fontSize: 11, color: '#888' }}>
        hit {d.regime_hit != null ? `${(d.regime_hit * 100).toFixed(1)}%` : '—'} · n {d.regime_n ?? '—'}
      </div>
    </div>
  )

  return (
    <div style={{ padding: 12, color: '#ddd', fontSize: 13, overflow: 'auto', height: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 10 }}>
        <strong>GBM-Predictor · Paper-Shadow <span style={{ color: '#6a9', fontSize: 11 }}>regime-adaptiv</span></strong>
        <span style={{ fontSize: 11, color: '#888' }}>
          gate ≥ {d.gate_threshold} · dir ≥ {d.dir_threshold} · TP +{d.tp_pct}% / SL −{d.sl_pct}% · ${d.trade_size_usd}×{d.leverage} · kein Echtgeld
        </span>
      </div>

      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 12 }}>
        {regBox}
        <Cell label="Equity (live)" value={`$${d.equity}`} color={col(d.equity - d.start_balance)} />
        <Cell label="unrealisiert" value={`${sgn(d.unrealized_usd)}$${d.unrealized_usd}`} color={col(d.unrealized_usd)} />
        <Cell label="realisiert" value={`${sgn(d.realized_usd)}$${d.realized_usd}`} color={col(d.realized_usd)} />
        <Cell label="Return" value={`${sgn(d.total_return_pct)}${d.total_return_pct}%`} color={col(d.total_return_pct)} />
        <Cell label="WR (TP/SL)" value={d.win_rate_pct != null ? `${d.win_rate_pct}%` : '—'} />
        <Cell label="Ø %/Trade" value={d.closed ? `${sgn(d.avg_pnl_pct)}${d.avg_pnl_pct}%` : '—'} color={col(d.avg_pnl_pct)} />
        <Cell label="Offen / Zu" value={`${d.open} / ${d.closed}`} />
        <Cell label="TP/SL/Timeout" value={`${d.wins}/${d.losses}/${d.drifts}`} />
      </div>

      {spark && <div style={{ marginBottom: 10 }}>{spark}</div>}

      <div style={{ background: '#15151b', borderRadius: 6, padding: '8px 10px', marginBottom: 12, fontSize: 12 }}>
        <span style={{ color: '#888' }}>Backtest (OOS, adaptives Vorzeichen): </span>
        AUC <b style={{ color: '#9c9' }}>{be.auc}</b> · Richtungstreffer <b style={{ color: '#9c9' }}>{be.dir_hit_pct}%</b>
        <span style={{ color: '#666' }}> — Paper sollte das treffen, wenn Slippage den Edge nicht frisst.</span>
      </div>

      <div style={{ display: 'flex', gap: 12, marginBottom: 6, fontSize: 12 }}>
        {['open', 'history'].map(t => (
          <span key={t} onClick={() => setTab(t)} style={{
            cursor: 'pointer', color: tab === t ? '#fff' : '#888',
            borderBottom: tab === t ? '2px solid #6a9' : 'none', paddingBottom: 2
          }}>
            {t === 'open' ? `Offen (${d.open})` : `Historie (${d.closed})`}
          </span>
        ))}
      </div>

      {tab === 'open' ? (
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead><tr style={{ color: '#888', textAlign: 'left' }}>
            <th>Coin</th><th>Seite</th><th>Entry</th><th>akt.</th><th>PnL %</th><th>PnL $</th><th>TP</th><th>SL</th><th>conf</th><th>seit</th>
          </tr></thead>
          <tbody>
            {(d.open_positions || []).map((p, i) => (
              <tr key={i} style={{ borderTop: '1px solid #2a2a33' }}>
                <td>{p.symbol}</td>
                <td style={{ color: p.side === 'long' ? '#3c3' : '#f77' }}>{p.side}</td>
                <td>{num(p.entry_px)}</td>
                <td>{num(p.current_px)}</td>
                <td style={{ color: col(p.live_pnl_pct) }}>{p.live_pnl_pct == null ? '—' : `${sgn(p.live_pnl_pct)}${p.live_pnl_pct}%`}</td>
                <td style={{ color: col(p.live_pnl_usd) }}>{p.live_pnl_usd == null ? '—' : `${sgn(p.live_pnl_usd)}$${p.live_pnl_usd}`}</td>
                <td style={{ color: '#6a9' }}>{num(p.tp_px)}</td>
                <td style={{ color: '#a66' }}>{num(p.sl_px)}</td>
                <td>{Number(p.conf).toFixed(3)}</td>
                <td style={{ color: '#888' }}>{new Date(p.opened_at).toLocaleTimeString('de-DE')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead><tr style={{ color: '#888', textAlign: 'left' }}>
            <th>Coin</th><th>Seite</th><th>Entry→Exit</th><th>PnL %</th><th>PnL $</th><th>Grund</th><th>zu</th>
          </tr></thead>
          <tbody>
            {(d.history || []).map((p, i) => (
              <tr key={i} style={{ borderTop: '1px solid #2a2a33' }}>
                <td>{p.symbol}</td>
                <td style={{ color: p.side === 'long' ? '#3c3' : '#f77' }}>{p.side}</td>
                <td>{num(p.entry_px)} → {num(p.exit_px)}</td>
                <td style={{ color: col(p.pnl_pct) }}>{sgn(p.pnl_pct)}{Number(p.pnl_pct).toFixed(2)}%</td>
                <td style={{ color: col(p.pnl_usd) }}>{sgn(p.pnl_usd)}${Number(p.pnl_usd).toFixed(2)}</td>
                <td style={{ color: p.status === 'win' ? '#3c3' : p.status === 'loss' ? '#f55' : '#aa8' }}>{p.status}</td>
                <td style={{ color: '#888' }}>{p.closed_at ? new Date(p.closed_at).toLocaleString('de-DE') : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
