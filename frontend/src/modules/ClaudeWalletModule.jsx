import { useEffect, useState } from 'react'
import api from '../utils/api'

// Claude Experimentier-Paper-Wallet — Strategie iterierbar, kein Echtgeld.
const col = (v) => (v > 0 ? '#3c3' : v < 0 ? '#f55' : '#aaa')
const sgn = (v) => (v > 0 ? '+' : '')
const num = (x, p = 5) => (x == null ? '—' : Number(x).toPrecision(p))
const dtf = (t) => (t ? new Date(t).toLocaleString('de-DE', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' }) : '—')
const vcol = (v) => (v === 'behalten' ? '#3c3' : v === 'drehen?' ? '#fa3' : v === 'wegwerfen' ? '#f55' : '#888')

export default function ClaudeWalletModule() {
  const [d, setD] = useState(null)
  const [err, setErr] = useState(null)
  const [tab, setTab] = useState('open')
  const [selStrat, setSelStrat] = useState(null)
  useEffect(() => {
    let alive = true
    const load = () => api.get('/api/v1/claudewallet/paper')
      .then(r => { if (alive) { setD(r.data); setErr(null) } })
      .catch(e => { if (alive) setErr(e?.message || 'Fehler') })
    load(); const t = setInterval(load, 5000)
    return () => { alive = false; clearInterval(t) }
  }, [])
  if (err) return <div style={{ padding: 12, color: '#f66' }}>Fehler: {err}</div>
  if (!d) return <div style={{ padding: 12, color: '#888' }}>lade…</div>

  const curve = d.equity_curve || []
  let spark = null
  if (curve.length > 1) {
    const ys = curve.map(p => p.equity)
    const min = Math.min(...ys, d.start_balance), max = Math.max(...ys, d.start_balance)
    const W = 320, H = 50, span = (max - min) || 1
    const pts = curve.map((p, i) => `${(i / (curve.length - 1)) * W},${H - ((p.equity - min) / span) * H}`).join(' ')
    const y0 = H - ((d.start_balance - min) / span) * H
    spark = <svg width={W} height={H} style={{ background: '#15151b', borderRadius: 6, width: '100%' }}>
      <line x1="0" y1={y0} x2={W} y2={y0} stroke="#444" strokeDasharray="3 3" />
      <polyline points={pts} fill="none" stroke={col(d.realized_usd)} strokeWidth="1.5" /></svg>
  }
  return (
    <div style={{ padding: 12, color: '#ddd', fontSize: 13, overflow: 'auto', height: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 10 }}>
        <strong>🧪 Claude Wallet · <span style={{ color: '#c9a', fontSize: 12 }}>{d.strategy}</span></strong>
        <span style={{ fontSize: 11, color: '#888' }}>${d.trade_size_usd}×≤{d.leverage_cap} · max {d.max_open} · TO {d.timeout_hours}h · kein Echtgeld</span>
      </div>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 10 }}>
        <Cell label="Equity" value={`$${d.equity}`} color={col(d.equity - d.start_balance)} />
        <Cell label="Return" value={`${sgn(d.total_return_pct)}${d.total_return_pct}%`} color={col(d.total_return_pct)} />
        <Cell label="real./unr." value={`${sgn(d.realized_usd)}$${d.realized_usd} / ${sgn(d.unrealized_usd)}$${d.unrealized_usd}`} />
        <Cell label="WR" value={d.win_rate_pct != null ? `${d.win_rate_pct}%` : '—'} />
        <Cell label="offen/zu" value={`${d.open} / ${d.closed}`} />
        <Cell label="TP/SL/TO" value={`${d.wins}/${d.losses}/${d.timeouts}`} />
      </div>
      {spark}
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', margin: '10px 0 4px' }}>
        {(d.strategies || []).map((st, i) => (
          <div key={i} onClick={() => setSelStrat(selStrat === st.strategy ? null : st.strategy)}
            style={{ background: selStrat === st.strategy ? '#23232e' : '#16161c', borderRadius: 6, padding: '6px 9px', borderTop: `2px solid ${col(st.realized)}`, minWidth: 130, cursor: 'pointer', outline: selStrat === st.strategy ? '1px solid #c9a' : 'none' }}>
            <div style={{ fontSize: 11, color: '#c9a', fontWeight: 600 }}>{st.strategy}</div>
            <div style={{ fontSize: 14, fontWeight: 600, color: col(st.realized) }}>{sgn(st.realized)}${st.realized}</div>
            <div style={{ fontSize: 10, color: '#888' }}>{st.closed} zu · {st.open} offen · WR {st.wr != null ? st.wr + '%' : '—'}</div>
            <div style={{ fontSize: 10, color: '#888' }}>TP {st.tp} / SL {st.sl} / TO {st.tmo}</div>
            <div style={{ fontSize: 10, fontWeight: 700, color: st.verdict === 'Edge+' ? '#3c3' : st.verdict === 'Edge−' ? '#f55' : '#888' }}>
              {st.verdict}{st.exp != null ? ` · ${sgn(st.exp)}$${st.exp}/Trade` : ''}</div>
          </div>
        ))}
      </div>
      {(d.edgemap || []).length > 0 && (
        <div style={{ margin: '8px 0 4px', padding: '6px 8px', background: '#141419', borderRadius: 6 }}>
          <div style={{ fontSize: 11, color: '#c9a', fontWeight: 600, marginBottom: 4 }}>🔬 Edge-Karte — Muster: <span style={{ color: '#3c3' }}>behalten</span> / <span style={{ color: '#fa3' }}>drehen?</span> / <span style={{ color: '#f55' }}>wegwerfen</span> / <span style={{ color: '#888' }}>zu früh</span></div>
          {(d.edgemap || []).map((e, i) => (
            <div key={i} style={{ fontSize: 10, color: '#aaa', marginBottom: 3, lineHeight: 1.5 }}>
              <b style={{ color: '#c9a' }}>{e.strategy}</b>{' '}
              <span style={{ color: vcol(e.overall.verdict), fontWeight: 600 }}>gesamt: {e.overall.verdict}</span>
              <span style={{ color: '#888' }}> ({e.overall.n}T{e.overall.exp != null ? `, ${sgn(e.overall.exp)}$${e.overall.exp}/T` : ''})</span>
              {(e.sides || []).map((s, j) => (
                <span key={j} style={{ marginLeft: 8 }}>· {s.side} <span style={{ color: vcol(s.verdict), fontWeight: 600 }}>{s.verdict}</span><span style={{ color: '#888' }}> ({s.n}T{s.exp != null ? `, ${sgn(s.exp)}$${s.exp}` : ''})</span>
                  {(s.tiers || []).map((t, k) => (
                    <span key={k} style={{ color: '#777', marginLeft: 4 }}>[{t.tier} {t.verdict} {t.n}T]</span>
                  ))}
                </span>
              ))}
            </div>
          ))}
        </div>
      )}
      <div style={{ display: 'flex', gap: 12, margin: '6px 0 4px', fontSize: 12, alignItems: 'center' }}>
        {['open', 'history'].map(t => (
          <span key={t} onClick={() => setTab(t)} style={{ cursor: 'pointer', color: tab === t ? '#fff' : '#888', borderBottom: tab === t ? '2px solid #c9a' : 'none', paddingBottom: 2 }}>
            {t === 'open' ? 'Offen' : 'Historie'}</span>))}
        <span style={{ fontSize: 11, color: '#c9a' }}>{selStrat ? `Filter: ${selStrat} (Box klicken zum Lösen)` : 'alle Modelle (Box klicken zum Filtern)'}</span>
      </div>
      {tab === 'open' ? (
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
          <thead><tr style={{ color: '#888', textAlign: 'left' }}><th>Coin</th><th></th><th>Trig</th><th>TP/SL</th><th>Hbl</th><th>PnL%</th><th>auf</th></tr></thead>
          <tbody>{(d.open_positions || []).filter(p => !selStrat || p.strategy === selStrat).map((p, i) => (
            <tr key={i} style={{ borderTop: '1px solid #2a2a33' }}>
              <td>{p.symbol}</td><td style={{ color: p.side === 'long' ? '#3c3' : '#f77', fontWeight: 700 }}>{p.side === 'long' ? '▲' : '▼'}</td>
              <td style={{ color: '#888' }}>{p.trigger_ret == null ? '—' : `${sgn(p.trigger_ret)}${Number(p.trigger_ret).toFixed(1)}%`}</td>
              <td style={{ color: '#888' }}>+{p.tp_pct}/-{p.sl_pct}</td><td style={{ color: '#9ac' }}>{p.leverage}x</td>
              <td style={{ color: col(p.live_pnl_pct) }}>{p.live_pnl_pct == null ? '—' : `${sgn(p.live_pnl_pct)}${p.live_pnl_pct}%`}</td>
              <td style={{ color: '#888' }}>{dtf(p.opened_at)}</td></tr>))}</tbody>
        </table>
      ) : (
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
          <thead><tr style={{ color: '#888', textAlign: 'left' }}><th>Coin</th><th></th><th>Entry→Exit</th><th>PnL%</th><th>PnL$</th><th>Grund</th><th>zu</th></tr></thead>
          <tbody>{(d.history || []).filter(p => !selStrat || p.strategy === selStrat).map((p, i) => (
            <tr key={i} style={{ borderTop: '1px solid #2a2a33' }}>
              <td>{p.symbol}</td><td style={{ color: p.side === 'long' ? '#3c3' : '#f77', fontWeight: 700 }}>{p.side === 'long' ? '▲' : '▼'}</td>
              <td>{num(p.entry_px)}→{num(p.exit_px)}</td>
              <td style={{ color: col(p.pnl_pct) }}>{sgn(p.pnl_pct)}{Number(p.pnl_pct).toFixed(2)}%</td>
              <td style={{ color: col(p.pnl_usd) }}>{sgn(p.pnl_usd)}${Number(p.pnl_usd).toFixed(2)}</td>
              <td style={{ color: p.exit_reason === 'TP' ? '#3c3' : p.exit_reason === 'SL' ? '#f55' : '#aa8' }}>{p.exit_reason}</td>
              <td style={{ color: '#888' }}>{dtf(p.closed_at)}</td></tr>))}</tbody>
        </table>
      )}
    </div>
  )
}
function Cell({ label, value, color }) {
  return <div style={{ background: '#1b1b22', borderRadius: 6, padding: '6px 9px', minWidth: 80 }}>
    <div style={{ fontSize: 10, color: '#888' }}>{label}</div>
    <div style={{ fontSize: 15, fontWeight: 600, color: color || '#ddd' }}>{value}</div></div>
}
