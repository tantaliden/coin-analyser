// Zeigt OHLCV + Kursaenderungen bei Hover ueber eine Candle
import { CHART_SETTINGS } from '../../config/chartSettings'

export default function CandleInfoBar({ candle, candles, index }) {
  if (!candle || !candles?.length) return null

  const prev1h = candles[Math.max(0, index - 60)]
  const prev4h = candles[Math.max(0, index - 240)]
  const first = candles[0]

  const pct = (ref) => {
    if (!ref) return null
    return ((candle.close - ref.close) / ref.close * 100).toFixed(2)
  }

  const pctClass = (v) => parseFloat(v) >= 0 ? 'text-green-400' : 'text-red-400'
  const c1h = pct(prev1h), c4h = pct(prev4h), cTotal = pct(first)

  return (
    <div className="flex items-center gap-1.5 px-2 py-0.5 text-[10px] bg-gray-900/60 border-t border-gray-700/50 font-mono">
      <span className="text-gray-500">O:</span>
      <span className={candle.close >= candle.open ? 'text-green-400' : 'text-red-400'}>{candle.open.toFixed(4)}</span>
      <span className="text-gray-500">H:</span>
      <span className="text-green-400">{candle.high.toFixed(4)}</span>
      <span className="text-gray-500">L:</span>
      <span className="text-red-400">{candle.low.toFixed(4)}</span>
      <span className="text-gray-500">C:</span>
      <span className={candle.close >= candle.open ? 'text-green-400' : 'text-red-400'}>{candle.close.toFixed(4)}</span>
      <span className="text-gray-600">|</span>
      <span className="text-gray-500">V:</span>
      <span className="text-blue-400">{Math.round(candle.volume).toLocaleString()}</span>
      {c1h && (
        <>
          <span className="text-gray-600">|</span>
          <span className="text-gray-500">1h:</span>
          <span className={pctClass(c1h)}>{c1h}%</span>
        </>
      )}
      {c4h && (
        <>
          <span className="text-gray-500">4h:</span>
          <span className={pctClass(c4h)}>{c4h}%</span>
        </>
      )}
      {cTotal && (
        <>
          <span className="text-gray-500">Ges:</span>
          <span className={pctClass(cTotal)}>{cTotal}%</span>
        </>
      )}
      {candle.funding != null && (
        <>
          <span className="text-gray-600">||</span>
          <span className="text-gray-500">Fnd:</span>
          <span className={candle.funding >= 0 ? 'text-emerald-300' : 'text-amber-300'}>
            {(candle.funding * 100).toFixed(4)}%
          </span>
        </>
      )}
      {candle.open_interest != null && (
        <>
          <span className="text-gray-500">OI:</span>
          <span className="text-cyan-300">{candle.open_interest.toLocaleString(undefined, {maximumFractionDigits: 0})}</span>
        </>
      )}
      {candle.premium != null && (
        <>
          <span className="text-gray-500">Prm:</span>
          <span className={candle.premium >= 0 ? 'text-emerald-300' : 'text-amber-300'}>
            {(candle.premium * 100).toFixed(3)}%
          </span>
        </>
      )}
      {candle.spread_bps != null && (
        <>
          <span className="text-gray-600">|</span>
          <span className="text-gray-500">Sprd:</span>
          <span className="text-gray-300">{candle.spread_bps.toFixed(1)}bps</span>
        </>
      )}
      {candle.book_imbalance_5 != null && (
        <>
          <span className="text-gray-500">Imb:</span>
          <span className={candle.book_imbalance_5 >= 0 ? 'text-emerald-300' : 'text-amber-300'}>
            {candle.book_imbalance_5.toFixed(2)}
          </span>
        </>
      )}
      {candle.taker_buy_base != null && candle.volume > 0 && (
        <>
          <span className="text-gray-500">Buy%:</span>
          <span className="text-purple-300">
            {((candle.taker_buy_base / candle.volume) * 100).toFixed(1)}%
          </span>
        </>
      )}
    </div>
  )
}
