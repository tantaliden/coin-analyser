// Kleines Ergebnis-Panel fuer Scan-Ergebnisse (GREEN/GREY/RED + Treffer-Liste)
export default function ScanResultsView({ scanResult }) {
  if (!scanResult) return null
  const stats = scanResult.statistics
  const results = scanResult.results || []

  const catColor = (cat) => cat === 'GREEN' ? 'text-green-400' :
    cat === 'GREY' ? 'text-gray-400' :
    cat === 'RED' ? 'text-red-400' : 'text-gray-500'

  return (
    <div className="space-y-2 text-xs">
      <div className="p-2 bg-gray-900/50 rounded border border-gray-700">
        <div className="text-gray-300 font-semibold mb-1">
          {scanResult.set_name} - {scanResult.total_found} Treffer
        </div>
        <div className="text-[10px] text-gray-500">
          {scanResult.period_days}T | {scanResult.symbols_scanned} Coins | Ziel {scanResult.target_percent}%
        </div>
        {stats?.total_with_label > 0 && (
          <div className="flex gap-2 mt-1 text-[10px]">
            <span className="text-green-400">GREEN {stats.green} ({stats.green_percent}%)</span>
            <span className="text-gray-400">GREY {stats.grey} ({stats.grey_percent}%)</span>
            <span className="text-red-400">RED {stats.red} ({stats.red_percent}%)</span>
          </div>
        )}
      </div>

      <div className="max-h-96 overflow-auto space-y-0.5">
        {results.map((r, i) => (
          <div key={i} className="flex items-center justify-between px-2 py-1 bg-gray-900/30 rounded">
            <span className="font-mono text-gray-200">{r.symbol}</span>
            <span className="text-gray-500 text-[10px]">{r.event_time.slice(5, 16).replace('T', ' ')}</span>
            <div className="flex items-center gap-2">
              {r.pct_label != null && (
                <span className={catColor(r.category)}>
                  {r.pct_label >= 0 ? '+' : ''}{r.pct_label.toFixed(1)}%
                </span>
              )}
              <span className="text-blue-400 font-mono">{(r.match_score * 100).toFixed(0)}%</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
