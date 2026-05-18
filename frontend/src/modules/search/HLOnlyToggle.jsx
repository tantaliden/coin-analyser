export default function HLOnlyToggle({ checked, onChange }) {
  return (
    <label className="flex items-center gap-2 cursor-pointer">
      <div className={`relative w-9 h-5 rounded-full transition-colors ${checked ? 'bg-blue-600' : 'bg-gray-600'}`}
        onClick={() => onChange(!checked)}>
        <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${checked ? 'translate-x-4' : 'translate-x-0.5'}`} />
      </div>
      <span className="text-xs text-gray-400">Nur Hyperliquid</span>
    </label>
  )
}
