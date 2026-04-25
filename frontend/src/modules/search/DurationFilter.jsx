export default function DurationFilter({ value, onChange, durations }) {
  return (
    <div>
      <label className="block text-gray-400 text-xs mb-1">Zeitfenster</label>
      <select
        value={value}
        onChange={e => onChange(parseInt(e.target.value))}
        className="input w-full text-sm"
      >
        {durations.map(d => (
          <option key={d} value={d}>{d} Min</option>
        ))}
      </select>
    </div>
  )
}
