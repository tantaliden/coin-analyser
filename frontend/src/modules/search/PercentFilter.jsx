export default function PercentFilter({ min, max, onChangeMin, onChangeMax }) {
  return (
    <div className="flex gap-2">
      <div className="flex-1">
        <label className="block text-gray-400 text-xs mb-1">Min %</label>
        <input
          type="number"
          value={min}
          onChange={e => onChangeMin(parseFloat(e.target.value))}
          className="input w-full text-sm"
          step="any"
        />
      </div>
      <div className="flex-1">
        <label className="block text-gray-400 text-xs mb-1">Max %</label>
        <input
          type="number"
          value={max == null ? '' : max}
          onChange={e => onChangeMax(e.target.value === '' ? null : parseFloat(e.target.value))}
          className="input w-full text-sm"
          step="any"
          placeholder="unbegrenzt"
        />
      </div>
    </div>
  )
}
