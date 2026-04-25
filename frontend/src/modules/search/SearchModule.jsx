import { useState, useEffect } from 'react'
import { Search } from 'lucide-react'
import { useSearchStore } from '../../stores/searchStore'
import { useConfigStore } from '../../stores/configStore'
import DirectionFilter from './DirectionFilter'
import PercentFilter from './PercentFilter'
import DurationFilter from './DurationFilter'
import DateFilter from './DateFilter'
import GroupFilter from './GroupFilter'
import AdvancedFilters from './AdvancedFilters'
import HLOnlyToggle from './HLOnlyToggle'

export default function SearchModule() {
  const { searchParams, setSearchParams, search, isSearching, searchError } = useSearchStore()
  const { getKlineMetricsDurations } = useConfigStore()
  const durations = getKlineMetricsDurations()

  const handleSearch = async (e) => {
    e.preventDefault()
    try { await search() } catch {}
  }

  return (
    <form onSubmit={handleSearch} className="space-y-3">
      <DirectionFilter
        direction={searchParams.direction}
        onChange={(d) => setSearchParams({ direction: d })}
      />

      <PercentFilter
        min={searchParams.targetPercent}
        max={searchParams.maxPercent}
        onChangeMin={(v) => setSearchParams({ targetPercent: v })}
        onChangeMax={(v) => setSearchParams({ maxPercent: v })}
      />

      <DurationFilter
        value={searchParams.durationMinutes}
        onChange={(v) => setSearchParams({ durationMinutes: v })}
        durations={durations}
      />

      <DateFilter
        startDate={searchParams.startDate}
        endDate={searchParams.endDate}
        onChange={(start, end) => setSearchParams({ startDate: start, endDate: end })}
      />

      <GroupFilter
        selectedIds={searchParams.groupIds}
        onChange={(ids) => setSearchParams({ groupIds: ids })}
      />

      <AdvancedFilters
        weekdays={searchParams.weekdays}
        hourStart={searchParams.hourStart}
        hourEnd={searchParams.hourEnd}
        onWeekdaysChange={(w) => setSearchParams({ weekdays: w })}
        onHoursChange={(s, e) => setSearchParams({ hourStart: s, hourEnd: e })}
      />

      <HLOnlyToggle
        checked={searchParams.hlOnly !== false}
        onChange={(v) => setSearchParams({ hlOnly: v })}
      />

      {searchError && (
        <div className="text-red-400 text-sm bg-red-900/30 p-2 rounded">{searchError}</div>
      )}

      <button
        type="submit"
        disabled={isSearching || !searchParams.startDate || !searchParams.endDate}
        className="btn btn-primary w-full flex items-center justify-center gap-2 disabled:opacity-50 text-sm"
      >
        <Search size={14} />
        {isSearching ? 'Suche laeuft...' : 'Suchen'}
      </button>
    </form>
  )
}
