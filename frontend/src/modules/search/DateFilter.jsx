import DateRangePicker from '../../components/DateRangePicker'

export default function DateFilter({ startDate, endDate, onChange }) {
  return (
    <DateRangePicker
      startDate={startDate}
      endDate={endDate}
      onChange={onChange}
    />
  )
}
