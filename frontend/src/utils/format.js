// Zeit-Formatierung
export const formatRelativeTime = (minutes) => {
  const abs = Math.abs(minutes)
  const d = Math.floor(abs / 1440)
  const h = Math.floor((abs % 1440) / 60)
  const m = abs % 60
  if (d > 0) return `${d}d ${String(h).padStart(2, '0')}h ${String(m).padStart(2, '0')}m`
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m`
  return `${m}m`
}

// Minuten zu DDHHMM String
export const minutesToDDHHMM = (totalMinutes) => {
  const days = Math.floor(Math.abs(totalMinutes) / 1440)
  const hours = Math.floor((Math.abs(totalMinutes) % 1440) / 60)
  const mins = Math.abs(totalMinutes) % 60
  return `${String(days).padStart(2, '0')}${String(hours).padStart(2, '0')}${String(mins).padStart(2, '0')}`
}

// DDHHMM String zu Minuten
export const ddhhmmToMinutes = (str) => {
  if (!str || str.length !== 6) return null
  const days = parseInt(str.substring(0, 2), 10)
  const hours = parseInt(str.substring(2, 4), 10)
  const mins = parseInt(str.substring(4, 6), 10)
  if (isNaN(days) || isNaN(hours) || isNaN(mins)) return null
  if (hours > 23 || mins > 59) return null
  return days * 1440 + hours * 60 + mins
}

// Prozent-Formatierung
export const formatPercent = (value, decimals = 2) => {
  if (value === null || value === undefined) return '-'
  return `${value >= 0 ? '+' : ''}${value.toFixed(decimals)}%`
}

// Volumen-Formatierung (K, M, B)
export const formatVolume = (value) => {
  if (value >= 1e9) return (value / 1e9).toFixed(2) + 'B'
  if (value >= 1e6) return (value / 1e6).toFixed(2) + 'M'
  if (value >= 1e3) return (value / 1e3).toFixed(2) + 'K'
  return value.toFixed(2)
}

// Datum-Formatierung
export const formatDate = (date) => {
  if (!date) return '-'
  const d = new Date(date)
  return d.toLocaleDateString('de-DE', { day: '2-digit', month: '2-digit', year: 'numeric' })
}

export const formatDateTime = (date) => {
  if (!date) return '-'
  const d = new Date(date)
  return d.toLocaleString('de-DE', { 
    day: '2-digit', month: '2-digit', year: 'numeric',
    hour: '2-digit', minute: '2-digit'
  })
}

// Zahl-Formatierung mit Dezimalstellen
export const formatNumber = (value, decimals = 2) => {
  if (value === null || value === undefined) return '-'
  const num = parseFloat(value)
  if (isNaN(num)) return '-'
  if (decimals === 0) return num.toLocaleString('de-DE', { maximumFractionDigits: 0 })
  return num.toLocaleString('de-DE', { minimumFractionDigits: decimals, maximumFractionDigits: decimals })
}
