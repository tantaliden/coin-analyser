// Syncron-Dienst: laedt Session-State vom Backend nach Login und
// speichert Aenderungen debounced wieder ins Backend.
// Backend ist Source of Truth — localStorage nur als Offline-Puffer.

import api from '../utils/api'
import { useSearchStore } from './searchStore'
import { useModuleStore } from './moduleStore'
import { useDrawingsStore } from './drawingsStore'

const DEBOUNCE_MS = 1500

let saveTimer = null
let hydrated = false
let saveInFlight = null

// Welche Felder aus searchStore auf Backend syncen
function extractSearch(state) {
  return {
    searchParams: state.searchParams,
    selectedEvents: state.selectedEvents,
    prehistoryMinutes: state.prehistoryMinutes,
    selectedSetId: state.selectedSetId,
    indicatorChain: state.indicatorChain,
  }
}

function extractModule(state) {
  return {
    activeModule: state.activeModule,
    layouts: state.layouts,
    enabledModules: state.enabledModules,
  }
}

async function loadFromBackend() {
  const res = await api.get('/api/v1/user/session-state')
  const data = res.data

  if (data.search && Object.keys(data.search).length > 0) {
    useSearchStore.setState((prev) => ({ ...prev, ...data.search }))
  }
  if (data.module && Object.keys(data.module).length > 0) {
    useModuleStore.setState((prev) => ({ ...prev, ...data.module }))
  }
  if (data.drawings && Object.keys(data.drawings).length > 0) {
    useDrawingsStore.getState().setAll(data.drawings)
  }
  hydrated = true
}

async function flushToBackend() {
  const search = extractSearch(useSearchStore.getState())
  const moduleState = extractModule(useModuleStore.getState())
  const drawings = useDrawingsStore.getState().drawings

  saveInFlight = api.put('/api/v1/user/session-state', {
    search, module: moduleState, drawings,
  })
  try { await saveInFlight } finally { saveInFlight = null }
}

function scheduleSave() {
  if (!hydrated) return  // Nicht speichern bevor geladen — sonst ueberschreiben wir Backend mit leerem State
  if (saveTimer) clearTimeout(saveTimer)
  saveTimer = setTimeout(flushToBackend, DEBOUNCE_MS)
}

// Subscribe: jede Aenderung in einem der drei Stores triggert debounced save
function installSubscriptions() {
  useSearchStore.subscribe(scheduleSave)
  useModuleStore.subscribe(scheduleSave)
  useDrawingsStore.subscribe(scheduleSave)
}

export async function initSessionSync() {
  await loadFromBackend()
  installSubscriptions()
}

// Optionaler sofortiger Flush (z.B. vor logout/navigate)
export async function forceFlush() {
  if (saveTimer) { clearTimeout(saveTimer); saveTimer = null }
  await flushToBackend()
}
