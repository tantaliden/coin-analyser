import { create } from 'zustand'
import api from '../utils/api'

// Debounce Helper
let saveTimeout = null
const debouncedSave = (saveFunc, delay = 2000) => {
  if (saveTimeout) clearTimeout(saveTimeout)
  saveTimeout = setTimeout(saveFunc, delay)
}

export const useSearchStore = create((set, get) => ({
  // Such-Filter
  searchParams: {
    direction: 'up',
    targetPercent: 5,
    maxPercent: 100,
    durationMinutes: 120,
    startDate: '',
    endDate: '',
    groupIds: [],
    weekdays: [],
    hourStart: -1,
    hourEnd: -1,
  },

  // Such-Ergebnisse (Primär)
  results: [],
  isSearching: false,
  searchError: null,
  loading: false,
  
  // Cascade: Sekundäre Ergebnisse nach Indikator-Filter
  cascadeResults: [],
  
  // Indikator-Chain (alle angewendeten Indikatoren)
  indicatorChain: [],
  setIndicatorChain: (chain) => set({ indicatorChain: chain }),
  
  // Ausgewähltes Set
  selectedSetId: null,
  setSelectedSetId: (id) => set({ selectedSetId: id }),
  
  // Prehistory (Vorlaufzeit)
  prehistoryMinutes: 720,
  setPrehistoryMinutes: (minutes) => set({ prehistoryMinutes: minutes }),
  
  // Ausgewählte Events (max 32)
  selectedEvents: [],
  maxSelection: 32,
  
  // Hauptsuche Checkbox
  useMainSearch: true,
  setUseMainSearch: (value) => set({ useMainSearch: value }),
  
  // === FILTER SETTERS ===
  
  setSearchParams: (params) => {
    set({ searchParams: { ...get().searchParams, ...params } })
  },

  setResults: (results) => set({ results }),

  // === SEARCH ===
  
  search: async () => {
    const { searchParams } = get()
    set({ isSearching: true, loading: true, searchError: null })

    if (!searchParams.startDate || !searchParams.endDate) {
      set({ searchError: 'Start- und Enddatum erforderlich', isSearching: false, loading: false })
      throw new Error('Start- und Enddatum erforderlich')
    }

    try {
      const params = {
        direction: searchParams.direction,
        min_percent: searchParams.targetPercent || 0,
        max_percent: searchParams.maxPercent || 999999,
        duration_minutes: searchParams.durationMinutes,
        start_date: searchParams.startDate,
        end_date: searchParams.endDate,
        limit: 100000
      }
      if (searchParams.groupIds?.length > 0) {
        params.groups = searchParams.groupIds.join(',')
      }
      if (searchParams.weekdays?.length > 0) {
        params.weekdays = searchParams.weekdays.join(',')
      }
      if (searchParams.hourStart >= 0 && searchParams.hourEnd >= 0) {
        params.hour_start = searchParams.hourStart
        params.hour_end = searchParams.hourEnd
      }

      const response = await api.get('/api/v1/search/events', { params })
      const results = response.data.results || []

      set({
        results,
        cascadeResults: [],
        selectedEvents: results.slice(0, 32),
        isSearching: false,
        loading: false,
      })
      
      // Cascade neu anwenden wenn Indikatoren aktiv
      const { indicatorChain, prehistoryMinutes } = get()
      const activeInds = indicatorChain.filter(i => i.is_active !== false)
      if (activeInds.length > 0 && prehistoryMinutes) {
        get().applyCascadeFilter(activeInds, prehistoryMinutes)
      }
      
      return response.data
    } catch (error) {
      const errorMsg = error.response?.data?.detail || error.message || 'Suche fehlgeschlagen'
      set({ searchError: errorMsg, isSearching: false, loading: false, results: [] })
      throw error
    }
  },

  // === CASCADE FILTERING ===
  
  applyCascadeFilter: async (indicatorItems, prehistory = null) => {
    const { results, prehistoryMinutes, maxSelection } = get()
    const preHist = prehistory || prehistoryMinutes
    
    if (!indicatorItems || indicatorItems.length === 0) {
      set({ cascadeResults: results, selectedEvents: results.slice(0, maxSelection) })
      return
    }
    
    if (!preHist) {
      console.error('[SEARCH] Cascade filter needs prehistoryMinutes')
      set({ cascadeResults: results })
      return
    }
    
    try {
      const response = await api.post('/api/v1/search/cascade', {
        events: results,
        indicators: indicatorItems,
        prehistory_minutes: preHist,
      })
      
      const filtered = response.data.results || []
      
      set({
        cascadeResults: filtered,
        selectedEvents: filtered.slice(0, maxSelection),
      })
      
      console.log(`[SEARCH] Cascade: ${results.length} → ${filtered.length} events (${response.data.match_rate}%)`)
    } catch (error) {
      console.error('[SEARCH] Cascade filter error:', error)
      set({ cascadeResults: results })
    }
  },

  // === EVENT SELECTION ===
  
  selectEvents: (events) => set({ selectedEvents: events }),

  toggleEvent: (event) => {
    const { selectedEvents, maxSelection } = get()
    const exists = selectedEvents.find(e => e.id === event.id)
    if (exists) {
      set({ selectedEvents: selectedEvents.filter(e => e.id !== event.id) })
    } else {
      if (selectedEvents.length >= maxSelection) return
      set({ selectedEvents: [...selectedEvents, event] })
    }
  },
  
  selectAll: () => {
    const { cascadeResults, results, maxSelection } = get()
    const toUse = cascadeResults.length > 0 ? cascadeResults : results
    set({ selectedEvents: toUse.slice(0, maxSelection) })
  },
  
  deselectAll: () => set({ selectedEvents: [] }),
  
  selectRandom: (count) => {
    const { cascadeResults, results } = get()
    const toUse = cascadeResults.length > 0 ? cascadeResults : results
    const shuffled = [...toUse].sort(() => Math.random() - 0.5)
    set({ selectedEvents: shuffled.slice(0, count) })
  },
  
  isSelected: (eventId) => {
    return get().selectedEvents.some(e => e.id === eventId)
  },
  
  // === HELPERS ===
  
  getActiveResults: () => {
    const { cascadeResults, results } = get()
    return cascadeResults.length > 0 ? cascadeResults : results
  },
  
  getSearchStats: () => {
    const { results, cascadeResults, indicatorChain } = get()
    const percentage = results.length > 0 
      ? Math.round((cascadeResults.length / results.length) * 100)
      : 0
    return {
      mainCount: results.length,
      cascadeCount: cascadeResults.length,
      filterCount: indicatorChain.length,
      percentageRemaining: percentage,
    }
  },

  clearResults: () => {
    set({ results: [], cascadeResults: [], selectedEvents: [], searchError: null })
  }
}))
