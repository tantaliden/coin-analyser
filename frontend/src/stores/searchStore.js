import { create } from 'zustand'
import api from '../utils/api'

export const useSearchStore = create((set, get) => ({
  // Suchparameter
  searchParams: {
    direction: 'up',
    targetPercent: 5,
    durationMinutes: 120,
    startDate: '',
    endDate: '',
    symbols: [],
    groupId: null
  },
  
  // Suchergebnisse
  results: [],
  isSearching: false,
  searchError: null,
  
  // Ausgewählte Events für Chart
  selectedEvents: [],
  
  // Chart Einstellungen
  prehistoryMinutes: 720,

  setSearchParams: (params) => {
    set({ searchParams: { ...get().searchParams, ...params } })
  },

  search: async () => {
    const { searchParams } = get()
    set({ isSearching: true, searchError: null })
    
    // Validierung
    if (!searchParams.startDate || !searchParams.endDate) {
      set({ searchError: 'Start- und Enddatum erforderlich', isSearching: false })
      throw new Error('Start- und Enddatum erforderlich')
    }
    
    try {
      const response = await api.post('/api/v1/search/events', {
        direction: searchParams.direction,
        target_percent: searchParams.targetPercent,
        duration_minutes: searchParams.durationMinutes,
        start_date: searchParams.startDate,
        end_date: searchParams.endDate,
        symbols: searchParams.symbols.length > 0 ? searchParams.symbols : null,
        limit: 1000
      })
      
      set({ 
        results: response.data.events || [], 
        isSearching: false 
      })
      return response.data
    } catch (error) {
      const errorMsg = error.response?.data?.detail || error.message || 'Suche fehlgeschlagen'
      set({ 
        searchError: errorMsg,
        isSearching: false,
        results: []
      })
      throw error
    }
  },

  selectEvents: (events) => {
    set({ selectedEvents: events })
  },

  toggleEvent: (event) => {
    const { selectedEvents } = get()
    const exists = selectedEvents.find(e => e.id === event.id)
    if (exists) {
      set({ selectedEvents: selectedEvents.filter(e => e.id !== event.id) })
    } else {
      set({ selectedEvents: [...selectedEvents, event] })
    }
  },

  setPrehistoryMinutes: (minutes) => {
    set({ prehistoryMinutes: minutes })
  },

  clearResults: () => {
    set({ results: [], selectedEvents: [], searchError: null })
  }
}))
