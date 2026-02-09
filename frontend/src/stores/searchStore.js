import { create } from 'zustand'
import api from '../utils/api'

export const useSearchStore = create((set, get) => ({
  // Suchparameter
  searchParams: {
    direction: 'up',
    targetPercent: 5,
    durationMinutes: 120,
    dateFrom: null,
    dateTo: null,
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

  // Suchparameter setzen
  setSearchParams: (params) => {
    set({ searchParams: { ...get().searchParams, ...params } })
  },

  // Suche ausführen
  search: async () => {
    const { searchParams } = get()
    set({ isSearching: true, searchError: null })
    
    try {
      const response = await api.post('/api/v1/search/events', {
        direction: searchParams.direction,
        target_percent: searchParams.targetPercent,
        duration_minutes: searchParams.durationMinutes,
        date_from: searchParams.dateFrom,
        date_to: searchParams.dateTo,
        symbols: searchParams.symbols.length > 0 ? searchParams.symbols : null,
        group_id: searchParams.groupId
      })
      
      set({ 
        results: response.data.events || [], 
        isSearching: false 
      })
      return response.data
    } catch (error) {
      set({ 
        searchError: error.response?.data?.detail || 'Suche fehlgeschlagen',
        isSearching: false,
        results: []
      })
      throw error
    }
  },

  // Events auswählen (für Chart)
  selectEvents: (events) => {
    set({ selectedEvents: events })
  },

  // Event hinzufügen/entfernen
  toggleEvent: (event) => {
    const { selectedEvents } = get()
    const exists = selectedEvents.find(e => e.id === event.id)
    if (exists) {
      set({ selectedEvents: selectedEvents.filter(e => e.id !== event.id) })
    } else {
      set({ selectedEvents: [...selectedEvents, event] })
    }
  },

  // Prehistory für Chart
  setPrehistoryMinutes: (minutes) => {
    set({ prehistoryMinutes: minutes })
  },

  // State zurücksetzen
  clearResults: () => {
    set({ results: [], selectedEvents: [], searchError: null })
  }
}))
