import { create } from 'zustand'
import api from '../utils/api'

export const useSearchStore = create((set, get) => ({
  searchParams: {
    direction: 'up',
    targetPercent: 5,
    maxPercent: 100,
    durationMinutes: 120,
    startDate: '',
    endDate: '',
    groupIds: []
  },

  results: [],
  isSearching: false,
  searchError: null,
  selectedEvents: [],
  prehistoryMinutes: 720,

  setSearchParams: (params) => {
    set({ searchParams: { ...get().searchParams, ...params } })
  },

  search: async () => {
    const { searchParams } = get()
    set({ isSearching: true, searchError: null })

    if (!searchParams.startDate || !searchParams.endDate) {
      set({ searchError: 'Start- und Enddatum erforderlich', isSearching: false })
      throw new Error('Start- und Enddatum erforderlich')
    }

    try {
      const params = {
        direction: searchParams.direction,
        min_percent: searchParams.targetPercent,
        max_percent: searchParams.maxPercent || 100,
        duration_minutes: searchParams.durationMinutes,
        start_date: searchParams.startDate,
        end_date: searchParams.endDate,
        limit: 10000
      }
      if (searchParams.groupIds?.length > 0) {
        params.groups = searchParams.groupIds.join(',')
      }

      const response = await api.get('/api/v1/search/events', { params })

      set({
        results: response.data.results || [],
        isSearching: false
      })
      return response.data
    } catch (error) {
      const errorMsg = error.response?.data?.detail || error.message || 'Suche fehlgeschlagen'
      set({ searchError: errorMsg, isSearching: false, results: [] })
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
      if (selectedEvents.length >= 32) return
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
