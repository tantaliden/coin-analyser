import { create } from 'zustand'
import { persist } from 'zustand/middleware'

// Persistiert Drawings pro Event-ID, ueberlebt Refresh.
export const useDrawingsStore = create(persist(
  (set, get) => ({
    drawings: {},  // { [eventId]: [drawing, ...] }

    setAll: (drawingsMap) => set({ drawings: drawingsMap || {} }),

    setForEvent: (eventId, list) => set(state => ({
      drawings: { ...state.drawings, [eventId]: list },
    })),

    addDrawing: (eventId, drawing) => set(state => ({
      drawings: {
        ...state.drawings,
        [eventId]: [...(state.drawings[eventId] || []), drawing],
      },
    })),

    removeDrawing: (eventId, index) => set(state => ({
      drawings: {
        ...state.drawings,
        [eventId]: (state.drawings[eventId] || []).filter((_, i) => i !== index),
      },
    })),

    clearForEvent: (eventId) => set(state => ({
      drawings: { ...state.drawings, [eventId]: [] },
    })),

    clearAll: () => set({ drawings: {} }),
  }),
  { name: 'drawings-store' }
))
