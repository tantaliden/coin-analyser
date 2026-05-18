import { create } from 'zustand'
import api from '../utils/api'

// Laedt Predictor-Settings vom Backend — Single Source of Truth.
// Kein Fallback: wenn Laden fehlschlaegt, bleibt settings null und Komponenten zeigen Fehler.
export const usePredictorSettingsStore = create((set, get) => ({
  settings: null,
  loading: false,
  error: null,

  load: async () => {
    if (get().settings || get().loading) return
    set({ loading: true, error: null })
    try {
      const res = await api.get('/api/v1/meta/predictor-settings')
      set({ settings: res.data, loading: false })
    } catch (err) {
      set({ error: err.response?.data?.detail || err.message, loading: false })
    }
  },

  // Pflicht-Getter: wenn Settings nicht geladen, Exception
  getFuzzyDefaults: () => {
    const s = get().settings
    if (!s) throw new Error('Predictor-Settings noch nicht geladen')
    return s.fuzzy_defaults
  },

  getInitialPointDefaults: () => {
    const s = get().settings
    if (!s) throw new Error('Predictor-Settings noch nicht geladen')
    return s.initial_point
  },

  getSetDefaults: () => {
    const s = get().settings
    if (!s) throw new Error('Predictor-Settings noch nicht geladen')
    return s.set_defaults
  },
}))
