import { create } from 'zustand'
import axios from 'axios'

export const useConfigStore = create((set, get) => ({
  config: null,
  isLoaded: false,
  loadError: null,

  loadConfig: async () => {
    try {
      const response = await axios.get('/api/v1/meta/config')
      const config = response.data
      
      // CSS Variables aus Config setzen
      if (config.ui?.theme?.colors) {
        const root = document.documentElement
        const c = config.ui.theme.colors
        Object.entries(c).forEach(([key, value]) => {
          root.style.setProperty(`--color-${key}`, value)
        })
      }
      
      set({ config, isLoaded: true, loadError: null })
      return config
    } catch (error) {
      // Kein Fallback - Fehler anzeigen
      const errorMsg = error.response?.data?.detail || error.message || 'Config konnte nicht geladen werden'
      set({ config: null, isLoaded: true, loadError: errorMsg })
      throw error
    }
  },

  getLabel: (key, lang = 'de') => {
    const { config } = get()
    if (!config?.labels) return key
    const label = config.labels[key]
    if (!label) return key
    return label[lang] || label.de || key
  },

  getModuleConfig: (moduleId) => {
    const { config } = get()
    return config?.modules?.[moduleId] || {}
  },

  getSearchDefaults: () => {
    const { config } = get()
    return config?.search || {}
  },

  getTimeframeOptions: () => {
    const { config } = get()
    return config?.timeframes?.chartOptions || ['1m', '5m', '15m', '1h', '4h', '1d']
  },

  getKlineMetricsDurations: () => {
    const { config } = get()
    return config?.klineMetricsDurations || []
  },

  getIndicatorFields: () => {
    const { config } = get()
    return config?.indicatorFields || []
  },

  getEventColors: () => {
    const { config } = get()
    return config?.eventColors || [
      '#3b82f6', '#22c55e', '#f59e0b', '#a855f7', '#ef4444', 
      '#06b6d4', '#f97316', '#ec4899', '#84cc16', '#14b8a6'
    ]
  },

  getEventColor: (index) => {
    const colors = get().getEventColors()
    return colors[index % colors.length]
  }
}))
