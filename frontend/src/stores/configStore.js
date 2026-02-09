import { create } from 'zustand'
import axios from 'axios'

export const useConfigStore = create((set, get) => ({
  config: null,
  isLoaded: false,

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
      
      set({ config, isLoaded: true })
      return config
    } catch (error) {
      console.error('Failed to load config:', error)
      // Fallback-Config
      set({ 
        config: { 
          labels: {}, 
          ui: {}, 
          modules: {},
          klineMetricsDurations: [30, 60, 90, 120, 180, 240, 300, 360, 420, 480, 540, 600]
        }, 
        isLoaded: true 
      })
    }
  },

  // Label Getter
  getLabel: (key, lang = 'de') => {
    const { config } = get()
    const label = config?.labels?.[key]
    if (!label) return key
    return label[lang] || label.de || key
  },

  // Module Config
  getModuleConfig: (moduleId) => get().config?.modules?.[moduleId] || {},

  // Search Defaults
  getSearchDefaults: () => get().config?.search || {},

  // Timeframe Options
  getTimeframeOptions: () => get().config?.timeframes?.chartOptions || ['1m', '5m', '15m', '1h', '4h', '1d'],

  // KLine Metrics Durations (für Event-Suche)
  getKlineMetricsDurations: () => get().config?.klineMetricsDurations || [30, 60, 90, 120, 180, 240, 300, 360, 420, 480, 540, 600],

  // Indicator Fields
  getIndicatorFields: () => get().config?.indicatorFields || [],

  // Event Colors für Charts
  getEventColors: () => get().config?.eventColors || [
    '#3b82f6', '#22c55e', '#f59e0b', '#a855f7', '#ef4444', 
    '#06b6d4', '#f97316', '#ec4899', '#84cc16', '#14b8a6'
  ],

  getEventColor: (index) => {
    const colors = get().getEventColors()
    return colors[index % colors.length]
  }
}))
