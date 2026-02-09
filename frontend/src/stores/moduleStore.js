import { create } from 'zustand'
import api from '../utils/api'

const DEFAULT_LAYOUTS = {
  lg: [
    { i: 'search', x: 0, y: 0, w: 4, h: 12 },
    { i: 'searchResults', x: 4, y: 0, w: 8, h: 12 },
    { i: 'chart', x: 0, y: 12, w: 12, h: 16 },
  ]
}

const MODULE_DEFAULTS = ['search', 'searchResults', 'chart']

export const useModuleStore = create((set, get) => ({
  activeModules: MODULE_DEFAULTS,
  currentLayout: DEFAULT_LAYOUTS,
  availableModules: [
    { id: 'search', label: 'Suche' },
    { id: 'searchResults', label: 'Suchergebnisse' },
    { id: 'chart', label: 'Chart' },
    { id: 'indicators', label: 'Indikatoren' },
    { id: 'sets', label: 'Indikator-Sets' },
    { id: 'groups', label: 'Coingruppen' },
    { id: 'wallet', label: 'Wallet' },
    { id: 'bot', label: 'Trading Bot' },
  ],

  // Modul öffnen
  openModule: (moduleId) => {
    const { activeModules, currentLayout } = get()
    if (activeModules.includes(moduleId)) return

    // Default Position für neues Modul
    const newLayout = {
      i: moduleId,
      x: 0,
      y: Math.max(...(currentLayout.lg?.map(l => l.y + l.h) || [0]), 0),
      w: 6,
      h: 10
    }

    set({
      activeModules: [...activeModules, moduleId],
      currentLayout: {
        ...currentLayout,
        lg: [...(currentLayout.lg || []), newLayout]
      }
    })
  },

  // Modul schließen
  closeModule: (moduleId) => {
    const { activeModules, currentLayout } = get()
    set({
      activeModules: activeModules.filter(id => id !== moduleId),
      currentLayout: {
        ...currentLayout,
        lg: (currentLayout.lg || []).filter(l => l.i !== moduleId)
      }
    })
  },

  // Layout ändern
  setCurrentLayout: (layouts) => {
    set({ currentLayout: layouts })
  },

  // Layout speichern
  saveLayout: async () => {
    const { currentLayout, activeModules } = get()
    try {
      await api.post('/api/v1/user/layout', { 
        layout: currentLayout, 
        activeModules 
      })
      return true
    } catch (error) {
      console.error('Failed to save layout:', error)
      return false
    }
  },

  // Layout vom Backend laden
  loadFromBackend: async () => {
    try {
      const response = await api.get('/api/v1/user/layout')
      if (response.data?.layout) {
        set({
          currentLayout: response.data.layout,
          activeModules: response.data.activeModules || MODULE_DEFAULTS
        })
      }
    } catch (error) {
      console.error('Failed to load layout:', error)
    }
  }
}))
