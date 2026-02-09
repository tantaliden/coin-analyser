import { create } from 'zustand'
import api from '../utils/api'

const DEFAULT_LAYOUTS = {
  lg: [
    { i: 'search', x: 0, y: 0, w: 3, h: 14 },
    { i: 'searchResults', x: 3, y: 0, w: 9, h: 14 },
    { i: 'chart', x: 0, y: 14, w: 12, h: 16 },
  ]
}

const MODULE_DEFAULTS = ['search', 'searchResults', 'chart']

// Nur fertige Module - keine Placeholder!
const AVAILABLE_MODULES = [
  { id: 'search', label: 'Suche' },
  { id: 'searchResults', label: 'Suchergebnisse' },
  { id: 'chart', label: 'Chart' },
]

export const useModuleStore = create((set, get) => ({
  activeModules: MODULE_DEFAULTS,
  currentLayout: DEFAULT_LAYOUTS,
  availableModules: AVAILABLE_MODULES,

  openModule: (moduleId) => {
    const { activeModules, currentLayout, availableModules } = get()
    
    // Nur registrierte Module erlauben
    if (!availableModules.find(m => m.id === moduleId)) return
    if (activeModules.includes(moduleId)) return

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

  setCurrentLayout: (layouts) => {
    set({ currentLayout: layouts })
  },

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
      // Kein Fallback - wenn Backend nicht erreichbar, Default-Layout bleibt
      console.error('Failed to load layout:', error)
    }
  }
}))
