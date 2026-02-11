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

// Alle Module - Labels aus naming.js über config
const AVAILABLE_MODULES = [
  { id: 'search', label: 'Suche' },
  { id: 'searchResults', label: 'Suchergebnisse' },
  { id: 'chart', label: 'Chart' },
  { id: 'indicators', label: 'Indikatoren' },
  { id: 'groups', label: 'Coin-Gruppen' },
  { id: 'wallet', label: 'Wallet' },
  { id: 'bot', label: 'Trading Bot' },
]

export const useModuleStore = create((set, get) => ({
  activeModules: MODULE_DEFAULTS,
  currentLayout: DEFAULT_LAYOUTS,
  availableModules: AVAILABLE_MODULES,

  openModule: (moduleId) => {
    const { activeModules, currentLayout, availableModules } = get()
    if (!availableModules.find(m => m.id === moduleId)) return
    if (activeModules.includes(moduleId)) return

    const maxY = Math.max(...(currentLayout.lg?.map(l => l.y + l.h) || [0]), 0)
    const newLayout = { i: moduleId, x: 0, y: maxY, w: 6, h: 12 }

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
      await api.put('/api/v1/user/state', {
        module_layouts: currentLayout.lg || []
      })
      return true
    } catch (error) {
      console.error('Failed to save layout:', error)
      return false
    }
  },

  loadFromBackend: async () => {
    try {
      const response = await api.get('/api/v1/user/state')
      const layouts = response.data?.module_layouts
      if (layouts && Array.isArray(layouts) && layouts.length > 0) {
        const moduleIds = layouts.map(l => l.i).filter(id =>
          AVAILABLE_MODULES.some(m => m.id === id)
        )
        set({
          currentLayout: { lg: layouts },
          activeModules: moduleIds.length > 0 ? moduleIds : MODULE_DEFAULTS
        })
      }
    } catch (error) {
      console.error('Failed to load layout:', error)
    }
  }
}))
