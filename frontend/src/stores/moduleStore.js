import { create } from 'zustand'
import api from '../utils/api'

const DEFAULT_LAYOUTS = {
  lg: [
    { i: 'search', x: 0, y: 0, w: 3, h: 14, minW: 2, minH: 6 },
    { i: 'searchResults', x: 3, y: 0, w: 9, h: 14, minW: 3, minH: 6 },
    { i: 'chart', x: 0, y: 14, w: 12, h: 16, minW: 4, minH: 8 },
  ]
}

const MODULE_DEFAULTS = ['search', 'searchResults', 'chart']

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
  savedLayouts: [],
  activeLayoutId: null,
  activeLayoutName: null,
  isLocked: false,

  setLocked: (locked) => set({ isLocked: locked }),

  openModule: (moduleId) => {
    const { activeModules, currentLayout, availableModules } = get()
    if (!availableModules.find(m => m.id === moduleId)) return
    if (activeModules.includes(moduleId)) return
    const maxY = Math.max(...(currentLayout.lg?.map(l => l.y + l.h) || [0]), 0)
    set({
      activeModules: [...activeModules, moduleId],
      currentLayout: {
        ...currentLayout,
        lg: [...(currentLayout.lg || []), { i: moduleId, x: 0, y: maxY, w: 6, h: 12, minW: 2, minH: 6 }]
      }
    })
  },

  closeModule: (moduleId) => {
    const { activeModules, currentLayout } = get()
    set({
      activeModules: activeModules.filter(id => id !== moduleId),
      currentLayout: { ...currentLayout, lg: (currentLayout.lg || []).filter(l => l.i !== moduleId) }
    })
  },

  setCurrentLayout: (layouts) => set({ currentLayout: layouts }),

  // Layout in user_layouts speichern (neues oder bestehendes aktualisieren)
  saveLayout: async (name) => {
    const { currentLayout, activeLayoutId } = get()
    const layoutData = currentLayout.lg || []
    try {
      if (activeLayoutId) {
        await api.put(`/api/v1/user/layouts/${activeLayoutId}`, { layout_data: layoutData })
      } else {
        const res = await api.post('/api/v1/user/layouts', { name: name || 'Standard', layout_data: layoutData, is_default: true })
        set({ activeLayoutId: res.data.id, activeLayoutName: name || 'Standard' })
      }
      await get().loadLayouts()
      return true
    } catch (e) { console.error('Save layout failed:', e); return false }
  },

  saveLayoutAs: async (name) => {
    const { currentLayout } = get()
    try {
      const res = await api.post('/api/v1/user/layouts', { name, layout_data: currentLayout.lg || [], is_default: false })
      set({ activeLayoutId: res.data.id, activeLayoutName: name })
      await get().loadLayouts()
      return true
    } catch (e) { console.error('Save layout as failed:', e); return false }
  },

  switchLayout: async (layoutId) => {
    const { savedLayouts } = get()
    const layout = savedLayouts.find(l => l.id === layoutId)
    if (!layout) return
    const moduleIds = layout.layout_data.map(l => l.i).filter(id => AVAILABLE_MODULES.some(m => m.id === id))
    set({
      currentLayout: { lg: layout.layout_data },
      activeModules: moduleIds.length > 0 ? moduleIds : MODULE_DEFAULTS,
      activeLayoutId: layout.id,
      activeLayoutName: layout.name
    })
    // Als Default setzen
    try { await api.put(`/api/v1/user/layouts/${layoutId}/default`) } catch {}
  },

  deleteLayout: async (layoutId) => {
    const { activeLayoutId } = get()
    try {
      await api.delete(`/api/v1/user/layouts/${layoutId}`)
      if (activeLayoutId === layoutId) {
        set({ activeLayoutId: null, activeLayoutName: null })
      }
      await get().loadLayouts()
    } catch (e) { console.error('Delete layout failed:', e) }
  },

  loadLayouts: async () => {
    try {
      const res = await api.get('/api/v1/user/layouts')
      set({ savedLayouts: res.data.layouts || [] })
    } catch {}
  },

  loadFromBackend: async () => {
    try {
      // Lade gespeicherte Layouts
      const layoutsRes = await api.get('/api/v1/user/layouts')
      const layouts = layoutsRes.data.layouts || []
      set({ savedLayouts: layouts })

      // Default-Layout laden (oder erstes)
      const defaultLayout = layouts.find(l => l.is_default) || layouts[0]
      if (defaultLayout && defaultLayout.layout_data?.length > 0) {
        const moduleIds = defaultLayout.layout_data.map(l => l.i).filter(id => AVAILABLE_MODULES.some(m => m.id === id))
        set({
          currentLayout: { lg: defaultLayout.layout_data },
          activeModules: moduleIds.length > 0 ? moduleIds : MODULE_DEFAULTS,
          activeLayoutId: defaultLayout.id,
          activeLayoutName: defaultLayout.name
        })
        return
      }

      // Fallback: user_state.module_layouts
      const stateRes = await api.get('/api/v1/user/state')
      const ml = stateRes.data?.module_layouts
      if (ml && Array.isArray(ml) && ml.length > 0) {
        const moduleIds = ml.map(l => l.i).filter(id => AVAILABLE_MODULES.some(m => m.id === id))
        set({
          currentLayout: { lg: ml },
          activeModules: moduleIds.length > 0 ? moduleIds : MODULE_DEFAULTS
        })
      }
    } catch (e) { console.error('Load from backend failed:', e) }
  }
}))
