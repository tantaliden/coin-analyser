import { useState, useRef, useEffect } from 'react'
import { LogOut, Plus, Save, ChevronDown, Lock, Unlock, Trash2, Copy } from 'lucide-react'
import { useAuthStore } from '../stores/authStore'
import { useModuleStore } from '../stores/moduleStore'

export default function Taskbar() {
  const { user, logout } = useAuthStore()
  const { availableModules, activeModules, openModule, saveLayout, saveLayoutAs,
    switchLayout, deleteLayout, savedLayouts, activeLayoutId, activeLayoutName,
    isLocked, setLocked } = useModuleStore()
  const [showModules, setShowModules] = useState(false)
  const [showLayouts, setShowLayouts] = useState(false)
  const [showSaveAs, setShowSaveAs] = useState(false)
  const [newName, setNewName] = useState('')
  const modRef = useRef(null)
  const layRef = useRef(null)

  // Click outside schließen
  useEffect(() => {
    const handler = (e) => {
      if (modRef.current && !modRef.current.contains(e.target)) setShowModules(false)
      if (layRef.current && !layRef.current.contains(e.target)) { setShowLayouts(false); setShowSaveAs(false) }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const inactiveModules = availableModules.filter(m => !activeModules.includes(m.id))

  const handleSave = async () => {
    if (!activeLayoutId) {
      setShowSaveAs(true); setShowLayouts(true); setNewName('Standard')
      return
    }
    await saveLayout()
  }

  const handleSaveAs = async () => {
    if (!newName.trim()) return
    await saveLayoutAs(newName.trim())
    setShowSaveAs(false); setNewName('')
  }

  return (
    <header className="bg-gray-800 border-b border-gray-700 px-3 py-1.5 flex items-center gap-2 text-xs flex-shrink-0">
      <span className="text-sm font-semibold text-white mr-2">Tresor</span>

      {/* Module hinzufügen */}
      <div className="relative" ref={modRef}>
        <button onClick={() => setShowModules(!showModules)}
          className="flex items-center gap-1 px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded">
          <Plus size={12} /> Modul
        </button>
        {showModules && (
          <div className="absolute top-full left-0 mt-1 bg-gray-800 border border-gray-700 rounded shadow-lg z-50 min-w-40">
            {inactiveModules.length === 0 ? (
              <div className="px-3 py-2 text-gray-400">Alle aktiv</div>
            ) : inactiveModules.map(m => (
              <button key={m.id} onClick={() => { openModule(m.id); setShowModules(false) }}
                className="w-full text-left px-3 py-1.5 hover:bg-gray-700">{m.label}</button>
            ))}
          </div>
        )}
      </div>

      {/* Layout Dropdown */}
      <div className="relative" ref={layRef}>
        <button onClick={() => setShowLayouts(!showLayouts)}
          className="flex items-center gap-1 px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded min-w-28">
          <span className="truncate">{activeLayoutName || 'Layout'}</span>
          <ChevronDown size={12} />
        </button>
        {showLayouts && (
          <div className="absolute top-full left-0 mt-1 bg-gray-800 border border-gray-700 rounded shadow-lg z-50 min-w-48">
            {savedLayouts.map(l => (
              <div key={l.id} className={`flex items-center justify-between px-3 py-1.5 hover:bg-gray-700 ${l.id === activeLayoutId ? 'bg-blue-900/30' : ''}`}>
                <button onClick={() => { switchLayout(l.id); setShowLayouts(false) }} className="flex-1 text-left truncate">
                  {l.name} {l.is_default && <span className="text-blue-400 ml-1">★</span>}
                </button>
                <button onClick={(e) => { e.stopPropagation(); deleteLayout(l.id) }}
                  className="p-0.5 text-zinc-500 hover:text-red-400 ml-2"><Trash2 size={10} /></button>
              </div>
            ))}
            {savedLayouts.length > 0 && <div className="border-t border-gray-700" />}
            {showSaveAs ? (
              <div className="px-3 py-2 flex items-center gap-1">
                <input value={newName} onChange={e => setNewName(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleSaveAs()}
                  placeholder="Name..." className="flex-1 bg-gray-700 rounded px-2 py-0.5 text-xs" autoFocus />
                <button onClick={handleSaveAs} className="px-2 py-0.5 bg-blue-600 rounded">OK</button>
              </div>
            ) : (
              <button onClick={() => { setShowSaveAs(true); setNewName('') }}
                className="w-full text-left px-3 py-1.5 hover:bg-gray-700 text-blue-400">
                <Copy size={10} className="inline mr-1" /> Speichern als...
              </button>
            )}
          </div>
        )}
      </div>

      {/* Speichern */}
      <button onClick={handleSave} className="flex items-center gap-1 px-2 py-1 bg-blue-600 hover:bg-blue-500 rounded" title="Layout speichern">
        <Save size={12} />
      </button>

      {/* Lock Toggle */}
      <button onClick={() => setLocked(!isLocked)}
        className={`flex items-center gap-1 px-2 py-1 rounded ${isLocked ? 'bg-yellow-600 hover:bg-yellow-500' : 'bg-gray-700 hover:bg-gray-600'}`}
        title={isLocked ? 'Layout entsperren' : 'Layout sperren'}>
        {isLocked ? <Lock size={12} /> : <Unlock size={12} />}
      </button>

      <div className="flex-1" />

      <span className="text-gray-400">{user?.email}</span>
      <button onClick={logout} className="flex items-center gap-1 px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded">
        <LogOut size={12} />
      </button>
    </header>
  )
}
