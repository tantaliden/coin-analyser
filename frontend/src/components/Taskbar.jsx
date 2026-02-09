import { useState } from 'react'
import { LogOut, Menu, User, Plus } from 'lucide-react'
import { useAuthStore } from '../stores/authStore'
import { useModuleStore } from '../stores/moduleStore'

export default function Taskbar() {
  const { user, logout } = useAuthStore()
  const { availableModules, activeModules, openModule, saveLayout } = useModuleStore()
  const [showModuleMenu, setShowModuleMenu] = useState(false)

  const inactiveModules = availableModules.filter(m => !activeModules.includes(m.id))

  return (
    <header className="bg-gray-800 border-b border-gray-700 px-4 py-2 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <h1 className="text-lg font-semibold text-white">Coin-Analyser</h1>
        
        {/* Module hinzufügen */}
        <div className="relative">
          <button
            onClick={() => setShowModuleMenu(!showModuleMenu)}
            className="flex items-center gap-1 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-sm"
          >
            <Plus size={14} />
            Modul
          </button>
          
          {showModuleMenu && (
            <div className="absolute top-full left-0 mt-1 bg-gray-800 border border-gray-700 rounded shadow-lg z-50 min-w-48">
              {inactiveModules.length === 0 ? (
                <div className="px-3 py-2 text-gray-400 text-sm">
                  Alle Module aktiv
                </div>
              ) : (
                inactiveModules.map(m => (
                  <button
                    key={m.id}
                    onClick={() => {
                      openModule(m.id)
                      setShowModuleMenu(false)
                    }}
                    className="w-full text-left px-3 py-2 hover:bg-gray-700 text-sm"
                  >
                    {m.label}
                  </button>
                ))
              )}
            </div>
          )}
        </div>

        <button
          onClick={saveLayout}
          className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 rounded text-sm"
        >
          Layout speichern
        </button>
      </div>

      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 text-gray-300">
          <User size={16} />
          <span className="text-sm">{user?.email || 'Benutzer'}</span>
        </div>
        
        <button
          onClick={logout}
          className="flex items-center gap-1 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-sm"
        >
          <LogOut size={14} />
          Logout
        </button>
      </div>
    </header>
  )
}
