import { useState, useRef, useEffect } from 'react'
import { LogOut, Plus, Save, ChevronDown, Lock, Unlock, Trash2, Copy, Settings, X, Eye, EyeOff, Check, AlertCircle, Loader2 } from 'lucide-react'
import { useAuthStore } from '../stores/authStore'
import { useModuleStore } from '../stores/moduleStore'
import api from '../utils/api'

function SettingsDialog({ onClose }) {
  const [settings, setSettings] = useState(null)
  const [loading, setLoading] = useState(true)
  const [editExchange, setEditExchange] = useState(null) // 'binance' | 'hyperliquid' | null
  const [apiKey, setApiKey] = useState('')
  const [apiSecret, setApiSecret] = useState('')
  const [showSecret, setShowSecret] = useState(false)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState(null) // {type: 'ok'|'error', text: '...'}
  const dialogRef = useRef(null)

  useEffect(() => {
    loadSettings()
  }, [])

  useEffect(() => {
    const handler = (e) => {
      if (dialogRef.current && !dialogRef.current.contains(e.target)) onClose()
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [onClose])

  const loadSettings = async () => {
    try {
      const { data } = await api.get('/api/v1/user/settings')
      setSettings(data)
    } catch (e) {
      console.error('Settings load error:', e)
    } finally {
      setLoading(false)
    }
  }

  const handleSave = async (exchange) => {
    if (!apiKey.trim() || !apiSecret.trim()) return
    setSaving(true)
    setMessage(null)
    try {
      const { data } = await api.put(`/api/v1/user/settings/${exchange}`, {
        api_key: apiKey.trim(),
        api_secret: apiSecret.trim()
      })
      if (data.status === 'ok') {
        setMessage({ type: 'ok', text: 'API Key gespeichert und verifiziert' })
        setEditExchange(null)
        setApiKey('')
        setApiSecret('')
        await loadSettings()
      } else {
        setMessage({ type: 'error', text: data.message })
      }
    } catch (e) {
      setMessage({ type: 'error', text: 'Verbindungsfehler' })
    } finally {
      setSaving(false)
    }
  }

  const startEdit = (exchange) => {
    setEditExchange(exchange)
    setApiKey('')
    setApiSecret('')
    setShowSecret(false)
    setMessage(null)
  }

  const exchanges = [
    { id: 'binance', label: 'Binance', keyLabel: 'API Key', secretLabel: 'API Secret' },
    { id: 'hyperliquid', label: 'Hyperliquid', keyLabel: 'Wallet Address', secretLabel: 'API Secret' }
  ]

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-[100]">
      <div ref={dialogRef} className="bg-gray-800 border border-gray-600 rounded-lg shadow-xl w-[480px] max-w-[95vw]">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
          <h2 className="text-sm font-semibold text-white">Einstellungen</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-white"><X size={16} /></button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4">
          <h3 className="text-xs text-gray-400 uppercase tracking-wider">API Verbindungen</h3>

          {loading ? (
            <div className="text-gray-400 text-xs py-4 text-center">Laden...</div>
          ) : settings && exchanges.map(ex => {
            const s = settings[ex.id]
            const isEditing = editExchange === ex.id
            return (
              <div key={ex.id} className="bg-gray-900/50 rounded-lg p-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-white">{ex.label}</span>
                    {s.configured && s.valid && (
                      <span className="flex items-center gap-1 text-xs text-emerald-400">
                        <Check size={12} /> Aktiv
                      </span>
                    )}
                    {s.configured && !s.valid && (
                      <span className="flex items-center gap-1 text-xs text-yellow-400">
                        <AlertCircle size={12} /> Ungültig
                      </span>
                    )}
                    {s.key_hint && (
                      <span className="text-xs text-gray-500">****{s.key_hint}</span>
                    )}
                  </div>
                  {!isEditing && (
                    <button onClick={() => startEdit(ex.id)}
                      className="text-xs px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded">
                      {s.configured ? 'Ändern' : 'Einrichten'}
                    </button>
                  )}
                </div>

                {isEditing && (
                  <div className="mt-3 space-y-2">
                    <div>
                      <label className="text-xs text-gray-400 block mb-1">{ex.keyLabel}</label>
                      <input value={apiKey} onChange={e => setApiKey(e.target.value)}
                        placeholder={ex.keyLabel}
                        className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-1.5 text-xs text-white focus:border-blue-500 outline-none"
                        autoFocus />
                    </div>
                    <div>
                      <label className="text-xs text-gray-400 block mb-1">{ex.secretLabel}</label>
                      <div className="relative">
                        <input value={apiSecret} onChange={e => setApiSecret(e.target.value)}
                          type={showSecret ? 'text' : 'password'}
                          placeholder={ex.secretLabel}
                          className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-1.5 text-xs text-white focus:border-blue-500 outline-none pr-8" />
                        <button onClick={() => setShowSecret(!showSecret)}
                          className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white">
                          {showSecret ? <EyeOff size={14} /> : <Eye size={14} />}
                        </button>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 pt-1">
                      <button onClick={() => handleSave(ex.id)} disabled={saving || !apiKey.trim() || !apiSecret.trim()}
                        className="flex items-center gap-1 px-3 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:text-gray-400 rounded text-xs font-medium">
                        {saving ? <><Loader2 size={12} className="animate-spin" /> Teste...</> : 'Testen & Speichern'}
                      </button>
                      <button onClick={() => { setEditExchange(null); setMessage(null) }}
                        className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs">
                        Abbrechen
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )
          })}

          {message && (
            <div className={`text-xs px-3 py-2 rounded ${message.type === 'ok' ? 'bg-emerald-900/40 text-emerald-400' : 'bg-red-900/40 text-red-400'}`}>
              {message.text}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default function Taskbar() {
  const { user, logout } = useAuthStore()
  const { availableModules, activeModules, openModule, saveLayout, saveLayoutAs,
    switchLayout, deleteLayout, savedLayouts, activeLayoutId, activeLayoutName,
    isLocked, setLocked } = useModuleStore()
  const [showModules, setShowModules] = useState(false)
  const [showLayouts, setShowLayouts] = useState(false)
  const [showSaveAs, setShowSaveAs] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
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

      <button onClick={() => setShowSettings(true)}
        className="flex items-center gap-1 text-gray-400 hover:text-white cursor-pointer">
        <Settings size={12} />
        <span>{user?.email}</span>
      </button>
      <button onClick={logout} className="flex items-center gap-1 px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded">
        <LogOut size={12} />
      </button>

      {showSettings && <SettingsDialog onClose={() => setShowSettings(false)} />}
    </header>
  )
}
