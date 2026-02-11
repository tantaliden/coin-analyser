import { useEffect, useState } from 'react'
import { useAuthStore } from './stores/authStore'
import { useConfigStore } from './stores/configStore'
import { useModuleStore } from './stores/moduleStore'
import Login from './components/Login'
import Dashboard from './components/Dashboard'

export default function App() {
  const { isAuthenticated, checkAuth } = useAuthStore()
  const { loadConfig, isLoaded, loadError } = useConfigStore()
  const { loadFromBackend } = useModuleStore()
  const [checking, setChecking] = useState(true)
  const [initError, setInitError] = useState(null)

  useEffect(() => {
    const init = async () => {
      try {
        await loadConfig()
        const isAuth = await checkAuth()
        if (isAuth) {
          await loadFromBackend()
        }
      } catch (error) {
        setInitError(error.message || 'Initialisierung fehlgeschlagen')
      }
      setChecking(false)
    }
    init()
  }, [])

  if (initError || loadError) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900">
        <div className="bg-red-900/50 border border-red-500 rounded-lg p-6 max-w-md">
          <h2 className="text-red-400 font-semibold mb-2">Fehler</h2>
          <p className="text-gray-300">{initError || loadError}</p>
          <p className="text-gray-500 text-sm mt-4">Backend nicht erreichbar. Bitte prüfe ob der Server läuft.</p>
          <button onClick={() => window.location.reload()}
            className="mt-4 px-4 py-2 bg-red-600 hover:bg-red-500 rounded text-white">
            Neu laden
          </button>
        </div>
      </div>
    )
  }

  if (checking || !isLoaded) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900">
        <div className="text-gray-400">Laden...</div>
      </div>
    )
  }

  return isAuthenticated ? <Dashboard /> : <Login />
}
