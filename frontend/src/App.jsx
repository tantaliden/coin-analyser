import { useEffect, useState } from 'react'
import { useAuthStore } from './stores/authStore'
import { useConfigStore } from './stores/configStore'
import Login from './components/Login'
import Dashboard from './components/Dashboard'

export default function App() {
  const { isAuthenticated, checkAuth } = useAuthStore()
  const { loadConfig, isLoaded } = useConfigStore()
  const [checking, setChecking] = useState(true)

  useEffect(() => {
    const init = async () => {
      await loadConfig()
      await checkAuth()
      setChecking(false)
    }
    init()
  }, [])

  if (checking || !isLoaded) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900">
        <div className="text-gray-400">Laden...</div>
      </div>
    )
  }

  return isAuthenticated ? <Dashboard /> : <Login />
}
