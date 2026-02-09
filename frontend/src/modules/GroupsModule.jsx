import { useState, useEffect } from 'react'
import { Plus, Trash2, Edit2 } from 'lucide-react'
import api from '../utils/api'

export default function GroupsModule() {
  const [groups, setGroups] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadGroups()
  }, [])

  const loadGroups = async () => {
    try {
      const response = await api.get('/api/v1/groups')
      setGroups(response.data || [])
    } catch (error) {
      console.error('Failed to load groups:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return <div className="text-gray-400 text-center py-8">Laden...</div>
  }

  return (
    <div className="space-y-2">
      <button className="btn btn-primary w-full flex items-center justify-center gap-2">
        <Plus size={14} /> Neue Gruppe
      </button>
      
      {groups.length === 0 ? (
        <div className="text-gray-400 text-center py-4">Keine Gruppen</div>
      ) : (
        <div className="space-y-1">
          {groups.map(g => (
            <div key={g.id} className="flex items-center justify-between p-2 bg-gray-800 rounded">
              <span>{g.name}</span>
              <span className="text-xs text-gray-500">{g.symbol_count || 0} Coins</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
