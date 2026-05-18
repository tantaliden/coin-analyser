import { useState, useEffect } from 'react'
import api from '../../utils/api'

export default function GroupFilter({ selectedIds, onChange }) {
  const [groups, setGroups] = useState([])

  useEffect(() => {
    api.get('/api/v1/groups').then(r => {
      if (!r.data.groups) throw new Error('API response missing groups field')
      setGroups(r.data.groups)
    })
  }, [])

  if (groups.length === 0) return null

  const toggle = (groupId) => {
    if (selectedIds.includes(groupId)) {
      onChange(selectedIds.filter(id => id !== groupId))
    } else {
      onChange([...selectedIds, groupId])
    }
  }

  return (
    <div>
      <label className="block text-gray-400 text-xs mb-1">Coin-Gruppen</label>
      <div className="flex flex-wrap gap-1">
        {groups.map(g => (
          <button
            key={g.id}
            type="button"
            onClick={() => toggle(g.id)}
            className={`px-2 py-1 rounded text-xs flex items-center gap-1 ${
              selectedIds.includes(g.id) ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'
            }`}
          >
            <div className="w-2 h-2 rounded-full" style={{ background: g.color }} />
            {g.name}
          </button>
        ))}
      </div>
    </div>
  )
}
