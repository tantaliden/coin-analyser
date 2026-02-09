import { X, Minus } from 'lucide-react'
import { useModuleStore } from '../stores/moduleStore'

export default function ModuleWrapper({ moduleId, title, children }) {
  const { closeModule } = useModuleStore()

  return (
    <div className="module-container">
      <div className="module-header">
        <h3>{title}</h3>
        <div className="flex gap-1">
          <button 
            onClick={() => closeModule(moduleId)}
            className="p-1 hover:bg-gray-700 rounded"
            title="Schließen"
          >
            <X size={14} />
          </button>
        </div>
      </div>
      <div className="module-content">
        {children}
      </div>
    </div>
  )
}
