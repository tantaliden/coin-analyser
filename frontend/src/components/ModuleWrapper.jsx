import { useModuleStore } from '../stores/moduleStore'

/**
 * Minimale Hülle um ein Modul.
 * - Kein sichtbarer Header, kein Titel, kein Close-Button.
 * - Oben 6px unsichtbare Drag-Zone (für react-grid-layout).
 * - Schließen erfolgt über die Taskbar.
 */
export default function ModuleWrapper({ moduleId, title, children }) {
  return (
    <div className="module-container">
      <div className="module-header" title={title} />
      <div className="module-content">{children}</div>
    </div>
  )
}
