import { useState } from 'react'
import { CHART_SETTINGS } from '../../config/chartSettings'
import {
  MousePointer2, Crosshair, TrendingUp, MoveUpRight, Minus, SeparatorVertical, AlignJustify,
  Square, ArrowUpDown, Timer,
  GitBranch, GitFork, Ruler,
  Highlighter, ChevronsUp, ChevronsDown, Spline, Scaling,
  ArrowUpCircle, ArrowDownCircle, Flag, AlertTriangle, Type,
  Eraser, Trash2, BarChart3, CandlestickChart, ChevronDown, Eye, EyeOff, SearchCheck, FolderOpen, Sparkles,
} from 'lucide-react'

const ICON_MAP = {
  MousePointer2, Crosshair, TrendingUp, MoveUpRight, Minus, SeparatorVertical, AlignJustify,
  Square, ArrowUpDown, Timer,
  GitBranch, GitFork, Ruler,
  Highlighter, ChevronsUp, ChevronsDown, Spline, Scaling,
  ArrowUpCircle, ArrowDownCircle, Flag, AlertTriangle, Type,
  Eraser,
}

export default function Toolbar({
  activeTool, setActiveTool,
  drawingColor, setDrawingColor,
  drawingWidth, setDrawingWidth,
  onClearAll,
  onToggleIndicators, onTogglePatterns,
  showIndicatorPanel, showPatternPanel,
  drawingsVisible, onToggleDrawingsVisible,
  onToggleCounterSearch, showCounterSearch,
  onToggleSetManager, showSetManager,
  onToggleAnomalies, showAnomalies,
}) {
  const [expandedGroup, setExpandedGroup] = useState(null)
  const tools = CHART_SETTINGS.drawing.tools
  const groups = CHART_SETTINGS.drawing.toolGroups
  const colors = CHART_SETTINGS.drawing.colors
  const widths = CHART_SETTINGS.drawing.lineWidths

  const toggleGroup = (groupId) => {
    setExpandedGroup(expandedGroup === groupId ? null : groupId)
  }

  return (
    <div className="border-b border-gray-700 bg-gray-800/50">
      {/* Main toolbar row */}
      <div className="flex items-center gap-1 px-2 py-1.5 flex-wrap">
        {/* Tool groups */}
        {groups.map(group => {
          const groupTools = tools.filter(t => t.group === group.id)
          const activeInGroup = groupTools.find(t => t.id === activeTool)
          const ActiveIcon = activeInGroup ? ICON_MAP[activeInGroup.icon] : ICON_MAP[groupTools[0]?.icon]

          return (
            <div key={group.id} className="relative">
              <div className="flex">
                {/* Primary button: use active tool from group or first */}
                <button
                  onClick={() => setActiveTool(activeInGroup?.id || groupTools[0]?.id)}
                  className={`p-1.5 rounded-l transition-colors ${
                    activeInGroup
                      ? group.id === 'marker' && activeTool === 'buyMarker' ? 'bg-green-600 text-white'
                        : group.id === 'marker' && activeTool === 'sellMarker' ? 'bg-red-600 text-white'
                        : 'bg-blue-600 text-white'
                      : 'bg-gray-700/50 text-gray-400 hover:bg-gray-600'
                  }`}
                  title={activeInGroup?.label || group.label}
                >
                  {ActiveIcon ? <ActiveIcon size={14} /> : <span className="text-[10px]">{group.label[0]}</span>}
                </button>
                {/* Dropdown toggle */}
                {groupTools.length > 1 && (
                  <button
                    onClick={() => toggleGroup(group.id)}
                    className={`px-0.5 py-1.5 rounded-r border-l border-gray-600/50 ${
                      activeInGroup ? 'bg-blue-600/80 text-white' : 'bg-gray-700/50 text-gray-500 hover:bg-gray-600'
                    }`}
                  >
                    <ChevronDown size={8} />
                  </button>
                )}
              </div>

              {/* Dropdown */}
              {expandedGroup === group.id && (
                <div className="absolute top-full left-0 mt-1 bg-gray-900 border border-gray-700 rounded shadow-lg z-50 min-w-40">
                  {groupTools.map(tool => {
                    const Icon = ICON_MAP[tool.icon]
                    return (
                      <button
                        key={tool.id}
                        onClick={() => { setActiveTool(tool.id); setExpandedGroup(null) }}
                        className={`w-full flex items-center gap-2 px-3 py-1.5 text-xs hover:bg-gray-800 ${
                          activeTool === tool.id ? 'text-blue-400 bg-blue-900/20' : 'text-gray-300'
                        }`}
                      >
                        {Icon && <Icon size={12} />}
                        {tool.label}
                      </button>
                    )
                  })}
                </div>
              )}
            </div>
          )
        })}

        <div className="w-px h-5 bg-gray-600 mx-1" />

        {/* Color Picker */}
        <div className="flex gap-0.5">
          {colors.map(color => (
            <button key={color} onClick={() => setDrawingColor(color)}
              className={`w-5 h-5 rounded-sm border-2 ${drawingColor === color ? 'border-white' : 'border-transparent hover:border-gray-500'}`}
              style={{ backgroundColor: color }}
            />
          ))}
        </div>

        <div className="w-px h-5 bg-gray-600 mx-1" />

        {/* Line Width */}
        <div className="flex gap-0.5">
          {widths.map(w => (
            <button key={w} onClick={() => setDrawingWidth(w)}
              className={`w-6 h-5 rounded flex items-center justify-center ${drawingWidth === w ? 'bg-blue-600' : 'bg-gray-700/50 hover:bg-gray-600'}`}
              title={`${w}px`}
            >
              <div className="bg-white rounded-full" style={{ width: w * 3, height: w }} />
            </button>
          ))}
        </div>

        <div className="w-px h-5 bg-gray-600 mx-1" />

        {/* Panel Toggles */}
        <button onClick={onToggleIndicators}
          className={`flex items-center gap-1 px-2 py-1 rounded text-xs ${showIndicatorPanel ? 'bg-blue-600 text-white' : 'bg-gray-700/50 text-gray-400 hover:bg-gray-600'}`}>
          <BarChart3 size={12} /> Indikatoren
        </button>
        <button onClick={onTogglePatterns}
          className={`flex items-center gap-1 px-2 py-1 rounded text-xs ${showPatternPanel ? 'bg-amber-600 text-white' : 'bg-gray-700/50 text-gray-400 hover:bg-gray-600'}`}>
          <CandlestickChart size={12} /> Muster
        </button>

        <button onClick={onToggleCounterSearch}
          className={`flex items-center gap-1 px-2 py-1 rounded text-xs ${showCounterSearch ? 'bg-green-600 text-white' : 'bg-gray-700/50 text-gray-400 hover:bg-gray-600'}`}>
          <SearchCheck size={12} /> Gegensuche
        </button>

        <button onClick={onToggleSetManager}
          className={`flex items-center gap-1 px-2 py-1 rounded text-xs ${showSetManager ? 'bg-indigo-600 text-white' : 'bg-gray-700/50 text-gray-400 hover:bg-gray-600'}`}>
          <FolderOpen size={12} /> Sets
        </button>

        <button onClick={onToggleAnomalies}
          className={`flex items-center gap-1 px-2 py-1 rounded text-xs ${showAnomalies ? 'bg-amber-600 text-white' : 'bg-gray-700/50 text-gray-400 hover:bg-gray-600'}`}>
          <Sparkles size={12} /> Anomalien
        </button>

        <div className="flex-1" />

        {/* Toggle drawings visibility */}
        <button onClick={onToggleDrawingsVisible}
          className={`flex items-center gap-1 px-2 py-1 rounded text-xs ${drawingsVisible ? 'bg-gray-700/50 text-gray-400' : 'bg-yellow-600/50 text-yellow-300'}`}
          title={drawingsVisible ? 'Zeichnungen ausblenden' : 'Zeichnungen einblenden'}>
          {drawingsVisible ? <Eye size={12} /> : <EyeOff size={12} />}
        </button>

        <button onClick={onClearAll}
          className="flex items-center gap-1 px-2 py-1 bg-gray-700/50 hover:bg-red-600/50 text-gray-400 hover:text-red-300 rounded text-xs">
          <Trash2 size={12} /> Alle loeschen
        </button>
      </div>
    </div>
  )
}
