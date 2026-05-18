import { getEventColor } from '../../config/chartSettings'
import ChartPanel from './ChartPanel'

export default function ChartGrid({
  displayEvents, chartData, mode, chartType, showVolume,
  activeTool, drawingColor, drawingWidth,
  drawings, onAddDrawing, onRemoveDrawing,
  activeIndicators, activeChartId, setActiveChartId,
  drawingsVisible, loading,
  hoveredPattern, onDoubleClick,
  syncedHoverTime, onSyncHover,
}) {
  if (loading) {
    return <div className="text-gray-400 text-center py-8">Laden...</div>
  }

  const gridClass = mode === 'grid'
    ? 'grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-2'
    : 'space-y-2'

  return (
    <div className={gridClass}>
      {displayEvents.map((event, idx) => {
        const data = chartData[event.id]
        if (!data) return null
        return (
          <ChartPanel
            key={event.id}
            event={event} data={data} eventIndex={idx}
            eventColor={getEventColor(idx)}
            chartType={chartType} showVolume={showVolume}
            isGridMode={mode === 'grid'}
            isActive={activeChartId === event.id}
            onClick={() => setActiveChartId(activeChartId === event.id ? null : event.id)}
            onDoubleClick={() => onDoubleClick(event.id)}
            activeTool={activeTool} drawingColor={drawingColor} drawingWidth={drawingWidth}
            drawings={drawings[event.id]}
            onAddDrawing={(d) => onAddDrawing(event.id, d)}
            onRemoveDrawing={(i) => onRemoveDrawing(event.id, i)}
            activeIndicators={activeIndicators}
            drawingsVisible={drawingsVisible}
            hoveredPattern={hoveredPattern}
            syncedHoverTime={syncedHoverTime}
            onSyncHover={onSyncHover}
          />
        )
      })}
    </div>
  )
}
