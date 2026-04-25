import { useCallback } from 'react'
import { Responsive, WidthProvider } from 'react-grid-layout'
import 'react-grid-layout/css/styles.css'
import 'react-resizable/css/styles.css'
import Taskbar from './Taskbar'
import ModuleWrapper from './ModuleWrapper'
import { useModuleStore } from '../stores/moduleStore'

import SearchModule from '../modules/search/SearchModule'
import SearchResultsModule from '../modules/results/SearchResultsModule'
import ChartViewModule from '../modules/chartview/ChartViewModule'
import IndicatorsModule from '../modules/indicators/IndicatorsModule'
import GroupsModule from '../modules/groups/GroupsModule'
import WalletModule from '../modules/wallet/WalletModule'
import BotModule from '../modules/bot/BotModule'
import MomentumModule from '../modules/MomentumModule'
import RLAgentModule from '../modules/RLAgentModule'

const ResponsiveGridLayout = WidthProvider(Responsive)

const MODULE_COMPONENTS = {
  search: SearchModule,
  searchResults: SearchResultsModule,
  chart: ChartViewModule,
  indicators: IndicatorsModule,
  groups: GroupsModule,
  wallet: WalletModule,
  bot: BotModule,
  momentum: MomentumModule,
  rlagent: RLAgentModule,
}

const MODULE_TITLES = {
  search: 'Suche',
  searchResults: 'Suchergebnisse',
  chart: 'Chart',
  indicators: 'Indikatoren',
  groups: 'Coin-Gruppen',
  wallet: 'Wallet',
  bot: 'Trading Bot',
  momentum: 'Momentum Scanner',
  rlagent: 'RL-Agent',
}

export default function Dashboard() {
  const { activeModules, currentLayout, setCurrentLayout, isLocked } = useModuleStore()

  const handleLayoutChange = useCallback((layout, allLayouts) => {
    if (!isLocked && allLayouts.lg) {
      setCurrentLayout(allLayouts)
    }
  }, [setCurrentLayout, isLocked])

  const filteredLayouts = {
    lg: (currentLayout.lg || []).filter(item => activeModules.includes(item.i))
  }

  return (
    <div className="h-screen flex flex-col">
      <Taskbar />
      <main className="flex-1 overflow-auto">
        <ResponsiveGridLayout
          className="layout"
          layouts={filteredLayouts}
          breakpoints={{ lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 }}
          cols={{ lg: 12, md: 10, sm: 6, xs: 4, xxs: 2 }}
          rowHeight={30}
          onLayoutChange={handleLayoutChange}
          draggableHandle=".module-header"
          margin={[0, 0]}
          containerPadding={[0, 0]}
          isDraggable={!isLocked}
          isResizable={!isLocked}
          resizeHandles={['se', 'sw', 'ne', 'nw', 'e', 'w', 'n', 's']}
        >
          {activeModules.map(moduleId => {
            const Component = MODULE_COMPONENTS[moduleId]
            if (!Component) return null
            return (
              <div key={moduleId}>
                <ModuleWrapper moduleId={moduleId} title={MODULE_TITLES[moduleId] || moduleId}>
                  <Component />
                </ModuleWrapper>
              </div>
            )
          })}
        </ResponsiveGridLayout>
      </main>
    </div>
  )
}
