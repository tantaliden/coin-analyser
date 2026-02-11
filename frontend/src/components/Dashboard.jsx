import { useCallback } from 'react'
import { Responsive, WidthProvider } from 'react-grid-layout'
import 'react-grid-layout/css/styles.css'
import 'react-resizable/css/styles.css'
import Taskbar from './Taskbar'
import ModuleWrapper from './ModuleWrapper'
import { useModuleStore } from '../stores/moduleStore'

// Alle Module importieren
import SearchModule from '../modules/SearchModule'
import SearchResultsModule from '../modules/SearchResultsModule'
import ChartModule from '../modules/ChartModule'
import IndicatorsModule from '../modules/indicators/IndicatorsModule'
import GroupsModule from '../modules/groups/GroupsModule'
import WalletModule from '../modules/wallet/WalletModule'
import BotModule from '../modules/bot/BotModule'

const ResponsiveGridLayout = WidthProvider(Responsive)

const MODULE_COMPONENTS = {
  search: SearchModule,
  searchResults: SearchResultsModule,
  chart: ChartModule,
  indicators: IndicatorsModule,
  groups: GroupsModule,
  wallet: WalletModule,
  bot: BotModule,
}

const MODULE_TITLES = {
  search: 'Suche',
  searchResults: 'Suchergebnisse',
  chart: 'Chart',
  indicators: 'Indikatoren',
  groups: 'Coin-Gruppen',
  wallet: 'Wallet',
  bot: 'Trading Bot',
}

export default function Dashboard() {
  const { activeModules, currentLayout, setCurrentLayout } = useModuleStore()

  const handleLayoutChange = useCallback((_, allLayouts) => {
    if (allLayouts.lg) {
      setCurrentLayout(allLayouts)
    }
  }, [setCurrentLayout])

  const filteredLayouts = {
    lg: (currentLayout.lg || []).filter(item => activeModules.includes(item.i))
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Taskbar />
      <main className="flex-1 p-2 overflow-auto">
        <ResponsiveGridLayout
          className="layout"
          layouts={filteredLayouts}
          breakpoints={{ lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 }}
          cols={{ lg: 12, md: 10, sm: 6, xs: 4, xxs: 2 }}
          rowHeight={30}
          onLayoutChange={handleLayoutChange}
          draggableHandle=".module-header"
          margin={[8, 8]}
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
