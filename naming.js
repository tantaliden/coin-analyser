/**
 * NAMING - Single Source of Truth FINAL VERSION
 * 
 * ALLE hardcodierten Werte sind HIER definiert:
 * - Indikator-Felder + Farben
 * - Timeframes und Aggregatoren
 * - Durations für kline_metrics (Backend)
 * - Operator und Referenzen
 * - Spalten für Search/Chart
 * - Farben (32x Overlap + Set-Farb-Optionen)
 * 
 * Alles andere LÄDT von hier!
 * 
 * Updated: 2025-12-24 CASCADE-INDICATORS
 */

const NAMING = {
  // === LANGUAGES ===
  languages: { de: 'Deutsch', en: 'English' },
  defaultLanguage: 'de',

  // === LABELS (DE + EN) ===
  labels: {
    appTitle: { de: 'Analyser', en: 'Analyser' },
    appSubtitle: { de: 'Trading Analysis Platform', en: 'Trading Analysis Platform' },
    login: { de: 'Anmelden', en: 'Login' },
    logout: { de: 'Abmelden', en: 'Logout' },
    save: { de: 'Speichern', en: 'Save' },
    delete: { de: 'Löschen', en: 'Delete' },
    cancel: { de: 'Abbrechen', en: 'Cancel' },
    add: { de: 'Hinzufügen', en: 'Add' },
    search: { de: 'Suchen', en: 'Search' },
    filter: { de: 'Filter', en: 'Filter' },
    moduleSearch: { de: 'Suche', en: 'Search' },
    moduleChart: { de: 'Chart', en: 'Chart' },
    moduleIndicators: { de: 'Indikatoren', en: 'Indicators' },
    moduleSets: { de: 'Indikator-Sets', en: 'Indicator Sets' },
    moduleEvents: { de: 'Events', en: 'Events' },
    moduleBot: { de: 'Trading Bot', en: 'Trading Bot' },
    moduleWallet: { de: 'Wallet', en: 'Wallet' },
    moduleGroups: { de: 'Coin-Gruppen', en: 'Coin Groups' },
  },

  // === DATABASES ===
  databases: {
    coins: {
      name: 'coins',
      tables: {
        klines: 'klines',
        ingestState: 'ingest_state',
        coinInfo: 'coin_info',
      },
      columns: {
        symbol: 'symbol',
        interval: 'interval',
        openTime: 'open_time',
        open: 'open',
        high: 'high',
        low: 'low',
        close: 'close',
        volume: 'volume',
        closeTime: 'close_time',
        trades: 'trades',
        quoteAssetVolume: 'quote_asset_volume',
        takerBuyBase: 'taker_buy_base',
        takerBuyQuote: 'taker_buy_quote',
      },
    },
    app: {
      name: 'analyser_app',
      tables: {
        users: 'users',
        userSessions: 'user_sessions',
        coinGroups: 'coin_groups',
        coinGroupMembers: 'coin_group_members',
        indicatorSets: 'indicator_sets',
        indicatorItems: 'indicator_items',
        indicatorProgress: 'indicator_progress',
        upcomingEvents: 'upcoming_events',
        tradingBotConfig: 'trading_bot_config',
        tradeHistory: 'trade_history',
        moduleLayouts: 'module_layouts',
      },
    },
  },

  // === INDICATOR FIELDS (von ChartModule) ===
  // Struktur: { key, label, color, type: 'base' oder 'calc' }
  indicatorFields: [
    // Base Values (direkt aus klines)
    { key: 'close', label: 'Close', color: '#3b82f6', type: 'base', dbColumn: 'close' },
    { key: 'volume', label: 'Volume', color: '#f59e0b', type: 'base', dbColumn: 'volume' },
    { key: 'trades', label: 'Trades', color: '#22c55e', type: 'base', dbColumn: 'trades' },
    { key: 'quoteVolume', label: 'QuoteVol', color: '#06b6d4', type: 'base', dbColumn: 'quote_asset_volume' },
    { key: 'takerBuyBase', label: 'TakerBase', color: '#8b5cf6', type: 'base', dbColumn: 'taker_buy_base' },
    { key: 'takerBuyQuote', label: 'TakerQuote', color: '#ec4899', type: 'base', dbColumn: 'taker_buy_quote' },
    // Calculated Values (mit gleitendem Durchschnitt)
    { key: 'volatility', label: 'Volatility', color: '#ef4444', type: 'calc' },
    { key: 'body', label: 'Body', color: '#84cc16', type: 'calc' },
    { key: 'bodyPercent', label: 'Body%', color: '#14b8a6', type: 'calc' },
    { key: 'volumeChangeAvg', label: 'VolChg%', color: '#f97316', type: 'calc' },
    { key: 'priceChangeAvg', label: 'PriceChg%', color: '#6366f1', type: 'calc' },
  ],

  // === INDICATOR OPERATIONS ===
  indicatorOperations: ['>', '<', '>=', '<=', 'between'],

  // === INDICATOR CONDITIONS (Referenzen) ===
  indicatorConditionReferences: ['absolute', 'average'],

  // === INDICATOR AGGREGATORS ===
  indicatorAggregators: ['1m', '10m', '15m', '30m', '1h'],

  // === CHART TIMEFRAMES ===
  candleTimeframes: [
    { key: '1m', label: '1m', minutes: 1 },
    { key: '10m', label: '10m', minutes: 10 },
    { key: '15m', label: '15m', minutes: 15 },
    { key: '30m', label: '30m', minutes: 30 },
    { key: '1h', label: '1h', minutes: 60 },
  ],

  // === AVERAGE PERIODS ===
  avgPeriods: [
    { key: 5, label: '5m' },
    { key: 10, label: '10m' },
    { key: 15, label: '15m' },
    { key: 30, label: '30m' },
    { key: 60, label: '60m' },
  ],

  // === KLINE_METRICS DURATIONS ===
  klineMetricsDurations: [30, 60, 90, 120, 180, 240, 300, 330, 360, 420, 480, 540, 600],

  // === KLINE_METRICS METRIC DURATIONS ===
  klineMetricsMetricDurations: [30, 60, 120, 240, 480],

  // === SEARCH RESULT COLUMNS ===
  searchResultColumns: [
    { key: 'symbol', label: 'Symbol', default: true },
    { key: 'event_start', label: 'Start', default: true },
    { key: 'event_end', label: 'Ende', default: false },
    { key: 'start_price', label: 'Start-Preis', default: false },
    { key: 'end_price', label: 'End-Preis', default: false },
    { key: 'change_percent', label: '%', default: true },
    { key: 'duration_minutes', label: 'Dauer (min)', default: true },
    { key: 'volume', label: 'Volume', default: false },
    { key: 'trades', label: 'Trades', default: false },
  ],

  // === COLORS ===
  
  // 32 helle Farben für Overlap-Chart (bis zu 32 Events gleichzeitig)
  // Dark Mode optimiert - KEINE Schwarz/Dunkelgrau
  overlapEventColors: [
    '#FF6B6B', '#FF8E72', '#FFA94D', '#FFB84D', '#FFD93D', '#FFED4E', '#C6FF1D', '#9EFF59',
    '#39FF14', '#2EFF71', '#00FF88', '#00FFB3', '#00FFD9', '#00E5FF', '#00BFFF', '#1E90FF',
    '#3B82F6', '#6366F1', '#8B5CF6', '#A78BFA', '#C084FC', '#D946EF', '#EC4899', '#F472B6',
    '#FB7185', '#FCA5A5', '#FBBF24', '#F59E0B', '#FB923C', '#F97316', '#EA580C', '#DC2626',
  ],

  // Farb-Optionen für Indikator-Sets (User wählt sich eine aus)
  setColorOptions: [
    { key: 'blue', label: 'Blau', color: '#3b82f6' },
    { key: 'green', label: 'Grün', color: '#22c55e' },
    { key: 'red', label: 'Rot', color: '#ef4444' },
    { key: 'yellow', label: 'Gelb', color: '#eab308' },
    { key: 'purple', label: 'Violett', color: '#a855f7' },
    { key: 'pink', label: 'Pink', color: '#ec4899' },
    { key: 'cyan', label: 'Cyan', color: '#06b6d4' },
    { key: 'orange', label: 'Orange', color: '#f97316' },
    { key: 'lime', label: 'Lime', color: '#84cc16' },
    { key: 'teal', label: 'Teal', color: '#14b8a6' },
    { key: 'indigo', label: 'Indigo', color: '#6366f1' },
    { key: 'rose', label: 'Rose', color: '#f43f5e' },
  ],

  // Event-Farben für Overlap-Chart (alte Palette, wird durch overlapEventColors ersetzt)
  eventColors: [
    '#3b82f6', '#22c55e', '#f59e0b', '#a855f7', '#ef4444', '#06b6d4',
    '#f97316', '#ec4899', '#84cc16', '#14b8a6', '#f43f5e', '#6366f1',
    '#78716c', '#0ea5e9', '#d946ef', '#facc15',
  ],

  // Indikator-Farben (Standard-Farbpalette)
  indicatorColors: [
    '#3b82f6', '#22c55e', '#ef4444', '#eab308',
    '#a855f7', '#ec4899', '#14b8a6', '#f97316',
  ],

  // === INDICATOR CHECK STATUS ===
  indicatorCheckStatuses: {
    match: 'match',           // Im erwarteten Range
    overRange: 'over_range',  // Zu aggressiv
    underRange: 'under_range',// Zu konservativ
  },

  // === API ROUTES ===
  api: {
    prefix: '/api/v1',
    routes: {
      health: '/meta/health',
      config: '/meta/config',
      symbols: '/meta/symbols',
      login: '/auth/login',
      refresh: '/auth/refresh',
      password: '/auth/password',
      me: '/auth/me',
      users: '/users',
      events: '/search/events',
      candles: '/search/candles',
      sets: '/indicators/sets',
      items: '/indicators/items',
      upcoming: '/events/upcoming',
      bot: '/trading/bot',
      positions: '/trading/positions',
      wallet: '/wallet',
      balance: '/wallet/balance',
      groups: '/groups',
      layout: '/layout',
    },
  },

  // === ROLES ===
  roles: { admin: 'admin', trader: 'trader' },

  // === STATUS ===
  eventStatus: { waiting: 'waiting', active: 'active', expired: 'expired', completed: 'completed' },
  tradeStatus: { open: 'open', tpHit: 'tp_hit', slHit: 'sl_hit', manualClose: 'manual_close' },

  // === TIMEFRAMES ===
  timeframes: {
    minutes: [1, 2, 5, 10, 15, 30],
    hours: [1, 2, 4, 6, 12],
    days: [1, 3, 7, 14, 30],
    primary: '1m',
    chartOptions: ['1m', '10m', '15m', '30m', '1h'],
  },

  // === KLINE FIELDS ===
  klineFields: [
    { key: 'open', dbColumn: 'open', labelKey: 'colOpen' },
    { key: 'high', dbColumn: 'high', labelKey: 'colHigh' },
    { key: 'low', dbColumn: 'low', labelKey: 'colLow' },
    { key: 'close', dbColumn: 'close', labelKey: 'colClose' },
    { key: 'volume', dbColumn: 'volume', labelKey: 'colVolume' },
    { key: 'trades', dbColumn: 'trades', labelKey: 'colTrades' },
    { key: 'quoteAssetVolume', dbColumn: 'quote_asset_volume', labelKey: 'colQuoteVolume' },
    { key: 'takerBuyBase', dbColumn: 'taker_buy_base', labelKey: 'colTakerBuyBase' },
    { key: 'takerBuyQuote', dbColumn: 'taker_buy_quote', labelKey: 'colTakerBuyQuote' },
  ],

  // === BINANCE ===
  binance: {
    quoteAsset: 'USDT',
    liveApiUrl: 'https://api.binance.com',
    visionBaseUrl: 'https://data.binance.vision/data/spot',
  },

  // === ERRORS ===
  errors: {
    unauthorized: 'ERR_UNAUTHORIZED',
    forbidden: 'ERR_FORBIDDEN',
    notFound: 'ERR_NOT_FOUND',
    invalidCredentials: 'ERR_INVALID_CREDENTIALS',
    tokenExpired: 'ERR_TOKEN_EXPIRED',
    dbConnectionFailed: 'ERR_DB_CONNECTION_FAILED',
    binanceApiFailed: 'ERR_BINANCE_API_FAILED',
  },

  // === HELPERS ===
  getIndicatorField: function(key) {
    return this.indicatorFields.find(f => f.key === key);
  },

  getIndicatorFieldsByType: function(type) {
    return this.indicatorFields.filter(f => f.type === type);
  },

  getCandleTimeframe: function(key) {
    return this.candleTimeframes.find(tf => tf.key === key);
  },

  getAvgPeriod: function(key) {
    return this.avgPeriods.find(p => p.key === key);
  },

  getSearchResultColumn: function(key) {
    return this.searchResultColumns.find(c => c.key === key);
  },

  getSetColorOption: function(key) {
    return this.setColorOptions.find(c => c.key === key);
  },

  getOverlapEventColor: function(index) {
    return this.overlapEventColors[index % this.overlapEventColors.length];
  },
};

module.exports = NAMING;
