// Zentrale Chart-Einstellungen - EINE Datei fuer alles Konfigurierbare

export const CHART_SETTINGS = {
  colors: {
    up: '#22c55e',
    down: '#ef4444',
    neutral: '#6b7280',
    grid: '#374151',
    text: '#9ca3af',
    background: 'transparent',
    crosshair: '#9ca3af',
    volumeUp: 'rgba(34,197,94,0.3)',
    volumeDown: 'rgba(239,68,68,0.3)',
  },

  eventColors: [
    '#3b82f6', '#22c55e', '#f59e0b', '#a855f7', '#ef4444',
    '#06b6d4', '#f97316', '#ec4899', '#84cc16', '#14b8a6',
    '#f43f5e', '#6366f1', '#78716c', '#0ea5e9', '#d946ef',
  ],

  timeframes: [
    { key: '1m', label: '1m', minutes: 1 },
    { key: '5m', label: '5m', minutes: 5 },
    { key: '15m', label: '15m', minutes: 15 },
    { key: '30m', label: '30m', minutes: 30 },
    { key: '1h', label: '1h', minutes: 60 },
    { key: '4h', label: '4h', minutes: 240 },
  ],

  drawing: {
    defaultColor: '#3b82f6',
    defaultLineWidth: 2,
    colors: ['#3b82f6', '#22c55e', '#ef4444', '#f59e0b', '#a855f7', '#ec4899', '#06b6d4', '#ffffff'],
    lineWidths: [1, 2, 3, 4],
    tools: [
      // Basis
      { id: 'cursor', label: 'Auswahl', icon: 'MousePointer2', group: 'basic' },
      { id: 'crosshair', label: 'Fadenkreuz', icon: 'Crosshair', group: 'basic' },

      // Linien
      { id: 'trendline', label: 'Trendlinie', icon: 'TrendingUp', group: 'lines' },
      { id: 'ray', label: 'Strahl', icon: 'MoveUpRight', group: 'lines' },
      { id: 'hline', label: 'Horizontal', icon: 'Minus', group: 'lines' },
      { id: 'vline', label: 'Vertikal', icon: 'SeparatorVertical', group: 'lines' },
      { id: 'channel', label: 'Kanal', icon: 'Rows3', group: 'lines' },

      // Bereiche & Formen
      { id: 'rect', label: 'Rechteck', icon: 'Square', group: 'shapes' },
      { id: 'priceRange', label: 'Preisbereich', icon: 'ArrowUpDown', group: 'shapes' },
      { id: 'timeRange', label: 'Zeitbereich', icon: 'Timer', group: 'shapes' },

      // Fibonacci & Messung
      { id: 'fibonacci', label: 'Fibonacci Retr.', icon: 'GitBranch', group: 'fib' },
      { id: 'fibExtension', label: 'Fibonacci Ext.', icon: 'GitFork', group: 'fib' },
      { id: 'measure', label: 'Messen (% + Zeit)', icon: 'Ruler', group: 'fib' },

      // Candle-Tools
      { id: 'candleHighlight', label: 'Candle markieren', icon: 'Highlighter', group: 'candle' },
      { id: 'candleHigh', label: 'Candle High', icon: 'ChevronsUp', group: 'candle' },
      { id: 'candleLow', label: 'Candle Low', icon: 'ChevronsDown', group: 'candle' },
      { id: 'candleRange', label: 'Candle-zu-Candle', icon: 'Spline', group: 'candle' },
      { id: 'wickToWick', label: 'Docht-zu-Docht', icon: 'Scaling', group: 'candle' },

      // Marker
      { id: 'initialPoint', label: 'Initialpunkt (Zeit-Anker)', icon: 'Star', group: 'marker' },
      { id: 'buyMarker', label: 'Buy Signal', icon: 'ArrowUpCircle', group: 'marker' },
      { id: 'sellMarker', label: 'Sell Signal', icon: 'ArrowDownCircle', group: 'marker' },
      { id: 'flag', label: 'Flagge', icon: 'Flag', group: 'marker' },
      { id: 'alert', label: 'Alert-Punkt', icon: 'AlertTriangle', group: 'marker' },
      { id: 'text', label: 'Text', icon: 'Type', group: 'marker' },

      // Loeschen
      { id: 'eraser', label: 'Einzeln loeschen', icon: 'Eraser', group: 'delete' },
    ],
    toolGroups: [
      { id: 'basic', label: 'Basis' },
      { id: 'lines', label: 'Linien' },
      { id: 'shapes', label: 'Bereiche' },
      { id: 'fib', label: 'Fibonacci' },
      { id: 'candle', label: 'Kerzen' },
      { id: 'marker', label: 'Marker' },
      { id: 'delete', label: 'Loeschen' },
    ],
    fibonacci: {
      levels: [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1],
      colors: ['#ef4444', '#f59e0b', '#22c55e', '#3b82f6', '#22c55e', '#f59e0b', '#ef4444'],
    },
    fibExtension: {
      levels: [0, 0.618, 1, 1.272, 1.618, 2, 2.618],
      colors: ['#6b7280', '#f59e0b', '#3b82f6', '#22c55e', '#a855f7', '#ec4899', '#ef4444'],
    },
  },

  indicators: {
    available: [
      { id: 'sma', label: 'SMA', params: [{ key: 'period', label: 'Periode', default: 20, min: 1, max: 500 }] },
      { id: 'ema', label: 'EMA', params: [{ key: 'period', label: 'Periode', default: 20, min: 1, max: 500 }] },
      { id: 'rsi', label: 'RSI', params: [{ key: 'period', label: 'Periode', default: 14, min: 1, max: 100 }], separate: true,
        thresholds: [{ value: 70, color: '#ef4444', label: 'Overbought' }, { value: 30, color: '#22c55e', label: 'Oversold' }] },
      { id: 'macd', label: 'MACD', params: [
        { key: 'fast', label: 'Fast', default: 12, min: 1, max: 100 },
        { key: 'slow', label: 'Slow', default: 26, min: 1, max: 200 },
        { key: 'signal', label: 'Signal', default: 9, min: 1, max: 50 },
      ], separate: true },
      { id: 'bollinger', label: 'Bollinger Bands', params: [
        { key: 'period', label: 'Periode', default: 20, min: 1, max: 200 },
        { key: 'stddev', label: 'StdDev', default: 2, min: 0.5, max: 5, step: 0.5 },
      ] },
      { id: 'vwap', label: 'VWAP', params: [] },
      { id: 'atr', label: 'ATR', params: [{ key: 'period', label: 'Periode', default: 14, min: 1, max: 100 }], separate: true },
      { id: 'stochastic', label: 'Stochastic', params: [
        { key: 'kPeriod', label: '%K', default: 14, min: 1, max: 100 },
        { key: 'dPeriod', label: '%D', default: 3, min: 1, max: 50 },
      ], separate: true, thresholds: [{ value: 80, color: '#ef4444' }, { value: 20, color: '#22c55e' }] },
      { id: 'volume', label: 'Volume', params: [], separate: true },
      { id: 'volume_sma', label: 'Volume SMA', params: [{ key: 'period', label: 'Periode', default: 20, min: 1, max: 200 }], separate: true },
      { id: 'trades', label: 'Trades', params: [], separate: true },
    ],
    maxOverlays: 10,
    defaultColors: ['#f59e0b', '#a855f7', '#06b6d4', '#ec4899', '#84cc16'],
  },

  candlePatterns: {
    available: [
      { id: 'doji', label: 'Doji', description: 'Open = Close, Unsicherheit' },
      { id: 'hammer', label: 'Hammer', description: 'Kleiner Body oben, langer unterer Docht' },
      { id: 'inverted_hammer', label: 'Inv. Hammer', description: 'Kleiner Body unten, langer oberer Docht' },
      { id: 'engulfing_bull', label: 'Bullish Engulfing', description: 'Gruene Kerze verschlingt vorherige rote' },
      { id: 'engulfing_bear', label: 'Bearish Engulfing', description: 'Rote Kerze verschlingt vorherige gruene' },
      { id: 'morning_star', label: 'Morning Star', description: '3-Kerzen Umkehrmuster (bullish)' },
      { id: 'evening_star', label: 'Evening Star', description: '3-Kerzen Umkehrmuster (bearish)' },
      { id: 'three_white', label: 'Three White Soldiers', description: '3 steigende gruene Kerzen' },
      { id: 'three_black', label: 'Three Black Crows', description: '3 fallende rote Kerzen' },
      { id: 'harami_bull', label: 'Bullish Harami', description: 'Kleine gruene Kerze in grosser roter' },
      { id: 'harami_bear', label: 'Bearish Harami', description: 'Kleine rote Kerze in grosser gruener' },
      { id: 'spinning_top', label: 'Spinning Top', description: 'Kleiner Body, lange Dochte beidseitig' },
      { id: 'marubozu_bull', label: 'Bull. Marubozu', description: 'Nur Body, keine Dochte (bullish)' },
      { id: 'marubozu_bear', label: 'Bear. Marubozu', description: 'Nur Body, keine Dochte (bearish)' },
      { id: 'tweezer_top', label: 'Tweezer Top', description: 'Zwei Kerzen mit gleichem High' },
      { id: 'tweezer_bottom', label: 'Tweezer Bottom', description: 'Zwei Kerzen mit gleichem Low' },
      { id: 'piercing', label: 'Piercing Line', description: 'Gruene Kerze schliesst ueber Mitte der vorherigen roten' },
      { id: 'dark_cloud', label: 'Dark Cloud Cover', description: 'Rote Kerze schliesst unter Mitte der vorherigen gruenen' },
    ],
    thresholds: {
      dojiBodyRatio: 0.05,
      hammerWickRatio: 2.0,
      engulfingMinBody: 0.01,
      marubozuWickRatio: 0.02,
      tweezerTolerance: 0.001,
    },
  },

  fuzzy: {
    global: { valueTolerance: 10, timeTolerance: 10, slopeTolerance: 15, ratioTolerance: 10 },
    min: { valueTolerance: 0, timeTolerance: 0, slopeTolerance: 0, ratioTolerance: 0 },
    max: { valueTolerance: 50, timeTolerance: 60, slopeTolerance: 50, ratioTolerance: 50 },
  },

  display: {
    maxCharts: 32,
    defaultGridCols: { sm: 1, md: 2, lg: 3, xl: 4 },
    defaultChartHeight: 300,
    gridChartHeight: 200,
    minChartHeight: 100,
    maxChartHeight: 600,
  },

  counterSearch: {
    defaultPeriodDays: 30,
    maxPeriodDays: 365,
    minMatchScore: 0.6,
  },
}

export const getEventColor = (index) =>
  CHART_SETTINGS.eventColors[index % CHART_SETTINGS.eventColors.length]

export const getTimeframeMinutes = (key) =>
  CHART_SETTINGS.timeframes.find(t => t.key === key)?.minutes || 1

// Marker-Farben
export const MARKER_COLORS = {
  buy: CHART_SETTINGS.colors.up,
  sell: CHART_SETTINGS.colors.down,
  alert: '#f59e0b',
  eventStart: '#3b82f680',
  patternHighlight: 'rgba(245,158,11,0.15)',
  patternMarker: '#f59e0b',
  crosshairBg: '#1e293b',
  filterBorderMatch: '#ffffff',
  filterBorderMiss: CHART_SETTINGS.colors.down,
  supportLine: CHART_SETTINGS.colors.up,
  resistanceLine: CHART_SETTINGS.colors.down,
  textOnColor: '#ffffff',
  textOnAlert: '#000000',
  axisLabel: '#6b7280',
  indicatorDefault: '#f59e0b',
  thresholdDefault: '#6b7280',
}


// Candle-Farb-Mapping (green/red/grey -> Hex)
export const CANDLE_COLOR_MAP = {
  green: CHART_SETTINGS.colors.up,
  red: CHART_SETTINGS.colors.down,
  grey: CHART_SETTINGS.colors.neutral,
}

// Gegensuche Einstellungen
export const COUNTER_SEARCH_SETTINGS = {
  defaultPeriodDays: 30,
  maxPeriodDays: 365,
  periodOptions: [7, 14, 30, 60, 90, 180, 365],
  minMatchScore: 0.6,
}

// Unscharfe-Defaults
export const FUZZY_DEFAULTS = {
  global: {
    valueTolerance: 10,   // +/- 10% um den Zielwert
    timeTolerance: 10,    // +/- 10 Minuten um den Zeitpunkt
    slopeTolerance: 15,   // +/- 15% Abweichung des Anstiegs
    ratioTolerance: 10,   // +/- 10% Abweichung des Verhaeltnisses
    useRange: false,      // absoluter Bereich statt Toleranz
    rangeMin: null,       // untere Grenze (wenn useRange)
    rangeMax: null,       // obere Grenze (wenn useRange)
  },
  ranges: {
    valueTolerance: { min: 0, max: 50, step: 1, label: 'Wert-Toleranz', unit: '%' },
    timeTolerance: { min: 0, max: 60, step: 1, label: 'Zeit-Toleranz', unit: 'm' },
    slopeTolerance: { min: 0, max: 50, step: 1, label: 'Anstieg-Toleranz', unit: '%' },
    ratioTolerance: { min: 0, max: 50, step: 1, label: 'Verhaeltnis-Toleranz', unit: '%' },
  },
}

// Standard-Indikatoren die beim Start geladen werden koennen
export const DEFAULT_INDICATORS = [
  { type: 'sma', label: 'SMA 200', params: { period: 200 }, visible: false },
  { type: 'sma', label: 'SMA 50', params: { period: 50 }, visible: false },
  { type: 'ema', label: 'EMA 20', params: { period: 20 }, visible: false },
  { type: 'ema', label: 'EMA 9', params: { period: 9 }, visible: false },
  { type: 'rsi', label: 'RSI 14', params: { period: 14 }, visible: false },
  { type: 'macd', label: 'MACD', params: { fast: 12, slow: 26, signal: 9 }, visible: false },
  { type: 'bollinger', label: 'Bollinger', params: { period: 20, stddev: 2 }, visible: false },
  { type: 'vwap', label: 'VWAP', params: {}, visible: false },
]

// Chart Layout-Konstanten (Pixel)
export const CHART_LAYOUT = {
  timeAxisH: 22,          // Hoehe der Zeitachse unten
  priceAxisW: 70,         // Breite der Preisachse rechts
  paneHMax: 60,           // Max Hoehe eines Sub-Panes
  paneHRatio: 0.12,       // Pane-Hoehe als Anteil der Restflaeche
  paneGridOpacity: '20',  // hex opacity fuer Sub-Pane Hintergrund
  paneLabelPadding: 4,
  priceGridCount: 6,      // Horizontale Gridlinien
  timeGridDiv: 8,         // Vertikale Gridlinien (candles/div)
  candleBodyRatio: 0.7,   // Body-Breite relativ zur Candle-Step-Breite
  pricePadding: 0.05,     // 5% Padding ueber/unter Preisbereich
  crosshairHeaderW: 360,  // OHLCV Header Breite
  crosshairHeaderH: 16,
}

// Marker-Dimensionen (Pixel)
export const MARKER_SIZES = {
  buySellTriangle: 8,     // Halbe Basis-Breite
  buySellHeight: 14,      // Hoehe Dreieck
  buySellOffset: 22,      // Abstand vom Candle bis Dreieck-Spitze
  buySellLabelOffset: 33,
  flagWidth: 14,
  flagHeight: 20,
  alertSize: 12,
  eraserThreshold: 25,    // Pixel-Radius fuer Eraser-Hit-Detection
}


// Schriften fuer Canvas-Rendering
export const CHART_FONTS = {
  axis: '10px Inter',
  axisSmall: '9px Inter',
  axisTiny: '8px Inter',
  indicatorLabel: '9px Inter',
  priceLabel: '10px Inter',
  measureLabel: '11px Inter',
  markerLabel: 'bold 9px Inter',
  alertLabel: 'bold 10px Inter',
  text: '12px Inter',
  supportLabel: 'bold 9px Inter',
  ohlcHeader: '10px Inter',
}


// Match-Modi fuer die Gegensuche mit mehreren Kriterien
export const MATCH_MODES = [
  { id: 'all', label: 'Alle (AND)', description: 'Jedes Kriterium muss erfuellt sein' },
  { id: 'atleast', label: 'Mindestens N von M', description: 'Event gilt wenn N der M Kriterien matchen' },
  { id: 'sequence', label: 'In Reihenfolge', description: 'Kriterien muessen in der angegebenen Zeit-Reihenfolge auftreten' },
]

export const MATCH_MODE_DEFAULT = 'all'

// Bereich-Operatoren fuer Range-Kriterien
export const RANGE_OPS = [
  { id: 'between', label: 'zwischen', needsMin: true, needsMax: true },
  { id: 'gt', label: 'groesser als', needsMin: true, needsMax: false },
  { id: 'lt', label: 'kleiner als', needsMin: false, needsMax: true },
]

// Live-Feedback Einstellungen
export const LIVE_FEEDBACK = {
  enabled: true,
  debounceMs: 600,       // Verzoegerung nach letzter Zeichnung
  maxCriteria: 10,        // Ab zu vielen Kriterien keine Live-Suche mehr
}

// Defaults fuer das Speichern von Indikator-Sets
export const SAVE_SET_SETTINGS = {
  defaultTargetPercent: 5.0,
  defaultDurationMinutes: 120,
  defaultDirection: 'up',
  defaultPrehistoryMinutes: 720,
}

// Fuzzy-Editor Einstellungen (fuer nachtraegliche Unschaerfe-Anpassung)
export const FUZZY_EDITOR_SETTINGS = {
  valueTolerance: { min: 0, max: 50, step: 1, label: 'Wert', unit: '%' },
  timeTolerance: { min: 0, max: 60, step: 1, label: 'Zeit', unit: 'min' },
  slopeTolerance: { min: 0, max: 50, step: 1, label: 'Anstieg', unit: '%' },
  ratioTolerance: { min: 0, max: 50, step: 1, label: 'Verhaeltnis', unit: '%' },
}


// Prae-Event Zeit-Fenster (Minuten vor Initialpunkt)
export const PRE_EVENT_WINDOW_SETTINGS = {
  defaultFromMinutes: 30,
  defaultToMinutes: 120,
  maxMinutes: 1440,
  stepMinutes: 5,
}

// Initialpunkt Settings
export const INITIAL_POINT_DEFAULTS = {
  windowMinutes: 30,
  matchMode: 'all',
  enforceSequence: false,
  fixedOffsetForFirst: true,
}


// Scan-UI (Backsearch) Einstellungen
export const SCAN_UI_SETTINGS = {
  defaultPeriodDays: 30,
  periodOptions: [7, 14, 30, 60, 90, 180, 365],
}


// Auto-Erkennung UI
export const ANOMALY_UI_SETTINGS = {
  description: 'Rolling-z-score ueber Volume, Trades, Body, Range, Wicks, Close-Delta',
  strongZ: 3.0,
  moderateZ: 2.0,
}
