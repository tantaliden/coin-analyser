/**
 * NAMING - Single Source of Truth
 * Alle Namen, Konstanten, Tabellen hier definiert
 */

const NAMING = {
  // Datenbank-Tabellen (coins DB)
  tables: {
    klines: 'klines',
    klineMetrics: 'kline_metrics',
    ingestState: 'ingest_state'
  },

  // Datenbank-Tabellen (analyser_app DB)
  appTables: {
    users: 'users',
    indicators: 'indicators',
    indicatorItems: 'indicator_items',
    indicatorSets: 'indicator_sets',
    events: 'events'
  },

  // Spalten
  columns: {
    klines: {
      symbol: 'symbol',
      interval: 'interval',
      openTime: 'open_time',
      open: 'open',
      high: 'high',
      low: 'low',
      close: 'close',
      volume: 'volume',
      closeTime: 'close_time',
      trades: 'trades'
    }
  },

  // Aggregates
  aggregates: {
    minutes: ['agg_2m', 'agg_5m', 'agg_10m', 'agg_15m', 'agg_30m'],
    hours: ['agg_1h', 'agg_2h', 'agg_4h', 'agg_6h', 'agg_8h', 'agg_12h'],
    days: ['agg_1d', 'agg_2d', 'agg_3d', 'agg_7d', 'agg_15d', 'agg_30d'],
    months: ['agg_1M', 'agg_2M', 'agg_3M', 'agg_6M'],
    years: ['agg_1y', 'agg_2y', 'agg_3y']
  },

  // API Routes
  api: {
    prefix: '/api/v1',
    routes: {
      auth: '/auth',
      users: '/users',
      search: '/search',
      candles: '/candles',
      indicators: '/indicators',
      meta: '/meta'
    }
  },

  // Fehler-Codes
  errors: {
    AUTH_FAILED: 'AUTH_FAILED',
    TOKEN_EXPIRED: 'TOKEN_EXPIRED',
    NOT_FOUND: 'NOT_FOUND',
    VALIDATION_ERROR: 'VALIDATION_ERROR',
    DB_ERROR: 'DB_ERROR'
  }
};

module.exports = NAMING;
