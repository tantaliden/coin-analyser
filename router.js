/**
 * ROUTER - Single Point of Entry
 * Lädt alle Module, verbindet NAMING + SETTINGS
 */

const path = require('path');
const NAMING = require('./naming');
const SETTINGS = require('./settings.json');

const ROOT = __dirname;

const PATHS = {
  root: ROOT,
  backend: path.join(ROOT, 'backend'),
  frontend: path.join(ROOT, 'frontend'),
  logs: path.join(ROOT, 'logs')
};

// Module-Loader
const modules = {
  auth: () => require('./backend/auth'),
  search: () => require('./backend/search'),
  candles: () => require('./backend/candles'),
  indicators: () => require('./backend/indicators'),
  meta: () => require('./backend/meta'),
  shared: () => require('./backend/shared')
};

const getModule = (name) => {
  if (!modules[name]) {
    throw new Error(`Module "${name}" not found in router`);
  }
  return modules[name]();
};

module.exports = {
  NAMING,
  SETTINGS,
  PATHS,
  getModule
};
