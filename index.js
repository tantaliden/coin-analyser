/**
 * INDEX - Entry Point
 * Startet die Anwendung über Router
 */

const { NAMING, SETTINGS, PATHS } = require('./router');

async function main() {
  console.log('========================================');
  console.log(`${SETTINGS.app.name} v${SETTINGS.app.version}`);
  console.log('========================================');
  console.log(`Environment: ${SETTINGS.app.environment}`);
  console.log(`API Port: ${SETTINGS.server.api.port}`);
  console.log(`Frontend Port: ${SETTINGS.server.frontend.port}`);
  console.log('');
  console.log('4-File Architecture loaded:');
  console.log('  - naming.js (Single Source of Truth)');
  console.log('  - router.js (Module Loader)');
  console.log('  - settings.json (Configuration)');
  console.log('  - index.js (Entry Point)');
  console.log('========================================');
}

if (require.main === module) {
  main().catch(console.error);
}

module.exports = main;
