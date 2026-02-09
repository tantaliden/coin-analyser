import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 3002,
    allowedHosts: ['tresor.tantaliden.com', 'localhost', '217.154.192.66'],
    proxy: { '/api': { target: 'http://localhost:8002', changeOrigin: true } }
  },
  build: { outDir: 'dist', sourcemap: false }
})
