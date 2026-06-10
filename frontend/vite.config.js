import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react()
  ],
  server: {
    port: 5173,
    proxy: {
      // Proxy POST /chat (the API) to FastAPI; GET /chat is the SPA chat page
      '/chat': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        bypass(req) {
          if (req.method === 'GET' && req.headers.accept?.includes('text/html')) {
            return '/index.html'
          }
        },
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/reset': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  }
})