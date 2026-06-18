import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react()
  ],
  build: {
    rollupOptions: {
      output: {
        // Long-lived vendor chunks (separate hashes from app code) so a
        // content edit doesn't bust react/icons in the browser cache. Both
        // pages use react + lucide, so these load on either route; the blog
        // and chat *page* code stay in their own lazy chunks (see App.jsx).
        // Function form (not the {name: [...]} map) so react-dom — whose path
        // contains "react" — lands in react-vendor instead of the entry chunk.
        manualChunks(id) {
          if (!id.includes('node_modules')) return;
          if (id.includes('lucide-react')) return 'icons';
          if (
            id.includes('node_modules/react-dom') ||
            id.includes('node_modules/react/') ||
            id.includes('node_modules/scheduler')
          ) return 'react-vendor';
        },
      },
    },
  },
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