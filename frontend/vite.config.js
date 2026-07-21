import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    chunkSizeWarningLimit: 800,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('recharts') || id.includes('d3-') || id.includes('victory')) {
            return 'recharts-vendor';
          }
          if (id.includes('react-dom') || id.includes('react-router')) {
            return 'react-vendor';
          }
        },
      },
    },
  },
})


