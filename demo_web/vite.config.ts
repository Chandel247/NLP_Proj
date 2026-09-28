import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// In development the API runs separately (python src/demo/server.py); in the
// demo itself FastAPI serves the built dist/ and this proxy is not involved.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://127.0.0.1:8000' } },
})
