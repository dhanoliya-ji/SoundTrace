import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// In dev, /api is proxied to the FastAPI server; in production FastAPI serves dist/ itself.
export default defineConfig({ plugins: [react()], server: { proxy: { '/api': 'http://127.0.0.1:8000' } } });
