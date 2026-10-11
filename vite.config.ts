import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { axomApiPlugin } from './server/vite-api.js';

export default defineConfig({
  plugins: [react(), axomApiPlugin()],
  server: {
    host: '0.0.0.0',
    port: 5173,
  },
  preview: {
    host: '0.0.0.0',
    port: 4173,
  },
});
